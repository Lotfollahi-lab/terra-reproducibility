"""
This script checks, on a GPU of the farm, the attention of TERRA's context encoder for a sample whose context is
padding only (every key of its attention rows masked): whether its outputs and the gradients are finite with
meta.zero_fully_masked_attention_rows false and true. Such samples occur in training (e.g. index-cell-only batches with
an index cell of at most 2 detected genes at mask ratio 0.6 and a padded or no metatoken). With false, the result
depends on the SDPA kernel, i.e. on the torch version and the GPU (zeros, or NaN that spread to the loss and, through
the gradients, to all weights); with true, TERRA outputs zeros for such rows and leaves all other rows unchanged.

The encoder is built from a run config with the constructor that training uses (`init_model`, through
`init_model_from_config`, which maps the config keys as training does), with the run's attention backend and bfloat16
autocast (meta.use_bfloat16) on a GPU. The batch comes from the run's tokenized dataset through the dataset class and
the mask collator of training (padded masks, `mask.pad_masks_to_batch_max`); then every context entry of sample 0 is
marked as padding, so its key mask is all False. Both settings use the same weights and batch. Reports the torch
version, the GPU and, per setting, whether the encoder outputs of the padded sample and of the other samples are finite
(the padded sample's outputs are not zero: its zero attention rows still pass through the residual and MLP blocks) and
whether all encoder gradients are finite; and whether the other samples' outputs are bit-identical between the
settings. Exits with an error unless true gives finite outputs and gradients and leaves the other samples
unchanged.

Usage (a 1-GPU job on the queue of the training runs, so that the GPU type matches; after tokenization, with any run
of the campaign, by default base_rep1, whose config is read from the manifest; `source datasets/<dataset>.env` first):
```bash
mkdir -p "${ARTIFACT_FOLDER_PATH}/logs"
bsub -G team361 -q training-parallel -n 4 -gpu "num=1:gmem=40000:mode=exclusive_process:block=yes" -M 32G \\
    -R "select[mem>32G] rusage[mem=32G] span[hosts=1]" -W 0:30 \\
    -o "${ARTIFACT_FOLDER_PATH}/logs/check_attention_kernel_%J.out" \\
    -e "${ARTIFACT_FOLDER_PATH}/logs/check_attention_kernel_%J.err" \\
    "source ${ENV_PATH}/bin/activate && PYTHONPATH=${CODE_PATH}/src python check_attention_kernel.py \\
     --artifact_folder_path ${ARTIFACT_FOLDER_PATH}"
```
`--run_name` picks another run of the manifest, `--config_file_path` a config file directly.
"""

import argparse
import copy
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

KEY = "zero_fully_masked_attention_rows"


def make_batch(config: dict, token_dict: dict, n_samples: int, device: torch.device) -> dict:
    """A batch of the run's tokenized dataset as training collates it, with the context of sample 0 padding only."""
    from datasets import load_from_disk
    from terra.datasets.cell_datasets import init_cell_dataset
    from terra.masks.block_masking import BlockMaskCollator
    from terra.masks.utils import padding_key_masks

    data, meta, mask = config["data"], config["meta"], config["mask"]
    if not mask.get("pad_masks_to_batch_max", False):
        raise ValueError("The check needs mask.pad_masks_to_batch_max (the key masks of padded contexts).")
    dataset = load_from_disk(data["tokenized_data_folder_path"]).select(range(n_samples))
    dataset.set_format(type="torch", columns=[column for column in dataset.column_names if column != "cell_id"])
    cell_dataset = init_cell_dataset(dataset=dataset,
                                     vocab_size=len(token_dict),
                                     seq_len_cell=data["seq_len_cell"],
                                     seq_len_neighborhood=data["seq_len_neighborhood"],
                                     tokenizer_type=data["tokenizer_type"],
                                     gt_type=meta["gt_type"],
                                     cell_pos_enc=meta["cell_pos_enc"],
                                     special_tokens=meta["special_tokens"],
                                     sampling_strategy=data["sampling_strategy"],
                                     n_nonzero_tokens_list=None,
                                     include_cell_id=False,
                                     sep_gene_tokens_neb=data["sep_gene_tokens_neb"],
                                     nz_spc=data["nz_spc"])
    n_special_tokens = len(meta["special_tokens"])
    collator = BlockMaskCollator(n_targets=mask["n_targets"],
                                 n_contexts=mask["n_contexts"],
                                 n_segments=data["n_segments"],
                                 seq_len_cell=data["seq_len_cell"],
                                 seq_len_neighborhood=data["seq_len_neighborhood"],
                                 n_special_tokens=n_special_tokens,
                                 per_block_mask_ratio=mask["per_block_mask_ratio"],
                                 sample_segments=mask["sample_segments"],
                                 sample_gene_masks=True,
                                 restrict_special_attention=mask["restrict_special_attention"],
                                 special_token_pad_ratio=0.0,
                                 pad_masks_to_batch_max=True)
    torch.manual_seed(0)
    udata, masks_enc, masks_pred, masks_attention, pad_special_tokens = collator(
        [cell_dataset[idx] for idx in range(n_samples)])
    masks_enc_valid = udata.pop("masks_enc_valid")
    masks_pred_valid = udata.pop("masks_pred_valid")
    masks_enc_valid[0][0] = False  # sample 0: every context entry (genes and metatoken) is padding
    enc_masks_attention, _ = padding_key_masks(masks_enc_valid[0], masks_pred_valid[0],
                                               n_special_tokens=n_special_tokens, new_spc=meta["new_spc"],
                                               pad_special_tokens=pad_special_tokens)
    return {"batch": {key: value.to(device) if torch.is_tensor(value) else value for key, value in udata.items()},
            "masks": [masks.to(device) for masks in masks_enc],
            "masks_attention": enc_masks_attention.to(device)}


def run_encoder(config: dict, token_dict: dict, zero_rows: bool, inputs: dict, state_dict: dict | None,
                device: torch.device) -> tuple[dict, torch.Tensor, dict]:
    """Forward and backward pass of the context encoder; returns the weights, the outputs and the gradient status."""
    from terra.utils.helper import init_model_from_config

    model_config = copy.deepcopy(config)
    model_config["meta"][KEY] = zero_rows
    torch.manual_seed(0)
    encoder, _ = init_model_from_config(model_config, token_dict, device)
    if state_dict is not None:
        encoder.load_state_dict(state_dict)
    encoder.train()
    torch.manual_seed(0)
    use_bfloat16 = config["meta"]["use_bfloat16"] and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
        z, _ = encoder(batch=inputs["batch"], masks=inputs["masks"], masks_attention=inputs["masks_attention"])
    z = z[0].float()  # one context mask: [B, L_ctx, D]
    z.sum().backward()
    gradients = [parameter.grad for parameter in encoder.parameters() if parameter.grad is not None]
    gradients_finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    return copy.deepcopy(encoder.state_dict()), z.detach(), {"gradients_finite": gradients_finite,
                                                             "n_gradients": len(gradients)}


def get_config_file_path(artifact_folder_path: str, run_name: str) -> str:
    """Config of a run of the manifest (configs/ can also hold the configs of earlier generations)."""
    manifest_df = pd.read_csv(Path(artifact_folder_path) / "manifest.tsv", sep="\t")
    rows = manifest_df[manifest_df["run_name"] == run_name]
    if len(rows) != 1:
        raise ValueError(f"Expected one manifest row of run {run_name!r}, found {len(rows)}.")
    return rows["config_file_path"].iloc[0]


def main(args):
    if args.config_file_path is None:
        args.config_file_path = get_config_file_path(args.artifact_folder_path, args.run_name)
    print(f"Run config: {args.config_file_path}")
    with open(args.config_file_path) as file:
        config = yaml.safe_load(file)
    with open(config["data"]["token_dict_folder_path"], "rb") as file:
        token_dict = pickle.load(file)
    import terra

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # TERRA ignores config keys it does not read: without the option, true behaves as false
    supported = any(KEY in file_path.read_text() for file_path in Path(terra.__file__).parent.rglob("*.py"))
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
          f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'no GPU (CPU)'}; TERRA {terra.__file__} "
          f"({'knows' if supported else 'does NOT know'} meta.{KEY}); attention_backend "
          f"{config['meta'].get('attention_backend')}, use_flash_attention "
          f"{config['meta'].get('use_flash_attention')}, bfloat16 autocast "
          f"{config['meta']['use_bfloat16'] and device.type == 'cuda'}")
    if not supported and not args.only_false:
        sys.exit(f"This TERRA version does not know meta.{KEY}: pin one that has it, or run with --only_false.")
    inputs = make_batch(config, token_dict, args.n_samples, device)
    print(f"Batch of {args.n_samples} samples from {config['data']['tokenized_data_folder_path']}: valid context keys "
          f"per sample {inputs['masks_attention'][:, 0, 0].sum(-1).tolist()} (sample 0: padding only)")

    # Same weights for both settings (those of the first)
    state_dict, outputs, statuses = None, {}, {}
    for zero_rows in [False] if args.only_false else [False, True]:
        state_dict, z, status = run_encoder(config, token_dict, zero_rows, inputs, state_dict, device)
        status.update(padded_finite=bool(torch.isfinite(z[0]).all()),
                      others_finite=bool(torch.isfinite(z[1:]).all()))
        outputs[zero_rows], statuses[zero_rows] = z, status
        print(f"meta.{KEY} {str(zero_rows).lower()}: padded sample outputs finite {status['padded_finite']}; other "
              f"samples finite {status['others_finite']}; gradients of {status['n_gradients']} encoder parameters "
              f"finite {status['gradients_finite']}")
    if args.only_false:
        return
    identical = torch.equal(outputs[False][1:], outputs[True][1:])
    print(f"Other samples bit-identical between false and true: {identical}")
    status = statuses[True]
    if not (status["padded_finite"] and status["others_finite"] and status["gradients_finite"] and identical):
        sys.exit(f"meta.{KEY} true does not give finite outputs and gradients with unchanged other samples on this "
                 f"setup, see above.")
    print(f"OK: meta.{KEY} true gives finite outputs and gradients and leaves the other samples unchanged.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument('--artifact_folder_path',
                              type=str,
                              help='Folder with manifest.tsv; the config of --run_name is checked.')
    config_group.add_argument('--config_file_path',
                              type=str,
                              help='Run config of the campaign (configs/<run id>.yaml); its tokenized dataset must '
                                   'exist.')
    parser.add_argument('--run_name',
                        type=str,
                        default='base_rep1',
                        help='Run of the manifest whose config is checked (with --artifact_folder_path).')
    parser.add_argument('--n_samples',
                        type=int,
                        default=4,
                        help='Samples in the batch (sample 0 gets a context of padding only).')
    parser.add_argument('--only_false',
                        action='store_true',
                        help=f'Only run meta.{KEY} false (for a TERRA version without the option).')
    args = parser.parse_args()

    main(args)
