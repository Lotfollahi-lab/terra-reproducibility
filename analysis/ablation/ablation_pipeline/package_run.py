"""
This script checks whether an ablation run has finished training and, if so, packages its run folder for
`harmonize_tokenize_embed_pipeline`, which expects `model_config.yaml`, `model_checkpoint.pt`, `token_dictionary.pkl`
and `norm_factors.csv` in the model folder (the mouse model folders held copies; here they are symlinks to the training
params, the latest checkpoint, the token dictionary and the normalization factors).

A run is finished when its latest checkpoint (`<state.write_tag>-latest.pth.tar`, i.e. `terra-latest.pth.tar`) was saved
at the end of the last epoch (`epoch == epochs - 1` without `iter_number`; mid-epoch saves carry `iter_number`).

Exit codes: 0 finished (and packaged unless --check_only), 3 not finished.

Usage:
```bash
python package_run.py --run_folder_path [RUN_FOLDER] --norm_factor_file_path [NORM_FACTORS] [--check_only]
```
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

NOT_FINISHED_EXIT_CODE = 3


def main(args):
    run_folder_path = Path(args.run_folder_path)
    params_file_path = run_folder_path / "params.yaml"
    if not params_file_path.exists():
        print(f"{run_folder_path} has no params.yaml yet.")
        sys.exit(NOT_FINISHED_EXIT_CODE)
    with open(params_file_path) as file:
        params = yaml.safe_load(file)
    checkpoint_file_path = run_folder_path / f"{params['state']['write_tag']}-latest.pth.tar"
    if not checkpoint_file_path.exists():
        print(f"{run_folder_path} has no latest checkpoint ({checkpoint_file_path.name}) yet.")
        sys.exit(NOT_FINISHED_EXIT_CODE)

    checkpoint = torch.load(checkpoint_file_path, map_location="cpu", weights_only=False)
    epochs = params["optimization"]["epochs"]
    if checkpoint["epoch"] != epochs - 1 or "iter_number" in checkpoint:
        print(f"{run_folder_path} is not finished: latest checkpoint is epoch {checkpoint['epoch']} "
              f"(iter_number {checkpoint.get('iter_number')}) of {epochs} epochs.")
        sys.exit(NOT_FINISHED_EXIT_CODE)
    print(f"{run_folder_path} finished {epochs} epochs (final loss {checkpoint.get('loss')}).")
    if args.check_only:
        return

    links = {"model_config.yaml": params_file_path,
             "model_checkpoint.pt": checkpoint_file_path,
             "token_dictionary.pkl": Path(params["data"]["token_dict_folder_path"]),
             "norm_factors.csv": Path(args.norm_factor_file_path)}
    for link_name, target_path in links.items():
        if not target_path.exists():
            raise FileNotFoundError(f"{target_path} does not exist.")
        link_path = run_folder_path / link_name
        link_path.unlink(missing_ok=True)
        link_path.symlink_to(target_path)
    print(f"Packaged {run_folder_path} for evaluation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--run_folder_path',
                        type=str,
                        required=True,
                        help='Run folder (state.folder_path of the run config).')
    parser.add_argument('--norm_factor_file_path',
                        type=str,
                        required=True,
                        help='Per-gene normalization factors of the dataset (linked into the run folder).')
    parser.add_argument('--check_only',
                        action='store_true',
                        help='Only check whether training has finished.')
    args = parser.parse_args()

    main(args)
