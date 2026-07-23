#!/usr/bin/env python3
"""Merge LoRA adapters and assemble an embeddable TERRA model folder.

Thin CLI wrapper around
``terra.training.finetune_self_supervised.prepare_finetuned_model``. The merge
logic (folding the LoRA adapters into the base weights and copying the config /
token dictionary) now lives in the ``terra`` package.
"""
import argparse

from terra.training.finetune_self_supervised import prepare_finetuned_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare a finetuned model for embedding generation.")
    parser.add_argument(
        "--finetuned_dir", type=str, required=True,
        help="Directory containing finetuned checkpoints (checkpoint_epoch_*.pt).")
    parser.add_argument(
        "--pretrained_dir", type=str, required=True,
        help="Directory with the pretrained model (model_config.yaml, token_dictionary.pkl).")
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for the prepared model.")
    parser.add_argument(
        "--checkpoint_epoch", type=int, default=None,
        help="Specific epoch checkpoint to use (default: final checkpoint).")
    parser.add_argument(
        "--use_peft", action="store_true",
        help="Set if PEFT (LoRA) was used during finetuning.")
    args = parser.parse_args()

    prepare_finetuned_model(
        finetuned_checkpoint_dir=args.finetuned_dir,
        pretrained_model_dir=args.pretrained_dir,
        output_dir=args.output_dir,
        checkpoint_epoch=args.checkpoint_epoch,
        use_peft=args.use_peft,
    )
