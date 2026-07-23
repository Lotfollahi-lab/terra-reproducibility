"""Self-supervised (I-JEPA) LoRA fine-tuning of TERRA on Visium data.

Thin CLI wrapper around ``terra.training.finetune_self_supervised``. Configure
the run in a YAML file (see ``config_Visium_IJEPA.yaml``) and launch with:

    python finetune_Visium_IJEPA.py --fname config_Visium_IJEPA.yaml

The heavy lifting — dataset loading/concatenation, LoRA setup, the I-JEPA
training loop, EMA target-encoder updates and checkpointing — now lives in the
``terra`` package.
"""
import argparse
from datetime import datetime

import yaml

from terra.training.finetune_self_supervised import finetune_self_supervised


def parse_arguments():
    """Load the fine-tuning config dict from a YAML file."""
    parser = argparse.ArgumentParser(
        description="Run TERRA self-supervised LoRA fine-tuning on Visium data.")
    parser.add_argument(
        "--fname", type=str, default="config_Visium_IJEPA.yaml",
        help="Path to the fine-tuning config YAML.")
    parsed = parser.parse_args()
    with open(parsed.fname) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = parse_arguments()
    run_name = "finetune_" + datetime.now().strftime("%d%m%Y_%H%M%S")
    finetune_self_supervised(args=config, run_name=run_name, use_wandb=True)
