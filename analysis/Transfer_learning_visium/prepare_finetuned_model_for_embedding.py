#!/usr/bin/env python3

import os
import shutil
import argparse
from pathlib import Path
import torch
import yaml
from collections import OrderedDict
import re


def merge_lora_weights(state_dict):
    print("  Merging LoRA adapters into base model...")
    
    # Extract base model weights and LoRA adapters
    base_weights = {}
    lora_a_weights = {}
    lora_b_weights = {}
    
    for key, value in state_dict.items():
        if 'base_layer.weight' in key:
            # Extract the layer name without PEFT wrapper
            clean_key = key.replace('base_model.model.', '').replace('.base_layer', '')
            base_weights[clean_key] = value
        elif 'base_layer.bias' in key:
            clean_key = key.replace('base_model.model.', '').replace('.base_layer', '')
            base_weights[clean_key] = value
        elif 'lora_A.default.weight' in key:
            clean_key = key.replace('base_model.model.', '').replace('.lora_A.default.weight', '.weight')
            lora_a_weights[clean_key] = value
        elif 'lora_B.default.weight' in key:
            clean_key = key.replace('base_model.model.', '').replace('.lora_B.default.weight', '.weight')
            lora_b_weights[clean_key] = value
        elif 'base_model.model.' in key:
            # Other weights (like norms) that don't have LoRA
            clean_key = key.replace('base_model.model.', '')
            base_weights[clean_key] = value
    
    # Merge LoRA weights into base weights
    merged_weights = {}
    n_merged = 0
    
    for key, base_weight in base_weights.items():
        if key in lora_a_weights and key in lora_b_weights:
            # Merge LoRA: W = W_base + (lora_B @ lora_A)
            lora_a = lora_a_weights[key]
            lora_b = lora_b_weights[key]
            
            # Compute LoRA delta: delta = lora_B @ lora_A
            # lora_A shape: [r, in_features]
            # lora_B shape: [out_features, r]
            # delta shape: [out_features, in_features]
            delta = torch.mm(lora_b, lora_a)
            
            # Merge into base weight
            merged_weight = base_weight + delta
            merged_weights[key] = merged_weight
            n_merged += 1
        else:
            # No LoRA adapter for this layer, just use base weight
            merged_weights[key] = base_weight
    
    print(f"  Merged {n_merged} LoRA adapter layers")
    print(f"  Total layers in merged model: {len(merged_weights)}")
    
    return merged_weights


def prepare_finetuned_model(
    finetuned_checkpoint_dir: str,
    pretrained_model_dir: str,
    output_dir: str,
    checkpoint_epoch: int = None,
    use_peft: bool = False
):
    finetuned_checkpoint_dir = Path(finetuned_checkpoint_dir)
    pretrained_model_dir = Path(pretrained_model_dir)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating model directory: {output_dir}")
    
    # Step 1: Copy model_config.yaml and token_dictionary.pkl from pretrained
    print("\n[1/3] Copying config files from pretrained model...")
    
    config_src = pretrained_model_dir / 'model_config.yaml'
    config_dst = output_dir / 'model_config.yaml'
    shutil.copy2(config_src, config_dst)
    print(f"  Copied model_config.yaml")
    
    token_dict_src = pretrained_model_dir / 'token_dictionary.pkl'
    token_dict_dst = output_dir / 'token_dictionary.pkl'
    shutil.copy2(token_dict_src, token_dict_dst)
    print(f"  Copied token_dictionary.pkl")
    
    # Step 2: Find the finetuned checkpoint
    print("\n[2/3] Loading finetuned checkpoint...")
    
    if checkpoint_epoch is not None:
        checkpoint_file = finetuned_checkpoint_dir / f'checkpoint_epoch_{checkpoint_epoch}.pt'
    else:
        # Find the highest epoch checkpoint
        checkpoints = list(finetuned_checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {finetuned_checkpoint_dir}")
        
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        checkpoint_file = checkpoints[-1]
    
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    
    print(f"  Loading: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Final loss: {checkpoint.get('loss', 'unknown'):.4f}")
    
    # Step 3: Process and save the model checkpoint
    print("\n[3/3] Preparing model checkpoint...")
    
    # Extract the state dicts
    encoder_state_dict = checkpoint['encoder']
    predictor_state_dict = checkpoint['predictor']
    target_encoder_state_dict = checkpoint['target_encoder']
    
    # If PEFT was used, merge the LoRA weights into the base model
    if use_peft:
        print("  PEFT detected - merging LoRA adapters into base model...")
        encoder_state_dict = merge_lora_weights(encoder_state_dict)
        predictor_state_dict = merge_lora_weights(predictor_state_dict)
        print("  LoRA weights successfully merged!")
        print("  target_encoder already contains merged fine-tuned weights from EMA")
    
    # Create the model checkpoint in the same format as pretrained
    model_checkpoint = {
        'encoder': encoder_state_dict,
        'predictor': predictor_state_dict,
        'target_encoder': target_encoder_state_dict,
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', 0.0),
    }
    
    checkpoint_dst = output_dir / 'model_checkpoint.pt'
    torch.save(model_checkpoint, checkpoint_dst)
    print(f"  Saved model_checkpoint.pt")
    
    # Save metadata
    metadata = {
        'source_checkpoint': str(checkpoint_file),
        'pretrained_model': str(pretrained_model_dir),
        'finetuned_epoch': checkpoint.get('epoch', 'unknown'),
        'final_loss': float(checkpoint.get('loss', 0.0)),
        'use_peft': use_peft,
    }
    
    metadata_file = output_dir / 'finetuning_metadata.yaml'
    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f)
    print(f"  Saved finetuning_metadata.yaml")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare a finetuned model for embedding generation')
    
    parser.add_argument(
        '--finetuned_dir',
        type=str,
        required=True,
        help='Directory containing finetuned checkpoints (e.g., checkpoint_epoch_*.pt)')
    
    parser.add_argument(
        '--pretrained_dir',
        type=str,
        required=True,
        help='Directory containing pretrained model (model_config.yaml, token_dictionary.pkl)')
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for prepared model')
    
    parser.add_argument(
        '--checkpoint_epoch',
        type=int,
        default=None,
        help='Specific epoch checkpoint to use (default: use final checkpoint)')
    
    parser.add_argument(
        '--use_peft',
        action='store_true',
        help='Set this flag if PEFT (LoRA) was used during finetuning')
    
    args = parser.parse_args()
    
    prepare_finetuned_model(
        finetuned_checkpoint_dir=args.finetuned_dir,
        pretrained_model_dir=args.pretrained_dir,
        output_dir=args.output_dir,
        checkpoint_epoch=args.checkpoint_epoch,
        use_peft=args.use_peft
    )
