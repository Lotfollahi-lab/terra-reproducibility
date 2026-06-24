### Adapt NEMO to visium data


###Import libraries
import os
import sys
import argparse
import pickle
import logging
import yaml
from ast import literal_eval
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

import copy

import numpy as np

import torch
import torch.nn.functional as F

import wandb


import os
TERRA_SRC = os.environ.get(
    'TERRA_SRC',
    '/lustre/scratch126/cellgen/lotfollahi/zh4/Codebase_of_Zichen/terra-main 3/src')
sys.path.insert(0, TERRA_SRC)

from datasets import Dataset, load_from_disk, concatenate_datasets

from app.utils.helper import init_model, init_opt
from nichejepa.datasets.cell_datasets import init_cell_dataset
from nichejepa.datasets.dataloaders import init_dataloader_and_sampler
from nichejepa.masks.block_masking  import BlockMaskCollator
from nichejepa.masks.utils import apply_masks
from nichejepa.utils.schedulers import (WarmupCosineSchedule,
                                         CosineWDSchedule)

# --- apply_peft
from peft import LoraConfig, get_peft_model
def apply_peft(model, peft_rank=128, peft_alpha=256, peft_dropout=0.04,
               peft_bias='none', peft_target_modules=None):
    if peft_target_modules is None:
        peft_target_modules = ['qkv', 'proj', 'fc1', 'fc2']
    peft_config = LoraConfig(r=peft_rank, lora_alpha=peft_alpha,
                             lora_dropout=peft_dropout, bias=peft_bias,
                             task_type=None, target_modules=peft_target_modules)
    peft_model = get_peft_model(model, peft_config)
    for name, p in peft_model.named_parameters():      # train only the adapters
        p.requires_grad = ('lora_' in name.lower())
    return peft_model

# --- load_checkpoint
def load_checkpoint(device, r_path, encoder, predictor, target_encoder,
                    opt, scaler, is_training=True):
    ckpt = torch.load(r_path, map_location=torch.device(device))
    epoch = ckpt.get('epoch', 0)
    iter_number = ckpt.get('iter_number', None)
    _strip = lambda sd: {k.replace('module.', ''): v for k, v in sd.items()}
    if encoder is not None:
        print('encoder load:', encoder.load_state_dict(_strip(ckpt['encoder'])))
    if predictor is not None:
        print('predictor load:', predictor.load_state_dict(_strip(ckpt['predictor'])))
    if target_encoder is not None:
        print('target_encoder load:', target_encoder.load_state_dict(_strip(ckpt['target_encoder'])))
    if opt is not None:
        opt.load_state_dict(ckpt['opt'])
    if scaler is not None:
        scaler.load_state_dict(ckpt['scaler'])
    del ckpt
    return encoder, predictor, target_encoder, opt, scaler, epoch, iter_number

#Error handling
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

# Global variables
_GLOBAL_SEED = 0
LOCAL_RANK = None

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

# Argument parsing function
def parse_arguments():
    """Parse config file name from command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run NicheJEPA I-JEPA finetuning on Visium data.')
    parser.add_argument('--fname', type=str, default='/lustre/scratch126/cellgen/lotfollahi/zh4/Codebase_of_Zichen/Zichen_NicheJEPA/Embryo_Figure/Finetune/config_Visium_IJEPA.yaml',
                        help='Name of the config file to load.')
    
    parser_args = parser.parse_args()
    
    # Load finetune config file
    with open(parser_args.fname, "r") as f:
        args = yaml.safe_load(f)
    
    return args

###Finetuning function
def finetune_ijepa(
        args: dict,
        dataset: Dataset,
        save_folder_path: str | None = None,
        LOCAL_RANK: int | None = None,
        WORLD_RANK: int | None = None,
        run_name: str | None = None,
    ):
    """
    Finetune model using I-JEPA architecture.

    Parameters
    -----------
    args:
        Dictionary containing the hyperparams from the config file.
    dataset:
        Finetune training dataset (tokenized).
    save_folder_path:
        Path for saving model artifacts.
    LOCAL_RANK:
        Local rank of the process.
    WORLD_RANK:
        World rank of the process.
    """
    
    ###Backend setup
    #set random seed
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)

    # CUDA settings
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_GLOBAL_SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    elif LOCAL_RANK is not None:
        device = torch.device(f"cuda:{LOCAL_RANK}")
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # Load finetune paths
    pretrained_checkpoint_path = Path(args['model']['pretrained_checkpoint_path'])
    model_config_file_path = pretrained_checkpoint_path / 'model_config.yaml'
    token_dictionary_file_path = pretrained_checkpoint_path / 'token_dictionary.pkl'
    model_checkpoint_path = pretrained_checkpoint_path / 'model_checkpoint.pt'

    # Load model config
    with open(model_config_file_path, 'r') as file:
        model_config = yaml.safe_load(file)

    # Load tokenisation parameters
    n_special_tokens = len(model_config['meta']['special_tokens'])
    seq_len = (
        model_config['data']['seq_len_cell'] +
        model_config['data']['seq_len_neighborhood'] +
        n_special_tokens)

    # Load token dictionary
    with open(token_dictionary_file_path, 'rb') as file:
        token_dict = pickle.load(file)
    vocab_size = len(token_dict)
    n_special_values = sum(1 for key in token_dict if "spv" in key)

    ###Prepare tokenized dataset and dataloader
    #Mask generator 
    mask_collator = BlockMaskCollator(
        n_targets=model_config['mask']['n_targets'],
        n_contexts=model_config['mask']['n_contexts'],
        n_segments=model_config['data']['n_segments'],
        seq_len_cell=model_config['data']['seq_len_cell'],
        seq_len_neighborhood=model_config['data']['seq_len_neighborhood'],
        n_special_tokens=n_special_tokens,
        per_block_mask_ratio=model_config['mask']['per_block_mask_ratio'],
        sample_segments=args['data'].get('sample_segments', False),
        sample_gene_masks=args['data'].get('sample_gene_masks', False)
        )
        
    #Initialize dataset 
    cell_dataset = init_cell_dataset(
        dataset=dataset,
        vocab_size=vocab_size,
        seq_len_cell=model_config['data']['seq_len_cell'],
        seq_len_neighborhood=model_config['data']['seq_len_neighborhood'],
        tokenizer_type=model_config['data']['tokenizer_type'],
        gt_type=model_config['meta']['gt_type'],
        cell_pos_enc=model_config['meta']['cell_pos_enc'],
        special_tokens=model_config['meta']['special_tokens'],
        sampling_strategy=None,
        n_nonzero_tokens_list=None,
        include_cell_id=True,
        sep_gene_tokens_neb=model_config['data']['sep_gene_tokens_neb'])

    # Batch data for training
    loader, _ = init_dataloader_and_sampler(
        cell_dataset=cell_dataset,
        batch_size=args['data']['batch_size'],
        distributed=False,
        world_size=1,
        rank=0,
        collate_fn=mask_collator,
        pin_memory=args['data']['pin_memory'],
        num_workers=args['data']['num_workers'],
        drop_last=args['data'].get('drop_last', True),
        persistent_workers=False,
        mega_batch_mult_max=1000)  # terra-main: new required arg; unused when distributed=False

    # Prepare finetune checkpoint paths
    if not save_folder_path:
        finetune_checkpoint_path = Path(args['model']['finetune_checkpoint_path'])
    else:
        finetune_checkpoint_path = Path(save_folder_path)
    
    # Set run name 
    if run_name:
        finetune_dir = finetune_checkpoint_path / run_name
    else:
        current_timestamp = (
            datetime.now().strftime("%d%m%Y_%H%M%S") +
            f"_{datetime.now().microsecond // 1000:03d}")
        finetune_dir = finetune_checkpoint_path / current_timestamp
    finetune_dir.mkdir(parents=True, exist_ok=True)
    
    # Save finetune config
    with open(finetune_dir / 'params.yaml', 'w') as f:
        yaml.dump(args, f)

    ###Initialize encoder, predictor, and target encoder
    logger.info("Initializing encoder, predictor, and target encoder...")
    
    #Initialize encoder and predictor
    encoder, predictor = init_model(
        gt_type=model_config['meta']['gt_type'],
        count_encoding=model_config['meta']['count_encoding'],
        n_value_bins=model_config['meta']['n_value_bins'],
        cell_pos_enc=model_config['meta']['cell_pos_enc'],
        device=device,
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_special_tokens=n_special_tokens,
        n_segments=model_config['data']['n_segments'],
        n_special_values=n_special_values,
        enc_emb_dim=model_config['meta']['enc_emb_dim'],
        enc_depth=model_config['meta']['enc_depth'],
        pred_emb_dim=model_config['meta']['pred_emb_dim'],
        pred_depth=model_config['meta']['pred_depth'],
        num_heads=model_config['meta']['num_heads'],
        mlp_ratio=model_config['meta']['mlp_ratio'],
        use_flash_attention=model_config['meta']['use_flash_attention'],
        api_version=model_config['meta']['api_version'],
        sep_gene_tokens_neb=model_config['data']['sep_gene_tokens_neb'],
        predict_gene=model_config['meta']['predict_gene'],
        pos_learnable=model_config['meta']['pos_learnable'],
        mlp_bias=model_config['meta'].get('mlp_bias', True))  # terra-main: value_embed bias (checkpoint has it)

    #Initialize target encoder
    target_encoder, _ = init_model(
        gt_type=model_config['meta']['gt_type'],
        count_encoding=model_config['meta']['count_encoding'],
        n_value_bins=model_config['meta']['n_value_bins'],
        cell_pos_enc=model_config['meta']['cell_pos_enc'],
        device=device,
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_special_tokens=n_special_tokens,
        n_segments=model_config['data']['n_segments'],
        n_special_values=n_special_values,
        enc_emb_dim=model_config['meta']['enc_emb_dim'],
        enc_depth=model_config['meta']['enc_depth'],
        pred_emb_dim=model_config['meta']['pred_emb_dim'],
        pred_depth=model_config['meta']['pred_depth'],
        num_heads=model_config['meta']['num_heads'],
        mlp_ratio=model_config['meta']['mlp_ratio'],
        use_flash_attention=model_config['meta']['use_flash_attention'],
        api_version=model_config['meta']['api_version'],
        sep_gene_tokens_neb=model_config['data']['sep_gene_tokens_neb'],
        predict_gene=model_config['meta']['predict_gene'],
        pos_learnable=model_config['meta']['pos_learnable'],
        mlp_bias=model_config['meta'].get('mlp_bias', True))  # terra-main: value_embed bias (checkpoint has it)

    # Load pretrained weights
    logger.info(f"Loading pretrained weights from {model_checkpoint_path}...")
    encoder, predictor, target_encoder, _, _, _, _ = load_checkpoint(
        device=device,
        r_path=model_checkpoint_path,
        encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder,
        opt=None,
        scaler=None,
        is_training=True)
    
    # Freeze target encoder 
    for p in target_encoder.parameters():
        p.requires_grad = False
    target_encoder.eval()
    
    ###Low rank adaptation (LoRA) 
    use_peft = args['finetune'].get('use_peft', False)
    
    # Apply PEFT
    if use_peft:
        logger.info("Applying LoRA adapters")
        
        # Get target modules from config
        peft_target_modules = args['finetune'].get('peft_target_modules', ['qkv', 'proj', 'fc1', 'fc2'])
        
        # Apply LoRA adapters to encoder
        encoder = apply_peft(
            model=encoder,
            peft_rank=args['finetune']['peft_rank'],
            peft_alpha=args['finetune']['peft_alpha'],
            peft_dropout=args['finetune']['peft_dropout'],
            peft_bias=args['finetune']['peft_bias'],
            peft_target_modules=peft_target_modules
        )

        # Apply LoRA adapters to predictor
        predictor = apply_peft(
            model=predictor,
            peft_rank=args['finetune']['peft_rank'],
            peft_alpha=args['finetune']['peft_alpha'],
            peft_dropout=args['finetune']['peft_dropout'],
            peft_bias=args['finetune']['peft_bias'],
            peft_target_modules=peft_target_modules
        )
            
        # Freeze base model params, keep LoRA params trainable
        for name, param in encoder.named_parameters():
            if 'lora_' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, param in predictor.named_parameters():
            if 'lora_' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        logger.info(f"LoRA adapters applied to encoder and predictor")
        logger.info(f"Target modules: {peft_target_modules}")
        n_trainable_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        n_trainable_pred = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters in encoder: {n_trainable_enc}")
        logger.info(f"Trainable parameters in predictor: {n_trainable_pred}")
        
        # Assert that parameters not containing 'lora_' are frozen
        for name, p in encoder.named_parameters():
            if 'lora_' not in name.lower():
                assert p.requires_grad == False, f"Parameter {name} should be frozen"
        for name, p in predictor.named_parameters():
            if 'lora_' not in name.lower():
                assert p.requires_grad == False, f"Parameter {name} should be frozen"

    #Move models to GPU
    encoder.to(device)
    predictor.to(device)
    target_encoder.to(device)
    
    ### Set up model training 
    # Get finetuning hyperparameters
    lr = args['finetune']['lr']
    start_lr = args['finetune'].get('start_lr', lr)
    final_lr = args['finetune'].get('final_lr', 1.0e-06)
    num_epochs = args['finetune']['num_epochs']
    warmup_epochs = args['finetune'].get('warmup_epochs', 0)
    weight_decay = args['finetune'].get('weight_decay', 0.04)
    final_weight_decay = args['finetune'].get('final_weight_decay', 0.4)
    ema_momentum = args['finetune'].get('ema_momentum', 0.996)
    final_ema_momentum = args['finetune'].get('final_ema_momentum', 1.0)
    loss_fn_type = args['finetune'].get('loss_fn_type', 'smooth_l1')
    clip_grad = args['finetune'].get('clip_grad', None)
    use_bfloat16 = args['finetune'].get('use_bfloat16', False)
    
    # Initialise optimizer and schedulers 
    optimizer, scaler, lr_schedule, wd_schedule = init_opt(
        encoder=encoder,
        predictor=predictor,
        iterations_per_epoch=len(loader),
        start_lr=start_lr,
        ref_lr=lr,
        warmup=warmup_epochs,
        num_epochs=num_epochs,
        wd=weight_decay,
        final_wd=final_weight_decay,
        final_lr=final_lr,
        use_bfloat16=use_bfloat16)

    # Confirm optimizer has trainable parameters
    opt_trainable_params = sum(p.numel() for g in optimizer.param_groups for p in g['params'] if p.requires_grad)
    logger.info(f"Optimizer trainable parameters: {opt_trainable_params}")
    
    # Momentum scheduler for EMA
    ipe = len(loader)
    ema = [ema_momentum, final_ema_momentum]
    mom_schedule = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs) for i in range(ipe*num_epochs))
    momentum_scheduler_iter = iter(mom_schedule)
    
    # Helper function to save checkpoints
    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': epoch_loss,
            'batch_size': args['data']['batch_size'],
            'world_size': 1,
            'lr': lr
        }

        # Save checkpoint
        save_path = finetune_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(save_dict, save_path)
        logger.info(f"Checkpoint saved to {save_path}")

    
    # Helper function to call encoder with PEFT handling
    def call_encoder(model, batch, masks=None, masks_attention=None):
        """Call encoder, handling PEFT wrapper if present."""
        if use_peft:
            # Access the base model directly to use custom forward signature
            return model.base_model.model(batch=batch, masks=masks, masks_attention=masks_attention)
        else:
            return model(batch=batch, masks=masks, masks_attention=masks_attention)
    
    # Helper function to call predictor with PEFT handling
    def call_predictor(model, z, token_emb, batch, masks_enc, masks_pred, masks_attention=None):
        """Call predictor, handling PEFT wrapper if present."""
        if use_peft:
            # Access the base model directly to use custom forward signature
            return model.base_model.model(
                z=z, token_emb=token_emb, batch=batch,
                masks_enc=masks_enc, masks_pred=masks_pred,
                masks_attention=masks_attention)
        else:
            return model(
                z=z, token_emb=token_emb, batch=batch,
                masks_enc=masks_enc, masks_pred=masks_pred,
                masks_attention=masks_attention)
    
    ### Training loop
    logger.info("Starting I-JEPA finetuning...")
    
    # Log initial configuration
    logger.info(f"Start LR: {start_lr}, Main LR: {lr}, Final LR: {final_lr}, Warmup epochs: {warmup_epochs}")
    logger.info(f"Weight decay: {weight_decay} -> {final_weight_decay}")
    logger.info(f"EMA momentum: {ema_momentum} -> {final_ema_momentum}")
    logger.info(f"Clip grad: {clip_grad}")
    
    # Training epochs
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        
        # Track gradient statistics for first batch of each epoch
        log_grad_stats = True

        # Set models to train mode
        encoder.train()
        predictor.train()
        
        # Iterate over batches
        for itr, (udata, masks_enc, masks_pred, masks_attention) in enumerate(tqdm(loader)):

            # Move data to GPU
            for key, val in udata.items():
                if isinstance(val, torch.Tensor):
                    udata[key] = val.to(device, non_blocking=True)

            # Move masks to GPU
            masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
            masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]
            masks_attention = masks_attention.to(device, non_blocking=True)
            
            # Update learning rate and weight decay
            _new_lr = lr_schedule.step()
            _new_wd = wd_schedule.step()
            
            # Forward pass - encode context and predict targets - calculate loss
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bfloat16):
                # Forward pass of target encoder 
                with torch.no_grad():
                    h, _ = target_encoder(batch=udata, masks_attention=masks_attention)
                    h = F.layer_norm(h, (h.size(-1),))
                    h = apply_masks(h, masks_pred, concat=False)

                # Forward pass of context encoder 
                z, token_emb = call_encoder(encoder, batch=udata, masks=masks_enc, masks_attention=None)

                # Forward pass of predictor 
                z = call_predictor(
                    predictor, z=z, token_emb=token_emb, batch=udata,
                    masks_enc=masks_enc, masks_pred=masks_pred, masks_attention=None)

                # Compute loss
                loss_exp = 1.0
                loss = 0.
                for zi, hi in zip(z, h):
                    if loss_fn_type == 'smooth_l1':
                        loss += F.smooth_l1_loss(zi, hi)
                    elif loss_fn_type == 'l1':
                        loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
                    elif loss_fn_type == 'mse':
                        loss += F.mse_loss(zi, hi)
                loss /= len(masks_pred)
            
            ### Backward pass
            # Ensure we actually have trainable parameters
            n_trainable_encoder = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
            n_trainable_predictor = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
            if (n_trainable_encoder + n_trainable_predictor) == 0:
                raise RuntimeError("No trainable parameters found in encoder or predictor before backward()."
                                   " This likely indicates PEFT adapters were inadvertently frozen.")
            
            # Backpropagation
            loss.backward()

            # Gradient clipping 
            if clip_grad is not None:
                grad_norm_enc = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                grad_norm_pred = torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                
                # Log gradient norms for first batch of each epoch
                if log_grad_stats:
                    logger.info(f"Epoch {epoch+1}, Batch {itr+1}: Grad norm - Encoder: {grad_norm_enc:.4f}, Predictor: {grad_norm_pred:.4f}")
                    logger.info(f"Current LR: {_new_lr:.6f}, WD: {_new_wd:.6f}")
                    log_grad_stats = False
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # EMA update of target encoder
            with torch.no_grad():

                # Get momentum value
                m = next(momentum_scheduler_iter)

                # EMA update with PEFT handling      
                if use_peft:

                    # Create temporary encoder copy
                    temp_encoder = copy.deepcopy(encoder)

                    # Merge and unload PEFT adapters
                    merged = temp_encoder.merge_and_unload()

                    # Get MERGED encoder state dict (critical: use merged, not temp_encoder)
                    enc_state_dict = merged.state_dict()

                    # Move enc_state_dict tensors to target_encoder device
                    device_target = next(target_encoder.parameters()).device
                    for name, tensor in enc_state_dict.items():
                        if isinstance(tensor, torch.Tensor):
                            enc_state_dict[name] = tensor.to(device_target)

                    # In-place EMA update on target_state_dict 
                    target_state_dict = target_encoder.state_dict()
                    for name, t in target_state_dict.items():
                        if name in enc_state_dict:
                            t.mul_(m).add_((1. - m) * enc_state_dict[name].detach())
                    target_encoder.load_state_dict(target_state_dict)

                    # Free temporary objects
                    del temp_encoder, merged, enc_state_dict
                else:
                    # Standard EMA update without PEFT
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)
            
            # Update running loss
            running_loss += loss.item()
            
            # Log batch loss every 10 batches
            if (itr + 1) % 10 == 0:
                avg_loss = running_loss / (itr + 1)
                wandb.log({
                    "batch_loss": loss.item(),
                    "avg_loss": avg_loss,
                    "batch": epoch * len(loader) + itr,
                    "learning_rate_batch": _new_lr,
                    "weight_decay_batch": _new_wd
                })
        
        # Epoch loss
        epoch_loss = running_loss / len(loader)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, LR: {_new_lr:.6f}")
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "learning_rate": _new_lr
        })
        if (epoch + 1) % args['finetune'].get('save_every', 10) == 0:
            save_checkpoint(epoch + 1)
    
    # Save final checkpoint
    save_checkpoint(num_epochs)
    logger.info("Finetuning complete!")

###Main function
if __name__ == '__main__':
    # Load args dictionary from finetune config file
    args = parse_arguments()
    
    # Load pretokenised finetune datasets 
    finetune_dataset_path = args['data']['finetune_dataset']
    
    # Multiple datasets - load and concatenate
    logger.info(f"Loading {len(finetune_dataset_path)} datasets...")
    datasets = []
    for i, path in enumerate(finetune_dataset_path):
        logger.info(f"Loading dataset {i+1}/{len(finetune_dataset_path)}: {path}")
        ds = load_from_disk(path)
        # Annotate each source file with a 'section' column so we can do group splits
        section_name = Path(path).name
        ds = ds.add_column('section', [section_name] * len(ds))
        datasets.append(ds)
    logger.info("Concatenating datasets...")
    dataset = concatenate_datasets(datasets)
    logger.info(f"Concatenated {len(datasets)} datasets")

    # Shuffle the dataset to mix sections across samples
    dataset = dataset.shuffle(seed=_GLOBAL_SEED)
    logger.info("Dataset shuffled to mix sections")

    # Set dataset format to PyTorch tensors.
    # Exclude non-tensorable metadata columns (e.g., 'section' strings) from the set_format columns
    cols = [c for c in dataset.column_names if c != 'section']
    dataset.set_format(
        type="torch",
        columns=cols,
        output_all_columns=True
    )
    logger.info(f"Dataset loaded with {len(dataset)} samples")
    logger.info(f"Sample data keys: {dataset[0].keys()}")

    # Create timestamp for run name
    current_timestamp = (
        datetime.now().strftime("%d%m%Y_%H%M%S") +
        f"_{datetime.now().microsecond // 1000:03d}")
    
    # Initialize wandb (let wandb sweep handle init if running as agent)
    if wandb.run is None:
        wandb.init(
            project="NicheJEPA_Finetuning",
            name=f"finetune_{current_timestamp}",
            config=args
        )
    else:
        # Running as sweep agent - update config with loaded args
        wandb.config.update(args, allow_val_change=True)
    
    # Finetune model using I-JEPA
    finetune_ijepa(
        args=args,
        dataset=dataset,
        run_name=f"finetune_{current_timestamp}",
    )
    
    # Finish wandb run
    wandb.finish()




