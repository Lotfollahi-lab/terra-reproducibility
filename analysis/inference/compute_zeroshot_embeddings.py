"""
Compute zero-shot embeddings for multiple models (terra, scgpt_spatial, novae, nicheformer, cellplm).

This script provides a unified CLI for computing zero-shot embeddings across different models.
It reads configuration from a YAML file and supports different input types depending on the model.

Usage:
    python compute_zeroshot_embeddings.py --config /path/to/config.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import anndata as ad
import yaml
import torch
# Add paths for model imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "source-code" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarking" / "scGPT-spatial"))
sys.path.insert(0, str(Path(__file__).parent / "nicheformer" / "src"))

# Import model libraries (will be imported conditionally)
try:
    from app.inference import (
        embed_dataset as terra_embed_dataset,
        harmonize_tokenize_embed_pipeline
    )
except ImportError:
    terra_embed_dataset = None
    harmonize_tokenize_embed_pipeline = None

try:
    import scgpt_spatial
except ImportError:
    scgpt_spatial = None

try:
    import novae
except ImportError:
    novae = None

try:
    from nicheformer.data import NicheformerDataset
    from nicheformer.models import Nicheformer
    import pytorch_lightning as pl
except ImportError:
    NicheformerDataset = None
    Nicheformer = None
    pl = None

try:
    from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
except ImportError:
    CellEmbeddingPipeline = None


def validate_config(
        config: dict
    ) -> None:
    """
    Validate configuration file.

    Parameters
    ----------
    config: dict
        Configuration dictionary.
    """
    required_keys = ['model', 'data']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    if 'name' not in config['model']:
        raise ValueError("Missing required config key: model.name")
    
    model_name = config['model']['name']
    valid_models = ['terra', 'scgpt_spatial', 'novae', 'nicheformer', 'cellplm']
    if model_name not in valid_models:
        raise ValueError(f"Invalid model name: {model_name}. Must be one of {valid_models}")
    
    if 'input_path' not in config['data']:
        raise ValueError("Missing required config key: data.input_path")
    
    input_path = Path(config['data']['input_path'])
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    # Validate input_type for terra
    if model_name == 'terra':
        input_type = config['data'].get('input_type', 'adata')
        if input_type not in ['adata', 'dataset']:
            raise ValueError(f"Invalid input_type for terra: {input_type}. Must be 'adata' or 'dataset'")
        
        if 'model_path' not in config['model']:
            raise ValueError("Missing required config key: model.model_path for terra model")
        model_path = Path(config['model']['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # Validate model-specific requirements
    if model_name == 'scgpt_spatial':
        if 'model_path' not in config['model']:
            raise ValueError("Missing required config key: model.model_path for scgpt_spatial model")
        model_path = Path(config['model']['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    if model_name == 'nicheformer':
        if not 'model_path' in config['model']:
            raise ValueError("Missing required config key: model.model_path for nicheformer model")
        model_path = Path(config['model']['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        base_path = model_path / "data" / "model_means"
        if not Path(base_path).exists():
            raise FileNotFoundError(f"Base path does not exist: {base_path}")
        
        if not 'technology_mean_name' in config['model']:
            raise ValueError("Missing required config key: model.technology_mean_name for nicheformer model")        
        technology_mean_path = base_path / config['model']['technology_mean_name']
        if not Path(technology_mean_path).exists():
            raise FileNotFoundError(f"Technology mean path does not exist: {technology_mean_path}")
        
        if not 'checkpoint_name' in config['model']:
            raise ValueError("Missing required config key: model.checkpoint_name for nicheformer model")
        checkpoint_path = model_path / "data" / config['model']['checkpoint_name']
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    if model_name == 'cellplm':
        if 'pretrain_prefix' not in config['model']:
            raise ValueError("Missing required config key: model.pretrain_prefix for cellplm model")
        if 'pretrain_directory' not in config['model']:
            raise ValueError("Missing required config key: model.pretrain_directory for cellplm model")
        pretrain_directory = Path(config['model']['pretrain_directory'])
        if not pretrain_directory.exists():
            raise FileNotFoundError(f"Pretrain directory does not exist: {pretrain_directory}")


def compute_terra_embeddings(
        config: dict
    ):
    """
    Compute embeddings using TERRA model.

    Parameters
    ----------
    config: dict
        Configuration dictionary.
    """
    from datasets import load_from_disk

    # Get data parameters
    input_type = config['data'].get('input_type', 'adata')
    input_path = Path(config['data']['input_path'])
    output_path = config['data'].get('output_path')
    if not output_path:
        raise ValueError("config.data.output_path is required")
    output_path = Path(output_path)

    # Get model parameters
    model_path = Path(config['model']['model_path'])

    # Get embedding parameters
    batch_size = config['embed'].get('batch_size', 128)
    num_workers = config['embed'].get('num_workers', 12)
    emb_layer = config['embed'].get('emb_layer', None)
    agg_excluded_genes = config['embed'].get('agg_excluded_genes', None)
    top_k = config['embed'].get('top_k', None)
    include_spatial_cell_emb = config['embed'].get('include_spatial_cell_emb', False)

    if input_type == 'dataset':
        # Load dataset and call embed_dataset directly
        print("Loading dataset...")
        dataset = load_from_disk(str(input_path))

        # Reorder dataset to match h5ad obs order so cell_ids are consistent across models
        adata_path = config['data'].get('adata_path')
        if adata_path:
            print("Loading AnnData for cell_id ordering...")
            adata_ref = sc.read_h5ad(adata_path, backed='r')
            h5ad_cell_ids = adata_ref.obs['cell_id'].tolist()
            del adata_ref
            dataset_cellid_to_idx = {cid: i for i, cid in enumerate(dataset['cell_id'])}
            reorder_indices = [dataset_cellid_to_idx[cid] for cid in h5ad_cell_ids]
            dataset = dataset.select(reorder_indices)
            print("Dataset reordered to match h5ad obs order.")

        # Extract cell_ids before set_format converts columns to torch tensors
        cell_ids = np.array(dataset['cell_id'])

        cols = [c for c in dataset.column_names]
        dataset.set_format(
            type="torch",
            columns=cols,
            output_all_columns=True
        )

        print("Computing embeddings from dataset...")
        output_embed = terra_embed_dataset(
            dataset=dataset,
            model_folder_path=str(model_path),
            emb_layer=emb_layer,
            agg_excluded_genes=agg_excluded_genes,
            top_k=top_k,
            batch_size=batch_size,
            num_workers=num_workers,
            include_spatial_cell_emb=include_spatial_cell_emb
        )

        output_path.mkdir(parents=True, exist_ok=True)

        filename = output_path / "terra_cell_emb.npy"
        np.save(filename, output_embed['cell_emb'])
        print(f"Saved TERRA cell embeddings to: {filename}")

        filename = output_path / "terra_nbr_emb.npy"
        np.save(filename, output_embed['neighborhood_emb'])
        print(f"Saved TERRA neighborhood embeddings to: {filename}")

        if include_spatial_cell_emb:
            filename = output_path / "terra_spatial_cell_emb.npy"
            np.save(filename, output_embed['spatial_cell_emb'])
            print(f"Saved TERRA spatial cell embeddings to: {filename}")

        np.save(output_path / "terra_cell_ids.npy", cell_ids)
        print(f"Saved TERRA cell IDs to: {output_path / 'terra_cell_ids.npy'}")

        return output_embed

    else:
        # Load adata
        print("Loading AnnData object...")
        adata = sc.read_h5ad(input_path)

        # Get sample_key if specified
        sample_key = config['data'].get('sample_key', None)

        print("Computing embeddings from AnnData (harmonize, tokenize, embed)...")
        adata = harmonize_tokenize_embed_pipeline(
            adata=adata,
            sample_key=sample_key,
            model_folder_path=str(model_path),
            emb_layer=emb_layer,
            agg_excluded_genes=agg_excluded_genes,
            top_k=top_k,
            batch_size=batch_size,
            num_workers=num_workers
        )

        output_path.mkdir(parents=True, exist_ok=True)

        filename = output_path / "terra_cell_emb.npy"
        np.save(filename, adata.obsm['cell_emb'])
        print(f"Saved TERRA cell embeddings to: {filename}")

        filename = output_path / "terra_nbr_emb.npy"
        np.save(filename, adata.obsm['neighborhood_emb'])
        print(f"Saved TERRA neighborhood embeddings to: {filename}")

        if include_spatial_cell_emb:
            filename = output_path / "terra_spatial_cell_emb.npy"
            np.save(filename, adata.obsm['spatial_cell_emb'])
            print(f"Saved TERRA spatial cell embeddings to: {filename}")

        np.save(output_path / "terra_cell_ids.npy", adata.obs['cell_id'].values)
        print(f"Saved TERRA cell IDs to: {output_path / 'terra_cell_ids.npy'}")

        return adata


def compute_scgpt_spatial_embeddings(
        config: dict
    ) -> ad.AnnData:
    """
    Compute embeddings using scGPT-spatial model.

    Parameters
    ----------
    config: dict
        Configuration dictionary.
    """
    # Get data parameters
    input_path = Path(config['data']['input_path'])
    output_path = config['data'].get('output_path')
    if not output_path:
        raise ValueError("config.data.output_path is required")
    output_path = Path(output_path)

    # Get model parameters
    model_path = config['model']['model_path']

    # Get embedding parameters
    batch_size = config['embed'].get('batch_size', 64)
    gene_col = config['embed'].get('gene_col', 'feature_name')

    import scipy.sparse as sp

    print("Loading AnnData object...")
    adata = sc.read_h5ad(input_path)

    # Set gene_col if needed
    if gene_col == 'index' or gene_col not in adata.var.columns:
        adata.var[gene_col] = adata.var.index

    # scGPT-spatial uses .A internally which is unsupported on newer scipy sparse views
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray().astype(np.float32)
    else:
        adata.X = np.asarray(adata.X).astype(np.float32)

    print("Computing scGPT-spatial embeddings...")
    adata = scgpt_spatial.tasks.embed_data(
        adata,
        model_dir=model_path,
        gene_col=gene_col,
        batch_size=batch_size,
        return_new_adata=False
    )

    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / "scgpt-spatial_emb.npy"
    np.save(filename, adata.obsm['X_scGPT'])
    print(f"Saved scGPT-spatial cell embeddings to: {filename}")
    np.save(output_path / "scgpt-spatial_cell_ids.npy", adata.obs['cell_id'].values)
    print(f"Saved scGPT-spatial cell IDs to: {output_path / 'scgpt-spatial_cell_ids.npy'}")

    return adata


def compute_novae_embeddings(
        config: dict
    ) -> ad.AnnData:
    """
    Compute embeddings using Novae model.

    Parameters
    ----------
    config: dict
        Configuration dictionary.
    """
    # Get data parameters
    input_path = Path(config['data']['input_path'])
    output_path = config['data'].get('output_path')
    if not output_path:
        raise ValueError("config.data.output_path is required")
    output_path = Path(output_path)

    # Get model parameters
    model_name = config['model']['pretrained_model']

    print("Loading AnnData object...")
    adata = sc.read_h5ad(input_path)

    # Derive per-slide key from cell_id (e.g. '1000_batch1_0' -> '1000batch1')
    adata.obs['batch'] = adata.obs['cell_id'].apply(
        lambda x: x.split('_')[0] + x.split('_')[1]
    )

    # Identify slides too small for Delaunay triangulation (qhull with -Qz needs >= 4 points).
    # Rather than dropping these cells entirely (which would break per-fold splits that
    # reference them), we exclude them from inference and fill their embeddings with zeros.
    all_cell_ids = adata.obs['cell_id'].values.copy()
    slide_counts = adata.obs['batch'].value_counts()
    small_slides = slide_counts[slide_counts < 4].index.tolist()
    if small_slides:
        dropped = adata.obs[adata.obs['batch'].isin(small_slides)]
        print(f"WARNING: {len(dropped)} cell(s) from {len(small_slides)} slide(s) have < 4 cells "
              f"and cannot be triangulated. Their embeddings will be set to zero.")
        for slide in small_slides:
            ids = dropped[dropped['batch'] == slide]['cell_id'].tolist()
            print(f"  slide={slide!r}: {ids}")
        adata_infer = adata[~adata.obs['batch'].isin(small_slides)].copy()
    else:
        adata_infer = adata

    print("Loading Novae model...")
    model = novae.Novae.from_pretrained(model_name)

    print("Computing Novae embeddings...")
    model.compute_representations(
        adata_infer,
        zero_shot=True,
        slide_key='batch'
    )

    # Build full embedding array in original cell order, zeros for excluded cells
    emb_dim = adata_infer.obsm['novae_latent'].shape[1]
    full_emb = np.zeros((len(all_cell_ids), emb_dim), dtype=np.float32)
    infer_cell_ids = adata_infer.obs['cell_id'].values
    infer_id_to_idx = {cid: i for i, cid in enumerate(infer_cell_ids)}
    for out_idx, cid in enumerate(all_cell_ids):
        if cid in infer_id_to_idx:
            full_emb[out_idx] = adata_infer.obsm['novae_latent'][infer_id_to_idx[cid]]

    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / "novae_emb.npy"
    np.save(filename, full_emb)
    print(f"Saved Novae embeddings to: {filename}")
    np.save(output_path / "novae_cell_ids.npy", all_cell_ids)
    print(f"Saved Novae cell IDs to: {output_path / 'novae_cell_ids.npy'}")

    return adata


def compute_nicheformer_embeddings(
        config: dict
    ) -> ad.AnnData:
    """Compute embeddings using Nicheformer model.

    Parameters
    ----------
    config: dict
        Configuration dictionary.
    """
    # Get data parameters
    input_path = Path(config['data']['input_path'])
    output_path = config['data'].get('output_path')
    if not output_path:
        raise ValueError("config.data.output_path is required")
    output_path = Path(output_path)

    # Get model parameters
    model_path = Path(config['model']['model_path'])
    base_path = model_path / "data" / "model_means"
    model_ref_path = base_path / "model.h5ad"
    technology_mean_path = base_path / config['model']['technology_mean_name']
    checkpoint_path = model_path / "data" / config['model']['checkpoint_name']
    technology_mean = np.load(technology_mean_path)

    # Get embedding parameters
    num_workers = config['embed'].get('num_workers', 4)
    batch_size = config['embed'].get('batch_size', 32)
    chunk_size = config['embed'].get('chunk_size', 1000)
    max_seq_len = config['embed'].get('max_seq_len', 1500)
    aux_tokens = config['embed'].get('aux_tokens', 30)
    embedding_layer = config['embed'].get('embedding_layer', -1)

    print("Loading AnnData object...")
    adata = sc.read_h5ad(input_path)
    
    # Load model reference data if needed (for formatting)
    model_ref = ad.read_h5ad(model_ref_path)
    adata = ad.concat([model_ref, adata], join='outer', axis=0)
    adata = adata[1:].copy()
    mask = adata.var_names.isin(model_ref.var_names)
    adata = adata[:, mask].copy()
    
    # Set metadata columns
    adata.obs['modality'] = config['embed'].get('modality', 4)  # spatial
    adata.obs['specie'] = config['embed'].get('specie', 5)  # human
    adata.obs['technology'] = config['embed'].get('technology', 9)  # xenium
    adata.obs['nicheformer_split'] = 'train'
        
    print("Creating Nicheformer dataset...")
    dataset = NicheformerDataset(
        adata=adata,
        technology_mean=technology_mean,
        split='train',
        max_seq_len=max_seq_len,
        aux_tokens=aux_tokens,
        chunk_size=chunk_size
    )
    
    print("Creating dataloader...")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print("Loading Nicheformer model...")
    model = Nicheformer.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        weights_only=False,
        strict=False
    )
    model.eval()
    
    print("Computing Nicheformer embeddings...")
    embeddings = []
    device = model.embeddings.weight.device
    
    try:
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # Get embeddings from the model
                emb = model.get_embeddings(
                    batch=batch,
                    layer=embedding_layer
                )
                embeddings.append(emb.cpu().numpy())
    except Exception as e:
        raise RuntimeError(f"Failed to compute Nicheformer embeddings: {e}")
    
    # Concatenate all embeddings
    embeddings = np.concatenate(embeddings, axis=0)
    adata.obsm['X_nicheformer'] = embeddings
    
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / "nicheformer_emb.npy"
    np.save(filename, embeddings)
    print(f"Saved Nicheformer embeddings to: {filename}")
    np.save(output_path / "nicheformer_cell_ids.npy", adata.obs['cell_id'].values)
    print(f"Saved Nicheformer cell IDs to: {output_path / 'nicheformer_cell_ids.npy'}")

    return adata


def compute_cellplm_embeddings(
        config: dict
    ) -> ad.AnnData:
    """
    Compute embeddings using CellPLM model.

    Parameters
    ----------
    config: dict
        Configuration dictionary.
    """
    # Get data parameters
    input_path = Path(config['data']['input_path'])
    output_path = config['data'].get('output_path')
    if not output_path:
        raise ValueError("config.data.output_path is required")
    output_path = Path(output_path)

    # Get model parameters
    pretrain_prefix = config['model'].get('pretrain_prefix')
    pretrain_directory = config['model'].get('pretrain_directory')
    
    # Get embedding parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading AnnData object...")
    adata = sc.read_h5ad(input_path)
    
    print("Initializing CellPLM pipeline...")
    pipeline = CellEmbeddingPipeline(
        pretrain_prefix=pretrain_prefix,
        pretrain_directory=pretrain_directory
    )
    print(pipeline.model)
    
    print("Computing CellPLM embeddings...")
    embeddings = pipeline.predict(
        adata,
        device=device
    )
    embeddings = embeddings.cpu().numpy()
    adata.obsm['X_cellplm'] = embeddings
    
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / "cellplm_emb.npy"
    np.save(filename, embeddings)
    print(f"Saved CellPLM embeddings to: {filename}")
    np.save(output_path / "cellplm_cell_ids.npy", adata.obs['cell_id'].values)
    print(f"Saved CellPLM cell IDs to: {output_path / 'cellplm_cell_ids.npy'}")

    return adata
    

def main():
    """Main function to compute zero-shot embeddings."""
    parser = argparse.ArgumentParser(
        description="Compute zero-shot embeddings for multiple models"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    args = parser.parse_args()
    
    # Load and validate config
    print("Loading configuration...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    validate_config(config)
    
    model_name = config['model']['name']
    print(f"Computing embeddings using model: {model_name}")
    
    # Route to appropriate model function
    if model_name == 'terra':
        result = compute_terra_embeddings(config)
    elif model_name == 'scgpt_spatial':
        result = compute_scgpt_spatial_embeddings(config)
    elif model_name == 'novae':
        result = compute_novae_embeddings(config)
    elif model_name == 'nicheformer':
        result = compute_nicheformer_embeddings(config)
    elif model_name == 'cellplm':
        result = compute_cellplm_embeddings(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print("Embedding computation complete!")
    
    return result


if __name__ == "__main__":
    main()
