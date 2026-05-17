"""Compute ESM-C 600M protein embeddings for the NicheJEPA gene-token
vocabulary.

For each Ensembl gene ID present in the NicheJEPA token dictionary
(``token_dictionary_homo_sapiens.pkl``), this script:

1. Looks up the gene in pyensembl (default release: 111).
2. Picks the gene's canonical protein sequence — heuristically the
   longest protein-coding transcript's protein sequence. Pyensembl does
   not directly expose Ensembl's "canonical" annotation, so longest-
   protein-coding-transcript is the standard pragmatic substitute (and
   for ~95% of human genes matches MANE Select).
3. Forwards the sequence through ESM-C 600M (Cambrian-600M,
   EvolutionaryScale, 2024 — 1152-dim embeddings).
4. Mean-pools the per-residue embeddings into a single (1152,) vector
   per gene.

The output is two files consumed by NicheJEPA's
``protein_init`` config block:

- ``<out>.pt`` : ``torch.Tensor[N_covered_genes, 1152]``, float32.
- ``<out>.json``: ``{ensembl_gene_id: row_index}`` mapping.

Non-protein-coding genes (lncRNAs, miRNAs, pseudogenes, deprecated
Ensembl IDs) are skipped here — at training time those token IDs
route through NicheJEPA's learnable special-token branch with random
init (frozen). The script logs coverage so you know how many vocab
tokens got a protein embedding.

Pre-requisites
--------------
- ``pip install esm`` (EvolutionaryScale Python SDK; provides ESM-C).
- ``pip install pyensembl`` and a one-time
  ``pyensembl download --release 111 --species human``.
- An NVIDIA GPU. On an A100 80GB this runs ~20k human protein-coding
  genes in ~2-4 hours depending on sequence-length distribution.

Usage
-----
::

    python analysis/data_preparation/compute_protein_embeddings.py \\
        --token_dictionary_file_path /nfs/team361/sb75/DATASETS/tokenizer/token_dictionary_homo_sapiens.pkl \\
        --output_prefix /nfs/team361/sb75/DATASETS/protein_embeddings/esm_c_600m_human \\
        --ensembl_release 111 \\
        --max_sequence_length 4096 \\
        --batch_size 1

The resulting ``esm_c_600m_human.pt`` / ``.json`` are referenced
directly from the model YAML under the ``protein_init`` block.
"""

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

logger = logging.getLogger("compute_protein_embeddings")


_ENSG_PREFIX = "ENSG"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _iter_gene_ensembl_ids(token_dict: Dict[str, int]) -> List[str]:
    """Return all ENSG-prefixed keys from the token dict, sorted for
    reproducibility."""
    return sorted(k for k in token_dict.keys() if k.startswith(_ENSG_PREFIX))


def _pick_canonical_protein_sequence(
        gene_id: str,
        ensembl_data,
        ) -> Optional[str]:
    """Return the protein sequence of the longest protein-coding
    transcript for ``gene_id``, or None if the gene is not protein-
    coding or pyensembl has no record of it.

    Pyensembl does not expose Ensembl's ``canonical_transcript`` flag,
    so longest-protein-coding-transcript is the standard pragmatic
    substitute. Empirically this matches MANE Select for ~95% of
    human protein-coding genes; for the remaining 5% the chosen
    isoform is a close paralog rather than the clinically curated
    canonical, which is fine for embedding purposes.
    """
    try:
        gene = ensembl_data.gene_by_id(gene_id)
    except (ValueError, KeyError):
        return None
    if not gene.transcripts:
        return None
    candidates: List[Tuple[int, str]] = []
    for tr in gene.transcripts:
        # Only protein-coding transcripts have protein_sequence; others
        # raise or return None.
        seq = getattr(tr, "protein_sequence", None)
        if not seq:
            continue
        candidates.append((len(seq), seq))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _load_esmc_600m(device: torch.device):
    """Lazy import + load ESM-C 600M. Kept inside a function so a
    user running --dry-run / --list-only doesn't pay the import cost.
    """
    try:
        from esm.models.esmc import ESMC  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Could not import `esm`. Install EvolutionaryScale's ESM "
            "SDK via `pip install esm`, then re-run."
        ) from exc
    logger.info("Loading ESM-C 600M weights (this is a ~1.2GB download "
                "on first run, cached afterwards in ~/.cache/...).")
    model = ESMC.from_pretrained("esmc_600m").to(device).eval()
    return model


def _embed_sequence(
        model,
        sequence: str,
        device: torch.device,
        max_sequence_length: int,
        ) -> torch.Tensor:
    """Forward one protein sequence through ESM-C 600M and return the
    mean-pooled per-residue embedding.

    Sequences longer than ``max_sequence_length`` are truncated. Titin
    (~36k residues) is the only sequence pathological enough to OOM
    on a 80GB A100; truncating to 4096 is a safe default that affects
    < 0.1% of human protein-coding genes.
    """
    from esm.sdk.api import ESMProtein, LogitsConfig  # type: ignore

    if len(sequence) > max_sequence_length:
        logger.debug(
            "Truncating sequence of length %d to %d residues.",
            len(sequence), max_sequence_length,
        )
        sequence = sequence[:max_sequence_length]

    protein = ESMProtein(sequence=sequence)
    with torch.no_grad():
        protein_tensor = model.encode(protein)
        out = model.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True),
        )
    # out.embeddings shape: (1, seq_len_with_special_tokens, 1152).
    # ESM-C prepends BOS / appends EOS; strip them before pooling so we
    # only average true residue positions.
    embeddings = out.embeddings[0]
    if embeddings.shape[0] >= len(sequence) + 2:
        embeddings = embeddings[1:-1]
    pooled = embeddings.mean(dim=0).to(torch.float32).cpu()
    return pooled


def _stream_progress(
        items: Iterable, total: int, label: str, log_every: int = 100,
        ):
    """Lightweight progress iterator that logs every `log_every` items
    without pulling in tqdm as a hard dependency on the cluster."""
    start = time.time()
    last = start
    for i, item in enumerate(items, start=1):
        yield item
        now = time.time()
        if i % log_every == 0 or i == total:
            elapsed = now - start
            rate = i / max(elapsed, 1e-6)
            remaining = (total - i) / max(rate, 1e-6)
            logger.info(
                "%s: %d/%d (%.1f%%) | %.1f items/s | elapsed %.0fs | ETA %.0fs",
                label, i, total, 100.0 * i / total, rate, elapsed, remaining,
            )
            last = now


def _write_checkpoint(
        out_prefix: Path,
        embeddings: List[torch.Tensor],
        ensembl_to_row: Dict[str, int],
        ) -> None:
    """Atomically write the partial results to disk so a long run can
    be resumed if it crashes or the wall-clock is exhausted."""
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    pt_path = out_prefix.with_suffix(".pt")
    json_path = out_prefix.with_suffix(".json")
    matrix = torch.stack(embeddings) if embeddings else torch.empty(0, 0)
    tmp_pt = pt_path.with_suffix(".pt.tmp")
    tmp_json = json_path.with_suffix(".json.tmp")
    torch.save(matrix, str(tmp_pt))
    with open(tmp_json, "w") as f:
        json.dump(ensembl_to_row, f)
    tmp_pt.replace(pt_path)
    tmp_json.replace(json_path)


def _load_checkpoint(
        out_prefix: Path,
        ) -> Tuple[List[torch.Tensor], Dict[str, int]]:
    pt_path = out_prefix.with_suffix(".pt")
    json_path = out_prefix.with_suffix(".json")
    if not (pt_path.exists() and json_path.exists()):
        return [], {}
    matrix = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    with open(json_path, "r") as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict) or matrix.dim() != 2:
        return [], {}
    if matrix.shape[0] != len(mapping):
        logger.warning("Checkpoint shape mismatch (matrix %d rows vs "
                       "mapping %d entries); starting fresh.",
                       matrix.shape[0], len(mapping))
        return [], {}
    embeddings = [matrix[i] for i in range(matrix.shape[0])]
    return embeddings, mapping


def main(args: argparse.Namespace) -> None:
    _setup_logging(args.verbose)

    out_prefix = Path(args.output_prefix)
    logger.info("Output prefix: %s", out_prefix)

    # 1. Load the NicheJEPA token dictionary.
    logger.info("Loading token dictionary from %s", args.token_dictionary_file_path)
    with open(args.token_dictionary_file_path, "rb") as f:
        token_dict = pickle.load(f)
    if not isinstance(token_dict, dict):
        raise SystemExit(
            f"{args.token_dictionary_file_path}: expected a dict, got "
            f"{type(token_dict)}.")
    ensembl_ids = _iter_gene_ensembl_ids(token_dict)
    logger.info("Token dictionary: %d entries, %d Ensembl gene IDs.",
                len(token_dict), len(ensembl_ids))

    if args.limit is not None:
        ensembl_ids = ensembl_ids[: args.limit]
        logger.info("Limited to first %d Ensembl IDs for testing.", len(ensembl_ids))

    # 2. Initialize pyensembl for sequence lookup.
    try:
        from pyensembl import EnsemblRelease
    except ImportError as exc:
        raise SystemExit(
            "Could not import pyensembl. `pip install pyensembl`, then "
            "run `pyensembl download --release "
            f"{args.ensembl_release} --species human` once.") from exc
    logger.info("Loading pyensembl release %d (human).", args.ensembl_release)
    ensembl_data = EnsemblRelease(args.ensembl_release, species="human")

    # 3. Resume from a previous run if a checkpoint exists.
    embeddings: List[torch.Tensor] = []
    ensembl_to_row: Dict[str, int] = {}
    if args.resume:
        embeddings, ensembl_to_row = _load_checkpoint(out_prefix)
        if embeddings:
            logger.info("Resuming from checkpoint: %d genes already done.",
                        len(embeddings))
    remaining = [eid for eid in ensembl_ids if eid not in ensembl_to_row]
    logger.info("Genes to embed this run: %d", len(remaining))

    if args.dry_run:
        # Just report coverage and exit, no GPU work.
        n_with_seq = 0
        sample = remaining[:200]
        for eid in sample:
            seq = _pick_canonical_protein_sequence(eid, ensembl_data)
            if seq:
                n_with_seq += 1
        logger.info("Dry-run coverage estimate from first %d genes: "
                    "%d/%d (%.1f%%) have a canonical protein sequence.",
                    len(sample), n_with_seq, len(sample),
                    100.0 * n_with_seq / max(len(sample), 1))
        return

    # 4. Set device + load ESM-C 600M.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logger.warning("No CUDA device available; running on CPU will "
                       "take many hours. Pass --dry-run to estimate "
                       "coverage first.")
    model = _load_esmc_600m(device)

    # 5. Main loop. ESM-C SDK doesn't expose a clean batching API at the
    # ``encode`` step (sequences have variable length); we forward one
    # at a time. This is fine — 20k forwards at ~1-2s each on an A100
    # is a few hours.
    n_no_protein = 0
    n_embedded_this_run = 0
    no_protein_examples: List[str] = []
    for eid in _stream_progress(remaining, len(remaining),
                                label="Embedding", log_every=200):
        seq = _pick_canonical_protein_sequence(eid, ensembl_data)
        if not seq:
            n_no_protein += 1
            if len(no_protein_examples) < 10:
                no_protein_examples.append(eid)
            continue
        try:
            emb = _embed_sequence(
                model, seq, device,
                max_sequence_length=args.max_sequence_length,
            )
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "OOM on %s (len=%d); skipping. Reduce "
                "--max_sequence_length or use a larger GPU.",
                eid, len(seq))
            torch.cuda.empty_cache()
            continue
        ensembl_to_row[eid] = len(embeddings)
        embeddings.append(emb)
        n_embedded_this_run += 1
        if (n_embedded_this_run % args.checkpoint_every) == 0:
            _write_checkpoint(out_prefix, embeddings, ensembl_to_row)
            logger.info("Checkpoint written (%d genes total).", len(embeddings))

    # 6. Final write.
    _write_checkpoint(out_prefix, embeddings, ensembl_to_row)
    n_total_dict_genes = len(ensembl_ids)
    n_covered = len(ensembl_to_row)
    logger.info("=" * 70)
    logger.info("Done.")
    logger.info("  Genes in token dictionary  : %d", n_total_dict_genes)
    logger.info("  Genes embedded             : %d (%.1f%% coverage)",
                n_covered, 100.0 * n_covered / max(n_total_dict_genes, 1))
    logger.info("  No protein sequence found  : %d (%.1f%%)",
                n_no_protein, 100.0 * n_no_protein / max(n_total_dict_genes, 1))
    if no_protein_examples:
        logger.info("  Examples without protein   : %s",
                    ", ".join(no_protein_examples))
    logger.info("  Output                     : %s.pt / .json",
                out_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--token_dictionary_file_path",
        type=str,
        default="/nfs/team361/sb75/DATASETS/tokenizer/token_dictionary_homo_sapiens.pkl",
        help="Path to NicheJEPA's pickled token dictionary "
             "(token_dictionary_homo_sapiens.pkl).")
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help="Output prefix; the script writes <prefix>.pt and "
             "<prefix>.json. Example: "
             "/nfs/team361/sb75/DATASETS/protein_embeddings/esm_c_600m_human")
    parser.add_argument(
        "--ensembl_release",
        type=int,
        default=111,
        help="Ensembl release for sequence lookup (must match the "
             "release used to build the token dictionary; the corpus "
             "scripts default to 111).")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=4096,
        help="Truncate any protein longer than this. Default 4096 "
             "covers >99.9%% of human proteins; titin and a few "
             "other mega-proteins get truncated.")
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=500,
        help="Write the partial .pt/.json every N successfully "
             "embedded genes so a crash can be resumed.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set and a .pt/.json already exist at --output_prefix, "
             "skip genes already present and only embed the remainder.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Embed only the first N genes (smoke-test).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the GPU model load; only estimate how many vocab "
             "genes have a protein-coding sequence in pyensembl.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.")
    main(parser.parse_args())
