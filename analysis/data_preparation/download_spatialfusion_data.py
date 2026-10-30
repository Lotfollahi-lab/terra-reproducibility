"""
This script downloads all data with pathologist niche labels from the SpatialFusion study (Yates et al., bioRxiv 2026,
doi:10.64898/2026.03.16.712056) to the bronze tier. These are the four preprocessed benchmark AnnData objects deposited
on Zenodo (record 20934900, concept DOI 10.5281/zenodo.20934899), each a single tissue section:

- `processed_OVCA.h5ad`: 10x Xenium Prime 5K (+100 custom genes) FFPE ovarian cancer; labels in `obs['path_region']`
- `preprocessed_BRCA.h5ad`: 10x Xenium FFPE breast cancer (313-gene panel); labels in `obs['path_region']`
- `preprocessed_PRAD.h5ad`: 10x Xenium Prime 5K FFPE prostate adenocarcinoma (HEST-1k TENX157); labels in
  `obs['region_annotation']`
- `preprocessed_LUAD.h5ad`: 10x Visium HD FFPE lung adenocarcinoma; labels in `obs['region_annotation']`

The label sources (per-cell region and cell type CSVs, QuPath GeoJSON region polygons) are additionally downloaded from
the authors' analysis repository (uhlerlab/spatialfusion-analysis) at a pinned commit, so that the labels can be
transferred to other preprocessings of the same sections. All files are verified against pinned checksums and
interrupted downloads are resumed. Finally, the Zenodo archive is extracted and the sources, checksums and a summary of
each AnnData object (shape, count matrices, label distributions) are written to `download_manifest.json`.

Usage:
```bash
python download_spatialfusion_data.py --output_directory_path /nfs/team361/sb75/DATASETS/bronze/spatialfusion
```
"""

import argparse
import datetime
import hashlib
import http.client
import json
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


CHUNK_SIZE = 5 * 1024 * 1024
USER_AGENT = "terra-reproducibility"

ZENODO_RECORD_ID = 20934900
ZENODO_FILES = {
    # file name: (size in bytes, MD5)
    "benchmark_processed_data.zip": (1620662647, "26c8b75de035d6de7a8305875861caa4"),
    "README_benchmark-dataset.md": (11200, "9981e3219b362e6ef9540926e5c016d9"),
}

GITHUB_REPOSITORY = "uhlerlab/spatialfusion-analysis"
GITHUB_COMMIT = "2c29b6130d601d606c6cd57b612a8c0b7e81f5df"
GITHUB_FILES = {
    # repository path: (size in bytes, git blob SHA-1)
    "Fig2/region_annotations_ovca.csv": (6441014, "9a1d5a95e3dbffe972a5ac30b6d9e09af17bffe5"),
    "Fig2/celltypes_ovca.csv": (16631226, "6aa7b518026bfc309ef830284c3bfdf5682bff8f"),
    "Fig3/breast_ffpe_region_annotations.csv": (2054454, "f7b7eeae4ed0ab29e4fcce89dd9a19ba0ca267f6"),
    "Fig3/celltypes_brca.csv": (4510530, "228c6949b62e8b5063679b6cc2507b88bc2ece9d"),
    "Fig4/Xenium_Prime_Human_Prostate_FFPE_he_image.geojson": (298604, "6144ea4d630b090ae705f4482a62aab0f37b4a36"),
    "Fig4/celltypes_prad.csv": (8688945, "6e8c470674839e8b6da9c8dd9993de7bc777a1cc"),
    "Fig4/Visium_HD_Human_Lung_Cancer_HD_Only_Experiment1_tissue_image.btf - 20x_BF_01.geojson": (
        164006, "3895233351cc6e68f0ddd07771d0501b7a8640a4"),
}

# Pathologist niche label columns of the benchmark AnnData objects
LABEL_KEYS = ["path_region", "region_annotation"]
# Xenium control probes and codewords that may be stored alongside genes
NON_GENE_FEATURE_PATTERN = r"^(NegControl|BLANK|UnassignedCodeword|DeprecatedCodeword|Intergenic)"


def compute_checksum(file_path: Path,
                     algorithm: str) -> str:
    """Compute the MD5 ('md5') or git blob SHA-1 ('git-sha1') checksum of a file."""
    if algorithm == "md5":
        hasher = hashlib.md5(usedforsecurity=False)
    elif algorithm == "git-sha1":
        hasher = hashlib.sha1(f"blob {file_path.stat().st_size}\0".encode(), usedforsecurity=False)
    else:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
    with open(file_path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(url: str,
                  file_path: Path,
                  size: int,
                  checksum: str,
                  algorithm: str,
                  n_attempts: int = 5):
    """Download a file and verify its size and checksum. Data is written to a `.part` file first so that interrupted
    downloads are resumed with an HTTP range request instead of restarted."""
    if file_path.exists() and compute_checksum(file_path, algorithm) == checksum:
        print(f"{file_path} already exists and is valid. Skipping download...")
        return

    file_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = file_path.with_name(f"{file_path.name}.part")
    for attempt in range(1, n_attempts + 1):
        try:
            offset = part_path.stat().st_size if part_path.exists() else 0
            if offset < size:
                print(f"Downloading {url} to {file_path} ({size / 1e6:.1f} MB, starting at byte {offset})...")
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                if offset > 0:
                    request.add_header("Range", f"bytes={offset}-")
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206:
                        # Server sent the whole file, e.g. because it ignored the range request
                        offset = 0
                    with open(part_path, "ab" if offset > 0 else "wb") as f:
                        n_downloaded_bytes, next_report = offset, offset + size // 10
                        while chunk := response.read(CHUNK_SIZE):
                            f.write(chunk)
                            n_downloaded_bytes += len(chunk)
                            if n_downloaded_bytes >= next_report:
                                print(f"{n_downloaded_bytes / size:.0%} downloaded...")
                                next_report += size // 10
            n_bytes = part_path.stat().st_size
            if n_bytes < size:
                raise OSError(f"Download interrupted after {n_bytes} of {size} bytes.")
            if n_bytes > size or compute_checksum(part_path, algorithm) != checksum:
                part_path.unlink()
                raise OSError(f"Size or {algorithm} checksum mismatch for {file_path.name}.")
            part_path.rename(file_path)
            print(f"Downloaded and verified {file_path}.")
            return
        except (OSError, http.client.HTTPException) as e:
            print(f"Attempt {attempt}/{n_attempts} failed: {e!r}")
            if attempt < n_attempts:
                time.sleep(60)
    raise RuntimeError(f"Failed to download {url} after {n_attempts} attempts.")


def extract_zip(zip_path: Path,
                output_directory_path: Path) -> list[Path]:
    """Extract a zip archive, skipping macOS metadata and members that were already extracted. Member CRCs are
    verified by `zipfile` during extraction."""
    extracted_paths = []
    with zipfile.ZipFile(zip_path) as zip_file:
        for member in zip_file.infolist():
            if member.is_dir() or member.filename.startswith("__MACOSX/") or member.filename.endswith(".DS_Store"):
                continue
            member_path = output_directory_path / member.filename
            if not member_path.resolve().is_relative_to(output_directory_path.resolve()):
                raise ValueError(f"Zip member {member.filename} would be extracted outside of {output_directory_path}.")
            if member_path.exists() and member_path.stat().st_size == member.file_size:
                print(f"{member_path} already extracted. Skipping...")
            else:
                print(f"Extracting {member.filename} ({member.file_size / 1e9:.2f} GB)...")
                zip_file.extract(member, output_directory_path)
            extracted_paths.append(member_path)
    return extracted_paths


def summarize_count_matrix(matrix) -> dict:
    """Summarize the first 1,000 rows of a (backed) matrix to check whether it contains raw counts."""
    import numpy as np
    import scipy.sparse as sp

    rows = matrix[:1000]
    values = rows.data if sp.issparse(rows) else np.asarray(rows).ravel()
    if values.size == 0:
        return {"dtype": str(values.dtype)}
    return {"dtype": str(values.dtype),
            "min": float(values.min()),
            "max": float(values.max()),
            "integer_valued": bool(np.all(np.mod(values, 1) == 0))}


def summarize_adata(adata_path: Path) -> dict:
    """Summarize an AnnData object in backed mode, i.e. without loading its count matrix into memory."""
    import anndata as ad
    import numpy as np

    adata = ad.read_h5ad(adata_path, backed="r")
    summary = {"file": adata_path.name,
               "n_obs": adata.n_obs,
               "n_vars": adata.n_vars,
               "n_non_gene_features": int(adata.var_names.astype(str).str.match(NON_GENE_FEATURE_PATTERN).sum()),
               "obs_columns": adata.obs.columns.tolist(),
               # Coordinate ranges reveal missing coordinates, e.g. the all-zero `obsm['spatial']` of PRAD
               "obsm_ranges": {key: {"min": np.asarray(adata.obsm[key]).min(axis=0).tolist(),
                                     "max": np.asarray(adata.obsm[key]).max(axis=0).tolist()}
                               for key in adata.obsm.keys()},
               "uns_keys": list(adata.uns.keys()),
               "X": summarize_count_matrix(adata.X),
               "layers": {key: summarize_count_matrix(adata.layers[key]) for key in adata.layers.keys()},
               "raw_X": summarize_count_matrix(adata.raw.X) if adata.raw is not None else None,
               "labels": {key: {label: int(n) for label, n in adata.obs[key].astype(str).value_counts().items()}
                          for key in LABEL_KEYS if key in adata.obs}}
    adata.file.close()
    return summary


def main(args):
    output_directory_path = Path(args.output_directory_path)
    output_directory_path.mkdir(parents=True, exist_ok=True)

    # Download benchmark AnnData objects from Zenodo and label sources from GitHub
    downloaded_files = []
    for file_name, (size, md5) in ZENODO_FILES.items():
        url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/{urllib.parse.quote(file_name)}/content"
        download_file(url, output_directory_path / file_name, size, md5, "md5")
        downloaded_files.append({"path": file_name, "url": url, "size": size, "md5": md5})
    for repository_path, (size, blob_sha) in GITHUB_FILES.items():
        url = (f"https://raw.githubusercontent.com/{GITHUB_REPOSITORY}/{GITHUB_COMMIT}/"
               f"{urllib.parse.quote(repository_path)}")
        file_path = Path("annotations") / repository_path
        download_file(url, output_directory_path / file_path, size, blob_sha, "git-sha1")
        downloaded_files.append({"path": str(file_path), "url": url, "size": size, "git_blob_sha1": blob_sha})

    extracted_paths = extract_zip(output_directory_path / "benchmark_processed_data.zip", output_directory_path)

    # Summarize AnnData objects to check count matrices and label distributions
    dataset_summaries = []
    for adata_path in [path for path in extracted_paths if path.suffix == ".h5ad"]:
        print(f"Summarizing {adata_path}...")
        try:
            summary = summarize_adata(adata_path)
        except Exception as e:  # The summary is informational only, so it should not fail the download
            summary = {"file": adata_path.name, "error": repr(e)}
        print(json.dumps(summary, indent=4))
        dataset_summaries.append(summary)

    manifest = {
        "download_date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "study": "Yates et al. SpatialFusion. bioRxiv (2026). doi:10.64898/2026.03.16.712056",
        "zenodo_record": f"https://zenodo.org/records/{ZENODO_RECORD_ID}",
        "github_commit": f"https://github.com/{GITHUB_REPOSITORY}/tree/{GITHUB_COMMIT}",
        "downloaded_files": downloaded_files,
        "extracted_files": [str(path.relative_to(output_directory_path)) for path in extracted_paths],
        "datasets": dataset_summaries,
    }
    manifest_path = output_directory_path / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
    print(f"Saved download manifest to {manifest_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--output_directory_path',
                        type=str,
                        default="/nfs/team361/sb75/DATASETS/bronze/spatialfusion",
                        help='Bronze directory path to download the SpatialFusion data to.')
    args = parser.parse_args()

    main(args)
