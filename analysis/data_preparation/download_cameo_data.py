"""
This script downloads the CAMEO multimodal benchmark datasets (Theis lab; aligned H&E image patches and gene expression
profiles) from the Hugging Face Hub to the bronze tier. The repositories are gated (access by manual approval):

- `theislab/CAMEO-Lung`: 10x Xenium lung, 52 Arrow shards (~26 GB); downloaded by default
- `theislab/CAMEO-Breast`: 10x Xenium breast, 82 Arrow shards (~41 GB)
- `theislab/CAMEO-Thymus`: 10x Visium thymus, 9 Arrow shards (~4.4 GB)

Only CAMEO-Lung is downloaded by default (access to the others is pending); add the others once access is approved:
`--repo_ids theislab/CAMEO-Breast theislab/CAMEO-Thymus` (or all three). `theislab-multimodal-ssl/CAMEO-Thymus-scvi-fixed`
can be added the same way. Each repository is a
`datasets.Dataset.save_to_disk` folder and is downloaded to `<output_directory_path>/<repository name>`; load it with
`datasets.load_from_disk`. Interrupted downloads are resumed. Each repository is pinned to the commit its revision
resolves to, every file is verified against the size and SHA-256 (git SHA-1 for small non-LFS files) that the Hub
records for it (files that fail are downloaded once more), and the commits, checksums and a summary of each dataset (rows, columns) are written to
`download_manifest.json`.

Access: a Hugging Face token of an account whose access requests were approved. Store it once, interactively, on the
farm head node: `--login` reads it without echo, checks it, saves it to the Hugging Face token file (`$HF_HOME/token`,
by default `~/.cache/huggingface/token`, readable only by you), where the download job finds it, and reports which
CAMEO repositories the account may download. Alternatively, export `HF_TOKEN`. The token is never printed, logged or
written to the manifest.

Usage:
```bash
python download_cameo_data.py --login                 # once, interactively (head node)
python download_cameo_data.py --check_access          # optional: check access with the stored token
bsub < download_cameo_data.lsf                        # download (normal queue)
python download_cameo_data.py --output_directory_path /nfs/team361/sb75/DATASETS/bronze/cameo   # or directly
```
"""

from __future__ import annotations

import argparse
import datetime
import getpass
import hashlib
import json
import sys
from pathlib import Path

# Gated CAMEO repositories; only those with approved access are downloaded by default
CAMEO_REPO_IDS = ["theislab/CAMEO-Lung", "theislab/CAMEO-Breast", "theislab/CAMEO-Thymus"]
DEFAULT_REPO_IDS = ["theislab/CAMEO-Lung"]
CHUNK_SIZE = 16 * 1024 * 1024


def check_access(repo_ids: list, token: str | None = None) -> dict | None:
    """
    Whether the token (default: the stored token or HF_TOKEN) may download files of each repository, as
    {repo_id: bool}; None if there is no valid token.
    """
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    try:
        user = HfApi().whoami(token=token)["name"]
    except Exception as error:  # noqa: BLE001 (no or invalid token)
        print(f"No valid Hugging Face token ({type(error).__name__}). Run `python download_cameo_data.py --login` "
              f"or export HF_TOKEN.")
        return None
    print(f"Hugging Face user: {user}")
    access = {}
    for repo_id in repo_ids:
        try:
            # A HEAD request for a small file: fails with GatedRepoError unless access was approved
            get_hf_file_metadata(hf_hub_url(repo_id, "state.json", repo_type="dataset"), token=token)
            access[repo_id] = True
            print(f"{repo_id}: access granted.")
        except GatedRepoError:
            access[repo_id] = False
            print(f"{repo_id}: NO ACCESS for {user}. Request it on https://huggingface.co/datasets/{repo_id} and "
                  f"wait for the approval.")
        except RepositoryNotFoundError:
            access[repo_id] = False
            print(f"{repo_id}: repository not found (or not visible to {user}).")
    return access


def login(repo_ids: list):
    """
    Read a token without echo, store it in the Hugging Face token file if it is valid, and report the access to every
    known CAMEO repository (and any requested one). Repositories without approved access can be downloaded once the
    approval arrives, without logging in again.
    """
    from huggingface_hub import login as hf_login
    from huggingface_hub.constants import HF_TOKEN_PATH

    if not sys.stdin.isatty():
        sys.exit("--login is interactive: run it in a terminal (e.g. on the head node), not in a batch job.")
    token = getpass.getpass("Hugging Face access token (read; input hidden): ").strip()
    if not token:
        sys.exit("No token entered.")
    access = check_access(CAMEO_REPO_IDS + [repo_id for repo_id in repo_ids if repo_id not in CAMEO_REPO_IDS],
                          token=token)
    if access is None:
        sys.exit("Token not stored: it is not valid.")
    hf_login(token=token, add_to_git_credential=False)
    print(f"Token stored in {HF_TOKEN_PATH}; the download job will use it.")
    print(f"Downloadable now: {[repo_id for repo_id, ok in access.items() if ok]}; "
          f"pending approval: {[repo_id for repo_id, ok in access.items() if not ok]}.")


def file_digest(file_path: Path, algorithm: str) -> str:
    """SHA-256 of the file, or its git blob SHA-1 (the Hub's checksum of non-LFS files)."""
    if algorithm == "sha256":
        digest = hashlib.sha256()
    else:
        digest = hashlib.sha1()
        digest.update(f"blob {file_path.stat().st_size}\0".encode())
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_dataset(dataset_path: Path) -> dict:
    """Rows, columns and features of a `save_to_disk` dataset (memory-mapped, nothing is loaded into memory)."""
    from datasets import load_from_disk

    dataset = load_from_disk(str(dataset_path))
    return {"n_rows": dataset.num_rows,
            "columns": dataset.column_names,
            "features": {name: str(feature) for name, feature in dataset.features.items()}}


def verify_files(repo_id: str, dataset_path: Path, siblings: list) -> tuple[dict, list]:
    """Size and checksum (SHA-256 for LFS files, git SHA-1 otherwise) of every file against the Hub's records."""
    files, failed = {}, []
    for sibling in siblings:
        file_path = dataset_path / sibling.rfilename
        if not file_path.is_file():
            failed.append(sibling.rfilename)
            continue
        size = file_path.stat().st_size
        if sibling.lfs is not None:
            algorithm, expected = "sha256", sibling.lfs.sha256
        else:
            algorithm, expected = "git_sha1", sibling.blob_id
        if (sibling.size is not None and size != sibling.size) or file_digest(file_path, algorithm) != expected:
            failed.append(sibling.rfilename)
            continue
        files[sibling.rfilename] = {"size": size, algorithm: expected}
    return files, failed


def download_repository(repo_id: str, output_directory_path: Path, revision: str, max_workers: int) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    # Pin the commit the revision resolves to, with the size and checksum of every file
    info = HfApi().dataset_info(repo_id, revision=revision, files_metadata=True)
    dataset_path = output_directory_path / repo_id.split("/")[-1]
    print(f"{repo_id}: commit {info.sha}, {len(info.siblings)} files, "
          f"{sum(sibling.size or 0 for sibling in info.siblings) / 1e9:.1f} GB -> {dataset_path}")

    # Resumable download of that commit; files that fail verification are deleted and downloaded once more
    for attempt in range(2):
        snapshot_download(repo_id=repo_id, repo_type="dataset", revision=info.sha, local_dir=dataset_path,
                          max_workers=max_workers)
        files, failed = verify_files(repo_id, dataset_path, info.siblings)
        if not failed:
            break
        print(f"{repo_id}: {len(failed)} files failed verification; downloading them again: {failed[:5]}")
        for rfilename in failed:
            (dataset_path / rfilename).unlink(missing_ok=True)
    else:
        raise ValueError(f"{repo_id}: files still fail verification after downloading them again: {failed[:5]}")
    print(f"{repo_id}: all {len(files)} files verified.")

    entry = {"revision": revision, "commit": info.sha, "path": str(dataset_path), "files": files}
    if (dataset_path / "state.json").is_file():
        entry["dataset"] = summarize_dataset(dataset_path)
        print(f"{repo_id}: {entry['dataset']['n_rows']} rows, columns {entry['dataset']['columns']}.")
    return entry


def main(args):
    if args.login:
        login(args.repo_ids)
        return
    if args.check_access:
        access = check_access(args.repo_ids)
        sys.exit(0 if access is not None and all(access.values()) else 1)

    import huggingface_hub

    # Fail before downloading anything if the token or an access approval of a requested repository is missing
    access = check_access(args.repo_ids)
    if access is None or not all(access.values()):
        sys.exit("Access check failed, see above (download the repositories with approved access only, e.g. with "
                 "the default --repo_ids).")
    output_directory_path = Path(args.output_directory_path)
    output_directory_path.mkdir(parents=True, exist_ok=True)
    manifest_file_path = output_directory_path / "download_manifest.json"
    manifest = json.loads(manifest_file_path.read_text()) if manifest_file_path.exists() else {"repositories": {}}
    for repo_id in args.repo_ids:
        manifest["repositories"][repo_id] = download_repository(repo_id, output_directory_path, args.revision,
                                                                args.max_workers)
        manifest["huggingface_hub_version"] = huggingface_hub.__version__
        manifest["date"] = datetime.datetime.now().isoformat(timespec="seconds")
        manifest_file_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved the manifest to {manifest_file_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--output_directory_path',
                        type=str,
                        default='/nfs/team361/sb75/DATASETS/bronze/cameo',
                        help='Bronze folder; each repository goes to a subfolder named after it.')
    parser.add_argument('--repo_ids',
                        nargs='+',
                        default=DEFAULT_REPO_IDS,
                        help=f'Hugging Face dataset repositories to download (default: CAMEO-Lung; known: {CAMEO_REPO_IDS}).')
    parser.add_argument('--revision',
                        type=str,
                        default='main',
                        help='Branch, tag or commit; the resolved commit is recorded in the manifest.')
    parser.add_argument('--max_workers',
                        type=int,
                        default=4,
                        help='Parallel file downloads.')
    parser.add_argument('--login',
                        action='store_true',
                        help='Interactively store a Hugging Face token (checked for access) and exit.')
    parser.add_argument('--check_access',
                        action='store_true',
                        help='Check access to the repositories with the stored token and exit.')
    args = parser.parse_args()

    main(args)
