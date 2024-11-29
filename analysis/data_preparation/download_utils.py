import os
import time
import tqdm
import requests
from pathlib import Path


CHUNK_SIZE = 5 * 1024 * 1024


def download_zipfiles(
        input_directory: Path = None,
        redownload: str = "False",
        urls: list[str] = [],
        file_names: list[str] = []
    ) -> list[Path]:
    raw_zipfile_paths = []
    for url in urls:
        raw_zipfile_name = f"{os.path.basename(url)}"
        raw_zipfile_path = input_directory / raw_zipfile_name
        raw_zipfile_paths.append(raw_zipfile_path)

        # download files if not already downloaded
        if not os.path.exists(raw_zipfile_path) or redownload == "True":
            # download files
            print(f"Downloading files from {url}...")
            download_url(url,
                        raw_zipfile_path)
        else:
            print(f"{raw_zipfile_path} already exists. Skipping download...")
    return raw_zipfile_paths


def download_url(url: str = None,
                 raw_file_path: Path = None):
    # poor man's retry
    _ATTEMPTS=3
    for attempt in range(1,_ATTEMPTS+1):
        try:
            r = requests.get(url, stream=True)
            total_size = int(r.headers.get("content-length", 0))
            with tqdm.tqdm(
                total=total_size, unit="B", unit_scale=True, desc=os.path.basename(url)
            ) as p:
                with open(raw_file_path, "wb") as f:
                    for chunk in r.iter_content(CHUNK_SIZE):
                        p.update(len(chunk))
                        f.write(chunk)
            return True
        except Exception as ex:
            # log error, wait 60s and try agaain
            print(f"Attempt #{attempt} failed to download. Error :{repr(ex)}")
            time.sleep(60)
    print(f"ERROR. Fail all {_ATTEMPTS} attempts")