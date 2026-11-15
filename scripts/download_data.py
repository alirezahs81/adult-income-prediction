"""Download the canonical UCI Adult train and test files."""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

URLS = {
    "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
}


def download(output_dir: Path, *, force: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in URLS.items():
        destination = output_dir / filename
        if destination.exists() and not force:
            print(f"Keeping existing file: {destination}")
            continue
        print(f"Downloading {filename}...")
        urlretrieve(url, destination)
        if destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded file is empty: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(args.output_dir, force=args.force)
