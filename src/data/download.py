"""Download the Kaggle Transfermarkt snapshot into data/raw/ (git-ignored)."""

from __future__ import annotations

import shutil
from pathlib import Path

from src.common.config import DATA_RAW, KAGGLE_DATASET
from src.common.exceptions import DataLoadError


def download_kaggle_dataset(destination: Path | None = None) -> Path:
    """Copy Kaggle Hub files into destination and return that directory."""
    destination = Path(destination) if destination is not None else DATA_RAW
    destination.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub
    except ImportError as exc:
        raise DataLoadError(
            "Install kagglehub and authenticate (`kagglehub login`) before downloading."
        ) from exc
    cache_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    if not cache_path.exists():
        raise DataLoadError(f"Kaggle Hub returned a missing path: {cache_path}")
    for source in cache_path.rglob("*"):
        if source.is_file():
            target = destination / source.name
            if not target.exists():
                shutil.copy2(source, target)
    return destination


def main() -> None:
    path = download_kaggle_dataset()
    print(f"Transfermarkt files are in {path}")


if __name__ == "__main__":
    main()
