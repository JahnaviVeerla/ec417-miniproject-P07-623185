#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib import request

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CSV_URLS = {
    "winequality-red.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "winequality-white.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
}
FALLBACK_ZIP_URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"


def download(url: str, destination: Path) -> None:
    print(f"Downloading {url} -> {destination}")
    with request.urlopen(url, timeout=60) as resp, open(destination, "wb") as out:
        shutil.copyfileobj(resp, out)


def ensure_csvs() -> None:
    missing = [name for name in CSV_URLS if not (RAW_DIR / name).exists()]
    if not missing:
        print("Wine CSV files already present in data/raw/")
        return

    for name, url in CSV_URLS.items():
        target = RAW_DIR / name
        if target.exists():
            continue
        try:
            download(url, target)
        except Exception as exc:
            print(f"Primary download for {name} failed: {exc}")
            zip_path = RAW_DIR / "wine_quality.zip"
            if zip_path.exists():
                zip_path.unlink()
            print(f"Falling back to ZIP archive: {FALLBACK_ZIP_URL}")
            download(FALLBACK_ZIP_URL, zip_path)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(RAW_DIR)
            # The zip contains CSVs inside a folder; keep both if they are present.
            for csv_name in CSV_URLS:
                candidate = RAW_DIR / csv_name
                if not candidate.exists():
                    for root, _, files in os.walk(RAW_DIR):
                        for fn in files:
                            if fn == csv_name:
                                source = Path(root) / fn
                                shutil.copy2(source, candidate)
                                break
            break


if __name__ == "__main__":
    ensure_csvs()
    print(f"Data directory ready: {RAW_DIR}")
