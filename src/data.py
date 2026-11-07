from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
LABEL_COLUMN = "good"


def load_wine_dataset(data_dir: str | Path) -> pd.DataFrame:
    data_path = Path(data_dir)
    red_path = data_path / "winequality-red.csv"
    white_path = data_path / "winequality-white.csv"
    if not red_path.exists() or not white_path.exists():
        raise FileNotFoundError(f"Expected CSV files in {data_path}. Run scripts/get_data.py first.")

    red = pd.read_csv(red_path, sep=";")
    white = pd.read_csv(white_path, sep=";")
    for df in (red, white):
        df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]
    red["is_red"] = 1
    white["is_red"] = 0

    df = pd.concat([red, white], ignore_index=True)
    df["good"] = (df["quality"] >= 7).astype(int)
    return df


def summarize_duplicates(df: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    duplicate_cols = FEATURE_COLUMNS + ["quality", "is_red"]
    duplicate_mask = df.duplicated(subset=duplicate_cols, keep=False)
    duplicates_before = int(duplicate_mask.sum())
    cleaned = df.drop_duplicates(subset=duplicate_cols).reset_index(drop=True)
    duplicates_after = int(cleaned.duplicated(subset=duplicate_cols, keep=False).sum())
    return duplicates_before, duplicates_after, cleaned


def duplicate_counts(df: pd.DataFrame) -> dict[str, int]:
    duplicate_cols = FEATURE_COLUMNS + ["quality", "is_red"]
    duplicate_rows = df.duplicated(subset=duplicate_cols, keep=False)
    redundant_rows = df.duplicated(subset=duplicate_cols, keep="first")
    return {
        "rows_before": int(len(df)),
        "rows_involved_in_duplicates": int(duplicate_rows.sum()),
        "redundant_rows_removed": int(redundant_rows.sum()),
        "rows_after": int(len(df) - redundant_rows.sum()),
    }


def make_split_indices(df: pd.DataFrame, seed: int = 0, project_root: str | Path | None = None) -> Dict[str, np.ndarray]:
    project_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    stratifier = df["good"].astype(str) + "_" + df["is_red"].astype(str)
    idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.4, random_state=seed, stratify=stratifier
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed, stratify=stratifier.iloc[temp_idx]
    )

    split_frame = pd.DataFrame(
        {
            "sample_index": list(train_idx) + list(val_idx) + list(test_idx),
            "split": ["train"] * len(train_idx) + ["val"] * len(val_idx) + ["test"] * len(test_idx),
        }
    )
    split_path = results_dir / f"split_seed{seed}.csv"
    split_frame.to_csv(split_path, index=False)

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def get_training_sets(df: pd.DataFrame, seed: int = 0, project_root: str | Path | None = None) -> Dict[str, pd.DataFrame | np.ndarray]:
    split = make_split_indices(df, seed=seed, project_root=project_root)
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]
    sets: Dict[str, pd.DataFrame | np.ndarray] = {}
    sets["X_train"] = X.iloc[split["train"]].reset_index(drop=True)
    sets["X_val"] = X.iloc[split["val"]].reset_index(drop=True)
    sets["X_test"] = X.iloc[split["test"]].reset_index(drop=True)
    sets["y_train"] = y.iloc[split["train"]].reset_index(drop=True)
    sets["y_val"] = y.iloc[split["val"]].reset_index(drop=True)
    sets["y_test"] = y.iloc[split["test"]].reset_index(drop=True)
    return sets
