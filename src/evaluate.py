from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


def classification_metrics(y_true: pd.Series | np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, (y_prob_arr >= 0.5).astype(int))),
        "roc_auc": float(roc_auc_score(y_true_arr, y_prob_arr)),
        "average_precision": float(average_precision_score(y_true_arr, y_prob_arr)),
    }


def save_confusion_matrix(y_true, y_prob, path: str | Path, title: str = "Confusion matrix") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    cm = confusion_matrix(np.asarray(y_true), y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    tick_labels = ["Bad", "Good"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_roc_pr_curves(y_true, y_prob, roc_path: str | Path, pr_path: str | Path) -> None:
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("ROC curve")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("PR curve")
    fig.tight_layout()
    fig.savefig(pr_path, dpi=200)
    plt.close(fig)


def append_metrics_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["model", "seed", "metric", "value", "notes"])
    if "model" not in df.columns:
        df["model"] = ""
    if "seed" not in df.columns:
        df["seed"] = ""
    if "metric" not in df.columns:
        df["metric"] = ""
    if "value" not in df.columns:
        df["value"] = np.nan
    if "notes" not in df.columns:
        df["notes"] = ""
    df.to_csv(path, index=False)
