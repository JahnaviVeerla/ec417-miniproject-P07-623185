from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from src.data import duplicate_counts, get_training_sets, load_wine_dataset, summarize_duplicates
from src.evaluate import append_metrics_csv
from src.models import (
    make_dummy_model,
    make_hist_gradient_boosting_model,
    make_logistic_model,
    make_random_forest_model,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]


def choose_threshold(y_true: pd.Series | np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in np.linspace(0.0, 1.0, 101):
        pred = (y_prob_arr >= threshold).astype(int)
        score = balanced_accuracy_score(y_true_arr, pred)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, float(best_score)


def score_on_threshold(y_true: pd.Series | np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    return float(balanced_accuracy_score(np.asarray(y_true), pred))


def evaluate_model_on_test(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    model_name: str,
    seed: int,
    config_count: int,
    tuning_metric: str,
    extra_meta: dict | None = None,
):
    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - fit_start

    val_prob = model.predict_proba(X_val)[:, 1]
    threshold, val_bal_acc = choose_threshold(y_val, val_prob)
    test_prob = model.predict_proba(X_test)[:, 1]

    bal_05 = score_on_threshold(y_test, test_prob, 0.5)
    bal_val = score_on_threshold(y_test, test_prob, threshold)
    roc_auc = roc_auc_score(y_test, test_prob)
    pr_auc = average_precision_score(y_test, test_prob)

    extra = extra_meta or {}
    notes = (
        f"threshold={threshold:.3f}; val_bal_acc={val_bal_acc:.4f}; "
        f"config_count={config_count}; tuning_metric={tuning_metric}; "
        f"train_time_s={training_time:.3f}; "
    )
    if "n_trees" in extra:
        notes += f"early_stopping_trees={extra['n_trees']}; early_stopping_source=sklearn_internal_validation; "
    if "class_weight" in extra:
        notes += f"class_weight={extra['class_weight']}; "

    metric_rows = [
        {"model": model_name, "seed": seed, "metric": "balanced_accuracy_0.5", "value": float(bal_05), "notes": notes},
        {"model": model_name, "seed": seed, "metric": "balanced_accuracy_val_threshold", "value": float(bal_val), "notes": notes},
        {"model": model_name, "seed": seed, "metric": "roc_auc", "value": float(roc_auc), "notes": notes},
        {"model": model_name, "seed": seed, "metric": "average_precision", "value": float(pr_auc), "notes": notes},
        {"model": model_name, "seed": seed, "metric": "training_time_seconds", "value": float(training_time), "notes": notes},
    ]

    if "n_trees" in extra:
        metric_rows.append({
            "model": model_name,
            "seed": seed,
            "metric": "early_stopping_trees",
            "value": int(extra["n_trees"]),
            "notes": notes,
        })
    return metric_rows, threshold, val_bal_acc, test_prob


def tune_logistic(X_train, y_train, X_val, y_val, seed: int):
    best_score = -np.inf
    best_model = None
    best_params = None
    grid = [0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0]
    for c in grid:
        model = make_logistic_model(random_state=seed)
        model.set_params(model__C=c)
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, val_prob)
        if score > best_score:
            best_score = score
            best_model = model
            best_params = {"C": c}
    return best_model, best_params, len(grid)


def tune_random_forest(X_train, y_train, X_val, y_val, seed: int):
    best_score = -np.inf
    best_model = None
    best_params = None
    grid = [
        {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"n_estimators": 600, "max_depth": 10, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1, "max_features": 0.5},
        {"n_estimators": 600, "max_depth": None, "min_samples_leaf": 2, "max_features": 0.5},
        {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 2, "max_features": "log2"},
        {"n_estimators": 600, "max_depth": 12, "min_samples_leaf": 3, "max_features": "log2"},
        {"n_estimators": 800, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
    ]
    for params in grid:
        model = make_random_forest_model(random_state=seed, **params)
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, val_prob)
        if score > best_score:
            best_score = score
            best_model = model
            best_params = params
    return best_model, best_params, len(grid)


def tune_gradient_boosting(X_train, y_train, X_val, y_val, seed: int):
    best_score = -np.inf
    best_model = None
    best_params = None
    grid = [
        {"learning_rate": 0.03, "max_depth": 3, "max_leaf_nodes": 15, "l2_regularization": 0.0},
        {"learning_rate": 0.03, "max_depth": 6, "max_leaf_nodes": 31, "l2_regularization": 0.0},
        {"learning_rate": 0.05, "max_depth": 3, "max_leaf_nodes": 15, "l2_regularization": 0.1},
        {"learning_rate": 0.05, "max_depth": 6, "max_leaf_nodes": 31, "l2_regularization": 0.0},
        {"learning_rate": 0.05, "max_depth": 8, "max_leaf_nodes": 31, "l2_regularization": 0.1},
        {"learning_rate": 0.1, "max_depth": 3, "max_leaf_nodes": 31, "l2_regularization": 0.0},
        {"learning_rate": 0.1, "max_depth": 6, "max_leaf_nodes": 63, "l2_regularization": 0.1},
        {"learning_rate": 0.1, "max_depth": 8, "max_leaf_nodes": 63, "l2_regularization": 0.0},
    ]
    for params in grid:
        model = make_hist_gradient_boosting_model(random_state=seed, **params)
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, val_prob)
        if score > best_score:
            best_score = score
            best_model = model
            best_params = params
    return best_model, best_params, len(grid)


def run_training_cycle(seed: int, project_root: Path):
    raw_df = load_wine_dataset(DATA_DIR)
    duplicate_summary = duplicate_counts(raw_df)
    df = raw_df
    duplicates_before, duplicates_after, df = summarize_duplicates(df)
    if duplicates_before != duplicate_summary["rows_involved_in_duplicates"] or duplicates_after != 0:
        raise ValueError("Duplicate-removal statistics are inconsistent.")
    if duplicate_summary["rows_before"] != 6497 or duplicate_summary["rows_after"] != 5320:
        raise ValueError(f"Unexpected wine dataset row counts: {duplicate_summary}")
    print(
        f"Seed {seed}: {duplicate_summary['rows_involved_in_duplicates']} rows involved in duplicates; "
        f"{duplicate_summary['redundant_rows_removed']} redundant rows removed "
        f"({duplicate_summary['rows_before']} -> {duplicate_summary['rows_after']})"
    )

    split = get_training_sets(df, seed=seed, project_root=project_root)
    assign = {
        "seed": seed,
        **duplicate_summary,
        "good_ratio_before": float(raw_df["good"].mean()),
        "good_ratio_after": float(df["good"].mean()),
    }
    rows = []
    configs = {}
    predictions = []
    split_frame = pd.read_csv(RESULTS_DIR / f"split_seed{seed}.csv")
    test_indices = split_frame.loc[split_frame["split"] == "test", "sample_index"].to_numpy()
    test_metadata = df.iloc[test_indices]
    expected_good = (test_metadata["quality"].to_numpy() >= 7).astype(int)
    if not np.array_equal(expected_good, split["y_test"].to_numpy()):
        raise AssertionError("Cleaned test metadata does not align with test labels.")

    # The test set is evaluated only once per seed after all tuning.
    trivial_model = make_dummy_model(strategy="prior")
    trivial_model.fit(split["X_train"], split["y_train"])
    test_prob = trivial_model.predict_proba(split["X_test"])[:, 1]
    threshold, _ = choose_threshold(split["y_val"], trivial_model.predict_proba(split["X_val"])[:, 1])
    predictions.extend(
        {
            "sample_index": int(index),
            "model": "trivial_prior",
            "y_true": int(y),
            "quality": int(quality),
            "is_red": int(is_red),
            "probability": float(prob),
            "threshold": float(threshold),
        }
        for index, y, quality, is_red, prob in zip(
            test_indices, split["y_test"], test_metadata["quality"], test_metadata["is_red"], test_prob
        )
    )
    if any(row["y_true"] != int(row["quality"] >= 7) for row in predictions):
        raise AssertionError("Prediction metadata labels disagree with wine quality.")
    rows.extend(
        [
            {"model": "trivial_prior", "seed": seed, "metric": "balanced_accuracy_0.5", "value": float(balanced_accuracy_score(split["y_test"], (test_prob >= 0.5).astype(int))), "notes": f"threshold={threshold:.3f}; val_bal_acc={choose_threshold(split['y_val'], trivial_model.predict_proba(split['X_val'])[:,1])[1]:.4f}; config_count=1; tuning_metric=prior"},
            {"model": "trivial_prior", "seed": seed, "metric": "balanced_accuracy_val_threshold", "value": float(balanced_accuracy_score(split["y_test"], (test_prob >= threshold).astype(int))), "notes": f"threshold={threshold:.3f}; val_bal_acc={choose_threshold(split['y_val'], trivial_model.predict_proba(split['X_val'])[:,1])[1]:.4f}; config_count=1; tuning_metric=prior"},
            {"model": "trivial_prior", "seed": seed, "metric": "roc_auc", "value": float(roc_auc_score(split["y_test"], test_prob)), "notes": "config_count=1; tuning_metric=prior"},
            {"model": "trivial_prior", "seed": seed, "metric": "average_precision", "value": float(average_precision_score(split["y_test"], test_prob)), "notes": "config_count=1; tuning_metric=prior"},
            {"model": "trivial_prior", "seed": seed, "metric": "training_time_seconds", "value": float(0.001), "notes": "config_count=1; tuning_metric=prior"},
        ]
    )

    for model_name, tuner in [
        ("logistic_regression", tune_logistic),
        ("random_forest", tune_random_forest),
        ("gradient_boosting", tune_gradient_boosting),
    ]:
        if model_name == "logistic_regression":
            tuned_model, best_params, config_count = tuner(split["X_train"], split["y_train"], split["X_val"], split["y_val"], seed)
            print(f"{model_name}: {config_count} validation configs; selection metric = PR-AUC")
            extra = {"class_weight": "balanced"}
        elif model_name == "random_forest":
            tuned_model, best_params, config_count = tuner(split["X_train"], split["y_train"], split["X_val"], split["y_val"], seed)
            print(f"{model_name}: {config_count} validation configs; selection metric = PR-AUC")
            extra = {"class_weight": "balanced_subsample"}
        else:
            tuned_model, best_params, config_count = tuner(split["X_train"], split["y_train"], split["X_val"], split["y_val"], seed)
            print(f"{model_name}: {config_count} validation configs; selection metric = PR-AUC; early stopping trees = {int(getattr(tuned_model, 'n_iter_', 0))}")
            extra = {"class_weight": "balanced", "n_trees": int(getattr(tuned_model, 'n_iter_', 0))}

        configs[model_name] = {"params": best_params, **extra}
        row_batch, threshold, val_bal_acc, test_prob = evaluate_model_on_test(
            tuned_model,
            split["X_train"], split["y_train"],
            split["X_val"], split["y_val"],
            split["X_test"], split["y_test"],
            model_name=model_name,
            seed=seed,
            config_count=config_count,
            tuning_metric="PR-AUC",
            extra_meta=extra,
        )
        rows.extend(row_batch)
        predictions.extend(
            {
                "sample_index": int(index),
                "model": model_name,
                "y_true": int(y),
                "quality": int(quality),
                "is_red": int(is_red),
                "probability": float(prob),
                "threshold": float(threshold),
            }
            for index, y, quality, is_red, prob in zip(
                test_indices, split["y_test"], test_metadata["quality"], test_metadata["is_red"], test_prob
            )
        )
        if any(row["y_true"] != int(row["quality"] >= 7) for row in predictions[-len(test_indices):]):
            raise AssertionError(f"{model_name} prediction metadata labels disagree with wine quality.")

    with open(RESULTS_DIR / f"predictions_seed{seed}.csv", "w", encoding="utf-8") as fh:
        pd.DataFrame(predictions).to_csv(fh, index=False)

    return rows, assign, configs


def main():
    all_rows = []
    seed_summaries = []
    model_configs = {}
    for seed in SEEDS:
        rows, seed_summary, configs = run_training_cycle(seed, ROOT)
        all_rows.extend(rows)
        seed_summaries.append(seed_summary)
        model_configs[str(seed)] = configs

    path = ROOT / "results" / "metrics.csv"
    append_metrics_csv(all_rows, path)
    with open(RESULTS_DIR / "seed_summary.json", "w", encoding="utf-8") as fh:
        json.dump(seed_summaries[0], fh, indent=2)
    with open(RESULTS_DIR / "model_configs.json", "w", encoding="utf-8") as fh:
        json.dump(model_configs, fh, indent=2)
    summary = pd.read_csv(path)

    summary["value"] = pd.to_numeric(summary["value"], errors="coerce")
    summary_table = (
        summary.groupby(["model", "metric"], as_index=False)["value"].agg(["mean", "std"])
    ).reset_index()
    print(summary_table.to_string(index=False))

    rf = summary[(summary["model"] == "random_forest") & (summary["metric"] == "average_precision")].sort_values("seed")
    gb = summary[(summary["model"] == "gradient_boosting") & (summary["metric"] == "average_precision")].sort_values("seed")
    diff = rf["value"].to_numpy() - gb["value"].to_numpy()
    print(f"Paired RF-vs-GB PR-AUC difference over 5 seeds: mean={diff.mean():.4f}, std={diff.std(ddof=1):.4f}")
    print(f"Saved metrics to {path}")


if __name__ == "__main__":
    main()
