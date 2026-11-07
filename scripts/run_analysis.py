from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import (
    FEATURE_COLUMNS,
    duplicate_counts,
    get_training_sets,
    load_wine_dataset,
    summarize_duplicates,
)
from src.models import make_hist_gradient_boosting_model, make_random_forest_model
from src.train import (
    SEEDS,
    choose_threshold,
    tune_gradient_boosting,
    tune_random_forest,
)

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = ("random_forest", "gradient_boosting")
MODEL_LABELS = {
    "trivial_prior": "Trivial prior",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}
LABELS = {feature: feature.replace("_", " ").title() for feature in FEATURE_COLUMNS}


def build_model(name: str, params: dict, seed: int):
    if name == "random_forest":
        return make_random_forest_model(random_state=seed, **params)
    if name == "gradient_boosting":
        return make_hist_gradient_boosting_model(random_state=seed, **params)
    raise ValueError(f"Unsupported model: {name}")


def save_dataset_summary(raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    type_summary = {}
    for is_red, label in ((0, "white"), (1, "red")):
        group = clean[clean["is_red"] == is_red]
        type_summary[label] = {
            "rows": int(len(group)),
            "good_wines": int(group["good"].sum()),
            "good_wine_rate": float(group["good"].mean()),
        }
    split = pd.read_csv(RESULTS / "split_seed0.csv")
    summary = {
        **duplicate_counts(raw),
        "good_wine_rate_before": float(raw["good"].mean()),
        "good_wine_rate_after": float(clean["good"].mean()),
        "good_wines_after": int(clean["good"].sum()),
        "wine_type_after": type_summary,
        "split_rows_seed0": {
            name: int((split["split"] == name).sum())
            for name in ("train", "val", "test")
        },
        "split_fractions_seed0": {
            name: float((split["split"] == name).mean())
            for name in ("train", "val", "test")
        },
        "features": FEATURE_COLUMNS,
        "is_red_is_model_feature": False,
    }
    (RESULTS / "dataset_summary.json").write_text(json.dumps(summary, indent=2))


def save_eda(raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    type_labels = {0: "White", 1: "Red"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for is_red, label, color in [(0, "White", "#4778a8"), (1, "Red", "#bd4b4b")]:
        values = raw.loc[raw["is_red"] == is_red, "alcohol"]
        sns.histplot(values, stat="density", element="step", fill=False, kde=True,
                     label=label, color=color, ax=axes[0])
    axes[0].set(title="Alcohol distribution by wine type", xlabel="Alcohol (% by volume)", ylabel="Density")
    axes[0].legend(title="Wine type (is_red)")
    counts = raw["is_red"].map(type_labels).value_counts().reindex(["Red", "White"])
    axes[1].bar(counts.index, counts.values, color=["#bd4b4b", "#4778a8"])
    axes[1].set(title="Rows by wine type (is_red)", xlabel="Wine type", ylabel="Rows")
    for i, count in enumerate(counts.values):
        axes[1].text(i, count, f"{count:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_wine_type.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 9))
    corr_data = clean[FEATURE_COLUMNS + ["quality"]].rename(columns=LABELS | {"quality": "Quality score"})
    sns.heatmap(corr_data.corr(), annot=True, fmt=".2f", cmap="vlag", center=0, square=True,
                linewidths=.4, annot_kws={"size": 7}, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Feature and quality correlations (deduplicated wines)")
    ax.tick_params(axis="x", labelrotation=40)
    ax.tick_params(axis="y", labelrotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_correlation_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=raw, x="quality", discrete=True, hue="is_red", multiple="dodge",
                 palette={0: "#4778a8", 1: "#bd4b4b"}, shrink=.8,
                 hue_order=[0, 1], ax=ax)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("Wine type (is_red)")
        for text, label in zip(legend.get_texts(), ["White", "Red"]):
            text.set_text(label)
    ax.set(title="Wine quality score by wine type", xlabel="Quality score (0-10)", ylabel="Rows")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_quality_by_type.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=clean, x="density", hue="good", common_norm=False, fill=False,
                palette={0: "#4778a8", 1: "#bd4b4b"}, ax=ax)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("Quality class")
        for text, label in zip(legend.get_texts(), ["Not good (quality < 7)", "Good (quality >= 7)"]):
            text.set_text(label)
    ax.set(title="Wine density by quality class (deduplicated data)",
           xlabel="Density (g/cm³)", ylabel="Density")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_density_by_class.png", dpi=180)
    plt.close(fig)


def save_seed0_curves_and_confusions(predictions: pd.DataFrame) -> None:
    seed0 = predictions[predictions["seed"] == 0]
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in MODEL_LABELS:
        model = seed0[seed0["model"] == name]
        RocCurveDisplay.from_predictions(model["y_true"], model["probability"], name=MODEL_LABELS[name], ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC curves by model (seed 0 held-out test set)")
    fig.tight_layout()
    fig.savefig(FIGURES / "roc_curves_seed0.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in MODEL_LABELS:
        model = seed0[seed0["model"] == name]
        PrecisionRecallDisplay.from_predictions(model["y_true"], model["probability"],
                                                name=MODEL_LABELS[name], ax=ax)
    ax.set_title("Precision-recall curves by model (seed 0 held-out test set)")
    fig.tight_layout()
    fig.savefig(FIGURES / "pr_curves_seed0.png", dpi=180)
    plt.close(fig)

    for name in MODEL_LABELS:
        if name == "trivial_prior":
            continue
        model = seed0[seed0["model"] == name]
        threshold = float(model["threshold"].iloc[0])
        predicted = (model["probability"].to_numpy() >= threshold).astype(int)
        matrix = confusion_matrix(model["y_true"], predicted)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
                    xticklabels=["Not good", "Good"], yticklabels=["Not good", "Good"], ax=ax)
        ax.set(title=f"{MODEL_LABELS[name]} (seed 0, threshold={threshold:.3f})",
               xlabel="Predicted class", ylabel="Actual class")
        fig.tight_layout()
        fig.savefig(FIGURES / f"confusion_matrix_{name}_seed0.png", dpi=180)
        plt.close(fig)


def run_importance_and_learning_curve(split: dict, configs: dict) -> pd.DataFrame:
    importance_rows = []
    curve_rows = []
    for name in MODELS:
        params = configs[name]["params"]
        model = build_model(name, params, 0)
        model.fit(split["X_train"], split["y_train"])
        permutation = permutation_importance(
            model, split["X_test"], split["y_test"], scoring="average_precision",
            n_repeats=10, random_state=0, n_jobs=-1,
        )
        impurity = getattr(model, "feature_importances_", None)
        for i, feature in enumerate(FEATURE_COLUMNS):
            importance_rows.append({
                "model": name,
                "feature": feature,
                "permutation_importance_mean": float(permutation.importances_mean[i]),
                "permutation_importance_std": float(permutation.importances_std[i]),
                "rf_impurity_importance": float(impurity[i]) if impurity is not None else np.nan,
                "permutation_repeats": 10,
                "evaluation_set": "seed0_test",
                "scoring": "average_precision",
            })

        fractions = [.2, .4, .6, .8, 1.0]
        for fraction in fractions:
            n = max(2, int(len(split["X_train"]) * fraction))
            subset = split["X_train"].iloc[:n]
            subset_y = split["y_train"].iloc[:n]
            curve_model = build_model(name, params, 0)
            curve_model.fit(subset, subset_y)
            score = average_precision_score(split["y_val"], curve_model.predict_proba(split["X_val"])[:, 1])
            curve_rows.append({
                "model": name, "training_fraction": fraction, "training_rows": n,
                "validation_pr_auc": float(score), "seed": 0,
            })

    importance = pd.DataFrame(importance_rows)
    importance.to_csv(RESULTS / "permutation_importance.csv", index=False)
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(RESULTS / "learning_curve.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    ordered = importance.pivot(index="feature", columns="model", values="permutation_importance_mean")
    rf_impurity = importance[importance["model"] == "random_forest"].set_index("feature")["rf_impurity_importance"]
    feature_order = ordered["random_forest"].sort_values().index
    y = np.arange(len(feature_order))
    axes[0].barh(y - .18, ordered.loc[feature_order, "random_forest"], height=.35,
                 xerr=importance[importance.model == "random_forest"].set_index("feature")
                 .loc[feature_order, "permutation_importance_std"],
                 label="RF permutation (test AP decrease)", color="#f0a43b")
    axes[0].barh(y + .18, rf_impurity.loc[feature_order], height=.35,
                 label="RF impurity (normalized; train fit)", color="#4778a8")
    axes[0].set(yticks=y, yticklabels=[LABELS[f] for f in feature_order], title="Random forest")
    axes[0].legend(fontsize=8)
    gb_order = ordered["gradient_boosting"].sort_values().index
    axes[1].barh(np.arange(len(gb_order)), ordered.loc[gb_order, "gradient_boosting"],
                 xerr=importance[importance.model == "gradient_boosting"].set_index("feature")
                 .loc[gb_order, "permutation_importance_std"], color="#45a878")
    axes[1].set(yticks=np.arange(len(gb_order)), yticklabels=[LABELS[f] for f in gb_order],
                title="Gradient boosting")
    fig.suptitle("Importance scales differ: test AP decrease vs normalized training impurity")
    fig.supxlabel("Decrease in average precision after permutation")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, group in curve.groupby("model"):
        ax.plot(group["training_rows"], group["validation_pr_auc"], marker="o",
                label=MODEL_LABELS[name])
    ax.set(title="Learning curve (seed 0): validation average precision",
           xlabel="Training rows", ylabel="Validation PR-AUC")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "learning_curve.png", dpi=180)
    plt.close(fig)

    gb_params = configs["gradient_boosting"]["params"].copy()
    learning_rate_rows = []
    for rate in [.01, .03, .05, .1, .2]:
        params = {**gb_params, "learning_rate": rate}
        model = build_model("gradient_boosting", params, 0)
        model.fit(split["X_train"], split["y_train"])
        score = average_precision_score(split["y_val"], model.predict_proba(split["X_val"])[:, 1])
        learning_rate_rows.append({
            "learning_rate": rate,
            "validation_pr_auc": float(score),
            "early_stopping_iterations": int(model.n_iter_),
            "early_stopping_source": "sklearn_internal_validation",
            "seed": 0,
        })
    rate_frame = pd.DataFrame(learning_rate_rows)
    rate_frame.to_csv(RESULTS / "gb_learning_rate_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rate_frame["learning_rate"], rate_frame["validation_pr_auc"], marker="o", color="#a04d88")
    ax.set(title="Gradient boosting validation curve (seed 0)",
           xlabel="Learning rate", ylabel="Validation PR-AUC")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "gb_learning_rate_curve.png", dpi=180)
    plt.close(fig)
    return importance


def run_domain_shift(clean: pd.DataFrame, importance: pd.DataFrame) -> None:
    rows = []
    comparison = clean.groupby("is_red")[FEATURE_COLUMNS].agg(["mean", "std"])
    domain_features = []
    for feature in FEATURE_COLUMNS:
        red_mean = float(comparison.loc[1, (feature, "mean")])
        white_mean = float(comparison.loc[0, (feature, "mean")])
        red_std = float(comparison.loc[1, (feature, "std")])
        white_std = float(comparison.loc[0, (feature, "std")])
        pooled = np.sqrt((red_std**2 + white_std**2) / 2)
        domain_features.append({
            "feature": feature,
            "red_mean": red_mean,
            "red_std": red_std,
            "white_mean": white_mean,
            "white_std": white_std,
            "standardized_mean_difference_red_minus_white": (red_mean - white_mean) / pooled if pooled else 0.0,
        })

    for seed in SEEDS:
        split_frame = pd.read_csv(RESULTS / f"split_seed{seed}.csv")
        split_indices = {
            part: split_frame.loc[split_frame["split"] == part, "sample_index"].to_numpy()
            for part in ("train", "val", "test")
        }
        for source_color, source_flag, target_color, target_flag in [
            ("red", 1, "white", 0), ("white", 0, "red", 1)
        ]:
            source_train_idx = split_indices["train"][
                clean.iloc[split_indices["train"]]["is_red"].to_numpy() == source_flag
            ]
            source_val_idx = split_indices["val"][
                clean.iloc[split_indices["val"]]["is_red"].to_numpy() == source_flag
            ]
            source_test_idx = split_indices["test"][
                clean.iloc[split_indices["test"]]["is_red"].to_numpy() == source_flag
            ]
            target_test_idx = split_indices["test"][
                clean.iloc[split_indices["test"]]["is_red"].to_numpy() == target_flag
            ]
            target_domain = clean.loc[clean["is_red"] == target_flag]
            for name in MODELS:
                tuner = tune_random_forest if name == "random_forest" else tune_gradient_boosting
                model, selected_params, config_count = tuner(
                    clean.iloc[source_train_idx][FEATURE_COLUMNS],
                    clean.iloc[source_train_idx]["good"],
                    clean.iloc[source_val_idx][FEATURE_COLUMNS],
                    clean.iloc[source_val_idx]["good"],
                    seed=seed,
                )
                source_val = clean.iloc[source_val_idx]
                source_val_prob = model.predict_proba(source_val[FEATURE_COLUMNS])[:, 1]
                threshold, _ = choose_threshold(source_val["good"], source_val_prob)
                source_test = clean.iloc[source_test_idx]
                target_test = clean.iloc[target_test_idx]
                for evaluation_color, test_frame, role in [
                    (source_color, source_test, "source_in_domain_test"),
                    (target_color, target_test, "target_domain_test"),
                    (target_color, target_domain, "target_full_domain"),
                ]:
                    probability = model.predict_proba(test_frame[FEATURE_COLUMNS])[:, 1]
                    pr_auc = float(average_precision_score(test_frame["good"], probability))
                    rows.append({
                        "seed": seed,
                        "model": name,
                        "direction": f"{source_color}_to_{target_color}",
                        "evaluation_domain": evaluation_color,
                        "evaluation_role": role,
                        "threshold_from_source_validation": threshold,
                        "balanced_accuracy_at_source_threshold": float(
                            balanced_accuracy_score(test_frame["good"], probability >= threshold)
                        ),
                        "roc_auc": float(roc_auc_score(test_frame["good"], probability)),
                        "pr_auc": pr_auc,
                        "pr_auc_lift_over_base_rate": pr_auc / float(test_frame["good"].mean()),
                        "good_wine_rate": float(test_frame["good"].mean()),
                        "positive_count": int(test_frame["good"].sum()),
                        "n_test": len(test_frame),
                        "source_train_rows": len(source_train_idx),
                        "config_count": config_count,
                        "tuning_metric": "average_precision",
                        "selected_params": json.dumps(selected_params, sort_keys=True),
                        "early_stopping_trees": int(model.n_iter_) if name == "gradient_boosting" else np.nan,
                    })

    domain = pd.DataFrame(rows)
    mixed = pd.read_csv(RESULTS / "metrics.csv")
    mixed = mixed[(mixed["model"].isin(MODELS)) & mixed["metric"].isin(
        ["balanced_accuracy_val_threshold", "roc_auc", "average_precision"]
    )].pivot(index=["seed", "model"], columns="metric", values="value").reset_index()
    mixed = mixed.rename(columns={
        "balanced_accuracy_val_threshold": "mixed_domain_test_balanced_accuracy",
        "roc_auc": "mixed_domain_test_roc_auc",
        "average_precision": "mixed_domain_test_pr_auc",
    })
    domain = domain.merge(mixed, on=["seed", "model"], validate="many_to_one")
    domain.to_csv(RESULTS / "domain_shift_metrics.csv", index=False)

    importance_pivot = importance.pivot(index="feature", columns="model",
                                        values="permutation_importance_mean").reset_index()
    importance_pivot["absolute_rf_gb_importance_difference"] = (
        importance_pivot["random_forest"] - importance_pivot["gradient_boosting"]
    ).abs()
    features = pd.DataFrame(domain_features).merge(importance_pivot, on="feature")
    features.to_csv(RESULTS / "domain_feature_comparison.csv", index=False)

    series = [
        ("mixed", "Mixed-domain in-domain test", "#4778a8"),
        ("target_slice", "Source-only model, target 20% test slice", "#45a878"),
        ("target_full", "Source-only model, entire target domain", "#e08a3b"),
    ]
    roles = {
        "target_slice": "target_domain_test",
        "target_full": "target_full_domain",
    }
    metric_fields = [
        ("balanced_accuracy_at_source_threshold", "mixed_domain_test_balanced_accuracy", "Balanced accuracy"),
        ("roc_auc", "mixed_domain_test_roc_auc", "ROC-AUC"),
        ("pr_auc", "mixed_domain_test_pr_auc", "PR-AUC"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (transfer_metric, mixed_metric, title) in zip(axes, metric_fields):
        x_labels = []
        grouped_values = {name: {"mean": [], "std": []} for name, _, _ in series}
        for direction in ["red_to_white", "white_to_red"]:
            for name in MODELS:
                short = "RF" if name == "random_forest" else "GB"
                x_labels.append(f"{short}\n{direction.replace('_to_', '→').title()}")
                rows_for_model = domain[(domain["direction"] == direction) & (domain["model"] == name)]
                grouped_values["mixed"]["mean"].append(float(rows_for_model[mixed_metric].mean()))
                grouped_values["mixed"]["std"].append(float(rows_for_model[mixed_metric].std()))
                for series_name in ("target_slice", "target_full"):
                    evaluation = rows_for_model[rows_for_model["evaluation_role"] == roles[series_name]]
                    grouped_values[series_name]["mean"].append(float(evaluation[transfer_metric].mean()))
                    grouped_values[series_name]["std"].append(float(evaluation[transfer_metric].std()))
        positions = np.arange(len(x_labels))
        width = .24
        for offset, (name, label, color) in enumerate(series):
            ax.bar(positions + (offset - 1) * width, grouped_values[name]["mean"], width=width,
                   yerr=grouped_values[name]["std"], capsize=2, label=label, color=color)
        ax.set(title=title, xticks=positions, xticklabels=x_labels, ylim=(0, 1),
               ylabel="Score" if ax is axes[0] else "")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.2)
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Mixed-domain baseline vs source-only and transfer tests (mean ± SD, 5 seeds)")
    fig.tight_layout()
    fig.savefig(FIGURES / "domain_shift.png", dpi=180)
    plt.close(fig)


def run_error_analysis(predictions: pd.DataFrame, clean: pd.DataFrame) -> None:
    rf_predictions = predictions[predictions["model"] == "random_forest"].copy()
    if not np.array_equal(
        rf_predictions["y_true"].to_numpy(),
        (rf_predictions["quality"].to_numpy() >= 7).astype(int),
    ):
        raise AssertionError("Actual good labels disagree with quality for saved RF predictions.")
    rf_predictions["predicted_good"] = (
        rf_predictions["probability"] >= rf_predictions["threshold"]
    ).astype(int)
    seed_metrics = []
    for seed, group in rf_predictions.groupby("seed", sort=True):
        actual = group["y_true"].to_numpy()
        predicted = group["predicted_good"].to_numpy()
        tp = int(((actual == 1) & (predicted == 1)).sum())
        fp = int(((actual == 0) & (predicted == 1)).sum())
        fn = int(((actual == 1) & (predicted == 0)).sum())
        tn = int(((actual == 0) & (predicted == 0)).sum())
        seed_metrics.append({
            "seed": int(seed),
            "threshold": float(group["threshold"].iloc[0]),
            "test_rows": int(len(group)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision_good": tp / (tp + fp) if tp + fp else 0.0,
            "recall_good": tp / (tp + fn) if tp + fn else 0.0,
        })
    if len(seed_metrics) != len(SEEDS):
        raise AssertionError(f"Expected RF predictions for all {len(SEEDS)} seeds.")
    seed0 = rf_predictions[rf_predictions["seed"] == 0].copy()
    metadata = clean[["quality", "good", "is_red", *FEATURE_COLUMNS]].copy()
    metadata.index.name = "sample_index"
    seed0 = seed0.drop(columns=["quality", "is_red"], errors="ignore").merge(
        metadata.reset_index(), on="sample_index", validate="one_to_one"
    )
    actual_good = seed0["y_true"].to_numpy()
    if not np.array_equal(actual_good, (seed0["quality"].to_numpy() >= 7).astype(int)):
        raise AssertionError("Actual labels disagree with quality after joining cleaned test metadata.")
    if not (seed0["y_true"].to_numpy() == seed0["good"].to_numpy()).all():
        raise AssertionError("Prediction labels disagree with cleaned dataset labels.")
    seed0["predicted_good"] = (seed0["probability"] >= seed0["threshold"]).astype(int)
    seed0["is_error"] = seed0["predicted_good"] != seed0["y_true"]
    seed0["error_type"] = np.select(
        [
            (seed0["y_true"] == 0) & (seed0["predicted_good"] == 1),
            (seed0["y_true"] == 1) & (seed0["predicted_good"] == 0),
        ],
        ["false_positive", "false_negative"],
        default="correct",
    )
    errors = seed0[seed0["is_error"]].copy()
    errors["distance_from_threshold"] = np.where(
        errors["error_type"] == "false_positive",
        errors["probability"] - errors["threshold"],
        errors["threshold"] - errors["probability"],
    )
    errors["wine_type"] = errors["is_red"].map({0: "White", 1: "Red"})
    errors["prediction"] = errors["predicted_good"].map({0: "Not good", 1: "Good"})
    errors["actual"] = errors["y_true"].map({0: "Not good", 1: "Good"})
    errors["error_type"] = errors["error_type"].map({
        "false_positive": "False positive", "false_negative": "False negative"
    })
    color_rates = {}
    grouped = seed0.groupby("is_red").agg(
        errors=("is_error", "sum"),
        false_positives=("error_type", lambda x: int((x == "false_positive").sum())),
        false_negatives=("error_type", lambda x: int((x == "false_negative").sum())),
        test_rows=("is_error", "size"),
        error_rate=("is_error", "mean"),
    ).rename(index={0: "white", 1: "red"})
    for color, group in grouped.iterrows():
        color_rates[color] = {
            "errors": int(group["errors"]),
            "false_positives": int(group["false_positives"]),
            "false_negatives": int(group["false_negatives"]),
            "test_rows": int(group["test_rows"]),
            "error_rate": float(group["error_rate"]),
        }
    quality_counts = {}
    for quality in sorted(seed0["quality"].unique()):
        group = seed0[seed0["quality"] == quality]
        quality_counts[str(int(quality))] = {
            "test_rows": int(len(group)),
            "errors": int(group["is_error"].sum()),
            "error_rate": float(group["is_error"].mean()),
            "false_positives": int((group["error_type"] == "false_positive").sum()),
            "false_negatives": int((group["error_type"] == "false_negative").sum()),
        }
    error_examples = pd.concat([
        errors[errors["error_type"] == "False positive"].sort_values(
            "distance_from_threshold", ascending=False
        ).head(5),
        errors[errors["error_type"] == "False negative"].sort_values(
            "distance_from_threshold", ascending=False
        ).head(5),
    ])
    fp_alcohol = error_examples.loc[
        error_examples["error_type"] == "False positive", "alcohol"
    ]
    fn_alcohol = error_examples.loc[
        error_examples["error_type"] == "False negative", "alcohol"
    ]
    metric_names = ("tp", "fp", "fn", "tn", "precision_good", "recall_good")
    all_seed_summary = {
        metric: {
            "mean": float(np.mean([row[metric] for row in seed_metrics])),
            "std": float(np.std([row[metric] for row in seed_metrics], ddof=1)),
        }
        for metric in metric_names
    }
    quality_6_or_7_errors = sum(
        quality_counts.get(str(quality), {}).get("errors", 0) for quality in (6, 7)
    )
    summary = {
        "model": "random_forest",
        "seed": 0,
        "threshold": float(seed0["threshold"].iloc[0]),
        "test_rows": int(len(seed0)),
        "errors": int(len(errors)),
        "error_rate": float(seed0["is_error"].mean()),
        "error_counts_by_quality": quality_counts,
        "error_rate_by_wine_type": color_rates,
        "quality_6_or_7_errors": quality_6_or_7_errors,
        "other_quality_errors": int(len(errors) - quality_6_or_7_errors),
        "quality_6_or_7_share_of_errors": (
            quality_6_or_7_errors / len(errors) if len(errors) else 0.0
        ),
        "alcohol_by_confident_error_group": {
            "false_positives": {
                "count": int(len(fp_alcohol)),
                "mean": float(fp_alcohol.mean()),
            },
            "false_negatives": {
                "count": int(len(fn_alcohol)),
                "mean": float(fn_alcohol.mean()),
            },
        },
        "good_class_confusion_and_precision_recall": {
            "seed_0": seed_metrics[0],
            "all_seeds": {
                "seed_count": len(seed_metrics),
                "per_seed": seed_metrics,
                "mean_std": all_seed_summary,
            },
        },
    }
    with (RESULTS / "error_analysis.json").open("w") as f:
        json.dump(summary, f, indent=2)
    columns = [
        "error_type", "sample_index", "wine_type", "quality", "actual", "prediction",
        "probability", "threshold", "distance_from_threshold", *FEATURE_COLUMNS,
    ]
    if len(error_examples) != 10 or error_examples["error_type"].value_counts().to_dict() != {
        "False positive": 5, "False negative": 5
    }:
        raise AssertionError("Expected five false positives and five false negatives for the examples table.")
    error_examples[columns].to_csv(RESULTS / "misclassified_wines.csv", index=False)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in FIGURES.glob("*.png"):
        path.unlink()
    raw = load_wine_dataset(ROOT / "data" / "raw")
    _, _, clean = summarize_duplicates(raw)
    save_dataset_summary(raw, clean)
    with (RESULTS / "model_configs.json").open() as f:
        configs = json.load(f)["0"]
    predictions = pd.concat(
        [pd.read_csv(RESULTS / f"predictions_seed{seed}.csv").assign(seed=seed) for seed in SEEDS],
        ignore_index=True,
    )
    actual_good = predictions["y_true"].to_numpy()
    if not np.array_equal(actual_good, (predictions["quality"].to_numpy() >= 7).astype(int)):
        raise AssertionError("Prediction metadata labels disagree with quality.")
    for seed, group in predictions.groupby("seed"):
        expected = clean.iloc[group["sample_index"].to_numpy()]
        if not np.array_equal(group["quality"].to_numpy(), expected["quality"].to_numpy()):
            raise AssertionError(f"Prediction quality metadata does not align with clean rows for seed {seed}.")
        if not np.array_equal(group["is_red"].to_numpy(), expected["is_red"].to_numpy()):
            raise AssertionError(f"Prediction wine-type metadata does not align with clean rows for seed {seed}.")
    save_eda(raw, clean)
    save_seed0_curves_and_confusions(predictions)
    sets = get_training_sets(clean, seed=0, project_root=ROOT)
    importance = run_importance_and_learning_curve(sets, configs)
    run_domain_shift(clean, importance)
    run_error_analysis(predictions, clean)
    print("Saved consolidated EDA, model-analysis, domain-shift, and error-analysis outputs.")


if __name__ == "__main__":
    main()
