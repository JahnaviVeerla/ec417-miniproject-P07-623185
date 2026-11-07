from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
LABELS = {
    "trivial_prior": "Trivial prior",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}


def add_image(ax, path: Path, title: str | None = None) -> None:
    image = plt.imread(path)
    ax.imshow(image)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


def style_table(table, size: float = 8) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(size)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#d9e7f5")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f3f6fa")


def main() -> None:
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    metrics["value"] = pd.to_numeric(metrics["value"], errors="coerce")
    summary = metrics[metrics["metric"].isin([
        "balanced_accuracy_0.5", "balanced_accuracy_val_threshold", "roc_auc",
        "average_precision", "training_time_seconds",
    ])].groupby(["model", "metric"])["value"].agg(["mean", "std"]).fillna(0)
    with (RESULTS / "seed_summary.json").open() as f:
        duplicates = json.load(f)
    error = json.loads((RESULTS / "error_analysis.json").read_text())
    domain = pd.read_csv(RESULTS / "domain_shift_metrics.csv")
    features = pd.read_csv(RESULTS / "domain_feature_comparison.csv")
    errors = pd.read_csv(RESULTS / "misclassified_wines.csv")

    with PdfPages(ROOT / "report" / "report.pdf", metadata={
        "Title": "EC417 Mini Project P07 - Results Report",
        "Author": "EC417 Mini Project P07",
        "Subject": "Wine-quality evaluation and additional analyses",
    }) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(.08, .95, "EC417 Mini Project P07", fontsize=22, weight="bold", color="#17365d")
        fig.text(.08, .915, "Wine Quality Classification — Results and Analysis", fontsize=14, color="#365f91")
        fig.text(.08, .86, "Protocol", fontsize=13, weight="bold")
        fig.text(.08, .825, "Binary target: quality >= 7. Deduplicated data are split 60/20/20, stratified by "
                 "quality class and wine type, over seeds 0–4. Model selection and thresholds use validation "
                 "data only; the reported test results are held out.", fontsize=9, wrap=True)
        fig.text(.08, .755, "Data cleaning", fontsize=13, weight="bold")
        fig.text(.08, .72, f"Raw rows: {duplicates['rows_before']:,}; rows involved in any duplicate "
                 f"(keep=False): {duplicates['rows_involved_in_duplicates']:,}; redundant rows removed "
                 f"(keep='first'): {duplicates['redundant_rows_removed']:,}; retained rows: "
                 f"{duplicates['rows_after']:,}.", fontsize=9, wrap=True)
        fig.text(.08, .65, "Test performance — mean ± standard deviation over five seeds", fontsize=12, weight="bold")
        models = ["trivial_prior", "logistic_regression", "random_forest", "gradient_boosting"]
        metrics_order = [
            ("balanced_accuracy_0.5", "BA @ 0.5"),
            ("balanced_accuracy_val_threshold", "BA @ val. threshold"),
            ("roc_auc", "ROC-AUC"), ("average_precision", "PR-AUC"),
        ]
        rows = []
        for model in models:
            rows.append([LABELS[model], *[
                f"{summary.loc[(model, metric), 'mean']:.3f} ± {summary.loc[(model, metric), 'std']:.3f}"
                for metric, _ in metrics_order
            ]])
        ax = fig.add_axes([.06, .44, .88, .18])
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=["Model", *(title for _, title in metrics_order)],
                         colWidths=[.25, .17, .24, .16, .15], loc="center", cellLoc="center")
        style_table(table, 8)
        gb_trees = metrics[(metrics.model == "gradient_boosting") &
                           (metrics.metric == "early_stopping_trees")].sort_values("seed")["value"].astype(int)
        fig.text(.08, .37, f"GB early-stopping iterations by seed: {', '.join(map(str, gb_trees))}. "
                 "Stopping uses sklearn's internal training split, not the project validation set.", fontsize=9)
        fig.text(.08, .31, "All results are from fresh, reproducible model runs; see README.md for domain-transfer, "
                 "feature-importance, learning-curve, and error-analysis details.", fontsize=9, wrap=True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(3, 2, figsize=(11.69, 8.27))
        fig.suptitle("Seed 0 test curves and validation-threshold confusion matrices", fontsize=14, weight="bold")
        add_image(axes[0, 0], FIGURES / "roc_curves_seed0.png", "ROC curves")
        add_image(axes[0, 1], FIGURES / "pr_curves_seed0.png", "Precision-recall curves")
        for ax, model in zip(axes[1:].flat, LABELS):
            add_image(ax, FIGURES / f"confusion_matrix_{model}_seed0.png", LABELS[model])
        fig.tight_layout(rect=[0, 0, 1, .95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        fig.suptitle("Exploratory data analysis", fontsize=14, weight="bold")
        eda = [
            ("eda_wine_type.png", "Wine type and alcohol"),
            ("eda_correlation_heatmap.png", "Annotated feature correlations"),
            ("eda_quality_by_type.png", "Quality by wine type"),
            ("eda_density_by_class.png", "Density by quality class"),
        ]
        for ax, (filename, title) in zip(axes.flat, eda):
            add_image(ax, FIGURES / filename, title)
        fig.tight_layout(rect=[0, 0, 1, .95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(11.69, 8.27))
        grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
        fig.suptitle("Feature importance and model diagnostics", fontsize=14, weight="bold")
        ax = fig.add_subplot(grid[0, :])
        add_image(ax, FIGURES / "feature_importance.png", "Permutation importance (10 test-set repeats) and RF impurity importance")
        ax = fig.add_subplot(grid[1, 0])
        add_image(ax, FIGURES / "learning_curve.png", "Learning curve")
        ax = fig.add_subplot(grid[1, 1])
        add_image(ax, FIGURES / "gb_learning_rate_curve.png", "GB learning-rate validation curve")
        fig.tight_layout(rect=[0, 0, 1, .95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(11.69, 8.27))
        grid = fig.add_gridspec(2, 1, height_ratios=[.9, 1.3])
        fig.suptitle("Red/white domain-shift experiment (five seeds)", fontsize=14, weight="bold")
        ax = fig.add_subplot(grid[0])
        add_image(ax, FIGURES / "domain_shift.png", "Bars show mean ± SD across five seed-specific test splits")
        display_rows = []
        for (model, direction), group in domain.groupby(["model", "direction"], sort=False):
            source = group[group["evaluation_role"] == "source_in_domain_test"]
            target = group[group["evaluation_role"] == "target_domain_test"]
            mixed = group.drop_duplicates("seed")
            for role, frame, score_fields in [
                ("Mixed in-domain", mixed, (
                    "mixed_domain_test_balanced_accuracy", "mixed_domain_test_roc_auc",
                    "mixed_domain_test_pr_auc",
                )),
                ("Source-only source test", source, (
                    "balanced_accuracy_at_source_threshold", "roc_auc", "pr_auc",
                )),
                ("Source-only target test", target, (
                    "balanced_accuracy_at_source_threshold", "roc_auc", "pr_auc",
                )),
            ]:
                target_stats = (
                    f"{target['good_wine_rate'].mean():.3f}±{target['good_wine_rate'].std():.3f}",
                    f"{target['positive_count'].mean():.0f} / {target['n_test'].mean():.0f}",
                ) if role == "Source-only target test" else ("—", "—")
                display_rows.append([
                    LABELS[model], direction.replace("_to_", " → "), role,
                    *(f"{frame[field].mean():.3f}±{frame[field].std():.3f}" for field in score_fields),
                    *target_stats,
                ])
        ax = fig.add_subplot(grid[1])
        ax.axis("off")
        table = ax.table(cellText=display_rows, colLabels=[
            "Model", "Transfer", "Evaluation subset",
            "BA", "ROC-AUC", "PR-AUC",
            "Target good rate\n(mean±SD)", "Positives / n",
        ], colWidths=[.13, .10, .20, .11, .11, .11, .13, .11], loc="center", cellLoc="center")
        style_table(table, 7.5)
        table.scale(1, 1.25)
        top_smd = features.assign(abs_smd=features["standardized_mean_difference_red_minus_white"].abs()) \
            .nlargest(4, "abs_smd")
        top_importance = features.nlargest(3, "absolute_rf_gb_importance_difference")
        fig.text(.04, .015, "Largest red/white distribution shifts (|standardized mean difference|): " +
                 "; ".join(f"{row.feature.replace('_', ' ')} ({row.standardized_mean_difference_red_minus_white:+.2f})"
                           for row in top_smd.itertuples()) +
                 ". Largest RF–GB permutation-importance differences: " +
                 ", ".join(row.feature.replace("_", " ") for row in top_importance.itertuples()) +
                 ". Each metric is mean±SD over five seeds; target prevalence and positive count are for its target test sets. "
                 "Thresholds are fixed on source validation.",
                 fontsize=8, wrap=True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(16, 9))
        fig.suptitle("Error analysis — random forest, seed 0 test set", fontsize=15, weight="bold")
        white = error["error_rate_by_wine_type"]["white"]
        red = error["error_rate_by_wine_type"]["red"]
        quality_summary = "; ".join(
            f"Q{quality}: {counts['errors']}/{counts['test_rows']} errors"
            for quality, counts in sorted(error["error_counts_by_quality"].items(), key=lambda item: int(item[0]))
        )
        boundary_errors = sum(
            error["error_counts_by_quality"].get(str(quality), {}).get("errors", 0)
            for quality in (6, 7)
        )
        fig.text(.04, .925,
                 f"Validation-chosen threshold: {error['threshold']:.3f}. {error['errors']} errors among "
                 f"{error['test_rows']} test rows ({error['error_rate']:.1%}). Quality 6 or 7 accounts for "
                 f"{boundary_errors} errors; by quality: {quality_summary}.",
                 fontsize=8.5, wrap=True)
        fig.text(.04, .875, f"Red errors: {red['errors']}/{red['test_rows']} "
                 f"({red['error_rate']:.1%}; {red['false_positives']} FP, {red['false_negatives']} FN). "
                 f"White errors: {white['errors']}/{white['test_rows']} "
                 f"({white['error_rate']:.1%}; {white['false_positives']} FP, {white['false_negatives']} FN). "
                 "Five confident false positives and five confident false negatives, ranked by distance from "
                 "the threshold within each group; all feature values are from the matching cleaned row.",
                 fontsize=8.5, wrap=True)
        shown = errors.copy()
        shown["probability"] = shown["probability"].map("{:.3f}".format)
        shown["distance_from_threshold"] = shown["distance_from_threshold"].map("{:.3f}".format)
        display_columns = [
            "error_type", "wine_type", "quality", "probability", "distance_from_threshold", *[
                "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol",
            ],
        ]
        display = shown[display_columns].copy()
        display = display.rename(columns={
            "error_type": "Error", "wine_type": "Wine", "quality": "Q",
            "probability": "P(good)", "distance_from_threshold": "Dist.",
            "fixed_acidity": "Fixed\nacid.", "volatile_acidity": "Volatile\nacid.",
            "citric_acid": "Citric\nacid", "residual_sugar": "Residual\nsugar",
            "chlorides": "Chlor.", "free_sulfur_dioxide": "Free SO2",
            "total_sulfur_dioxide": "Total SO2", "density": "Density",
            "sulphates": "Sulph.", "alcohol": "Alcohol",
        })
        display["Error"] = display["Error"].replace({
            "False positive": "FP", "False negative": "FN"
        })
        for column in display.columns[5:]:
            display[column] = display[column].map(lambda value: f"{value:.4g}")
        ax = fig.add_axes([.025, .08, .95, .72])
        ax.axis("off")
        widths = [.07, .05, .035, .055, .05] + [.067] * 11
        table = ax.table(cellText=display.values, colLabels=display.columns,
                         colWidths=widths, loc="center", cellLoc="center")
        style_table(table, 6.4)
        table.scale(1, 2.0)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Generated six-page report: {ROOT / 'report' / 'report.pdf'}")


if __name__ == "__main__":
    main()
