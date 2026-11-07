from __future__ import annotations

import json
import textwrap
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
    "trivial_prior": "Dummy prior",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}


def mean_sd(values: pd.Series) -> str:
    return f"{values.mean():.3f} ± {values.std(ddof=1):.3f}"


def add_image(ax, path: Path, title: str | None = None) -> None:
    ax.imshow(plt.imread(path))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, weight="bold", pad=5)


def style_table(table, size: float = 8) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(size)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#d9e7f5")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f3f6fa")


def image_page(pdf: PdfPages, title: str, subtitle: str, images: list[tuple[str, str]]) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(11.69, 8.27))
    if len(images) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=16, weight="bold", color="#17365d", y=.96)
    fig.text(.06, .90, subtitle, fontsize=9, wrap=True)
    for ax, (filename, label) in zip(axes, images):
        add_image(ax, FIGURES / filename, label)
    fig.tight_layout(rect=[.03, .04, .97, .87])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    metrics["value"] = pd.to_numeric(metrics["value"], errors="raise")
    dataset = json.loads((RESULTS / "dataset_summary.json").read_text())
    error = json.loads((RESULTS / "error_analysis.json").read_text())
    domain = pd.read_csv(RESULTS / "domain_shift_metrics.csv")
    domain_features = pd.read_csv(RESULTS / "domain_feature_comparison.csv")
    importance = pd.read_csv(RESULTS / "permutation_importance.csv")
    examples = pd.read_csv(RESULTS / "misclassified_wines.csv")
    rf_importance = importance[importance["model"] == "random_forest"].sort_values(
        "permutation_importance_mean", ascending=False
    )
    alcohol_rank = rf_importance["feature"].tolist().index("alcohol") + 1
    paired_rf = metrics[
        (metrics["model"] == "random_forest")
        & (metrics["metric"] == "average_precision")
    ].sort_values("seed")
    paired_gb = metrics[
        (metrics["model"] == "gradient_boosting")
        & (metrics["metric"] == "average_precision")
    ].sort_values("seed")
    if not np.array_equal(paired_rf["seed"].to_numpy(), paired_gb["seed"].to_numpy()):
        raise ValueError("RF and GB PR-AUC rows do not share the same seeds.")
    paired = paired_rf["value"].to_numpy() - paired_gb["value"].to_numpy()
    if not (paired > 0).all():
        raise ValueError("The all-seed RF PR-AUC headline is not supported.")
    seeds = sorted(int(seed) for seed in metrics["seed"].unique())
    seed_count = len(seeds)
    mixed_roc_mean = {
        model: float(metrics[
            (metrics["model"] == model) & (metrics["metric"] == "roc_auc")
        ]["value"].mean())
        for model in ("random_forest", "gradient_boosting")
    }
    full_target = domain[domain["evaluation_role"] == "target_full_domain"]
    transfer_roc = {
        (model, direction): float(full_target[
            (full_target["model"] == model) & (full_target["direction"] == direction)
        ]["roc_auc"].mean())
        for model in ("random_forest", "gradient_boosting")
        for direction in ("red_to_white", "white_to_red")
    }
    white_to_red_ba = {
        model: float(full_target[
            (full_target["model"] == model) & (full_target["direction"] == "white_to_red")
        ]["balanced_accuracy_at_source_threshold"].mean())
        for model in ("random_forest", "gradient_boosting")
    }

    ROOT.joinpath("report").mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "report" / "report.pdf"
    with PdfPages(report_path, metadata={
        "Title": "EC417 Mini Project P07 - Wine Quality Results",
        "Author": "EC417 Mini Project P07",
        "Subject": "Reproducible wine-quality classification results",
    }) as pdf:
        # 1. Problem, data, method, and compact results.
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(.07, .95, "EC417 Mini Project P07", fontsize=22, weight="bold", color="#17365d")
        fig.text(.07, .915, "Wine Quality Classification — Final Results", fontsize=14, color="#365f91")
        fig.text(.07, .865, "Problem and metric", fontsize=13, weight="bold")
        problem_text = textwrap.fill(
            f"Predict quality ≥ 7 from physicochemical measures in red and white Vinho Verde wines. "
            f"Only {dataset['good_wine_rate_after']:.1%} of cleaned rows are positive, so PR-AUC "
            "(average precision) is primary: it foregrounds positive-class retrieval and reflects the "
            "low base rate. Balanced accuracy is also reported, but its value at a fixed 0.5 cutoff "
            "depends on that operating point. The RF uses balanced_subsample class weights, so it is "
            "not an unweighted forest; its 0.5 cutoff is not assumed to be calibrated.",
            width=105,
        )
        fig.text(
            .07, .835, problem_text, fontsize=9.5, va="top",
        )
        fig.text(.07, .72, "Data and protocol", fontsize=13, weight="bold")
        data_text = textwrap.fill(
            f"UCI Wine Quality; CC BY 4.0. Rows before/after exact de-duplication: "
            f"{dataset['rows_before']:,} → {dataset['rows_after']:,}. "
            f"Rows involved in duplicate groups (keep=False): {dataset['rows_involved_in_duplicates']:,}; "
            f"redundant rows removed (keep='first'): {dataset['redundant_rows_removed']:,}. "
            f"Each of {seed_count} seeds uses a stratified "
            f"{dataset['split_fractions_seed0']['train']:.0%}/"
            f"{dataset['split_fractions_seed0']['val']:.0%}/"
            f"{dataset['split_fractions_seed0']['test']:.0%} train/validation/test split after "
            "de-duplication. Indices always resolve against the cleaned dataframe.",
            width=105,
        )
        fig.text(
            .07, .69, data_text, fontsize=9.5, va="top",
        )
        fig.text(.07, .57, "Method", fontsize=13, weight="bold")
        method_text = textwrap.fill(
            "Dummy prior baseline; logistic regression reference; random forest main model and "
            "histogram gradient boosting comparator. Configurations use validation PR-AUC; the sole "
            "operating-point improvement is to select threshold by validation balanced accuracy. "
            "The 11 physicochemical columns are inputs; is_red is not.",
            width=105,
        )
        fig.text(
            .07, .54, method_text, fontsize=9.5, va="top",
        )
        fig.text(.09, .44, "Raw red + white rows → de-duplicate → stratified split → select model on validation\n"
                 "                                                        → choose threshold on validation → test once",
                 fontsize=9, family="monospace", color="#365f91")

        model_order = [
            "trivial_prior", "logistic_regression", "random_forest", "gradient_boosting",
        ]
        metric_order = [
            ("balanced_accuracy_0.5", "BA @ 0.5"),
            ("balanced_accuracy_val_threshold", "BA @ val."),
            ("roc_auc", "ROC-AUC"),
            ("average_precision", "PR-AUC"),
            ("training_time_seconds", "Fit s"),
        ]
        table_rows = []
        for model in model_order:
            group = metrics[metrics["model"] == model]
            table_rows.append([
                LABELS[model],
                *[
                    mean_sd(group.loc[group["metric"] == metric, "value"])
                    for metric, _ in metric_order
                ],
            ])
        fig.text(.07, .385, f"Results — mean ± SD, {seed_count} seeds", fontsize=12, weight="bold")
        ax = fig.add_axes([.055, .20, .89, .15])
        ax.axis("off")
        result_table = ax.table(
            cellText=table_rows,
            colLabels=["Model", *(title for _, title in metric_order)],
            colWidths=[.25, .15, .16, .14, .14, .14],
            loc="center",
            cellLoc="center",
        )
        style_table(result_table, 7.5)
        early = metrics[
            (metrics["model"] == "gradient_boosting")
            & (metrics["metric"] == "early_stopping_trees")
        ].sort_values("seed")
        fig.text(
            .07, .12,
            "RF exceeded GB in PR-AUC on every paired seed. "
            f"Paired RF−GB differences: {', '.join(f'{value:+.3f}' for value in paired)}. "
            f"GB early stopping uses sklearn's internal validation split; iterations by seed: "
            f"{', '.join(str(int(value)) for value in early['value'])}.",
            fontsize=8.5, wrap=True,
        )
        fig.text(.07, .045, "See README.md for complete per-seed tables and secondary analyses.", fontsize=9)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 2. Seed-0 aggregate curves, not per-seed plots.
        image_page(
            pdf,
            "Results: held-out ranking curves",
            (
                "All models are overlaid on the seed-0 test set; selection and threshold choice used validation only. "
                f"RF−GB paired PR-AUC differences: {', '.join(f'{value:+.3f}' for value in paired)}. "
                "Possible explanations—not facts—include the small de-duplicated sample, noisy sensory labels, "
                "unstable GB early stopping, and the small tuning budget."
            ),
            [
                ("roc_curves_seed0.png", "ROC curves"),
                ("pr_curves_seed0.png", "Precision-recall curves"),
            ],
        )

        # 3. Feature interpretation and diagnostics.
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle("Feature importance and model diagnostics", fontsize=16, weight="bold", color="#17365d")
        fig.text(
            .055, .905,
            "Permutation importance is the test-set average-precision decrease (10 repeats); RF impurity "
            "importance is a normalized training-fit scale. Compare rankings within a method, not magnitudes across methods.",
            fontsize=9, wrap=True,
        )
        ax = fig.add_axes([.06, .46, .88, .41])
        add_image(ax, FIGURES / "feature_importance.png", "Permutation and impurity importance")
        ax = fig.add_axes([.045, .05, .45, .36])
        add_image(ax, FIGURES / "learning_curve.png", "Learning curve")
        ax = fig.add_axes([.515, .05, .45, .36])
        add_image(ax, FIGURES / "gb_learning_rate_curve.png", "GB learning-rate validation curve")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 4. Domain-shift figure plus the full-target-domain metrics.
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle("Domain shift: source-only models and target domains", fontsize=16, weight="bold", color="#17365d")
        fig.text(
            .045, .905,
            "Threshold is selected on source validation and fixed for target evaluation. The figure contrasts the mixed-domain test reference, the original target 20% test slice, and the entire cleaned target domain. Compare PR-AUC with its lift over target prevalence because base rates differ.",
            fontsize=8.5, wrap=True,
        )
        ax = fig.add_axes([.04, .47, .92, .40])
        add_image(ax, FIGURES / "domain_shift.png", "Mean ± SD over the saved seeds")
        full = domain[domain["evaluation_role"] == "target_full_domain"]
        rows = []
        for (model, direction), group in full.groupby(["model", "direction"], sort=False):
            first = group.sort_values("seed").iloc[0]
            rows.append([
                f"{LABELS[model]}, {direction.replace('_to_', ' → ')}",
                mean_sd(group["balanced_accuracy_at_source_threshold"]),
                mean_sd(group["roc_auc"]),
                mean_sd(group["pr_auc"]),
                mean_sd(group["pr_auc_lift_over_base_rate"]),
                f"{group['good_wine_rate'].mean():.3f}",
                f"{int(first['positive_count'])}/{int(first['n_test'])}",
            ])
        ax = fig.add_axes([.035, .12, .93, .26])
        ax.axis("off")
        domain_table = ax.table(
            cellText=rows,
            colLabels=[
                "Source-only model", "BA", "ROC-AUC", "PR-AUC",
                "PR-AUC lift", "Target rate", "Positives / rows",
            ],
            colWidths=[.21, .12, .12, .12, .14, .12, .15],
            loc="center",
            cellLoc="center",
        )
        style_table(domain_table, 7.5)
        top_shift = domain_features.assign(
            abs_shift=domain_features["standardized_mean_difference_red_minus_white"].abs()
        ).nlargest(3, "abs_shift")
        shift_names = ", ".join(feature.replace("_", " ") for feature in top_shift["feature"])
        fig.text(
            .05, .045,
            f"Red→white ROC-AUC falls by {mixed_roc_mean['random_forest'] - transfer_roc[('random_forest', 'red_to_white')]:.3f} "
            f"(RF) and {mixed_roc_mean['gradient_boosting'] - transfer_roc[('gradient_boosting', 'red_to_white')]:.3f} "
            " (GB) versus mixed-domain results; RF/GB target ranges overlap, so no transfer winner is claimed. "
            f"White→red BA is {white_to_red_ba['random_forest']:.3f} (RF) / "
            f"{white_to_red_ba['gradient_boosting']:.3f} (GB), near chance, while ROC-AUC is "
            f"{transfer_roc[('random_forest', 'white_to_red')]:.3f} / "
            f"{transfer_roc[('gradient_boosting', 'white_to_red')]:.3f}. Largest distribution shifts: " + shift_names +
            ". These feature shifts are hypotheses, not causal findings.",
            fontsize=8, wrap=True,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 5. Error analysis and one ten-row, full-feature table.
        fig = plt.figure(figsize=(16, 9))
        fig.suptitle("Error analysis — seed-0 random forest", fontsize=16, weight="bold", color="#17365d")
        rf = error["good_class_confusion_and_precision_recall"]["seed_0"]
        red = error["error_rate_by_wine_type"]["red"]
        white = error["error_rate_by_wine_type"]["white"]
        boundary_errors = error["quality_6_or_7_errors"]
        alcohol = error["alcohol_by_confident_error_group"]
        fig.text(
            .035, .92,
            f"Validation threshold {error['threshold']:.3f}; {error['errors']}/{error['test_rows']} errors. "
            f"Quality 6 or 7: {boundary_errors} errors ({error['quality_6_or_7_share_of_errors']:.1%}). "
            f"Red {red['errors']}/{red['test_rows']} ({red['error_rate']:.1%}); "
            f"white {white['errors']}/{white['test_rows']} ({white['error_rate']:.1%}). "
            f"Good-class TP/FP/FN/TN={rf['tp']}/{rf['fp']}/{rf['fn']}/{rf['tn']}; "
            f"precision={rf['precision_good']:.3f} (share of predicted-good wines that are good), "
            f"recall={rf['recall_good']:.3f} (share of good wines found).",
            fontsize=8.5, wrap=True,
        )
        fig.text(
            .035, .865,
            f"The ten rows below are five false positives and five false negatives, ranked by distance "
            f"from the threshold within each group. FP mean alcohol={alcohol['false_positives']['mean']:.2f}; "
            f"FN mean alcohol={alcohol['false_negatives']['mean']:.2f}. FP alcohol is higher on average, "
            f"consistent with alcohol being a dominant RF signal (permutation-importance rank {alcohol_rank}); "
            "this is based on only ten examples and does not establish cause.",
            fontsize=8.5, wrap=True,
        )
        display = examples.copy()
        display["error_type"] = display["error_type"].replace({
            "False positive": "FP", "False negative": "FN",
        })
        display_columns = [
            "error_type", "wine_type", "quality", "probability",
            "distance_from_threshold", "fixed_acidity", "volatile_acidity",
            "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide",
            "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol",
        ]
        display = display[display_columns].rename(columns={
            "error_type": "Err", "wine_type": "Type", "quality": "Q",
            "probability": "P(good)", "distance_from_threshold": "Dist.",
            "fixed_acidity": "Fixed acid.", "volatile_acidity": "Volatile acid.",
            "citric_acid": "Citric acid", "residual_sugar": "Residual sugar",
            "chlorides": "Chlor.", "free_sulfur_dioxide": "Free SO2",
            "total_sulfur_dioxide": "Total SO2", "density": "Density",
            "sulphates": "Sulph.", "alcohol": "Alcohol",
        })
        for column in display.columns[3:]:
            display[column] = pd.to_numeric(display[column]).map(lambda value: f"{value:.3g}")
        ax = fig.add_axes([.025, .13, .95, .68])
        ax.axis("off")
        error_table = ax.table(
            cellText=display.values,
            colLabels=display.columns,
            colWidths=[.045, .045, .025, .05, .05] + [.066] * 11,
            loc="center",
            cellLoc="center",
        )
        style_table(error_table, 6.1)
        error_table.scale(1, 1.8)
        fig.text(
            .035, .055,
            "Labels and feature values join through the cleaned sample_index; code asserts actual_good == (quality >= 7) for each row.",
            fontsize=8,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 6. Limits, reproducibility, AI-use, and next steps.
        reproducibility_path = RESULTS / "reproducibility.json"
        if reproducibility_path.exists():
            reproducibility = json.loads(reproducibility_path.read_text())
            runtime = (
                f"{reproducibility['clone_install_and_run_seconds']:.0f} seconds "
                f"({reproducibility['clone_install_and_run_seconds'] / 60:.1f} minutes)"
            )
            hardware = (
                f"{reproducibility['cpu_count']} logical CPUs, {reproducibility['memory_gib']:.1f} GiB RAM; "
                f"{reproducibility['platform']}; Python {reproducibility['python_version']}."
            )
        else:
            runtime = "not recorded yet"
            hardware = "See results/reproducibility.json after the fresh-clone check."
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(.07, .95, "Limitations, reproducibility, and next steps", fontsize=18, weight="bold", color="#17365d")
        limitations = [
            "Exact duplicates were removed; near-duplicates and source-level dependence may remain.",
            "Sensory labels are subjective/noisy; a binary threshold discards ordinal information.",
            (
                "This is one Vinho Verde dataset. The original red target slice has only "
                f"{int(domain[(domain['direction'] == 'white_to_red') & (domain['evaluation_role'] == 'target_domain_test')]['positive_count'].iloc[0])} positives."
            ),
            "Seeds overlap in their samples, so seed SD understates sampling uncertainty.",
            "Tuning grids are small; GB early stopping varies substantially and uses sklearn's internal split.",
            "is_red is not a feature. Educational use only; no commercial or human-impact decision use.",
        ]
        fig.text(.075, .895, "Limitations and ethics", fontsize=12, weight="bold")
        fig.text(
            .085, .855, "\n".join(f"• {line}" for line in limitations),
            fontsize=9.2, linespacing=1.5, va="top",
        )
        fig.text(.075, .68, "Reproduce", fontsize=12, weight="bold")
        fig.text(
            .085, .64,
            "From a fresh clone: python3 -m venv .venv; .venv/bin/pip install -r requirements.txt; "
            "./run_all.sh. This downloads UCI data, trains the saved seeds, regenerates CSV/JSON results "
            "and figures, writes README.md, and builds this report.",
            fontsize=9.2, wrap=True, va="top",
        )
        fig.text(.085, .57, f"Fresh-clone total (clone, environment, install, run): {runtime}. Hardware/software: {hardware}",
                 fontsize=9.2, wrap=True, va="top")
        fig.text(.075, .48, "What I would try next", fontsize=12, weight="bold")
        next_steps = [
            "Allocate a larger, predeclared validation budget to GB hyperparameter tuning.",
            "Use a cross-validated early-stopping scheme that respects the project's split.",
            "Evaluate probability calibration before choosing a probability-based operating threshold.",
            "Treat quality as ordinal and compare ordinal regression against this binary target.",
        ]
        fig.text(
            .085, .44, "\n".join(f"• {line}" for line in next_steps),
            fontsize=9.2, linespacing=1.5, va="top",
        )
        fig.text(.075, .31, "AI use — author TODO", fontsize=12, weight="bold")
        fig.text(
            .085, .27,
            "OpenAI Codex generated the code scaffold, experiments, and drafts; Claude was used for "
            "reviewing results and advice.\n\n"
            "TODO: The author must fill in what they personally verified, the evidence/method used, and "
            "what was changed after reviewing AI output. No personal verification is claimed here.",
            fontsize=9.2, wrap=True, va="top",
        )
        fig.text(.075, .14, "References", fontsize=12, weight="bold")
        fig.text(
            .085, .105,
            "Cortez et al. (2009), Decision Support Systems 47(4), 547–553, "
            "doi:10.1016/j.dss.2009.05.016; Breiman (2001), Machine Learning 45, 5–32; "
            "Friedman (2001), Annals of Statistics 29(5), 1189–1232; scikit-learn documentation "
            "for RandomForestClassifier, HistGradientBoostingClassifier, permutation importance, and average precision.",
            fontsize=8.5, wrap=True, va="top",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Generated six-page report: {report_path}")


if __name__ == "__main__":
    main()
