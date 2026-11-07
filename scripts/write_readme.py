from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPOSITORY_URL = "https://github.com/JahnaviVeerla/ec417-miniproject-P07-623185.git"
LABELS = {
    "trivial_prior": "Dummy prior",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}
METRIC_LABELS = {
    "balanced_accuracy_0.5": "BA @ 0.5",
    "balanced_accuracy_val_threshold": "BA @ validation threshold",
    "roc_auc": "ROC-AUC",
    "average_precision": "PR-AUC",
    "training_time_seconds": "Fit time (s)",
}


def mean_sd(values: pd.Series | list[float]) -> tuple[float, float]:
    values = pd.Series(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def cell(values: pd.Series | list[float], digits: int = 3) -> str:
    mean, std = mean_sd(values)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def load_outputs() -> tuple[pd.DataFrame, dict, dict, pd.DataFrame, dict]:
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    metrics["value"] = pd.to_numeric(metrics["value"], errors="raise")
    dataset = json.loads((RESULTS / "dataset_summary.json").read_text())
    errors = json.loads((RESULTS / "error_analysis.json").read_text())
    domain = pd.read_csv(RESULTS / "domain_shift_metrics.csv")
    features = pd.read_csv(RESULTS / "domain_feature_comparison.csv")
    importance = pd.read_csv(RESULTS / "permutation_importance.csv")
    return metrics, dataset, errors, domain, {
        "features": features,
        "importance": importance,
    }


def model_row(metrics: pd.DataFrame, model: str) -> list[str]:
    rows = metrics[metrics["model"] == model]
    return [
        LABELS[model],
        *[
            cell(rows.loc[rows["metric"] == metric, "value"])
            for metric in METRIC_LABELS
        ],
    ]


def domain_triplet(frame: pd.DataFrame, columns: tuple[str, str, str]) -> str:
    return " / ".join(cell(frame[column]) for column in columns)


def source_slice_table(domain: pd.DataFrame) -> str:
    rows = []
    for model in ("random_forest", "gradient_boosting"):
        for direction in ("red_to_white", "white_to_red"):
            group = domain[(domain["model"] == model) & (domain["direction"] == direction)]
            mixed = group.drop_duplicates("seed")
            source = group[group["evaluation_role"] == "source_in_domain_test"]
            target = group[group["evaluation_role"] == "target_domain_test"]
            target_first = target.sort_values("seed").iloc[0]
            rows.append([
                f"{LABELS[model]}, {direction.replace('_to_', ' → ')}",
                domain_triplet(mixed, (
                    "mixed_domain_test_balanced_accuracy",
                    "mixed_domain_test_roc_auc",
                    "mixed_domain_test_pr_auc",
                )),
                domain_triplet(source, (
                    "balanced_accuracy_at_source_threshold", "roc_auc", "pr_auc",
                )),
                domain_triplet(target, (
                    "balanced_accuracy_at_source_threshold", "roc_auc", "pr_auc",
                )),
                cell(target["pr_auc_lift_over_base_rate"]),
                f"{target['good_wine_rate'].mean():.3f}",
                f"{int(target_first['positive_count'])} / {int(target_first['n_test'])}",
            ])
    return table(
        [
            "Model and direction",
            "Mixed-domain test<br>BA / ROC-AUC / PR-AUC",
            "Source-only source test<br>BA / ROC-AUC / PR-AUC",
            "Source-only target 20% test slice<br>BA / ROC-AUC / PR-AUC",
            "Target-slice PR-AUC lift",
            "Target good rate",
            "Positives / rows",
        ],
        rows,
    )


def full_domain_table(domain: pd.DataFrame) -> str:
    rows = []
    for model in ("random_forest", "gradient_boosting"):
        for direction in ("red_to_white", "white_to_red"):
            group = domain[
                (domain["model"] == model)
                & (domain["direction"] == direction)
                & (domain["evaluation_role"] == "target_full_domain")
            ].sort_values("seed")
            first = group.iloc[0]
            rows.append([
                f"{LABELS[model]}, {direction.replace('_to_', ' → ')}",
                cell(group["balanced_accuracy_at_source_threshold"]),
                cell(group["roc_auc"]),
                cell(group["pr_auc"]),
                cell(group["pr_auc_lift_over_base_rate"]),
                f"{group['good_wine_rate'].mean():.3f}",
                f"{int(first['positive_count'])} / {int(first['n_test'])}",
            ])
    return table(
        [
            "Source-only model and direction",
            "BA at fixed threshold",
            "ROC-AUC",
            "PR-AUC",
            "PR-AUC / target rate (lift)",
            "Target good rate",
            "Positives / all target rows",
        ],
        rows,
    )


def generate_readme() -> str:
    metrics, dataset, error, domain, extras = load_outputs()
    features = extras["features"]
    importance = extras["importance"]
    seeds = sorted(int(seed) for seed in metrics["seed"].unique())
    seed_count = len(seeds)
    rf_pr = metrics[
        (metrics["model"] == "random_forest")
        & (metrics["metric"] == "average_precision")
    ].sort_values("seed")
    gb_pr = metrics[
        (metrics["model"] == "gradient_boosting")
        & (metrics["metric"] == "average_precision")
    ].sort_values("seed")
    if not np.array_equal(rf_pr["seed"].to_numpy(), gb_pr["seed"].to_numpy()):
        raise ValueError("RF and GB PR-AUC rows do not share the same seed set.")
    paired = rf_pr["value"].to_numpy() - gb_pr["value"].to_numpy()
    if not (paired > 0).all():
        raise ValueError("The claimed RF PR-AUC win on every seed is not supported.")

    positive_rate = dataset["good_wine_rate_after"]
    seed_error = error["good_class_confusion_and_precision_recall"]["seed_0"]
    all_seed_error = error["good_class_confusion_and_precision_recall"]["all_seeds"]
    mean_std = all_seed_error["mean_std"]
    per_seed_text = ", ".join(
        f"seed {seed}: {difference:+.3f}"
        for seed, difference in zip(seeds, paired)
    )

    model_rows = [model_row(metrics, model) for model in LABELS]
    ba_means = {
        model: {
            metric: float(metrics[
                (metrics["model"] == model) & (metrics["metric"] == metric)
            ]["value"].mean())
            for metric in ("balanced_accuracy_0.5", "balanced_accuracy_val_threshold")
        }
        for model in ("logistic_regression", "random_forest", "gradient_boosting")
    }
    result_table = table(
        ["Model", *METRIC_LABELS.values()],
        model_rows,
    )

    quality_rows = []
    for quality, counts in sorted(
        error["error_counts_by_quality"].items(), key=lambda item: int(item[0])
    ):
        quality_rows.append([
            quality,
            str(counts["test_rows"]),
            str(counts["errors"]),
            f"{counts['error_rate']:.1%}",
            str(counts["false_positives"]),
            str(counts["false_negatives"]),
        ])
    quality_table = table(
        ["Quality", "Test rows", "Errors", "Error rate", "FP", "FN"],
        quality_rows,
    )

    red = error["error_rate_by_wine_type"]["red"]
    white = error["error_rate_by_wine_type"]["white"]
    boundary_share = error["quality_6_or_7_share_of_errors"]
    fp_alcohol = error["alcohol_by_confident_error_group"]["false_positives"]
    fn_alcohol = error["alcohol_by_confident_error_group"]["false_negatives"]
    rf_importance = importance[importance["model"] == "random_forest"].sort_values(
        "permutation_importance_mean", ascending=False
    )
    alcohol_rank = rf_importance["feature"].tolist().index("alcohol") + 1
    threshold = error["threshold"]

    seed0_confusion = [
        str(seed_error[key]) for key in ("tp", "fp", "fn", "tn")
    ]
    confusion_rows = [
        ["Seed 0", *seed0_confusion,
         f"{seed_error['precision_good']:.3f}", f"{seed_error['recall_good']:.3f}"],
        ["Mean ± SD, all seeds", *[
            f"{mean_std[key]['mean']:.1f} ± {mean_std[key]['std']:.1f}"
            for key in ("tp", "fp", "fn", "tn")
        ], f"{mean_std['precision_good']['mean']:.3f} ± {mean_std['precision_good']['std']:.3f}",
         f"{mean_std['recall_good']['mean']:.3f} ± {mean_std['recall_good']['std']:.3f}"],
    ]
    confusion_table = table(
        ["RF good class", "TP", "FP", "FN", "TN", "Precision", "Recall"],
        confusion_rows,
    )

    early = metrics[
        (metrics["model"] == "gradient_boosting")
        & (metrics["metric"] == "early_stopping_trees")
    ].sort_values("seed")
    early_trees = ", ".join(
        f"{int(row.seed)}: {int(row.value)}"
        for row in early.itertuples()
    )
    early_min = int(early["value"].min())
    early_max = int(early["value"].max())

    shift_rows = []
    for feature in features.assign(
        abs_shift=features["standardized_mean_difference_red_minus_white"].abs()
    ).nlargest(3, "abs_shift").itertuples():
        shift_rows.append(
            f"{feature.feature.replace('_', ' ')} "
            f"(red − white standardized mean difference "
            f"{feature.standardized_mean_difference_red_minus_white:+.2f})"
        )
    importance_shift = features.assign(
        importance_gap=features["absolute_rf_gb_importance_difference"]
    ).nlargest(3, "importance_gap")
    importance_text = ", ".join(
        row.feature.replace("_", " ")
        for row in importance_shift.itertuples()
    )
    features_text = ", ".join(feature.replace("_", " ") for feature in dataset["features"])

    red_white_roc = {}
    white_red = {}
    for model in ("random_forest", "gradient_boosting"):
        in_domain = metrics[
            (metrics["model"] == model) & (metrics["metric"] == "roc_auc")
        ]["value"]
        full_red_white = domain[
            (domain["model"] == model)
            & (domain["direction"] == "red_to_white")
            & (domain["evaluation_role"] == "target_full_domain")
        ]
        full_white_red = domain[
            (domain["model"] == model)
            & (domain["direction"] == "white_to_red")
            & (domain["evaluation_role"] == "target_full_domain")
        ]
        red_white_roc[model] = {
            "in_domain": float(in_domain.mean()),
            "target": float(full_red_white["roc_auc"].mean()),
            "loss": float(in_domain.mean() - full_red_white["roc_auc"].mean()),
            "std": float(full_red_white["roc_auc"].std(ddof=1)),
        }
        white_red[model] = {
            "ba": float(full_white_red["balanced_accuracy_at_source_threshold"].mean()),
            "roc": float(full_white_red["roc_auc"].mean()),
            "pr_lift": float(full_white_red["pr_auc_lift_over_base_rate"].mean()),
        }
    roc_gap = abs(
        red_white_roc["random_forest"]["target"]
        - red_white_roc["gradient_boosting"]["target"]
    )

    reproducibility_path = RESULTS / "reproducibility.json"
    if reproducibility_path.exists():
        reproducibility = json.loads(reproducibility_path.read_text())
        runtime = (
            f"{reproducibility['clone_install_and_run_seconds']:.0f} seconds "
            f"({reproducibility['clone_install_and_run_seconds'] / 60:.1f} minutes)"
        )
        hardware = (
            f"{reproducibility['cpu_count']} logical CPUs, "
            f"{reproducibility['memory_gib']:.1f} GiB RAM; "
            f"{reproducibility['platform']}; Python {reproducibility['python_version']}."
        )
        python_311_test = reproducibility.get("python_311_requirements_install")
    else:
        runtime = "not recorded yet for a clean-clone install-and-run"
        hardware = "See the measured fresh-clone run recorded in results/reproducibility.json."
        python_311_test = None

    if python_311_test:
        python_compatibility = (
            f"A fresh virtual environment using Python {python_311_test['python_version']} was tested; "
            f"`pip install -r requirements.txt` {python_311_test['status']}. "
            f"Reason: {python_311_test['reason']}. The full workflow was tested on Python "
            f"{reproducibility['python_version']} only; compatibility with other versions is not claimed."
        )
    else:
        python_compatibility = (
            f"The full workflow was tested on Python {reproducibility['python_version']} only; "
            "compatibility with other Python versions is not claimed."
        )

    return f"""# 1. EC417 Mini Project P07 — Wine Quality Classification

**Headline:** On each of the {seed_count} held-out seeds, the random forest exceeded gradient boosting in PR-AUC; paired differences (RF − GB) were {per_seed_text}.

## 2. Problem, motivation, and metric

The task is to identify wines with sensory quality at least 7 from physicochemical measurements. It is a compact way to study how objective lab measures act as proxies for subjective sensory judgments, and where that proxy may fail. This is an educational imbalanced-classification exercise, not a basis for commercial or consumer decisions. The positive class is {positive_rate:.1%} of the cleaned data, so PR-AUC (average precision) is the primary ranking metric: it focuses attention on positive-class retrieval and has a prevalence-dependent baseline. ROC-AUC is reported as a complementary ranking measure.

Balanced accuracy averages sensitivity and specificity at a particular threshold. The BA @ 0.5 column applies the same fixed probability cutoff to every model; mean scores are LR {ba_means['logistic_regression']['balanced_accuracy_0.5']:.3f}, RF {ba_means['random_forest']['balanced_accuracy_0.5']:.3f}, and GB {ba_means['gradient_boosting']['balanced_accuracy_0.5']:.3f}. These are valid fixed-cutoff results, not each model's best operating point or a ranking metric. The validation-threshold column instead uses each model's threshold selected on validation (mean BA: LR {ba_means['logistic_regression']['balanced_accuracy_val_threshold']:.3f}, RF {ba_means['random_forest']['balanced_accuracy_val_threshold']:.3f}, GB {ba_means['gradient_boosting']['balanced_accuracy_val_threshold']:.3f}). The RF is class-weighted with `class_weight="balanced_subsample"`. Its probability scores were not calibrated to a 0.5 cutoff in this project. Calibration was not tested, so no cause for the difference is claimed; a reliability diagram would help assess calibration.

## 3. Data

The red and white Vinho Verde wines are from the [UCI Wine Quality repository](https://archive.ics.uci.edu/dataset/186/wine+quality), introduced by Cortez et al. The repository lists the dataset under CC BY 4.0; see [LICENSE](./LICENSE) for this project's code license. The modeling data have {dataset['rows_before']:,} rows before exact de-duplication and {dataset['rows_after']:,} after. Good-wine prevalence is {dataset['good_wine_rate_before']:.1%} before and {positive_rate:.1%} after cleaning.

| Duplicate accounting | Rows |
|---|---:|
| Before cleaning | {dataset['rows_before']:,} |
| Rows in any duplicate group (`keep=False`) | {dataset['rows_involved_in_duplicates']:,} |
| Redundant rows removed (`keep="first"`) | {dataset['redundant_rows_removed']:,} |
| After cleaning | {dataset['rows_after']:,} |

| Cleaned domain | Rows | Good wines | Good-wine rate |
|---|---:|---:|---:|
| Red | {dataset['wine_type_after']['red']['rows']:,} | {dataset['wine_type_after']['red']['good_wines']:,} | {dataset['wine_type_after']['red']['good_wine_rate']:.1%} |
| White | {dataset['wine_type_after']['white']['rows']:,} | {dataset['wine_type_after']['white']['good_wines']:,} | {dataset['wine_type_after']['white']['good_wine_rate']:.1%} |

Each seed stratifies a {dataset['split_fractions_seed0']['train']:.0%}/{dataset['split_fractions_seed0']['val']:.0%}/{dataset['split_fractions_seed0']['test']:.0%} train/validation/test split on `(good, is_red)`. Duplicate rows are removed before splitting. The saved split indices address the cleaned, reset-index dataset; no identical cleaned sample can occur in two partitions. Seeds used: {', '.join(map(str, seeds))}.

## 4. Method

The baseline is a dummy prior classifier; logistic regression is a linear reference. The main model is a random forest, compared with histogram gradient boosting. The one operating-point improvement is to select the classification threshold using validation balanced accuracy instead of assuming 0.5. Model configurations are chosen by validation PR-AUC; the test set is held out.

```text
Raw red + white rows
        ↓ exact de-duplication (features + quality + wine type)
Cleaned rows → stratified train / validation / test split
        ↓ model selection by validation PR-AUC
Validation-only threshold selection → one held-out test evaluation
```

The model inputs are the following {len(dataset['features'])} physicochemical columns: {features_text}. `is_red` is used for stratification, descriptive summaries, and transfer evaluation; **it is not a model feature**.

Gradient boosting early stopping uses sklearn's internal validation split (`validation_fraction`), not the project's validation set. Project validation is still used for configuration selection and threshold choice. Early-stopping iterations by seed, read from `metrics.csv`: {early_trees} ({early_min}–{early_max} iterations).

## 5. Results

Values below are mean ± sample standard deviation over the {seed_count} seeds, read from `results/metrics.csv`. Fit time is the recorded final-model fit time, not the whole experiment runtime.

{result_table}

The RF beat GB in PR-AUC on every paired seed. Paired differences (RF − GB): {per_seed_text}; mean ± SD: {np.mean(paired):+.3f} ± {np.std(paired, ddof=1):.3f}. Possible explanations—not established causes—include the small de-duplicated dataset, noisy subjective sensory labels, unstable GB early stopping ({early_min}–{early_max} trees across the recorded seeds), and the small tuning grid.

At the validation-chosen RF threshold ({threshold:.3f}), the seed-0 good-class precision/recall and confusion counts are shown below; the second line is mean ± SD over the saved predictions for all seeds. These values are computed directly from `results/predictions_seed*.csv`.

{confusion_table}

Precision is the share of predicted-good wines that are truly good; recall is the share of all truly good wines found. At the selected threshold, seed-0 precision is {seed_error['precision_good']:.3f} and recall is {seed_error['recall_good']:.3f}; across seeds they are {mean_std['precision_good']['mean']:.3f} ± {mean_std['precision_good']['std']:.3f} and {mean_std['recall_good']['mean']:.3f} ± {mean_std['recall_good']['std']:.3f}. The chosen threshold balances class recall rather than optimizing either precision or recall alone.

Key figures: [seed-0 ROC curves](./results/figures/roc_curves_seed0.png) and [seed-0 precision-recall curves](./results/figures/pr_curves_seed0.png) overlay all models. Validation-threshold seed-0 confusion matrices are generated for logistic regression, RF, and GB; no trivial-prior confusion figure is retained. [Feature importance](./results/figures/feature_importance.png) is shown with separate scales. The concise report contains only selected figures.

### Domain shift

For each direction, seed, and model, fitting and selection use source-color rows only. The threshold is chosen on source validation and held fixed. The first table preserves the original 20%-test-slice transfer evaluation; the second extends the same source-only fits to **every cleaned wine in the target color**. Neither target set is used to train, tune, or select the threshold. Cells in the first table show balanced accuracy at that threshold / ROC-AUC / PR-AUC (mean ± SD across seeds).

#### Original target test-slice evaluation

{source_slice_table(domain)}

#### Entire cleaned target-domain evaluation

{full_domain_table(domain)}

For red → white, the mixed-domain in-domain ROC-AUC minus full-white transfer ROC-AUC is {red_white_roc['random_forest']['loss']:.3f} for RF and {red_white_roc['gradient_boosting']['loss']:.3f} for GB. Full-target ROC-AUC means are {red_white_roc['random_forest']['target']:.3f} ± {red_white_roc['random_forest']['std']:.3f} (RF) and {red_white_roc['gradient_boosting']['target']:.3f} ± {red_white_roc['gradient_boosting']['std']:.3f} (GB); their mean gap is {roc_gap:.3f} and their one-SD ranges overlap, so the five seeds do not support claiming RF transfers better. For white → red, fixed-threshold BA is {white_red['random_forest']['ba']:.3f} (RF) and {white_red['gradient_boosting']['ba']:.3f} (GB), near chance, while ROC-AUC is {white_red['random_forest']['roc']:.3f} and {white_red['gradient_boosting']['roc']:.3f}; the ranking retains some signal but the source threshold fails to transfer. Use PR-AUC lift over the target good-wine rate for cross-domain precision-recall comparisons, since the base rates differ.

The largest red/white distribution shifts in the cleaned data are {', '.join(shift_rows)}. The largest RF–GB permutation-importance gaps include {importance_text}. These are descriptive leads for hypotheses about transfer, not evidence that a specific feature shift causes the performance change.

## 6. Error analysis

The metadata alignment bug was caused by applying split indices from the cleaned/reset-index dataframe to raw-row metadata. Prediction metadata is now checked against the cleaned row, and every saved row asserts `actual_good == (quality >= 7)`. The error table is regenerated from saved seed-0 RF predictions and the cleaned data.

Seed-0 RF errors by quality:

{quality_table}

Quality 6 or 7 accounts for {error['quality_6_or_7_errors']} of {error['errors']} errors ({boundary_share:.1%}); other qualities account for {error['other_quality_errors']}. Red wines have {red['errors']}/{red['test_rows']} errors ({red['error_rate']:.1%}); white wines have {white['errors']}/{white['test_rows']} ({white['error_rate']:.1%}).

`results/misclassified_wines.csv` contains five confident false positives and five confident false negatives, each group sorted by distance from the threshold. In these ten examples, false positives have mean alcohol {fp_alcohol['mean']:.2f} and false negatives {fn_alcohol['mean']:.2f}. The false positives are higher-alcohol and false negatives lower-alcohol on average, consistent with alcohol being a dominant RF signal (its permutation importance rank is {alcohol_rank}); this is a pattern from ten selected examples, not evidence of causality or a population-wide effect.

## 7. Limitations and ethics

- Exact duplicate rows were removed before splitting; this changes the sample mix and does not eliminate all near-duplicate or source-level dependence.
- Sensory quality labels are subjective and noisy; the task is binary after thresholding an ordinal 0–10 score.
- This is one Vinho Verde dataset, not evidence of performance on other regions, vintages, producers, or populations.
- The red target test slice in the original transfer experiment contains only {int(domain[(domain['direction'] == 'white_to_red') & (domain['evaluation_role'] == 'target_domain_test')]['positive_count'].iloc[0])} positives; full-domain evaluation improves coverage but still uses one dataset.
- The {seed_count} seeds reuse overlapping samples, so seed-to-seed standard deviations understate sampling uncertainty.
- Tuning grids are small; GB stopping is based on sklearn's internal split and varies between {early_min} and {early_max} trees.
- `is_red` is not a model input. The experiment is for education and must not be used to price, certify, recommend, or make decisions about wines or people.

## 8. Reproduce

From a fresh clone, use the following commands. `run_all.sh` creates a virtual environment and installs the pinned requirements if `.venv` is absent; the explicit install command is shown for clarity.

```bash
git clone {REPOSITORY_URL}
cd ec417-miniproject-P07-623185
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run_all.sh
```

The run downloads the UCI CSVs, trains all saved seeds, regenerates result CSV/JSON files and figures, writes this README, and builds [report/report.pdf](./report/report.pdf). The measured fresh-clone total (clone, environment creation, dependency install, and `./run_all.sh`) took {runtime}. Hardware/software: {hardware} {python_compatibility}

### Run on Google Colab

The following commands are **untested on Colab**:

```bash
!git clone {REPOSITORY_URL}
%cd ec417-miniproject-P07-623185
!pip install -r requirements.txt
!./run_all.sh
```

Useful outputs: `results/metrics.csv`, `results/predictions_seed*.csv`, `results/domain_shift_metrics.csv`, `results/error_analysis.json`, and `results/figures/`. To check the report/demo materials, see [demo outline](./report/demo_outline.md) and [viva questions](./docs/viva_questions.md).

## 9. AI-use statement

The coding assistant was an AI assistant using Copilot SDK in VS Code. It helped create the code scaffold, run experiments, generate the README and report, and fix errors. Claude (Anthropic) was used to review results, spot issues (including the index-alignment bug and weighted-RF wording), and provide advice.

> **TODO (student): list what I personally verified, for example the leakage-free split, the label/index alignment fix, the numbers in the results table, and the viva answers.**

## 10. References

1. P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis. “Modeling wine preferences by data mining from physicochemical properties.” *Decision Support Systems* 47(4), 547–553 (2009). [DOI: 10.1016/j.dss.2009.05.016](https://doi.org/10.1016/j.dss.2009.05.016); [UCI dataset](https://doi.org/10.24432/C56S3T).
2. L. Breiman. “Random Forests.” *Machine Learning* 45, 5–32 (2001). [DOI: 10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324).
3. J. H. Friedman. “Greedy Function Approximation: A Gradient Boosting Machine.” *The Annals of Statistics* 29(5), 1189–1232 (2001). [DOI: 10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451).
4. scikit-learn documentation used for [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [`HistGradientBoostingClassifier` early stopping](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting), [permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html), and [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
"""


if __name__ == "__main__":
    (ROOT / "README.md").write_text(generate_readme(), encoding="utf-8")
    print(f"Generated README from saved results: {ROOT / 'README.md'}")
