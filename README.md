# EC417 Mini Project P07: Wine Quality Classification

This project predicts whether a wine is good (`quality >= 7`) from its physicochemical measurements. The results below are from fresh, reproducible model runs; the five-seed test summary is generated from `results/metrics.csv`.

## Data cleaning and split

The UCI Wine Quality data contain 6,497 red and white wine rows before cleaning. Duplicate identity is computed over the 11 physicochemical features, quality score, and wine type:

| Duplicate accounting | Rows |
|---|---:|
| Rows before cleaning | 6,497 |
| Rows involved in any duplicate group (`keep=False`) | 2,169 |
| Redundant rows removed (`keep="first"`) | 1,177 |
| Rows after cleaning | 5,320 |

These are two different duplicate counts: 2,169 rows belong to repeated groups, while removing all but one row per group removes 1,177 rows. Splits are stratified on the pair `(good, is_red)` and use 60/20/20 train/validation/test partitions for seeds 0–4. Indices are saved in `results/split_seed{seed}.csv`.

## Model results

Models use the same cleaned dataset and seed-specific split. RF and gradient boosting each select among eight configurations by validation PR-AUC. For each fitted model, the operating threshold is selected on validation by balanced accuracy and fixed before test evaluation. Values are mean ± standard deviation across five held-out test sets.

The models use the 11 physicochemical feature columns only. `is_red` is used for stratification, descriptive analysis, and the red/white transfer experiment; it is **not** a model input feature.

| Model | BA @ 0.5 | BA @ validation threshold | ROC-AUC | PR-AUC | Fit time (s) |
|---|---:|---:|---:|---:|---:|
| Trivial prior | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.190 ± 0.000 | 0.001 ± 0.000 |
| Logistic regression | 0.759 ± 0.013 | 0.756 ± 0.010 | 0.817 ± 0.008 | 0.499 ± 0.017 | 0.015 ± 0.001 |
| Random forest | 0.638 ± 0.014 | **0.783 ± 0.011** | **0.847 ± 0.009** | **0.565 ± 0.018** | 2.701 ± 0.066 |
| Gradient boosting | 0.761 ± 0.011 | 0.764 ± 0.009 | 0.836 ± 0.006 | 0.531 ± 0.024 | 0.419 ± 0.135 |

The random forest gives the best ranking metrics and validation-threshold balanced accuracy. Logistic regression has stronger BA at the default 0.5 cutoff than RF because class weighting changes its operating point; thresholds are therefore tuned on validation for the fair BA comparison.

### Gradient-boosting early stopping

`HistGradientBoostingClassifier` uses sklearn's **internal validation split for early stopping** (`validation_fraction=0.2`); it does not use the project's validation set for stopping. Model selection and threshold choice still use the project's validation set. The selected boosting iterations are saved as `early_stopping_trees` metric rows in `results/metrics.csv`:

| Seed | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| GB iterations | 235 | 44 | 38 | 138 | 90 |

## Additional analysis

All the following diagnostic curves use seed 0 unless noted. `scripts/run_analysis.py` regenerates them from the real fitted runs and held-out data.

### EDA and figures

- `eda_correlation_heatmap.png`: annotated correlation values with readable feature labels.
- `eda_wine_type.png` and `eda_quality_by_type.png`: wine-type plots label `is_red=1` as Red and `is_red=0` as White.
- `eda_density_by_class.png`: density distributions for `quality < 7` and `quality >= 7`.
- `roc_curves_seed0.png` and `pr_curves_seed0.png`: all four models overlaid on the seed-0 test set.
- `confusion_matrix_<model>_seed0.png`: one test confusion matrix per model, using its seed-0 validation-chosen threshold.

Per-seed ROC, PR, and confusion-matrix figures (including the old most-frequent baseline figures) are removed.

### Feature importance and curves

Permutation importance is computed on the seed-0 test set with **10 repeats**, scored by average precision. The RF impurity importance is also shown in `feature_importance.png` (`results/permutation_importance.csv` contains the values). Permutation importance (change in test average precision) and impurity importance (normalized training split criterion) are different scales; compare rankings within each method, not the bar magnitudes across methods. Alcohol is the highest permutation-importance feature for both RF (0.188) and GB (0.230); RF next ranks chlorides (0.049) and total sulfur dioxide (0.047), while GB next ranks chlorides (0.036) and free sulfur dioxide (0.034).

`learning_curve.png` plots validation PR-AUC against training-set size:

| Training rows | 638 | 1,276 | 1,915 | 2,553 | 3,192 |
|---|---:|---:|---:|---:|---:|
| RF validation PR-AUC | 0.494 | 0.511 | 0.523 | 0.545 | 0.541 |
| GB validation PR-AUC | 0.501 | 0.521 | 0.522 | 0.524 | 0.556 |

`gb_learning_rate_curve.png` varies the GB learning rate and reports validation PR-AUC. Rates 0.01, 0.03, 0.05, 0.10, and 0.20 scored 0.547, 0.556, 0.554, 0.550, and 0.550, respectively. Early-stopping iterations for these diagnostic fits are in `results/gb_learning_rate_curve.csv`.

### Error analysis

The original error table joined test predictions to raw-row metadata using indices from the deduplicated/reset-index data, so sample metadata could refer to a different wine. This is fixed by joining through the cleaned dataset's `sample_index`; assertions now verify for every prediction that `actual_good == (quality >= 7)` and that the wine type matches the cleaned row.

For the seed-0 RF at its validation-selected threshold (0.190), 268 of 1,064 test wines were misclassified (25.2%). Errors by quality are:

| Quality | Test rows | Errors | Error rate | False positives | False negatives |
|---:|---:|---:|---:|---:|---:|
| 3 | 6 | 0 | 0.0% | 0 | 0 |
| 4 | 35 | 9 | 25.7% | 9 | 0 |
| 5 | 368 | 35 | 9.5% | 35 | 0 |
| 6 | 453 | 184 | 40.6% | 184 | 0 |
| 7 | 171 | 38 | 22.2% | 0 | 38 |
| 8 | 31 | 2 | 6.5% | 0 | 2 |

Red wines had 58 errors among 272 (21.3%; 43 false positives, 15 false negatives); white wines had 210 errors among 792 (26.5%; 185 false positives, 25 false negatives). `results/misclassified_wines.csv` contains one table with five confident false positives followed by five confident false negatives, ranked by distance from threshold within each error group. The records contain the matching cleaned row's quality and all 11 feature values; the report displays them in one wide table.

### Red/white domain shift

For each direction and seed, RF and GB configurations are selected by PR-AUC using only source-domain train/validation rows (eight configurations); the decision threshold is then selected on source validation and held fixed for source-test and target-test scoring. The target test results are compared with the same-seed mixed-domain test results. Metrics are mean ± standard deviation across seeds 0–4; triplets are `balanced accuracy at source-validation threshold / ROC-AUC / PR-AUC`.

| Model and transfer | Mixed-domain in-domain test | Source-only source test | Source-only target test | Target good-wine prevalence | Target positives (of n) |
|---|---:|---:|---:|---:|---:|
| RF, red → white | 0.783 ± 0.011 / 0.847 ± 0.009 / 0.565 ± 0.018 | 0.771 ± 0.019 / 0.845 ± 0.023 / 0.528 ± 0.069 | 0.660 ± 0.052 / 0.737 ± 0.050 / 0.404 ± 0.054 | 0.208 ± 0.000 | 165 (n=792) |
| GB, red → white | 0.764 ± 0.009 / 0.836 ± 0.006 / 0.531 ± 0.024 | 0.754 ± 0.018 / 0.834 ± 0.022 / 0.498 ± 0.084 | 0.664 ± 0.026 / 0.737 ± 0.006 / 0.377 ± 0.016 | 0.208 ± 0.000 | 165 (n=792) |
| RF, white → red | 0.783 ± 0.011 / 0.847 ± 0.009 / 0.565 ± 0.018 | 0.774 ± 0.014 / 0.844 ± 0.010 / 0.572 ± 0.026 | 0.533 ± 0.031 / 0.713 ± 0.045 / 0.289 ± 0.044 | 0.136 ± 0.000 | 37 (n=272) |
| GB, white → red | 0.764 ± 0.009 / 0.836 ± 0.006 / 0.531 ± 0.024 | 0.758 ± 0.010 / 0.832 ± 0.005 / 0.545 ± 0.020 | 0.524 ± 0.012 / 0.687 ± 0.060 / 0.287 ± 0.045 | 0.136 ± 0.000 | 37 (n=272) |

The target-set class counts are stable across the stratified test seeds: each white target test set contains 165 good wines among 792 (20.8%); each red target test set contains 37 among 272 (13.6%). The source-only models select and hold their thresholds separately per seed. The largest red/white standardized mean differences are total sulfur dioxide (-2.34), volatile acidity (+1.67), free sulfur dioxide (-1.33), and density (+1.20), based on the deduplicated full dataset. RF and GB permutation importance differs most for alcohol, total sulfur dioxide, and density. Per-seed metrics, thresholds, selected configurations, good-wine rates, and positive counts are saved in `results/domain_shift_metrics.csv`; full feature distribution and importance comparisons are in `results/domain_feature_comparison.csv`.

## Reproduce results

From the project root:

```bash
./run_all.sh
```

The script downloads the source files, runs all five seed-specific model comparisons, regenerates the analysis CSVs and figures, then produces `report/report.pdf`. The PDF is six pages and contains only the overview, key figures, domain comparison, and concise error-analysis examples.

## Key files

- `src/data.py`: data loading, duplicate accounting, and split creation
- `src/train.py`: model tuning, test evaluation, per-seed metrics and predictions
- `scripts/run_analysis.py`: EDA, importance, learning/validation curves, domain shift, and error analysis
- `scripts/make_report.py`: six-page PDF report generation
- `results/metrics.csv`: per-seed test metrics, including GB early-stopping iterations
- `results/domain_shift_metrics.csv`: source/target transfer results and mixed-domain comparisons
- `results/permutation_importance.csv`, `results/learning_curve.csv`, `results/gb_learning_rate_curve.csv`: numerical diagnostic outputs
- `results/error_analysis.json`, `results/misclassified_wines.csv`: error-by-quality and wine-type summaries, plus five false positives and five false negatives
- `results/figures/`: regenerated EDA and model-analysis figures
- `report/report.pdf`: concise report with key figures only

## Data source

P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis. “Modeling wine preferences by data mining from physicochemical properties.” *Decision Support Systems*, 47(4):547–553, 2009. Dataset: [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality).
