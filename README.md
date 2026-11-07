# 1. EC417 Mini Project P07 — Wine Quality Classification

**Headline:** On each of the 5 held-out seeds, the random forest exceeded gradient boosting in PR-AUC; paired differences (RF − GB) were seed 0: +0.021, seed 1: +0.037, seed 2: +0.050, seed 3: +0.034, seed 4: +0.031.

## 2. Problem, motivation, and metric

The task is to identify wines with sensory quality at least 7 from physicochemical measurements. It is a compact way to study how objective lab measures act as proxies for subjective sensory judgments, and where that proxy may fail. This is an educational imbalanced-classification exercise, not a basis for commercial or consumer decisions. The positive class is 19.0% of the cleaned data, so PR-AUC (average precision) is the primary ranking metric: it focuses attention on positive-class retrieval and has a prevalence-dependent baseline. ROC-AUC is reported as a complementary ranking measure.

Balanced accuracy averages sensitivity and specificity at a particular threshold. The BA @ 0.5 column applies the same fixed probability cutoff to every model; mean scores are LR 0.759, RF 0.638, and GB 0.761. These are valid fixed-cutoff results, not each model's best operating point or a ranking metric. The validation-threshold column instead uses each model's threshold selected on validation (mean BA: LR 0.756, RF 0.783, GB 0.764). The RF is class-weighted with `class_weight="balanced_subsample"`. Its probability scores were not calibrated to a 0.5 cutoff in this project. Calibration was not tested, so no cause for the difference is claimed; a reliability diagram would help assess calibration.

## 3. Data

The red and white Vinho Verde wines are from the [UCI Wine Quality repository](https://archive.ics.uci.edu/dataset/186/wine+quality), introduced by Cortez et al. The repository lists the dataset under CC BY 4.0; see [LICENSE](./LICENSE) for this project's code license. The modeling data have 6,497 rows before exact de-duplication and 5,320 after. Good-wine prevalence is 19.7% before and 19.0% after cleaning.

| Duplicate accounting | Rows |
|---|---:|
| Before cleaning | 6,497 |
| Rows in any duplicate group (`keep=False`) | 2,169 |
| Redundant rows removed (`keep="first"`) | 1,177 |
| After cleaning | 5,320 |

| Cleaned domain | Rows | Good wines | Good-wine rate |
|---|---:|---:|---:|
| Red | 1,359 | 184 | 13.5% |
| White | 3,961 | 825 | 20.8% |

Each seed stratifies a 60%/20%/20% train/validation/test split on `(good, is_red)`. Duplicate rows are removed before splitting. The saved split indices address the cleaned, reset-index dataset; no identical cleaned sample can occur in two partitions. Seeds used: 0, 1, 2, 3, 4.

## 4. Method

The baseline is a dummy prior classifier; logistic regression is a linear reference. The main model is a random forest, compared with histogram gradient boosting. The one operating-point improvement is to select the classification threshold using validation balanced accuracy instead of assuming 0.5. Model configurations are chosen by validation PR-AUC; the test set is held out.

```text
Raw red + white rows
        ↓ exact de-duplication (features + quality + wine type)
Cleaned rows → stratified train / validation / test split
        ↓ model selection by validation PR-AUC
Validation-only threshold selection → one held-out test evaluation
```

The model inputs are the following 11 physicochemical columns: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol. `is_red` is used for stratification, descriptive summaries, and transfer evaluation; **it is not a model feature**.

Gradient boosting early stopping uses sklearn's internal validation split (`validation_fraction`), not the project's validation set. Project validation is still used for configuration selection and threshold choice. Early-stopping iterations by seed, read from `metrics.csv`: 0: 235, 1: 44, 2: 38, 3: 138, 4: 90 (38–235 iterations).

## 5. Results

Values below are mean ± sample standard deviation over the 5 seeds, read from `results/metrics.csv`. Fit time is the recorded final-model fit time, not the whole experiment runtime.

| Model | BA @ 0.5 | BA @ validation threshold | ROC-AUC | PR-AUC | Fit time (s) |
| --- | --- | --- | --- | --- | --- |
| Dummy prior | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.190 ± 0.000 | 0.001 ± 0.000 |
| Logistic regression | 0.759 ± 0.013 | 0.756 ± 0.010 | 0.817 ± 0.008 | 0.499 ± 0.017 | 0.015 ± 0.001 |
| Random forest | 0.638 ± 0.014 | 0.783 ± 0.011 | 0.847 ± 0.009 | 0.565 ± 0.018 | 2.889 ± 0.154 |
| Gradient boosting | 0.761 ± 0.011 | 0.764 ± 0.009 | 0.836 ± 0.006 | 0.531 ± 0.024 | 0.460 ± 0.151 |

The RF beat GB in PR-AUC on every paired seed. Paired differences (RF − GB): seed 0: +0.021, seed 1: +0.037, seed 2: +0.050, seed 3: +0.034, seed 4: +0.031; mean ± SD: +0.035 ± 0.010. Possible explanations—not established causes—include the small de-duplicated dataset, noisy subjective sensory labels, unstable GB early stopping (38–235 trees across the recorded seeds), and the small tuning grid.

At the validation-chosen RF threshold (0.190), the seed-0 good-class precision/recall and confusion counts are shown below; the second line is mean ± SD over the saved predictions for all seeds. These values are computed directly from `results/predictions_seed*.csv`.

| RF good class | TP | FP | FN | TN | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| Seed 0 | 162 | 228 | 40 | 634 | 0.415 | 0.802 |
| Mean ± SD, all seeds | 165.8 ± 6.4 | 220.4 ± 36.3 | 36.2 ± 6.4 | 641.6 ± 36.3 | 0.432 ± 0.031 | 0.821 ± 0.032 |

Precision is the share of predicted-good wines that are truly good; recall is the share of all truly good wines found. At the selected threshold, seed-0 precision is 0.415 and recall is 0.802; across seeds they are 0.432 ± 0.031 and 0.821 ± 0.032. The chosen threshold balances class recall rather than optimizing either precision or recall alone.

Key figures: [seed-0 ROC curves](./results/figures/roc_curves_seed0.png) and [seed-0 precision-recall curves](./results/figures/pr_curves_seed0.png) overlay all models. Validation-threshold seed-0 confusion matrices are generated for logistic regression, RF, and GB; no trivial-prior confusion figure is retained. [Feature importance](./results/figures/feature_importance.png) is shown with separate scales. The concise report contains only selected figures.

### Domain shift

For each direction, seed, and model, fitting and selection use source-color rows only. The threshold is chosen on source validation and held fixed. The first table preserves the original 20%-test-slice transfer evaluation; the second extends the same source-only fits to **every cleaned wine in the target color**. Neither target set is used to train, tune, or select the threshold. Cells in the first table show balanced accuracy at that threshold / ROC-AUC / PR-AUC (mean ± SD across seeds).

#### Original target test-slice evaluation

| Model and direction | Mixed-domain test<br>BA / ROC-AUC / PR-AUC | Source-only source test<br>BA / ROC-AUC / PR-AUC | Source-only target 20% test slice<br>BA / ROC-AUC / PR-AUC | Target-slice PR-AUC lift | Target good rate | Positives / rows |
| --- | --- | --- | --- | --- | --- | --- |
| Random forest, red → white | 0.783 ± 0.011 / 0.847 ± 0.009 / 0.565 ± 0.018 | 0.771 ± 0.019 / 0.845 ± 0.023 / 0.528 ± 0.069 | 0.660 ± 0.052 / 0.737 ± 0.050 / 0.404 ± 0.054 | 1.940 ± 0.258 | 0.208 | 165 / 792 |
| Random forest, white → red | 0.783 ± 0.011 / 0.847 ± 0.009 / 0.565 ± 0.018 | 0.774 ± 0.014 / 0.844 ± 0.010 / 0.572 ± 0.026 | 0.533 ± 0.031 / 0.713 ± 0.045 / 0.289 ± 0.044 | 2.121 ± 0.326 | 0.136 | 37 / 272 |
| Gradient boosting, red → white | 0.764 ± 0.009 / 0.836 ± 0.006 / 0.531 ± 0.024 | 0.754 ± 0.018 / 0.834 ± 0.022 / 0.498 ± 0.084 | 0.664 ± 0.026 / 0.737 ± 0.006 / 0.377 ± 0.016 | 1.807 ± 0.076 | 0.208 | 165 / 792 |
| Gradient boosting, white → red | 0.764 ± 0.009 / 0.836 ± 0.006 / 0.531 ± 0.024 | 0.758 ± 0.010 / 0.832 ± 0.005 / 0.545 ± 0.020 | 0.524 ± 0.012 / 0.687 ± 0.060 / 0.287 ± 0.045 | 2.110 ± 0.331 | 0.136 | 37 / 272 |

#### Entire cleaned target-domain evaluation

| Source-only model and direction | BA at fixed threshold | ROC-AUC | PR-AUC | PR-AUC / target rate (lift) | Target good rate | Positives / all target rows |
| --- | --- | --- | --- | --- | --- | --- |
| Random forest, red → white | 0.661 ± 0.048 | 0.738 ± 0.046 | 0.394 ± 0.051 | 1.889 ± 0.245 | 0.208 | 825 / 3961 |
| Random forest, white → red | 0.533 ± 0.015 | 0.770 ± 0.022 | 0.315 ± 0.026 | 2.324 ± 0.188 | 0.135 | 184 / 1359 |
| Gradient boosting, red → white | 0.658 ± 0.019 | 0.734 ± 0.010 | 0.380 ± 0.018 | 1.823 ± 0.085 | 0.208 | 825 / 3961 |
| Gradient boosting, white → red | 0.532 ± 0.010 | 0.746 ± 0.038 | 0.307 ± 0.022 | 2.265 ± 0.161 | 0.135 | 184 / 1359 |

For red → white, the mixed-domain in-domain ROC-AUC minus full-white transfer ROC-AUC is 0.109 for RF and 0.101 for GB. Full-target ROC-AUC means are 0.738 ± 0.046 (RF) and 0.734 ± 0.010 (GB); their mean gap is 0.004 and their one-SD ranges overlap, so the five seeds do not support claiming RF transfers better. For white → red, fixed-threshold BA is 0.533 (RF) and 0.532 (GB), near chance, while ROC-AUC is 0.770 and 0.746; the ranking retains some signal but the source threshold fails to transfer. Use PR-AUC lift over the target good-wine rate for cross-domain precision-recall comparisons, since the base rates differ.

The largest red/white distribution shifts in the cleaned data are total sulfur dioxide (red − white standardized mean difference -2.34), volatile acidity (red − white standardized mean difference +1.67), free sulfur dioxide (red − white standardized mean difference -1.33). The largest RF–GB permutation-importance gaps include alcohol, total sulfur dioxide, density. These are descriptive leads for hypotheses about transfer, not evidence that a specific feature shift causes the performance change.

## 6. Error analysis

The metadata alignment bug was caused by applying split indices from the cleaned/reset-index dataframe to raw-row metadata. Prediction metadata is now checked against the cleaned row, and every saved row asserts `actual_good == (quality >= 7)`. The error table is regenerated from saved seed-0 RF predictions and the cleaned data.

Seed-0 RF errors by quality:

| Quality | Test rows | Errors | Error rate | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 3 | 6 | 0 | 0.0% | 0 | 0 |
| 4 | 35 | 9 | 25.7% | 9 | 0 |
| 5 | 368 | 35 | 9.5% | 35 | 0 |
| 6 | 453 | 184 | 40.6% | 184 | 0 |
| 7 | 171 | 38 | 22.2% | 0 | 38 |
| 8 | 31 | 2 | 6.5% | 0 | 2 |

Quality 6 or 7 accounts for 222 of 268 errors (82.8%); other qualities account for 46. Red wines have 58/272 errors (21.3%); white wines have 210/792 (26.5%).

`results/misclassified_wines.csv` contains five confident false positives and five confident false negatives, each group sorted by distance from the threshold. In these ten examples, false positives have mean alcohol 12.63 and false negatives 9.46. The false positives are higher-alcohol and false negatives lower-alcohol on average, consistent with alcohol being a dominant RF signal (its permutation importance rank is 1); this is a pattern from ten selected examples, not evidence of causality or a population-wide effect.

## 7. Limitations and ethics

- Exact duplicate rows were removed before splitting; this changes the sample mix and does not eliminate all near-duplicate or source-level dependence.
- Sensory quality labels are subjective and noisy; the task is binary after thresholding an ordinal 0–10 score.
- This is one Vinho Verde dataset, not evidence of performance on other regions, vintages, producers, or populations.
- The red target test slice in the original transfer experiment contains only 37 positives; full-domain evaluation improves coverage but still uses one dataset.
- The 5 seeds reuse overlapping samples, so seed-to-seed standard deviations understate sampling uncertainty.
- Tuning grids are small; GB stopping is based on sklearn's internal split and varies between 38 and 235 trees.
- `is_red` is not a model input. The experiment is for education and must not be used to price, certify, recommend, or make decisions about wines or people.

## 8. Reproduce

From a fresh clone, use the following commands. `run_all.sh` creates a virtual environment and installs the pinned requirements if `.venv` is absent; the explicit install command is shown for clarity.

```bash
git clone https://github.com/JahnaviVeerla/ec417-miniproject-P07-623185.git
cd ec417-miniproject-P07-623185
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run_all.sh
```

The run downloads the UCI CSVs, trains all saved seeds, regenerates result CSV/JSON files and figures, writes this README, and builds [report/report.pdf](./report/report.pdf). The measured fresh-clone total (clone, environment creation, dependency install, and `./run_all.sh`) took 383 seconds (6.4 minutes). Hardware/software: 32 logical CPUs, 30.8 GiB RAM; Linux-4.18.0-514.el8.x86_64-x86_64-with-glibc2.28; Python 3.14.7. A fresh virtual environment using Python 3.11.4 was tested; `pip install -r requirements.txt` failed. Reason: Pinned contourpy==1.4.0 is incompatible: pip reports Requires-Python >=3.12. The full workflow was tested on Python 3.14.7 only; compatibility with other versions is not claimed.

### Run on Google Colab

The following commands are **untested on Colab**:

```bash
!git clone https://github.com/JahnaviVeerla/ec417-miniproject-P07-623185.git
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
