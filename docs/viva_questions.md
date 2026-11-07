# Viva questions and grounded answers

## 1. What is the prediction task?

The target is `good = (quality >= 7)`, derived from the sensory quality score; the inputs are physicochemical columns. The cutoff is a project definition, not an objective measure of wine value.  
**Where:** [`src/data.py`](../src/data.py) — `load_wine_dataset()`.

## 2. Why remove duplicates before splitting?

Repeated rows could otherwise appear in both training and test partitions, making evaluation overly optimistic. The project removes exact duplicates using the measurements, quality, and wine type before creating any split.  
**Where:** [`src/data.py`](../src/data.py) — `duplicate_counts()`, `summarize_duplicates()`, `make_split_indices()`.

## 3. How do you know the splits are stratified and leakage is controlled?

The split stratifier is the pair `(good, is_red)`, and split indices are generated only after de-duplication. The saved indices refer to the cleaned/reset-index dataframe; the test set is not used for tuning or threshold selection.  
**Where:** [`src/data.py`](../src/data.py) — `make_split_indices()`; [`src/train.py`](../src/train.py) — `run_training_cycle()`.

## 4. What caused the error-analysis index bug, and what prevents it returning?

Split indices were made against the cleaned dataframe but metadata was joined from raw rows, so a valid index could identify a different wine. The analysis now joins through cleaned `sample_index` and asserts `actual_good == (quality >= 7)` for every row.  
**Where:** [`scripts/run_analysis.py`](../scripts/run_analysis.py) — `run_error_analysis()` and `main()`.

## 5. How is the decision threshold selected?

The model's validation probabilities are evaluated over candidate thresholds, and the threshold maximizing validation balanced accuracy is retained. The test probabilities are evaluated only after selection; no test label is used to choose the cutoff.  
**Where:** [`src/train.py`](../src/train.py) — `choose_threshold()`, `evaluate_model_on_test()`.

## 6. What do balanced accuracy and PR-AUC measure?

Balanced accuracy averages sensitivity and specificity, so each class contributes equally at a chosen threshold. PR-AUC (average precision) summarizes the precision/recall ranking and is especially informative here because positives are a minority; its baseline depends on prevalence.  
**Where:** [`src/train.py`](../src/train.py) — `evaluate_model_on_test()`; [`README.md`](../README.md) — metric rationale and saved results.

## 7. Why is balanced accuracy at 0.5 not the whole story?

Balanced accuracy at 0.5 applies the same fixed probability cutoff to LR, RF, and GB; their mean values are 0.759, 0.638, and 0.761. At the separate validation-chosen thresholds, mean BA is 0.756, 0.783, and 0.764. The RF is class-weighted (`class_weight="balanced_subsample"`); its seed-0 threshold is 0.19, and its scores were not calibrated to a 0.5 cutoff in this project. Calibration was not tested, so do not claim a cause for the difference; a reliability diagram could assess calibration.  
**Where:** [`src/models.py`](../src/models.py) — `make_random_forest_model()`; [`results/metrics.csv`](../results/metrics.csv).

## 8. Why did the random forest beat gradient boosting on PR-AUC?

The saved five-seed results show RF ahead on every paired seed. Possible explanations are hypotheses: the cleaned sample is small, sensory labels are noisy, GB early stopping varies by seed, and the tuning grid is small; this experiment does not isolate a cause.  
**Where:** [`results/metrics.csv`](../results/metrics.csv); [`src/train.py`](../src/train.py) — `tune_random_forest()`, `tune_gradient_boosting()`.

## 9. What does GB early stopping do here?

`HistGradientBoostingClassifier` uses sklearn's internal validation fraction and patience to stop its iterations. Project validation is still used for configuration selection and threshold choice; the internal split is not the project validation set.  
**Where:** [`src/models.py`](../src/models.py) — `make_hist_gradient_boosting_model()`; [`src/train.py`](../src/train.py) — `tune_gradient_boosting()`.

## 10. How do impurity and permutation importance differ?

Impurity importance sums training split-criterion improvements and is normalized within the fitted forest. Permutation importance measures the change in held-out average precision after shuffling a feature; these are different scales and should not be compared by bar magnitude.  
**Where:** [`scripts/run_analysis.py`](../scripts/run_analysis.py) — `run_importance_and_learning_curve()`; [`results/permutation_importance.csv`](../results/permutation_importance.csv).

## 11. How were the confident false positives and negatives selected?

The error table uses saved seed-zero RF predictions at the validation-selected threshold. It takes five of each error type, sorted within group by distance from the threshold, and attaches features from the matching cleaned row.  
**Where:** [`scripts/run_analysis.py`](../scripts/run_analysis.py) — `run_error_analysis()`; [`results/misclassified_wines.csv`](../results/misclassified_wines.csv).

## 12. What patterns appear in the errors?

The quality breakdown and red/white rates are computed from seed-zero predictions and cleaned metadata. The ten selected false positives have higher mean alcohol than the selected false negatives; that is consistent with alcohol influencing the model, not proof of a general causal pattern.  
**Where:** [`results/error_analysis.json`](../results/error_analysis.json); [`results/misclassified_wines.csv`](../results/misclassified_wines.csv).

## 13. How is red/white domain transfer evaluated?

For each direction, source-only tuning and threshold selection use source-color rows; the fixed threshold is then evaluated on the target test slice and all cleaned target-color rows. Target PR-AUC is interpreted with target prevalence and PR-AUC/base-rate lift.  
**Where:** [`scripts/run_analysis.py`](../scripts/run_analysis.py) — `run_domain_shift()`; [`results/domain_shift_metrics.csv`](../results/domain_shift_metrics.csv).

## 14. How should the domain-shift results be interpreted?

Red-to-white ROC-AUC drops relative to the mixed-domain reference, while RF and GB target summaries overlap across the saved seeds; there is no evidence here to claim RF transfers better. White-to-red fixed-threshold BA is near chance even though ROC-AUC retains some ranking signal.  
**Where:** [`results/domain_shift_metrics.csv`](../results/domain_shift_metrics.csv); [`README.md`](../README.md) — domain-shift tables and interpretation.

## 15. What are the main limitations and next steps?

This is one dataset with subjective labels, duplicate removal, overlapping seed samples, small tuning grids, and limited red target positives in the original slice. Consider larger GB tuning, cross-validated early stopping, calibration, or ordinal regression; use the model only for education.  
**Where:** [`README.md`](../README.md) — limitations and next steps; [`scripts/make_report.py`](../scripts/make_report.py) — report generation.
