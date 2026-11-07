# Five-minute demo outline

1. **Question and motivation** — Predict whether Vinho Verde wine receives a quality score of at least seven from physicochemical measurements.  
   *Speaker note:* Frame this as an educational imbalanced-classification experiment, not a consumer or commercial decision tool.

2. **Dataset and target** — Introduce the UCI red/white dataset, the binary label, and the low positive-class prevalence.  
   *Speaker note:* Explain that quality is subjective and thresholding an ordinal score discards information.

3. **Cleaning and split** — Show duplicate handling and the stratified train/validation/test protocol.  
   *Speaker note:* Emphasize that duplicate removal happens before splitting and indices always refer to the cleaned dataframe.

4. **Models and selection** — Present the dummy baseline, logistic reference, random forest, and gradient boosting.  
   *Speaker note:* Model configurations use validation PR-AUC and thresholds use validation balanced accuracy; test is held out.

5. **Headline results** — Compare the saved metrics and paired RF-versus-GB PR-AUC differences.  
   *Speaker note:* Be honest about the result on each seed and describe possible explanations only as hypotheses.

6. **Ranking and operating point** — Show the seed-zero ROC and precision-recall overlays, then explain the selected threshold.  
   *Speaker note:* BA at 0.5 uses the same fixed cutoff across models; the class-weighted RF has BA 0.638 there and 0.783 at its validation-selected threshold (seed-zero cutoff 0.19). Scores were not calibrated here; a reliability diagram could assess calibration, and no cause is claimed.

7. **Feature behavior and diagnostics** — Show permutation versus impurity importance and the learning/GB validation curves.  
   *Speaker note:* Point out that the two importance methods are different scales and early stopping uses sklearn's internal split.

8. **Domain shift** — Compare source-only transfer to the target test slice and the entire cleaned target domain.  
   *Speaker note:* Read PR-AUC alongside target prevalence and its lift; do not claim one model transfers better without evidence.

9. **Error analysis** — Show the ten confident errors and the quality/type summaries.  
   *Speaker note:* Explain the cleaned-index assertion and treat the alcohol pattern as a hypothesis from ten examples.

10. **Limitations and next steps** — Close with dataset scope, uncertainty, ethics, and possible extensions.  
    *Speaker note:* Suggest broader GB tuning, cross-validated early stopping, calibration, or ordinal regression, then invite questions.
