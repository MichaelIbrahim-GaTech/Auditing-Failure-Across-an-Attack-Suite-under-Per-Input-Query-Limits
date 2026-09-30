# Interpretation tables

`run_interpretation.py` derives these tables without new audit trials.
All exact moments mean analytic finite-population reference calculations in
double precision, not rational-arithmetic certificates or deployment estimates.
The frozen matrices and source summaries are identified by SHA256 in validation.json.

- positive_count_descriptors.csv: all six populations, at k=4 and 16. Singleton
  fractions and S summaries condition on S>0; residual failure conditions on a
  negative screen. Expected calls use independent uniform attack orders.
- count_detection_contrasts.csv: the twelve matched count/detection designs in
  completion_results/main_summary.csv. Point estimates are interval midpoints.
  Difference standard errors use independent audit replications. Detection-only
  oracle intervals use equation (15); count oracle widths preserve the original
  sharp-versus-upper-bound certificate label. These are not cost-matched frontiers.
- low_prevalence_inference_comparison.csv: theta=.01 in both specified synthetic
  templates at all three original budgets. Exact point-estimator variance is
  separated from observed confidence width and certification probability.
  Certification is a valid one-sided 95% upper bound <=.05; Monte Carlo limits
  and widths come unchanged from the original 300-replication study.
- threshold_design_sensitivity.csv: frozen primary StrongREJECT score matrix
  rethresholded at .25/.5/.75, B=18,000, k=4, f=.5. Changing threshold changes the
  target. This evaluates exact variance and expected unit-outcome stopping cost;
  it does not evaluate confidence intervals or establish their ranking stability.
  Undefined variance ratios for theta=1 are empty; both variances there are zero.

Full auditing fixes B//d records. Completion uses the fixed allocation formula;
saved evaluations are not reinvested. No allocation is selected using outcomes.
The validation independently checks the first-hit and combinatorial cost formulas,
the two equivalent variance formulas, all twelve threshold histograms, and exact
moments against the existing threshold=.5 summaries.
