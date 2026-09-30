# Measured digits finite-suite stress experiment

Run `python ../empirical.py` from this directory, or run the script by absolute path. The script uses the bundled scikit-learn digits dataset; no external dataset download is required.

The experiment trains logistic regression and a random forest on the same stratified training split (1,078 inputs) and evaluates their clean predictions on the same test split (719 inputs). It retains clean-correct test inputs separately for each classifier. Every retained input is evaluated under 98 deterministic transformations: each of the 49 possible 2-by-2 patches is overwritten with either pixel value 0 or pixel value 1 after scaling pixels by 16.

A suite failure means a transformed prediction differs from the original dataset label. This is an operational wrong-label stress test. The transformations may change digit semantics, so these measurements do not establish label-preserving adversarial robustness. The two classifiers' eligible pools differ; their estimates are conditional on their respective clean-correct test pools.

`empirical_summary.json` contains the full configuration, environment versions, elapsed times, measured results, distributions, and exact expected detection probabilities. Each classifier's `*_failure_matrix.npz` contains its binary failure matrix, transformed predictions, original labels, count S of failing attacks, the complete clean predictions, and dataset/test row identifiers. The `*_test_metadata.csv` files include every test input; attack outcomes are blank for clean-incorrect inputs that were not evaluated. `attack_suite.csv` defines the column order of both failure matrices. `dataset_split.npz` contains the exact train and test indices.

The empirical audit population is the fixed, finite eligible test pool with uniform input sampling **with replacement**. Conditional on a sampled input with S failing attacks, the simulated count from a uniformly sampled subset of k distinct attacks is drawn using the exact Hypergeom(98, S, k) law. This reproduces the count law of subset sampling without rerunning the classifier. It does not create additional independently measured images.

`audit_histograms.npz` stores 3,000 simulated audit trials, 100 repetitions for each of two models, three input sample sizes m = 100, 500, 2,000, and five attack budgets k = 1, 4, 16, 64, 98. Its `histograms` array has axes `(model, m, k, repetition, t)` and shape `(2, 3, 5, 100, 99)`. Each row stores the number of input draws with t observed failing attacks. Entries with t > k are zero. The file also stores model names and axis values. Each trial uses a separate RNG seed sequence `[20260920, model_index, m, k, repetition]`. `audit_trials.csv` contains per-trial summaries.

`empirical_s_distribution.pdf/png` shows the measured S distributions including S = 0. `empirical_detection_curve.pdf/png` shows the exact probability of detecting at least one suite failure versus k; dotted horizontal lines give each pool's fully enumerated suite-failure probability. Numerical plot data are saved in `s_distribution.csv` and `detection_curve.csv`.

Consistency checks passed for failure matrix entries against stored predictions and labels, S row sums and histograms, eligible indices, clean correctness, and every simulated trial's total and support.
