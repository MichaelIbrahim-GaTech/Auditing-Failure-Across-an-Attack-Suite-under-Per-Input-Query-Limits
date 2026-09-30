# Completion audit study

The results support a valid way to remove the ambiguity caused by incomplete suite observations. They do **not** establish that screening followed by random completion uniformly improves statistical efficiency, confidence-interval width, certification power, or actual query cost over a fixed number of complete first-success audits.

## Design and interpretation

All six measured populations from `audit_data.load_populations` are included. The four StrongREJECT populations use d = 36 and available budget B = 18,000; the two digits classifiers use d = 98 and B = 19,600. The main grid uses k = 4, 16 and screen-budget fractions f = .25, .50, .75, retaining only f >= k/d. This outcome-independent feasibility rule excludes the four StrongREJECT k = 16, f = .25 cases. The public loop assigns design IDs before applying the skip, preserving the independent random streams of all retained designs.

The allocation is m = floor(fB/k) and R = min(m, floor((B-mk)/(d-k))). No saved calls are reinvested. Each screened input has an independent uniform attack order. Screening stops at its first success or k calls. A uniform subset of min(R,M) negative screens is continued to its first success or suite exhaustion. The exact bound m k + R(d-k) is recorded separately from the available budget. The full baseline fixes floor(B/d) inputs before observing results and also stops each input at its first success. Count projection and the regularized linear estimator use floor(B/k) inputs and exactly k calls per input. The detection-only baseline stops at its first success or k calls and uses the support-envelope interval.

The main experiment has 74 design groups and 300 repetitions per group, giving 22,200 reported trial rows. The controlled experiment has 180 groups and 54,000 rows. It uses only the prespecified GPT-4o-mini and Forest templates, keeps each template's conditional distribution S | S > 0, and replaces the zero mass to set theta in {.001, .01, .05, .2, .5, .9}. Budgets are d times {50, 200, 800}, k = 4, f = .5. These are **synthetic prevalence mixtures**, not additional model measurements or estimates of real deployment prevalences.

Every experiment samples inputs with replacement from its specified population. Simulated first-hit order statistics and hypergeometric counts have the exact law induced by the frozen outcome population and a uniform attack order or subset. Query costs are replayed attack-call counts, not fresh paid calls or measured inference runtimes. Repeated draws do not supply additional independently collected source data.

## Findings

With half the budget reserved for screening:

- Logistic: the full baseline has empirical MSE .000336, mean interval width .07565, and mean calls 4,658. Completion at k = 16 has MSE .000208, width .08222, and mean calls 10,128. Its lower MSE comes with a wider conservative interval and more than twice the mean calls. At k = 4 completion has worse MSE .000490, width .10583, and mean calls 11,491.
- Forest: the full baseline has MSE .000724, width .11701, and mean calls 7,973. Completion at k = 16 has MSE .000710, width .15157, and mean calls 13,111. The small empirical MSE difference should not be interpreted without its Monte Carlo uncertainty; the exact variances are .000829 for full and .000720 for completion.
- The nearly saturated StrongREJECT populations make complete first-success auditing particularly inexpensive. The full baseline uses approximately 1,125 to 3,401 calls on average despite its 18,000-call reserved ceiling. Comparing only maximum budget would hide this practical difference.

For synthetic theta = .01, the event of interest is a valid 95% upper bound at most .05:

| Template | Budget | Full certification probability (95% MC interval) | Completion certification probability (95% MC interval) |
|---|---:|---:|---:|
| GPT-4o-mini | 7,200 | .9267 [.8911, .9535] | .4967 [.4387, .5547] |
| Forest | 19,600 | .9467 [.9148, .9692] | .4500 [.3928, .5082] |

At those same budgets, full/completion mean widths are .03308/.05879 for the GPT-4o-mini template and .03296/.05816 for the Forest template. Mean calls are nearly the full available budget for both methods because the synthetic prevalence is small. Simultaneous CP completion intervals therefore lose substantial certification power in these examples. At the largest tested budget both methods attain high certification probabilities. A globally regularized linear estimator can retain substantial approximation bias near small theta; it does not certify theta <= .05 in these theta = .01 cases.

Completion's observed coverage ranges from .9833 to 1 in the measured study and from .9800 to 1 in the controlled study. These empirical rates are diagnostic checks, not substitutes for the finite-sample coverage proof. No false certifications were observed for the tested theta > .05 values, but zero events in 300 repetitions still have a positive upper Monte Carlo confidence bound.

## Uncertainty and saved outputs

`main_summary.csv` and `controlled_summary.csv` include bias, variance, MSE, raw-estimator bias/MSE, interval width, calls, and Monte Carlo standard errors. Coverage, one-sided upper-bound coverage, certification, and precision probabilities include exact binomial 95% Monte Carlo intervals. Sample variance's Monte Carlo SE and RMSE's delta-method SE are large-repetition approximations. For completion and full auditing, population-specific exact means, variances, MSEs, and expected costs are also supplied. The linear estimator's exact risk columns refer to its **raw** sample mean; its reported point estimate is clipped to [0,1], so raw risk must not be substituted for clipped-estimator MSE. Projection and detection midpoint MSEs are explicitly operational choices under partial identification.

`tested_budget_thresholds.csv` reports the smallest **tested** budget at which the lower 95% Monte Carlo limit for a success probability reaches .90. The prespecified events are interval width at most .10 and a 95% upper bound at most .05. Missing values mean this criterion was not met on the three-budget grid. They do not prove that no intermediate or larger budget works, and the values are not optimized minimal budgets.

`main_cost_width.pdf/png` reports every retained main allocation with a logarithmic width axis. Completion error bars show 1.96 Monte Carlo standard errors of the mean width. `controlled_certification.pdf/png` shows all tested synthetic prevalences and budgets, with exact binomial Monte Carlo intervals. The count-projection upper bound is the upper endpoint of its two-sided 95% confidence set, a conservative 95% upper bound. Completion, full, detection, and linear bounds use the dedicated one-sided constructions documented in the code.

## Verification

The first-hit distribution agrees with direct enumeration of all permutations for a five-attack suite, with maximum discrepancy 1.12e-15. A 40,000-repetition nondegenerate completion check agrees with exact mean, variance, and expected cost within .51 Monte Carlo standard errors. The two equivalent exact variance formulas differ by less than 2e-18 in that check. The public interval routine explicitly tests that realizing M <= R does not trigger ordinary full-cohort CP when the design did not guarantee it. All projected confidence LPs in the study solved successfully; no fallback interval was used. Additional independent exact theory checks are preserved in `theory_validation.json`.

Run `python run_completion.py --repetitions 300` from the parent directory to regenerate the study and figures. Source modules are `completion_audit.py`, `run_completion.py`, `confidence.py`, `audit_inference.py`, and `audit_data.py`. `study_design.json` contains the complete fixed design, and `run_metadata.json` records versions and elapsed time.
