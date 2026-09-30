# Held-out learned attack order

One fixed 50-row fully labeled pilot is used per population. The four StrongREJECT pilots use the same context IDs. The greedy order maximizes newly covered pilot rows and breaks ties by original column index. The order is computed solely from pilot outcomes and applied unchanged to held-out rows. This generated protocol records the analysis; it does not establish external preregistration. The greedy rule is classical min-sum set cover (Feige, Lovasz, and Tetali, Algorithmica 2004), with pilot-positive rows as elements and attacks as covering sets; its pilot approximation guarantee does not imply held-out transfer. No model calls or Monte Carlo audit trials are performed.

All results condition on this split and on the uniform empirical distribution over its held-out rows. These are reference-population expectations, not measured deployment savings or confidence intervals for transfer to new populations. Attack outcomes have unit cost; original procedure/API costs may differ. Pilots cost 50d revealed outcomes even when a row already has a success. Pilot labels are not reused in the audit estimate.

## Exact mean cost comparison

| Population | Held-out rows | Failure rate | Random mean | Learned mean | Pilot cost | Break-even cohort |
|---|---:|---:|---:|---:|---:|---:|
| Dolphin | 263 | 1.0000 | 2.249 | 1.156 | 1800 | 1647 |
| GPT-3.5 | 263 | 0.9962 | 3.466 | 1.878 | 1800 | 1134 |
| GPT-4o-mini | 263 | 0.9886 | 6.980 | 2.837 | 1800 | 435 |
| Llama-3.1 | 263 | 0.9886 | 6.222 | 2.011 | 1800 | 428 |
| Logistic | 644 | 0.9270 | 23.271 | 12.528 | 4900 | 457 |
| Forest | 651 | 0.7911 | 39.949 | 31.057 | 4900 | 552 |

Break-even is the smallest fixed audit cohort whose expected saved evaluations cover the pilot, allowing equality. The strict-saving threshold is also stored. Individual learned costs are paired with each held-out row's exact expected random-order cost; a paired difference is not a realized random-order draw.

## Two cost scenarios

Scenario A holds the audit cohort at floor(B/d). The same sampled rows have identical suite labels, prevalence estimate, and Clopper--Pearson interval under either order. Learned ordering adds a fully charged pilot outside the common audit ceiling, so its total hard ceiling is B+50d. Expected costs include that pilot.

Scenario B keeps the total hard ceiling B. The uniform method audits floor(B/d) records; the learned method pays for the pilot and audits floor(B/d)-50 independent held-out records. Any cost saving is accompanied by this smaller fixed audit cohort. No saved evaluations are reinvested. Scenario summaries report the exact binomial variance and expected two-sided 95% CP width at each fixed cohort size. This is a transparent fixed-cohort comparison, not an optimized allocation policy.

| Population | Uniform total cost | A learned total cost | B learned total cost | Uniform CP width | B learned CP width |
|---|---:|---:|---:|---:|---:|
| Dolphin | 1124.5 | 2377.9 | 2320.2 | 0.00735 | 0.00816 |
| GPT-3.5 | 1733.0 | 2739.2 | 2645.2 | 0.01314 | 0.01405 |
| GPT-4o-mini | 3490.1 | 3218.3 | 3076.4 | 0.02066 | 0.02189 |
| Llama-3.1 | 3111.2 | 2805.7 | 2705.1 | 0.02066 | 0.02189 |
| Logistic | 4654.1 | 7405.6 | 6779.2 | 0.07655 | 0.08906 |
| Forest | 7989.9 | 11111.4 | 9558.5 | 0.11669 | 0.13524 |

For this fixed split, learned ordering lowers the held-out per-record mean cost in all six populations. After charging the pilot, only GPT-4o-mini and Llama-3.1 have lower total expected cost at the tested cohort sizes, in either scenario. Scenario A changes cost but preserves inference on a common cohort. Scenario B reduces the audit cohort from 500 to 450 for StrongREJECT and 200 to 150 for Digits; nonsaturated prevalence variances therefore increase by 11.1% and 33.3%, respectively, and expected CP widths increase for every population. A favorable learned order is not by itself an end-to-end cost or precision improvement.

The Scenario B precision penalty is imposed by the conditional held-out-reference design, which excludes pilot labels; it is not an intrinsic cost of learning an order. If pilot and follow-up records instead are independent draws from the same population P and the total full-label cohort r is fixed, all r labels can be pooled. The order learned from pilot outcomes cannot change a fully determined suite label. Unconditionally, the pooled success count is Binomial(r, theta), so the mean retains variance theta(1-theta)/r and the ordinary Clopper--Pearson interval. This same-population pooling observation is analytic; it is not an empirical result for the separate conditional held-out target used here.

Expected costs, prevalence, and variances have rational representations in the CSVs. Expected CP widths are finite binomial sums using double-precision quantiles and probabilities; they are not interval-arithmetic certificates. Leakage, greedy tie handling, suite-label invariance, all-zero costs, small exhaustive permutation checks, and CP boundary checks are recorded in validation.json.

Run: `python run_heldout_ordering.py`. Inputs and code hashes appear in provenance.json. Split membership, ordered attack IDs, row-level paired costs, and both scenario summaries are retained.
