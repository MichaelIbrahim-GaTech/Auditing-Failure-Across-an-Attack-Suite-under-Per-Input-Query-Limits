# Exact decision consequences

Run `python decision_consequences.py` after the original finite-suite witness exists at `experiments/results/finite_suite_results.json`. The module uses only the Python standard library and no randomness. These are analytic constructed examples, with no empirical or Monte Carlo trials, model calls, new labels, or deployment measurements.

## Certification

The module reads the original d=64, k=4 rational witness, independently checks all hypergeometric probabilities and moments through degree four, and mixes both laws with an all-zero vector using weight 3/50. The resulting complete-suite failure probabilities are 768/59375 (about 1.2935%) and 3/50 (6%). Their symmetric vector lifts have identical transcript laws for every outcome-only adaptive capped audit. Consequently any gate with false-pass probability at most 5% uniformly above the 5% threshold passes this particular good law with probability at most 5%, for every finite number of inputs.

At a common reserved ceiling of 12,800 unit-cost outcomes, cap-four screening can use 3,200 inputs and complete labeling can use 200. The binomial gate X <= 4 has boundary false-pass probability 0.0264468000091201; X <= 5 would give 0.0623424950422953. Its exact power at the good population is 0.8805593633426956. This is a valid concrete comparator, not a claim of optimal power. Complete labels may stop after their first success without exceeding the stated ceiling. Input acquisition has no separate cost.

The zero-detection theorem addresses the worst-case probability of a specific event and its common endpoint, whereas this gate also passes some positive-label transcripts. The complete-label and uniform-screen zero-detection probabilities at the constructed good population are recorded in `exact_checks.json` to make the distinction explicit.

## Pilot cost bound

On an input with s>0 successes, uniform ordering costs (d+1)/(s+1) <= (d+1)/2 in expectation, and any first-success labeler costs at least one query. All-zero inputs cost d under both orders. Therefore any ordering can save at most theta*(d-1)/2 queries per fresh input in expectation relative to uniform ordering. Paying an extra fully queried 50-record pilot requires a same fresh cohort of at least ceil(100*d/(theta*(d-1))) even to break even. At d=36 and theta=1/100 this is 10,286 inputs. This is necessary, not sufficient; it assumes the extra pilot cost is recovered only from savings on the same fresh unpooled cohort. It does not claim this overhead for a comparison that pools pilot labels into an equally sized inference sample.

## Files and provenance

`mixed_witness.csv` and `common_count_law.csv` give exact rational masses. `certification_comparison.csv` distinguishes the cap-four universal power upper bound from the full-label exact pass probability. `ordering_pilot_bound.csv` contains the single analytic pilot example. `exact_checks.json` preserves exact binomial fractions, checks, and scope. `provenance.json` hashes the source module, canonical exact witness, and each result file except itself. Numeric displays are conversions of exact fractions; no normal approximation or floating-point CDF is used. The mixed populations are constructed from an existing certificate, not newly collected observations. Witness hashing selects only the exact fields used here and excludes execution timings in the original numerical output; its canonical serialization is recorded in provenance.
