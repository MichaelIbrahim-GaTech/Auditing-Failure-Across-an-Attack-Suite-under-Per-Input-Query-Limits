"""Deterministic interpretation tables for the frozen attack-suite audit study.

This script does not fit a model or draw additional audit replications. It derives
success-count descriptors and exact design moments from archived matrices, and
selects explicitly labeled comparisons from the existing simulation summaries.
Changing the StrongREJECT score threshold changes the operational failure target.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

from completion_audit import allocation, exact_completion_moments, first_hit_distribution
from audit_data import load_populations


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "interpretation_results"
SENSITIVITY_BUDGET = 18_000
SENSITIVITY_K = 4
SENSITIVITY_FRACTION = .5
THRESHOLDS = (.25, .5, .75)


def no_hit(d, s, k):
    """Exact-integer combinatorial ratio converted once to double precision."""
    return math.comb(d - int(s), k) / math.comb(d, k) if d - int(s) >= k else 0.


def descriptors(S, d, k):
    S = np.asarray(S, dtype=int)
    positive = S[S > 0]
    if not len(positive):
        raise ValueError("These descriptive tables require at least one positive row")
    theta = float(len(positive) / len(S))
    q = np.array([no_hit(d, s, k) for s in S])
    a = float(1 - q.mean())
    positive_miss = float(q[S > 0].mean())
    residual = float((theta - a) / (1 - a)) if a < 1 else 0.
    full_cost = float(np.mean([d if s == 0 else (d + 1) / (s + 1) for s in S]))
    screen_cost = float(sum(np.mean([no_hit(d, s, j) for s in S]) for j in range(k)))
    return dict(n=len(S), d=d, k=k, positive_rows=len(positive), theta=theta,
                positive_singleton_fraction=float(np.mean(positive == 1)),
                positive_median_S=float(np.median(positive)),
                positive_mean_S=float(np.mean(positive)),
                positive_screen_miss_probability=positive_miss,
                detection_probability=a, negative_screen_probability=1-a,
                residual_failure_probability=float(np.clip(residual, 0, 1)),
                expected_full_calls_per_record=full_cost,
                expected_screen_calls_per_record=screen_cost)


def independent_completion(mu, d, k, m, R):
    """Check moments via the alternative variance and direct conditional costs."""
    s = np.arange(d + 1)
    q = np.array([no_hit(d, x, k) for x in s])
    p = float(mu @ q)
    a = 1 - p
    theta = float(1 - mu[0])
    b = (theta - a) / p if p else 0.
    b = float(np.clip(b, 0, 1))
    M = np.arange(m + 1)
    mass = binom.pmf(M, m, p)
    variance = a * (1-a) * (1-b)**2 / m
    variance += b * (1-b) * np.sum(mass[1:] * M[1:]**2 / np.minimum(R, M[1:])) / m**2
    screen_cost = sum(float(mu @ np.array([no_hit(d, x, j) for x in s])) for j in range(k))
    remaining_cost = np.array([d-k if x == 0 else (d-k+1)/(x+1) for x in s])
    conditional_remaining = float(mu @ (q * remaining_cost)) / p if p else 0.
    calls = m * screen_cost + float(mass @ np.minimum(R, M)) * conditional_remaining
    return float(variance), float(calls)


def source_hashes():
    inputs = ["run_interpretation.py", "audit_data.py", "completion_audit.py",
              "strongreject_audit_results/primary_score_matrices.npz",
              "strongreject_audit_results/threshold_sensitivity.csv",
              "strongreject_results/histograms.csv",
              "empirical_results/logistic_regression_failure_matrix.npz",
              "empirical_results/random_forest_failure_matrix.npz",
              "completion_results/main_summary.csv", "completion_results/controlled_summary.csv",
              "oracle_results/oracle.csv"]
    return {str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in inputs}


def build_tables():
    OUT.mkdir(exist_ok=True)
    populations = load_populations(ROOT)
    names = {p["identifier"]: p["name"] for p in populations}
    description = []
    largest_detection_residual = 0.
    largest_cost_residual = 0.
    for pop in populations:
        first = first_hit_distribution(pop["mu"])
        for k in (4, 16):
            row = dict(population=pop["name"], identifier=pop["identifier"], family=pop["family"],
                       **descriptors(pop["S"], pop["d"], k))
            largest_detection_residual = max(largest_detection_residual,
                                            abs(row["detection_probability"] - first[:k].sum()))
            alternative_cost = float(first @ np.minimum(np.arange(1, pop["d"] + 2), pop["d"]))
            largest_cost_residual = max(largest_cost_residual,
                                       abs(row["expected_full_calls_per_record"] - alternative_cost))
            description.append(row)
    desc = pd.DataFrame(description)
    desc.to_csv(OUT / "positive_count_descriptors.csv", index=False)

    main = pd.read_csv(ROOT / "completion_results/main_summary.csv")
    oracle = pd.read_csv(ROOT / "oracle_results/oracle.csv")
    common = ["population", "d", "theta", "budget", "k", "m", "repetitions"]
    quantities = ["mean_width", "mean_width_mcse", "mse", "mse_mcse", "bias", "bias_mcse",
                  "mean_calls", "mean_calls_mcse", "coverage", "coverage_mc_lower", "coverage_mc_upper",
                  "effective_budget_ceiling"]
    contrasts = None
    for method, prefix in (("count_midpoint", "count"), ("detection_midpoint", "detection")):
        selected = main.loc[main.method == method, common + quantities].copy()
        selected = selected.rename(columns={c: f"{prefix}_{c}" for c in quantities})
        contrasts = selected if contrasts is None else contrasts.merge(selected, on=common, validate="one_to_one")
    for quantity in ("mean_width", "mse", "mean_calls"):
        contrasts[f"count_minus_detection_{quantity}"] = contrasts[f"count_{quantity}"] - contrasts[f"detection_{quantity}"]
        contrasts[f"difference_{quantity}_mcse_independent"] = np.hypot(
            contrasts[f"count_{quantity}_mcse"], contrasts[f"detection_{quantity}_mcse"])
    contrasts["count_to_detection_actual_cost_ratio"] = contrasts.count_mean_calls / contrasts.detection_mean_calls
    contrasts = contrasts.merge(desc[["population", "k", "detection_probability", "expected_screen_calls_per_record"]],
                                on=["population", "k"], validate="one_to_one")
    a = contrasts.detection_probability
    contrasts["oracle_detection_lower"] = a
    contrasts["oracle_detection_upper"] = np.minimum(1., contrasts.d * a / contrasts.k)
    contrasts["oracle_detection_width"] = contrasts.oracle_detection_upper - a
    contrasts["expected_detection_calls"] = contrasts.m * contrasts.expected_screen_calls_per_record
    contrasts = contrasts.merge(oracle.rename(columns={"name": "population", "width": "oracle_count_width_upper",
                                                      "certification": "oracle_count_certification"})[
        ["population", "k", "oracle_count_width_upper", "oracle_count_certification"]],
        on=["population", "k"], validate="one_to_one")
    contrasts.to_csv(OUT / "count_detection_contrasts.csv", index=False)

    controlled = pd.read_csv(ROOT / "completion_results/controlled_summary.csv")
    low_common = ["population", "d", "theta", "budget", "repetitions"]
    low_quantities = ["k", "screen_fraction", "m", "R", "exact_variance", "mse", "mse_mcse",
                      "mean_width", "mean_width_mcse", "mean_calls", "mean_calls_mcse", "expected_calls",
                      "coverage", "coverage_mc_lower", "coverage_mc_upper", "certification_probability",
                      "certification_probability_mc_lower", "certification_probability_mc_upper"]
    low = None
    for method in ("full", "completion"):
        selected = controlled.loc[(controlled.theta == .01) & (controlled.method == method), low_common + low_quantities]
        selected = selected.rename(columns={c: f"{method}_{c}" for c in low_quantities})
        low = selected if low is None else low.merge(selected, on=low_common, validate="one_to_one")
    low["completion_to_full_exact_variance_ratio"] = low.completion_exact_variance / low.full_exact_variance
    low["completion_to_full_mean_width_ratio"] = low.completion_mean_width / low.full_mean_width
    low["completion_minus_full_certification_probability"] = low.completion_certification_probability - low.full_certification_probability
    low.to_csv(OUT / "low_prevalence_inference_comparison.csv", index=False)

    scores = np.load(ROOT / "strongreject_audit_results/primary_score_matrices.npz", allow_pickle=False)
    upstream_summary = pd.read_csv(ROOT / "strongreject_audit_results/threshold_sensitivity.csv")
    stored_histograms = pd.read_csv(ROOT / "strongreject_results/histograms.csv")
    sensitivity = []
    moment_residuals = []
    max_existing_moment_residual = 0.
    for index, identifier in enumerate(scores["models"]):
        identifier = str(identifier)
        for threshold in THRESHOLDS:
            S = (scores["scores"][index] >= threshold).sum(axis=1).astype(int)
            d, k, budget = 36, SENSITIVITY_K, SENSITIVITY_BUDGET
            details = descriptors(S, d, k)
            mu = np.bincount(S, minlength=d+1) / len(S)
            first = first_hit_distribution(mu)
            m, R = allocation(budget, d, k, SENSITIVITY_FRACTION)
            moments = exact_completion_moments(first, m, R, k)
            independent_variance, independent_cost = independent_completion(mu, d, k, m, R)
            moment_residuals.append((abs(moments["exact_variance"]-independent_variance),
                                     abs(moments["expected_calls"]-independent_cost)))
            reference = upstream_summary.loc[(upstream_summary.model == identifier) & (upstream_summary.threshold == threshold)]
            assert len(reference) == 1
            assert int(reference.iloc[0].union_count) == int(np.count_nonzero(S))
            assert abs(float(reference.iloc[0].mean_S) - S.mean()) < 1e-12
            saved = stored_histograms.loc[(stored_histograms.suite == "primary_36") &
                                         (stored_histograms.model == identifier) &
                                         (stored_histograms.threshold == threshold)].sort_values("S")
            assert len(saved) == d + 1
            assert np.array_equal(saved["count"].to_numpy(), np.bincount(S, minlength=d+1))
            full_m = budget // d
            full_variance = details["theta"] * (1-details["theta"]) / full_m
            row = dict(population=names[identifier], identifier=identifier, score_threshold=threshold,
                       target="archived_score_at_least_threshold", suite="primary_36", budget=budget,
                       screen_fraction=SENSITIVITY_FRACTION, full_m=full_m,
                       completion_m=m, completion_R=R, **details,
                       full_exact_variance=full_variance,
                       full_expected_calls=full_m*details["expected_full_calls_per_record"],
                       completion_exact_variance=moments["exact_variance"],
                       completion_expected_calls=moments["expected_calls"],
                       completion_reserved_ceiling=m*k+R*(d-k),
                       full_reserved_ceiling=full_m*d,
                       completion_to_full_variance_ratio=(moments["exact_variance"]/full_variance
                                                         if full_variance else np.nan))
            row["completion_to_full_expected_calls_ratio"] = row["completion_expected_calls"]/row["full_expected_calls"]
            sensitivity.append(row)
            if threshold == .5:
                old = main.loc[(main.population == names[identifier]) & (main.method == "completion") &
                               (main.k == k) & (main.screen_fraction == SENSITIVITY_FRACTION)]
                assert len(old) == 1
                for column in ("exact_variance", "expected_calls"):
                    max_existing_moment_residual = max(max_existing_moment_residual,
                        abs(float(old.iloc[0][column]) - moments[column]))
    pd.DataFrame(sensitivity).to_csv(OUT / "threshold_design_sensitivity.csv", index=False)

    assert largest_detection_residual < 1e-12
    assert largest_cost_residual < 1e-12
    assert max(r[0] for r in moment_residuals) < 1e-12
    assert max(r[1] for r in moment_residuals) < 1e-8
    assert max_existing_moment_residual < 1e-8
    validation = dict(
        analysis="Deterministic reanalysis; no new model calls or audit simulations",
        sampling_target="Uniform distribution over retained frozen rows; with-replacement audit records",
        threshold_design=dict(budget=SENSITIVITY_BUDGET, k=SENSITIVITY_K,
                              screen_fraction=SENSITIVITY_FRACTION, thresholds=THRESHOLDS),
        max_detection_formula_residual=largest_detection_residual,
        max_full_cost_formula_residual=largest_cost_residual,
        max_completion_variance_formula_residual=max(r[0] for r in moment_residuals),
        max_completion_cost_formula_residual=max(r[1] for r in moment_residuals),
        max_existing_summary_moment_residual=max_existing_moment_residual,
        threshold_histograms_match_archived_counts=True,
        threshold_descriptors_match_existing_sensitivity=True,
        comparison_replications_are_independent=True,
        source_sha256=source_hashes())
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    readme = """# Interpretation tables

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
"""
    (OUT / "README.md").write_text(readme)
    print(json.dumps({"output": str(OUT), "tables": 4,
                      "new_model_calls": 0, "new_audit_replications": 0,
                      "validation": "passed"}))


if __name__ == "__main__":
    build_tables()
