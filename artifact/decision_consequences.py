"""Exact decision consequences of the existing finite-suite results.

This deterministic, standard-library-only calculation uses an existing rational
witness. It constructs populations, not measurements or Monte Carlo trials.
"""

import argparse
import csv
from fractions import Fraction
import hashlib
import json
from math import comb
from pathlib import Path


F = Fraction
ROOT = Path(__file__).resolve().parent
WITNESS_RELATIVE = Path("experiments/results/finite_suite_results.json")


def require(condition, message):
    """Keep validation active even when Python is invoked with -O."""
    if not condition:
        raise AssertionError(message)


def choose(n, k):
    return comb(n, k) if 0 <= k <= n else 0


def count_law(law, d, k):
    """Exact hypergeometric mixture probabilities for c=0,...,k."""
    denominator = comb(d, k)
    return [sum((weight * F(choose(s, c) * choose(d - s, k - c),
                            denominator)
                 for s, weight in law.items()), F(0))
            for c in range(k + 1)]


def validate_law(law, d):
    require(all(isinstance(s, int) and 0 <= s <= d for s in law),
            "Support outside the finite suite")
    require(all(weight >= 0 for weight in law.values()), "Negative mass")
    require(sum(law.values(), F(0)) == 1, "Mass does not sum to one")


def mix_with_zero(law, weight):
    mixed = {s: weight * mass for s, mass in law.items()}
    mixed[0] = mixed.get(0, F(0)) + 1 - weight
    return mixed


def binomial_cdf(c, n, probability):
    """Exact rational lower-tail probability, with no beta-quantile routine."""
    require(0 <= probability <= 1, "Invalid binomial probability")
    return sum((F(comb(n, j)) * probability ** j *
                (1 - probability) ** (n - j)
                for j in range(min(c, n) + 1)), F(0))


def critical_count(n, threshold, delta):
    """Largest nonrandom lower-tail count threshold satisfying level delta."""
    c = -1
    while c < n and binomial_cdf(c + 1, n, threshold) <= delta:
        c += 1
    return c


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fraction_record(value):
    return {"exact": str(value), "decimal": float(value)}


def main(output_dir="consequence_results", witness_path=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    witness_path = Path(witness_path) if witness_path else ROOT / WITNESS_RELATIVE
    records = json.loads(witness_path.read_text(encoding="utf-8"))
    d, k = 64, 4
    candidates = [record for record in records
                  if record["d"] == d and record["k"] == k]
    require(len(candidates) == 1, "Expected one original d=64,k=4 witness")
    record = candidates[0]
    # The full numerical result file contains execution timings. Hash only the
    # exact witness fields used here, so provenance survives faithful reruns.
    witness_fields = ["d", "k", "support", "weights_A_exact", "weights_B_exact",
                      "target_A_exact", "target_B_exact", "common_count_distribution_exact"]
    canonical_witness = {name: record[name] for name in witness_fields}
    canonical_witness_bytes = json.dumps(canonical_witness, sort_keys=True,
                                         separators=(",", ":")).encode("utf-8")
    support = record["support"]
    require(len(support) == len(set(support)), "Duplicate support point")
    for name in ("A", "B"):
        require(len(record[f"weights_{name}_exact"]) == len(support),
                "Mismatched witness support and mass lengths")
    original_high = dict(zip(support, map(F, record["weights_A_exact"])))
    original_low = dict(zip(support, map(F, record["weights_B_exact"])))
    for law in (original_high, original_low):
        validate_law(law, d)
    original_q_high = count_law(original_high, d, k)
    original_q_low = count_law(original_low, d, k)
    require(original_q_high == original_q_low,
            "Original witness does not match the complete count law")
    require(original_q_high == list(map(F, record["common_count_distribution_exact"])),
            "Stored and independently recomputed count laws differ")
    require(1 - original_high.get(0, F(0)) == F(record["target_A_exact"]) == 1,
            "Unexpected original high target")
    require(1 - original_low.get(0, F(0)) == F(record["target_B_exact"]) == F(512, 2375),
            "Unexpected original low target")

    mixing_weight, threshold, delta = F(3, 50), F(1, 20), F(1, 20)
    good = mix_with_zero(original_low, mixing_weight)
    bad = mix_with_zero(original_high, mixing_weight)
    for law in (good, bad):
        validate_law(law, d)
    good_q, bad_q = count_law(good, d, k), count_law(bad, d, k)
    require(good_q == bad_q, "Mixed populations have different count laws")
    require(sum(good_q, F(0)) == 1, "Count probabilities do not sum to one")
    theta_good, theta_bad = 1 - good[0], 1 - bad[0]
    require(theta_good == F(768, 59375), "Unexpected mixed low target")
    require(theta_bad == F(3, 50), "Unexpected mixed high target")
    require(theta_good < threshold < theta_bad, "Gate does not separate targets")
    moment_checks = []
    for degree in range(k + 1):
        original_a = sum((mass * s ** degree for s, mass in original_high.items()), F(0))
        original_b = sum((mass * s ** degree for s, mass in original_low.items()), F(0))
        good_moment = sum((mass * s ** degree for s, mass in good.items()), F(0))
        bad_moment = sum((mass * s ** degree for s, mass in bad.items()), F(0))
        require(original_a == original_b and good_moment == bad_moment,
                f"Moment mismatch at degree {degree}")
        moment_checks.append({"degree": degree, "original_common_moment": str(original_a),
                              "mixed_common_moment": str(good_moment),
                              "exact_discrepancy": "0"})

    full_inputs, capped_inputs = 200, 3200
    budget = full_inputs * d
    require(budget == capped_inputs * k == 12800, "Unmatched query ceilings")
    c = critical_count(full_inputs, threshold, delta)
    boundary_cdf = binomial_cdf(c, full_inputs, threshold)
    next_cdf = binomial_cdf(c + 1, full_inputs, threshold)
    good_power = binomial_cdf(c, full_inputs, theta_good)
    bad_power = binomial_cdf(c, full_inputs, theta_bad)
    require(c == 4 and boundary_cdf <= delta < next_cdf,
            "Incorrect nonrandom binomial critical count")
    require(good_power > delta, "Constructed comparator has no power separation")
    full_zero_power = (1 - theta_good) ** full_inputs
    capped_zero_power = good_q[0] ** capped_inputs

    mixed_rows = [{"s": s,
                   "original_low_mass_exact": str(original_low[s]),
                   "original_high_mass_exact": str(original_high[s]),
                   "good_mass_exact": str(good[s]), "bad_mass_exact": str(bad[s])}
                  for s in support]
    count_rows = [{"c": index, "original_common_probability_exact": str(original_q_high[index]),
                   "good_probability_exact": str(good_q[index]),
                   "bad_probability_exact": str(bad_q[index]),
                   "exact_discrepancy": "0"}
                  for index in range(k + 1)]
    certification_rows = []
    for design, depth, inputs, rule, kind, probability in [
            ("every_valid_cap4_gate", k, capped_inputs, "any uniformly valid gate",
             "universal_upper_bound", delta),
            ("complete_labels_binomial_gate", d, full_inputs, f"X <= {c}",
             "exact_pass_probability", good_power)]:
        certification_rows.append({"design": design, "d": d, "per_input_cap": depth,
                                   "inputs": inputs, "reserved_query_ceiling": budget,
                                   "threshold_exact": str(threshold), "delta_exact": str(delta),
                                   "theta_good_exact": str(theta_good),
                                   "theta_bad_exact": str(theta_bad), "pass_rule": rule,
                                   "probability_kind": kind, "good_pass_probability": float(probability),
                                   "good_pass_probability_exact": str(probability)})

    # Uniform order costs (d+1)/(s+1) on a vector with s>0 successes;
    # any first-success labeler costs at least one there. Zero vectors cost d.
    ordering_d, ordering_theta, pilot_inputs = 36, F(1, 100), 50
    per_positive_bound = F(ordering_d - 1, 2)
    conditional_bound_checks = []
    for s in range(1, ordering_d + 1):
        uniform_cost = F(ordering_d + 1, s + 1)
        best_possible_saving = uniform_cost - 1
        require(best_possible_saving <= per_positive_bound,
                "Conditional ordering saving exceeds claimed bound")
        conditional_bound_checks.append(str(best_possible_saving))
    saving_bound = ordering_theta * per_positive_bound
    pilot_cost = pilot_inputs * ordering_d
    ratio = F(pilot_cost) / saving_bound
    minimum_cohort = -(-ratio.numerator // ratio.denominator)
    require(minimum_cohort == 10286, "Unexpected necessary pilot break-even cohort")
    require((minimum_cohort - 1) * saving_bound < pilot_cost <= minimum_cohort * saving_bound,
            "Incorrect ceiling for necessary cohort size")
    require(ratio == F(100 * ordering_d, 1) / (ordering_theta * (ordering_d - 1)),
            "Pilot threshold formula mismatch")
    ordering_rows = [{"d": ordering_d, "theta_exact": str(ordering_theta),
                      "pilot_inputs": pilot_inputs, "extra_pilot_query_cost": pilot_cost,
                      "saving_per_fresh_input_upper_bound_exact": str(saving_bound),
                      "saving_per_fresh_input_upper_bound": float(saving_bound),
                      "necessary_cohort_ratio_exact": str(ratio),
                      "necessary_same_fresh_cohort": minimum_cohort,
                      "threshold_kind": "necessary_for_expected_break_even_not_sufficient",
                      "pilot_labels_pooled": False}]
    write_csv(out / "mixed_witness.csv", mixed_rows)
    write_csv(out / "common_count_law.csv", count_rows)
    write_csv(out / "certification_comparison.csv", certification_rows)
    write_csv(out / "ordering_pilot_bound.csv", ordering_rows)

    exact_checks = {
        "status": "passed", "evidence_type": "analytic_constructed_example",
        "empirical_trials": 0, "monte_carlo_trials": 0,
        "numeric_method": "Exact rational arithmetic; floating values are display conversions.",
        "original_witness": {"d": d, "k": k, "support": support,
                             "normalization_verified": True, "nonnegativity_verified": True,
                             "count_law_verified_at_every_c": True,
                             "moment_checks": moment_checks},
        "mixed_witness": {"mixing_weight": str(mixing_weight),
                          "theta_good": fraction_record(theta_good),
                          "theta_bad": fraction_record(theta_bad),
                          "threshold": str(threshold), "delta": str(delta),
                          "normalization_verified": True, "nonnegativity_verified": True,
                          "count_law_verified_at_every_c": True,
                          "common_count_law": [str(value) for value in good_q],
                          "capped_pass_probability_bound": str(delta),
                          "bound_applies_for_every_finite_input_count": True},
        "full_label_gate": {"inputs": full_inputs, "critical_count": c,
                            "boundary_false_pass_cdf": fraction_record(boundary_cdf),
                            "next_count_boundary_cdf": fraction_record(next_cdf),
                            "boundary_cdf_at_most_delta": True,
                            "next_count_cdf_exceeds_delta": True,
                            "good_pass_probability": fraction_record(good_power),
                            "bad_pass_probability": fraction_record(bad_power),
                            "scope": "Valid concrete nonrandom binomial gate; no claim of optimal power."},
        "matched_budget": {"reserved_query_ceiling": budget, "full_inputs": full_inputs,
                           "full_depth": d, "capped_inputs": capped_inputs, "cap": k,
                           "ceilings_equal": True,
                           "scope": "Unit-cost revealed outcomes; input acquisition has no separate cost."},
        "zero_detection_reconciliation": {
            "full_label_zero_probability": fraction_record(full_zero_power),
            "uniform_cap4_zero_probability_exact_expression": f"({good_q[0]})**{capped_inputs}",
            "uniform_cap4_zero_probability": float(capped_zero_power),
            "scope": "The full gate accepts up to four positive labels. The zero-detection theorem "
                     "optimizes a different event and its common upper endpoint, not general certification power."},
        "ordering_pilot_bound": {
            "d": ordering_d, "theta": str(ordering_theta), "extra_pilot_inputs": pilot_inputs,
            "extra_pilot_cost": pilot_cost,
            "per_positive_saving_upper_bound": str(per_positive_bound),
            "conditional_maximum_savings_for_s_1_through_d": conditional_bound_checks,
            "per_fresh_input_saving_upper_bound": str(saving_bound),
            "necessary_cohort_ratio": str(ratio), "necessary_same_fresh_cohort": minimum_cohort,
            "previous_cohort_cannot_cover_pilot_even_at_bound": True,
            "scope": "Necessary, not sufficient, for expected break-even when an extra fully queried "
                     "50-record pilot is charged against savings on the same fresh cohort. Pilot labels "
                     "are unpooled; this does not compare equally sized pooled-label designs."},
        "statistical_scope": "Independent fixed-vector inputs; outcome-only audit; symmetric lifts of "
                             "the supplied count laws; cap per input; arbitrary adaptive labeled queries "
                             "and randomized gates. Power obstruction is at this constructed good law, "
                             "not at every law below the threshold. No model calls or empirical observations."
    }
    write_json(out / "exact_checks.json", exact_checks)
    (out / "README.md").write_text(
        "# Exact decision consequences\n\n"
        "Run `python decision_consequences.py` after the original finite-suite witness exists at "
        "`experiments/results/finite_suite_results.json`. The module uses only the Python standard "
        "library and no randomness. These are analytic constructed examples, with no empirical or "
        "Monte Carlo trials, model calls, new labels, or deployment measurements.\n\n"
        "## Certification\n\n"
        "The module reads the original d=64, k=4 rational witness, independently checks all "
        "hypergeometric probabilities and moments through degree four, and mixes both laws with "
        "an all-zero vector using weight 3/50. The resulting complete-suite failure probabilities "
        "are 768/59375 (about 1.2935%) and 3/50 (6%). Their symmetric vector lifts have identical "
        "transcript laws for every outcome-only adaptive capped audit. Consequently any gate with "
        "false-pass probability at most 5% uniformly above the 5% threshold passes this particular "
        "good law with probability at most 5%, for every finite number of inputs.\n\n"
        "At a common reserved ceiling of 12,800 unit-cost outcomes, cap-four screening can use "
        "3,200 inputs and complete labeling can use 200. The binomial gate X <= 4 has boundary "
        "false-pass probability 0.0264468000091201; X <= 5 would give 0.0623424950422953. "
        "Its exact power at the good population is 0.8805593633426956. This is a valid concrete "
        "comparator, not a claim of optimal power. Complete labels may stop after their first "
        "success without exceeding the stated ceiling. Input acquisition has no separate cost.\n\n"
        "The zero-detection theorem addresses the worst-case probability of a specific event "
        "and its common endpoint, whereas this gate also passes some positive-label transcripts. "
        "The complete-label and uniform-screen zero-detection probabilities at the constructed "
        "good population are recorded in `exact_checks.json` to make the distinction explicit.\n\n"
        "## Pilot cost bound\n\n"
        "On an input with s>0 successes, uniform ordering costs (d+1)/(s+1) <= (d+1)/2 in "
        "expectation, and any first-success labeler costs at least one query. All-zero inputs "
        "cost d under both orders. Therefore any ordering can save at most theta*(d-1)/2 "
        "queries per fresh input in expectation relative to uniform ordering. Paying an extra "
        "fully queried 50-record pilot requires a same fresh cohort of at least "
        "ceil(100*d/(theta*(d-1))) even to break even. At d=36 and theta=1/100 this is 10,286 "
        "inputs. This is necessary, not sufficient; it assumes the extra pilot cost is recovered "
        "only from savings on the same fresh unpooled cohort. It does not claim this overhead "
        "for a comparison that pools pilot labels into an equally sized inference sample.\n\n"
        "## Files and provenance\n\n"
        "`mixed_witness.csv` and `common_count_law.csv` give exact rational masses. "
        "`certification_comparison.csv` distinguishes the cap-four universal power upper bound "
        "from the full-label exact pass probability. `ordering_pilot_bound.csv` contains the "
        "single analytic pilot example. `exact_checks.json` preserves exact binomial fractions, "
        "checks, and scope. `provenance.json` hashes the source module, canonical exact witness, "
        "and each result file except itself. Numeric displays are conversions of exact "
        "fractions; no normal approximation or floating-point CDF is used. The mixed populations "
        "are constructed from an existing certificate, not newly collected observations. "
        "Witness hashing selects only the exact fields used here and excludes execution timings "
        "in the original numerical output; its canonical serialization is recorded in provenance.\n",
        encoding="utf-8")
    result_names = ["certification_comparison.csv", "mixed_witness.csv", "common_count_law.csv",
                    "ordering_pilot_bound.csv", "exact_checks.json", "README.md"]
    write_json(out / "provenance.json", {
        "evidence_type": "analytic_constructed_example", "empirical_trials": 0,
        "monte_carlo_trials": 0, "deterministic": True,
        "source_hashes": {"decision_consequences.py": sha256(Path(__file__))},
        "witness_source": WITNESS_RELATIVE.as_posix(),
        "witness_canonical_fields": witness_fields,
        "witness_canonical_sha256": hashlib.sha256(canonical_witness_bytes).hexdigest(),
        "witness_canonical_serialization": "UTF-8 JSON; sort_keys=True; separators=(',', ':'); no final newline",
        "result_hashes": {name: sha256(out / name) for name in result_names},
        "witness_selection": {"d": d, "k": k},
        "hash_scope": "Source and result hashes use raw bytes. The witness hash uses canonical "
                      "exact fields, excluding runtime measurements. provenance.json is excluded "
                      "from result hashes to avoid a self-reference."
    })
    summary = {"status": "passed", "theta_good": float(theta_good),
               "theta_bad": float(theta_bad), "capped_power_upper_bound": float(delta),
               "full_label_power": float(good_power), "critical_count": c,
               "matched_query_ceiling": budget,
               "necessary_same_fresh_cohort_for_pilot": minimum_cohort,
               "empirical_trials": 0, "monte_carlo_trials": 0}
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="consequence_results")
    parser.add_argument("--witness-path", default=None)
    arguments = parser.parse_args()
    main(arguments.output_dir, arguments.witness_path)
