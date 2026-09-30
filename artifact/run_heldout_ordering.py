"""One fixed held-out attack-order experiment on six frozen outcome matrices.

No model calls or Monte Carlo audit trials are performed. A fully observed
50-row pilot determines a greedy coverage order; all reported audit targets
and costs refer to the disjoint held-out empirical population. The expected
random-order costs and prevalence variances are evaluated as rational numbers.
Expected Clopper--Pearson widths use finite binomial enumeration in floating
point, not a numerical coverage certificate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from scipy.special import betaincinv
from scipy.stats import binom

ROOT = Path(__file__).resolve().parent
SEED = 20260923
PILOT_ROWS = 50
DELTA = .05


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fraction_fields(prefix, value):
    value = Fraction(value)
    return {prefix: float(value), prefix + "_fraction": str(value)}


def greedy_coverage_order(pilot):
    """Use only pilot outcomes; break every tie by original column index."""
    pilot = np.asarray(pilot, dtype=bool)
    remaining = list(range(pilot.shape[1]))
    uncovered = np.ones(len(pilot), dtype=bool)
    order, trace = [], []
    while remaining:
        gains = pilot[uncovered][:, remaining].sum(axis=0)
        largest = int(gains.max())
        # Remaining indices stay sorted, so argmax implements the fixed tie rule.
        chosen = remaining[int(np.argmax(gains))]
        trace.append(dict(rank=len(order) + 1, coordinate_index=chosen,
                          pilot_newly_covered=largest,
                          pilot_uncovered_before=int(uncovered.sum()),
                          maximizing_tie_count=int(np.sum(gains == largest))))
        order.append(chosen)
        uncovered &= ~pilot[:, chosen]
        remaining.remove(chosen)
    return np.asarray(order, dtype=int), trace


def ordered_first_cost(matrix, order):
    matrix = np.asarray(matrix, dtype=bool)
    positive = matrix.any(axis=1)
    first = np.argmax(matrix[:, order], axis=1) + 1
    return np.where(positive, first, matrix.shape[1]).astype(int)


def uniform_row_cost(s, d):
    return Fraction(d) if s == 0 else Fraction(d + 1, int(s) + 1)


def expected_cp_width(theta, n, delta=DELTA):
    """Exact finite binomial sum, with floating-point probabilities/quantiles."""
    if n < 1:
        raise ValueError("A positive fixed cohort is required")
    x = np.arange(n + 1)
    lower = np.zeros(n + 1)
    upper = np.ones(n + 1)
    lower[1:] = betaincinv(x[1:], n - x[1:] + 1, delta / 2)
    upper[:-1] = betaincinv(x[:-1] + 1, n - x[:-1], 1 - delta / 2)
    weights = binom.pmf(x, n, float(theta))
    assert abs(float(weights.sum()) - 1) < 1e-12
    return float(np.dot(weights, upper - lower))


def load_matrices(root):
    score_path = root / "strongreject_audit_results/primary_score_matrices.npz"
    populations = []
    with np.load(score_path, allow_pickle=False) as archive:
        names = ["Dolphin", "GPT-3.5", "GPT-4o-mini", "Llama-3.1"]
        methods = [str(x) for x in archive["methods"]]
        for index, identifier in enumerate(archive["models"]):
            populations.append(dict(name=names[index], identifier=str(identifier),
                family="StrongREJECT", source_path=score_path,
                matrix=(archive["scores"][index] >= .5).astype(np.uint8),
                row_ids=[str(x) for x in archive["context_ids"][index]],
                attack_ids=methods, budget=18000, split_stream=0))
    suite_path = root / "empirical_results/attack_suite.csv"
    with suite_path.open(newline="") as stream:
        specs = list(csv.DictReader(stream))
    specs.sort(key=lambda item: int(item["attack_id"]))
    assert [int(item["attack_id"]) for item in specs] == list(range(98))
    ids = ["patch_r{row}_c{column}_fill{fill_value}".format(**item) for item in specs]
    for split_stream, (identifier, name) in enumerate(
            [("logistic_regression", "Logistic"), ("random_forest", "Forest")], 1):
        path = root / f"empirical_results/{identifier}_failure_matrix.npz"
        with np.load(path, allow_pickle=False) as archive:
            populations.append(dict(name=name, identifier=identifier, family="Digits",
                source_path=path, matrix=archive["failure_matrix"].copy(),
                row_ids=[str(int(x)) for x in archive["eligible_dataset_indices"]],
                attack_ids=ids, budget=19600, split_stream=split_stream))
    source_paths = sorted(set([p["source_path"] for p in populations] + [suite_path]))
    return populations, source_paths


def pilot_indices(population, shared_context_ids):
    ids = population["row_ids"]
    if population["family"] == "StrongREJECT":
        selected_ids = shared_context_ids
        return np.asarray([i for i, row_id in enumerate(ids) if row_id in selected_ids], dtype=int)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, population["split_stream"]]))
    return np.sort(rng.permutation(len(ids))[:PILOT_ROWS])


def audit_order_validation(pilot, order, trace):
    remaining = set(range(pilot.shape[1]))
    uncovered = set(range(len(pilot)))
    for rank, chosen in enumerate(order):
        # Independent set-based computation checks coverage gains and tie handling.
        gain_sets = {j: {i for i in uncovered if pilot[i, j]} for j in remaining}
        best_gain = max(len(value) for value in gain_sets.values())
        best_index = min(j for j, value in gain_sets.items() if len(value) == best_gain)
        assert int(chosen) == best_index
        assert trace[rank]["pilot_newly_covered"] == best_gain
        uncovered -= gain_sets[int(chosen)]
        remaining.remove(int(chosen))
    assert not remaining
    return True


def analytic_checks():
    # Exhaust all vectors and permutations for a small suite to check the
    # random-order formula and the learned first-success cost implementation.
    d = 5
    orders = list(itertools.permutations(range(d)))
    for bits in itertools.product([0, 1], repeat=d):
        row = np.asarray([bits], dtype=np.uint8)
        costs = [int(ordered_first_cost(row, order)[0]) for order in orders]
        assert Fraction(sum(costs), len(costs)) == uniform_row_cost(sum(bits), d)
    pilot = np.asarray([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
    order, trace = greedy_coverage_order(pilot)
    assert order.tolist() == [1, 2, 0, 3]
    audit_order_validation(pilot, order, trace)
    assert greedy_coverage_order(np.zeros((3, 4), dtype=np.uint8))[0].tolist() == [0, 1, 2, 3]
    for n in [1, 7, 200, 500]:
        boundary = 1 - (DELTA / 2) ** (1 / n)
        assert abs(expected_cp_width(Fraction(0), n) - boundary) < 1e-12
        assert abs(expected_cp_width(Fraction(1), n) - boundary) < 1e-12
    return dict(exhaustive_uniform_cost_vectors=2 ** d, permutations_per_vector=len(orders),
                greedy_hand_example="passed", all_zero_tie_order="passed",
                cp_boundary_checks=8)


def run(root=ROOT, output=None):
    root = Path(root)
    output = root / "ordering_results" if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    populations, source_paths = load_matrices(root)
    strong = [p for p in populations if p["family"] == "StrongREJECT"]
    common = set(strong[0]["row_ids"])
    assert all(set(p["row_ids"]) == common for p in strong)
    canonical_ids = sorted(common)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 0]))
    shared_pilot_ids = {canonical_ids[i] for i in rng.permutation(len(canonical_ids))[:PILOT_ROWS]}
    protocol = dict(seed=SEED, pilot_rows_per_population=PILOT_ROWS,
        strongreject_score_threshold=.5, error_probability=DELTA,
        split="fixed NumPy PCG64 SeedSequence([20260923, stream]); first 50 permuted rows",
        strongreject_split="stream 0 over sorted common context IDs; same 50 IDs for all four models",
        digits_split="stream 1 for logistic and 2 for forest, in archived eligible-row order",
        order="greedy coverage of pilot positives, ties by original coordinate index; all d attacks retained",
        pilot_measurement="fully reveal 50*d entries; do not early stop pilot",
        reference="uniform distribution over disjoint held-out rows, conditional on this fixed split",
        audit_sampling="hypothetical iid draws with replacement from held-out reference; fixed cohort size",
        scenario_a="same cohort r=floor(B/d); learner pilot is separate and included in expected total cost; learned total ceiling is B+50*d",
        scenario_b="same end-to-end ceiling B; uniform r=floor(B/d), learned r-50; pilot labels excluded from audit estimate",
        no_saved_cost_reinvestment=True, no_model_calls=True, no_monte_carlo_trials=True,
        numeric_precision="rational expected costs/theta/variance; floating-point CP quantiles and finite binomial enumeration")
    write_json(output / "study_protocol.json", protocol)
    splits, orders, paired_rows, summaries, scenarios = [], [], [], [], []
    validations = {}
    frozen_orders = {}
    for pop in populations:
        matrix = np.asarray(pop["matrix"], dtype=np.uint8)
        n, d = matrix.shape
        assert len(set(pop["row_ids"])) == n and np.isin(matrix, [0, 1]).all()
        assert len(pop["attack_ids"]) == d and len(set(pop["attack_ids"])) == d
        pilot_idx = pilot_indices(pop, shared_pilot_ids)
        heldout_idx = np.setdiff1d(np.arange(n), pilot_idx)
        assert len(pilot_idx) == PILOT_ROWS
        assert not np.intersect1d(pilot_idx, heldout_idx).size
        pilot = matrix[pilot_idx].copy()
        # Freeze the order before any held-out outcome is supplied to cost evaluation.
        order, trace = greedy_coverage_order(pilot)
        audit_order_validation(pilot, order, trace)
        heldout = matrix[heldout_idx]
        modified = matrix.copy()
        modified[heldout_idx] = 1 - modified[heldout_idx]
        assert np.array_equal(greedy_coverage_order(modified[pilot_idx])[0], order)
        S = heldout.sum(axis=1).astype(int)
        labels = S > 0
        learned_costs = ordered_first_cost(heldout, order)
        assert np.array_equal(heldout[:, order].any(axis=1), labels)
        assert np.all(learned_costs[~labels] == d)
        assert np.all((learned_costs >= 1) & (learned_costs <= d))
        uniform_costs = [uniform_row_cost(s, d) for s in S]
        uniform_mean = sum(uniform_costs, Fraction(0)) / len(heldout)
        learned_mean = Fraction(int(learned_costs.sum()), len(heldout))
        saving = uniform_mean - learned_mean
        theta = Fraction(int(labels.sum()), len(labels))
        pilot_cost = PILOT_ROWS * d
        break_even = None if saving <= 0 else (Fraction(pilot_cost) / saving).__ceil__()
        strict_break_even = None if saving <= 0 else (Fraction(pilot_cost) / saving).__floor__() + 1
        r = pop["budget"] // d
        common_info = dict(population=pop["name"], identifier=pop["identifier"],
                           family=pop["family"], d=d, source_rows=n,
                           pilot_rows=PILOT_ROWS, heldout_rows=len(heldout),
                           available_budget=pop["budget"])
        summaries.append(dict(**common_info, **fraction_fields("theta_heldout", theta),
            **fraction_fields("uniform_mean_cost", uniform_mean),
            **fraction_fields("learned_mean_cost", learned_mean),
            **fraction_fields("saving_per_audit_record", saving),
            relative_audit_cost_reduction=float(saving / uniform_mean),
            pilot_cost=pilot_cost, expected_cost_break_even_cohort=break_even,
            expected_cost_strict_saving_cohort=strict_break_even,
            heldout_rows_learned_cheaper=sum(Fraction(int(l)) < u for l, u in zip(learned_costs, uniform_costs)),
            heldout_rows_learned_equal=sum(Fraction(int(l)) == u for l, u in zip(learned_costs, uniform_costs)),
            heldout_rows_learned_costlier=sum(Fraction(int(l)) > u for l, u in zip(learned_costs, uniform_costs))))
        for scenario in ["A_same_audit_cohort", "B_same_total_ceiling"]:
            for method in ["uniform_order", "learned_order"]:
                uses_pilot = method == "learned_order"
                cohort = r - (PILOT_ROWS if uses_pilot and scenario.startswith("B") else 0)
                assert cohort > 0
                expected_audit_cost = cohort * (learned_mean if uses_pilot else uniform_mean)
                charged_pilot = pilot_cost if uses_pilot else 0
                total_cost = expected_audit_cost + charged_pilot
                total_ceiling = cohort * d + charged_pilot
                if scenario.startswith("B"):
                    assert total_ceiling <= pop["budget"]
                scenarios.append(dict(**common_info, scenario=scenario, method=method,
                    audit_cohort=cohort, pilot_cost_charged=charged_pilot,
                    audit_reserved_ceiling=cohort * d, total_reserved_ceiling=total_ceiling,
                    **fraction_fields("expected_audit_cost", expected_audit_cost),
                    **fraction_fields("expected_total_cost", total_cost),
                    **fraction_fields("prevalence_variance", theta * (1 - theta) / cohort),
                    expected_cp95_width=expected_cp_width(theta, cohort),
                    **fraction_fields("theta_heldout", theta)))
        pilot_set = set(pilot_idx.tolist())
        for source_index, row_id in enumerate(pop["row_ids"]):
            splits.append(dict(population=pop["name"], source_row_index=source_index,
                               source_row_id=row_id, membership="pilot" if source_index in pilot_set else "heldout"))
        for step in trace:
            orders.append(dict(population=pop["name"], attack_id=pop["attack_ids"][step["coordinate_index"]], **step))
        for i, source_index in enumerate(heldout_idx):
            difference = Fraction(int(learned_costs[i])) - uniform_costs[i]
            paired_rows.append(dict(population=pop["name"], source_row_index=int(source_index),
                source_row_id=pop["row_ids"][source_index], S=int(S[i]), suite_failure=int(labels[i]),
                learned_cost=int(learned_costs[i]), **fraction_fields("uniform_expected_cost", uniform_costs[i]),
                **fraction_fields("learned_minus_uniform_expected_cost", difference)))
        validations[pop["name"]] = dict(pilot_heldout_disjoint=True,
            complete_partition=sorted(pilot_idx.tolist() + heldout_idx.tolist()) == list(range(n)),
            heldout_outcome_flip_does_not_change_order=True,
            independently_recomputed_greedy_ties=True,
            full_suite_labels_invariant=True, all_zero_cost_equals_d=True,
            pilot_binary_sha256=hashlib.sha256(pilot.tobytes()).hexdigest(),
            heldout_binary_sha256=hashlib.sha256(heldout.tobytes()).hexdigest(),
            scenario_a_same_labels_and_cp_for_any_common_cohort=True)
        frozen_orders[pop["name"]] = dict(identifier=pop["identifier"],
            pilot_indices=pilot_idx.tolist(), heldout_indices=heldout_idx.tolist(),
            ordered_coordinate_indices=order.tolist(),
            ordered_attack_ids=[pop["attack_ids"][j] for j in order])
    checks = analytic_checks()
    checks.update(populations=validations, common_strongreject_pilot_ids=True,
                  fixed_split_conditional_results=True, no_heldout_order_optimization=True)
    write_csv(output / "population_summary.csv", summaries)
    write_csv(output / "scenario_summary.csv", scenarios)
    write_csv(output / "split_indices.csv", splits)
    write_csv(output / "attack_orders.csv", orders)
    write_csv(output / "heldout_row_costs.csv", paired_rows)
    write_json(output / "frozen_orders.json", frozen_orders)
    write_json(output / "validation.json", checks)
    write_json(output / "provenance.json", dict(
        source_sha256={str(p.relative_to(root)): sha256(p) for p in source_paths},
        script_sha256=sha256(Path(__file__)), numpy_version=np.__version__, scipy_version=scipy.__version__,
        binary_matrix_encoding="row-major uint8 0/1; pilot and heldout rows sorted by original source row index",
        source_rows="StrongREJECT context hash; Digits original load_digits dataset index"))
    lines = ["# Held-out learned attack order", "",
        "One fixed 50-row fully labeled pilot is used per population. The four StrongREJECT pilots use the same context IDs. The greedy order maximizes newly covered pilot rows and breaks ties by original column index. The order is computed solely from pilot outcomes and applied unchanged to held-out rows. This generated protocol records the analysis; it does not establish external preregistration. The greedy rule is classical min-sum set cover (Feige, Lovasz, and Tetali, Algorithmica 2004), with pilot-positive rows as elements and attacks as covering sets; its pilot approximation guarantee does not imply held-out transfer. No model calls or Monte Carlo audit trials are performed.", "",
        "All results condition on this split and on the uniform empirical distribution over its held-out rows. These are reference-population expectations, not measured deployment savings or confidence intervals for transfer to new populations. Attack outcomes have unit cost; original procedure/API costs may differ. Pilots cost 50d revealed outcomes even when a row already has a success. Pilot labels are not reused in the audit estimate.", "",
        "## Exact mean cost comparison", "",
        "| Population | Held-out rows | Failure rate | Random mean | Learned mean | Pilot cost | Break-even cohort |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['population']} | {row['heldout_rows']} | {row['theta_heldout']:.4f} | {row['uniform_mean_cost']:.3f} | {row['learned_mean_cost']:.3f} | {row['pilot_cost']} | {row['expected_cost_break_even_cohort']} |")
    lines += ["", "Break-even is the smallest fixed audit cohort whose expected saved evaluations cover the pilot, allowing equality. The strict-saving threshold is also stored. Individual learned costs are paired with each held-out row's exact expected random-order cost; a paired difference is not a realized random-order draw.", "",
        "## Two cost scenarios", "",
        "Scenario A holds the audit cohort at floor(B/d). The same sampled rows have identical suite labels, prevalence estimate, and Clopper--Pearson interval under either order. Learned ordering adds a fully charged pilot outside the common audit ceiling, so its total hard ceiling is B+50d. Expected costs include that pilot.", "",
        "Scenario B keeps the total hard ceiling B. The uniform method audits floor(B/d) records; the learned method pays for the pilot and audits floor(B/d)-50 independent held-out records. Any cost saving is accompanied by this smaller fixed audit cohort. No saved evaluations are reinvested. Scenario summaries report the exact binomial variance and expected two-sided 95% CP width at each fixed cohort size. This is a transparent fixed-cohort comparison, not an optimized allocation policy.", "",
        "| Population | Uniform total cost | A learned total cost | B learned total cost | Uniform CP width | B learned CP width |",
        "|---|---:|---:|---:|---:|---:|"]
    for pop in populations:
        selected = {(r["scenario"], r["method"]): r for r in scenarios if r["population"] == pop["name"]}
        uniform = selected[("A_same_audit_cohort", "uniform_order")]
        learned_a = selected[("A_same_audit_cohort", "learned_order")]
        learned_b = selected[("B_same_total_ceiling", "learned_order")]
        lines.append(f"| {pop['name']} | {uniform['expected_total_cost']:.1f} | {learned_a['expected_total_cost']:.1f} | {learned_b['expected_total_cost']:.1f} | {uniform['expected_cp95_width']:.5f} | {learned_b['expected_cp95_width']:.5f} |")
    lines += ["", "For this fixed split, learned ordering lowers the held-out per-record mean cost in all six populations. After charging the pilot, only GPT-4o-mini and Llama-3.1 have lower total expected cost at the tested cohort sizes, in either scenario. Scenario A changes cost but preserves inference on a common cohort. Scenario B reduces the audit cohort from 500 to 450 for StrongREJECT and 200 to 150 for Digits; nonsaturated prevalence variances therefore increase by 11.1% and 33.3%, respectively, and expected CP widths increase for every population. A favorable learned order is not by itself an end-to-end cost or precision improvement.", "",
        "The Scenario B precision penalty is imposed by the conditional held-out-reference design, which excludes pilot labels; it is not an intrinsic cost of learning an order. If pilot and follow-up records instead are independent draws from the same population P and the total full-label cohort r is fixed, all r labels can be pooled. The order learned from pilot outcomes cannot change a fully determined suite label. Unconditionally, the pooled success count is Binomial(r, theta), so the mean retains variance theta(1-theta)/r and the ordinary Clopper--Pearson interval. This same-population pooling observation is analytic; it is not an empirical result for the separate conditional held-out target used here.", "",
        "Expected costs, prevalence, and variances have rational representations in the CSVs. Expected CP widths are finite binomial sums using double-precision quantiles and probabilities; they are not interval-arithmetic certificates. Leakage, greedy tie handling, suite-label invariance, all-zero costs, small exhaustive permutation checks, and CP boundary checks are recorded in validation.json.", "",
        "Run: `python run_heldout_ordering.py`. Inputs and code hashes appear in provenance.json. Split membership, ordered attack IDs, row-level paired costs, and both scenario summaries are retained.", ""]
    (output / "README.md").write_text("\n".join(lines))
    print(json.dumps(dict(populations=len(populations), heldout_rows=len(paired_rows),
                         scenario_rows=len(scenarios), output=str(output)), sort_keys=True))
    return summaries, scenarios


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.root, args.output)
