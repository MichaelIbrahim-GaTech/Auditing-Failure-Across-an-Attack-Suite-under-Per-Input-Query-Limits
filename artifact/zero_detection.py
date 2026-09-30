"""Exact worst-case zero-detection analysis under a hard total query budget.

Assumptions: an unlimited IID input stream; d fixed binary outcomes per input;
unit query costs; outcome-only selection; at most B queries. Audits may randomize,
interleave, revisit inputs, stop early, and use every previous observed bit.

The minimax no-success probability at fixed suite prevalence theta is
    (1-theta)**r * (1-theta*b/d),  B = r*d + b, 0 <= b < d.
It is attained by r fully determined fresh inputs plus a uniform b-coordinate
subset on one more fresh input. Zero-success upper bounds below apply to THIS
attaining design, not to every arbitrary audit with the same budget ceiling.
The result optimizes this event probability, not complete confidence procedures,
MSE, or expected-cost allocation. No literature-novelty claim is made here.

Run ``python zero_detection.py`` for a small exact-arithmetic validation study
and predetermined numerical examples. Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import operator
from fractions import Fraction
from functools import lru_cache
from pathlib import Path


def _integer(value, name, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _probability(value, name, strict=False):
    # A decimal float such as .05 denotes its displayed decimal value. Fractions
    # are accepted directly when exact numerator/denominator control is wanted.
    try:
        result = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{name} must be a finite probability") from exc
    if not (0 < result < 1 if strict else 0 <= result <= 1):
        raise ValueError(f"{name} must lie {'strictly ' if strict else ''}between 0 and 1")
    return result


def minimax_zero_probability(B, d, theta):
    """Exact Fraction minimax probability of no observed success at theta.

    The minimax quantifiers are inf over permitted audits, sup over populations
    with this theta. This is an upper envelope for the attaining design only.
    """
    B, d = _integer(B, "B"), _integer(d, "d", 1)
    theta = _probability(theta, "theta")
    r, b = divmod(B, d)
    # Fraction(0) ** 0 is 1, as required when theta=1 and r=0.
    return (1 - theta) ** r * (1 - theta * Fraction(b, d))


def zero_upper_bracket(B, d, delta=Fraction(1, 20), bits=80):
    """Exact rational bracket for the sharp common zero-success endpoint.

    Use [0,U] on no successes and [0,1] otherwise with the attaining design.
    U=sup{theta in [0,1]: minimax_zero_probability(B,d,theta) >= delta}.
    Returns (lo, hi), hi-lo <= 2**(-bits), unless both equal the exact endpoint.
    hi is an outward, coverage-conservative endpoint. This uses exact rational
    evaluations, not floating-point root-finding or beta-quantile inversion.
    """
    B, d = _integer(B, "B"), _integer(d, "d", 1)
    bits = _integer(bits, "bits", 1)
    delta = _probability(delta, "delta", strict=True)
    if B == 0 or minimax_zero_probability(B, d, 1) >= delta:
        return Fraction(1), Fraction(1)
    if B < d:
        endpoint = Fraction(d, B) * (1 - delta)
        return endpoint, endpoint
    if B == d:
        return 1 - delta, 1 - delta
    lo, hi = Fraction(0), Fraction(1)
    for _ in range(bits):
        mid = (lo + hi) / 2
        value = minimax_zero_probability(B, d, mid)
        if value == delta:
            return mid, mid
        if value > delta:
            lo = mid
        else:
            hi = mid
    return lo, hi


def _outward_float(value):
    """Smallest adjacent float at or above an exact rational endpoint."""
    result = float(value)
    if Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return min(1.0, result)


def zero_detection_upper_bound(B, d, delta=Fraction(1, 20), bits=80):
    """Coverage-conservative float endpoint for the attaining design."""
    return _outward_float(zero_upper_bracket(B, d, delta, bits)[1])


def allocation_zero_probability(depths, d, theta):
    """Exact zero probability under symmetric singleton populations."""
    d, theta = _integer(d, "d", 1), _probability(theta, "theta")
    answer = Fraction(1)
    for depth in depths:
        depth = _integer(depth, "depth")
        if depth > d:
            raise ValueError("A depth cannot exceed d")
        answer *= 1 - theta * Fraction(depth, d)
    return answer


def integer_allocations(total, d, minimum=1):
    """Enumerate every unordered positive depth allocation of exactly total.

    Permutation and input-label copies have the same probability under the
    singleton population, so omitting those copies loses no possible product.
    """
    if total == 0:
        yield ()
        return
    for first in range(minimum, min(d, total) + 1):
        for tail in integer_allocations(total - first, d, first):
            yield (first,) + tail


def exhaustive_allocation_minimum(B, d, theta):
    """Enumerate all integer allocations using AT MOST B queries."""
    B, d = _integer(B, "B"), _integer(d, "d", 1)
    theta = _probability(theta, "theta")
    best, witness, count = Fraction(1), (), 0
    for total in range(B + 1):
        for depths in integer_allocations(total, d):
            count += 1
            probability = allocation_zero_probability(depths, d, theta)
            if probability < best:
                best, witness = probability, depths
    return best, witness, count


def bellman_zero_probability(B, d, theta):
    """Exact adaptive-depth Bellman minimum on the all-zero path.

    A state stores queried depths of independent inputs whose observed bits
    were all zero. After t zero bits on one input, the conditional probability
    the next bit is zero is (d-theta*(t+1))/(d-theta*t). Every step can stop,
    start a fresh input, or return to ANY unfinished input. Label choices are
    immaterial for the singleton population. Randomized choices are convex
    combinations of these action values and cannot improve their minimum.

    Impossible zero branches (conditional survival zero) terminate with value
    zero rather than conditioning on an event of probability zero.
    """
    B, d = _integer(B, "B"), _integer(d, "d", 1)
    theta = _probability(theta, "theta")

    @lru_cache(maxsize=None)
    def solve(remaining, depths):
        if remaining == 0:
            return Fraction(1)
        answer = Fraction(1)  # Stopping or wasting the remaining budget.
        # index=-1 starts a new input; one copy per existing depth suffices.
        actions = [(-1, 0)]
        seen = set()
        for index, depth in enumerate(depths):
            if depth < d and depth not in seen:
                actions.append((index, depth))
                seen.add(depth)
        for index, depth in actions:
            survival = (d - theta * (depth + 1)) / (d - theta * depth)
            if survival == 0:
                candidate = Fraction(0)
            else:
                updated = list(depths)
                if index == -1:
                    updated.append(1)
                else:
                    updated[index] += 1
                candidate = survival * solve(remaining - 1, tuple(sorted(updated)))
            answer = min(answer, candidate)
        return answer

    value = solve(B, ())
    return value, solve.cache_info().currsize


def _partial_upper(m, k, d, delta, bits=80):
    # (1-theta*k/d)**m <= delta, with exact rational root bracketing.
    root_hi = zero_upper_bracket(m, 1, delta, bits)[1]
    return _outward_float(min(Fraction(1), root_hi * Fraction(d, k)))


def _require(condition, explanation):
    if not condition:
        raise AssertionError(explanation)


def validation_records():
    """Small independently formulated checks, including all boundary cases."""
    rows = []
    thetas = (Fraction(0), Fraction(1, 5), Fraction(1, 2), Fraction(4, 5), Fraction(1))
    allocation_evaluations, bellman_states = 0, 0
    for d in range(1, 5):
        for B in range(8):
            for theta in thetas:
                brute, witness, allocation_count = exhaustive_allocation_minimum(B, d, theta)
                dynamic, state_count = bellman_zero_probability(B, d, theta)
                closed = minimax_zero_probability(B, d, theta)
                _require(brute == dynamic == closed, f"Allocation/Bellman mismatch: {(B,d,theta)}")
                allocation_evaluations += allocation_count
                bellman_states += state_count
                rows.append(dict(B=B, d=d, theta=str(theta), exact_minimum=str(brute),
                                 formula=str(closed), bellman=str(dynamic),
                                 minimizing_depths=json.dumps(witness),
                                 allocations_checked=allocation_count, bellman_states=state_count))

    # Direct enumeration of every possible successful-position set, independent
    # of both the hypergeometric expression and the singleton product formula.
    from itertools import combinations
    subset_checks = 0
    for d in range(1, 7):
        for b in range(d):
            subsets = list(combinations(range(d), b))
            for s in range(1, d + 1):
                positives = set(range(s))
                missed = sum(not positives.intersection(subset) for subset in subsets)
                probability = Fraction(missed, len(subsets))
                _require(probability <= 1 - Fraction(b, d), "Uniform-subset envelope failed")
                if s == 1:
                    _require(probability == 1 - Fraction(b, d), "Singleton did not attain envelope")
                subset_checks += 1

    # Explicit inversion boundaries and analytic formulas.
    exact_cases = ((0, 4, Fraction(1, 20), Fraction(1)),
                   (1, 4, Fraction(1, 20), Fraction(1)),
                   (3, 4, Fraction(1, 4), Fraction(1)),
                   (3, 4, Fraction(1, 2), Fraction(2, 3)),
                   (4, 4, Fraction(1, 20), Fraction(19, 20)),
                   (8, 4, Fraction(1, 4), Fraction(1, 2)))
    for B, d, delta, expected in exact_cases:
        _require(zero_upper_bracket(B, d, delta) == (expected, expected),
                 f"Exact inverse mismatch: {(B,d,delta)}")
    inverse_cases = ((1, 1), (2, 1), (9, 4), (17, 5), (61, 64),
                     (64, 64), (65, 64), (2000, 64), (2048, 64), (7200, 36), (19600, 98))
    for B, d in inverse_cases:
        delta = Fraction(1, 20)
        lo, hi = zero_upper_bracket(B, d, delta)
        _require(hi - lo <= Fraction(1, 2**80), "Inverse bracket too wide")
        _require(minimax_zero_probability(B, d, lo) >= delta, "Lower root bracket failed")
        _require(minimax_zero_probability(B, d, hi) <= delta, "Upper root bracket failed")
        rounded = zero_detection_upper_bound(B, d, delta)
        _require(Fraction.from_float(rounded) >= hi, "Float endpoint rounded inward")
        if hi < 1:
            _require(minimax_zero_probability(B, d, (hi + 1) / 2) < delta,
                     "Zero-event confidence coverage check failed")
    # If a proposed common endpoint u is too small, an intermediate prevalence
    # exceeds u while its zero-event probability exceeds delta: coverage fails.
    proposed_u, counterexample_theta = Fraction(9, 10), Fraction(37, 40)
    _require(counterexample_theta > proposed_u and
             minimax_zero_probability(4, 4, counterexample_theta) > Fraction(1, 20),
             "Sharpness counterexample failed")
    invalid_cases = ((-1, 4, Fraction(1, 2)), (2, 0, Fraction(1, 2)),
                     (2.5, 4, Fraction(1, 2)), (2, 4, Fraction(3, 2)))
    for args in invalid_cases:
        try:
            minimax_zero_probability(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid parameter accepted: {args}")
    summary = dict(status="passed", exact_allocation_bellman_cases=len(rows),
                   exact_allocation_products_checked=allocation_evaluations,
                   bellman_states_evaluated=bellman_states, subset_envelope_cases=subset_checks,
                   exact_inverse_boundary_cases=len(exact_cases), inverse_bracket_cases=len(inverse_cases),
                   invalid_parameter_cases=len(invalid_cases), fraction_bisection_bits=80,
                   arithmetic="Exact fractions for checks and root brackets; outward-rounded display floats",
                   scope="Small exhaustive checks support the written proof; they do not enumerate arbitrary infinite policy classes.")
    return rows, summary


def example_records():
    """Predetermined boundary cases and budgets already used in the manuscript."""
    settings = (("no_queries", 0, 64), ("insufficient_for_nontrivial_95_percent", 60, 64),
                ("first_nontrivial_95_percent_budget", 61, 64), ("one_full_input", 64, 64),
                ("one_full_plus_one_query", 65, 64), ("paper_64_attacks_2000_queries", 2000, 64),
                ("divisible_64_attack_budget", 2048, 64),
                ("controlled_strongreject_budget", 7200, 36),
                ("controlled_classifier_budget", 19600, 98))
    rows, delta = [], Fraction(1, 20)
    for name, B, d in settings:
        r, b = divmod(B, d)
        lo, hi = zero_upper_bracket(B, d, delta)
        row = dict(example=name, B=B, d=d, r=r, b=b, delta=str(delta),
                   sharp_zero_upper=zero_detection_upper_bound(B, d, delta),
                   endpoint_lower_exact=str(lo), endpoint_upper_exact=str(hi),
                   floor_full_only_upper=zero_detection_upper_bound(r*d, d, delta),
                   uniform_k4_upper="")
        if B > 0 and B % 4 == 0:
            row["uniform_k4_upper"] = _partial_upper(B // 4, 4, d, delta)
        rows.append(row)
    return rows


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "zero_detection_results")
    args = parser.parse_args(argv)
    rows, summary = validation_records()
    examples = example_records()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "allocation_checks.csv", rows)
    _write_csv(args.output_dir / "bound_examples.csv", examples)
    summary["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary["examples"] = len(examples)
    (args.output_dir / "validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
