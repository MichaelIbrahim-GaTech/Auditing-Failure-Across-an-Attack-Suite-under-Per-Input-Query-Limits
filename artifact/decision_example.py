"""Exact budget calculation for a proposed, suite-specific audit contract.

This is an illustrative design calculation, not an agent experiment. A 5% suite
failure threshold is a declared example, not a recommended deployment standard.
Only the zero-detection branch is certified; other observations return [0,1].
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

from zero_detection import minimax_zero_probability, zero_upper_bracket


def first_integer(predicate):
    """First nonnegative integer satisfying a monotone Boolean predicate."""
    if predicate(0):
        return 0
    lo, hi = 0, 1
    while not predicate(hi):
        lo, hi = hi, 2 * hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if predicate(mid):
            hi = mid
        else:
            lo = mid
    return hi


def screen_bracket(n, d, k, delta, bits=80):
    """Exact bracket for the zero-detection uniform-screen upper endpoint."""
    fun = lambda theta: (1 - theta * F(k, d)) ** n
    if fun(F(1)) >= delta:
        return F(1), F(1)
    lo, hi = F(0), F(1)
    for _ in range(bits):
        mid = (lo + hi) / 2
        if fun(mid) >= delta:
            lo = mid
        else:
            hi = mid
    return lo, hi


def main(output_dir="decision_results"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    d, k, tau, delta = 36, 4, F(1, 20), F(1, 20)
    budget = first_integer(lambda B: minimax_zero_probability(B, d, tau) <= delta)
    full_n = first_integer(lambda n: (1 - tau) ** n <= delta)
    screen_n = first_integer(lambda n: (1 - tau * F(k, d)) ** n <= delta)
    assert (budget, full_n, screen_n) == (2103, 59, 538)
    assert minimax_zero_probability(budget - 1, d, tau) > delta
    assert (1 - tau) ** (full_n - 1) > delta
    assert (1 - tau * F(k, d)) ** (screen_n - 1) > delta
    rows, exact = [], []
    cases = [("complete_plus_remainder", budget - 1),
             ("complete_plus_remainder", budget),
             ("complete_labels", (full_n - 1) * d),
             ("complete_labels", full_n * d),
             ("uniform_screen", (screen_n - 1) * k),
             ("uniform_screen", screen_n * k)]
    for design, B in cases:
        if design == "uniform_screen":
            n = B // k
            prob = (1 - tau * F(k, d)) ** n
            lo, hi = screen_bracket(n, d, k, delta)
            full, remainder = 0, 0
        else:
            full, remainder = divmod(B, d)
            n = full + int(remainder > 0)
            prob = minimax_zero_probability(B, d, tau)
            lo, hi = zero_upper_bracket(B, d, delta)
        assert hi - lo <= F(1, 2**80)
        assert (hi <= tau) == (prob <= delta)
        displayed_hi = float(hi)
        if F.from_float(displayed_hi) < hi:
            displayed_hi = math.nextafter(displayed_hi, math.inf)
        rows.append(dict(design=design, budget=B, inputs=n,
                         complete_inputs=full, remainder_depth=remainder,
                         d=d, k=k if design == "uniform_screen" else "",
                         threshold=float(tau), alpha=float(delta),
                         worst_zero_probability_at_threshold=float(prob),
                         upper_endpoint=displayed_hi,
                         passes_zero_detection_gate=prob <= delta))
        exact.append(dict(design=design, budget=B,
                          probability_at_threshold=str(prob),
                          endpoint_lower=str(lo), endpoint_upper=str(hi)))
    with (out / "budget_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "exact_checks.json").write_text(json.dumps(dict(
        status="passed", d=d, k=k, threshold=str(tau), alpha=str(delta),
        minimum_total_budget=budget, minimum_complete_inputs=full_n,
        minimum_equal_depth_screen_inputs=screen_n, boundary_checks=3,
        root_brackets=exact,
        scope="Common deterministic zero-detection endpoint; attaining design. "
              "The equal-depth screen comparison does not claim optimality "
              "over all per-input-capped protocols. No deployment data."), indent=2)+"\n")
    (out / "provenance.json").write_text(json.dumps({name: hashlib.sha256(
        Path(name).read_bytes()).hexdigest()
        for name in ["decision_example.py", "zero_detection.py"]}, indent=2)+"\n")
    (out / "README.md").write_text(
        "# Proposed audit-contract decision calculation\n\n"
        "Run `python decision_example.py`. The script uses exact fractions and "
        "80-bit rational root brackets. The six rows compare the last failing "
        "and first passing designs for three specified allocation families. "
        "The total-budget optimum follows the manuscript theorem; equal-depth "
        "screening is a particular comparator. The threshold .05 is illustrative. "
        "All designs report [0,U] after zero detections and [0,1] otherwise. "
        "This is a mathematical decision example, with no new agent measurements.\n")
    print(json.dumps({"minimum_total_budget": budget,
                      "full_labels_budget": full_n * d,
                      "equal_depth_screen_budget": screen_n * k,
                      "attaining_zero_endpoint": rows[1]["upper_endpoint"],
                      "validation": "passed"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="decision_results")
    main(parser.parse_args().output_dir)
