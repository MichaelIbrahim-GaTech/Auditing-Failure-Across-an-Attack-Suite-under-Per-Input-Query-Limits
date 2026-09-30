"""Independent small-cohort enumeration for two-stage auditing.

Enumerates full negative labels and then hypergeometric selection rather than
assuming the selected-label binomial law used by the confidence procedure.
"""
from pathlib import Path
from functools import lru_cache
from itertools import combinations
from math import comb
import json

import numpy as np
from scipy.stats import beta, binom, hypergeom


ROOT = Path(__file__).resolve().parent


@lru_cache(None)
def cp(x, n, alpha):
    if n == 0:
        return (0., 1.)
    lo = 0. if x == 0 else float(beta.ppf(alpha / 2, x, n-x+1))
    hi = 1. if x == n else float(beta.ppf(1-alpha / 2, x+1, n-x))
    return lo, hi


@lru_cache(None)
def interval(n, m, x, r, R, delta=.05):
    if R >= m:
        return cp(n+x, m, delta)
    al, au = cp(n, m, delta/2)
    bl, bu = cp(x, r, delta/2)
    return al+(1-al)*bl, au+(1-au)*bu


def states(m, R):
    rows = []
    cohort_error = 0.
    for n in range(m+1):
        M = m-n
        r = min(R, M)
        for z in range(M+1):
            choices = [(0, 1.)] if M == 0 else [
                (x, float(hypergeom.pmf(x, M, z, r)))
                for x in range(max(0, r-(M-z)), min(z, r)+1)
            ]
            conditional_mean = 0.
            for x, probability in choices:
                estimate = 1. if M == 0 else n/m+(M/m)*x/r
                lo, hi = interval(n, m, x, r, R)
                conditional_mean += probability*estimate
                rows.append((n, M, z, probability, estimate, lo, hi))
            cohort_error = max(cohort_error, abs(conditional_mean-(n+z)/m))
    return rows, cohort_error


def validate_estimator():
    grid = [0., .001, .01, .05, .1, .25, .5, .75, .9, .95, .99, .999, 1.]
    max_mass_error = max_mean_error = max_var_error = max_cohort_error = 0.
    minimum_coverage = 1.
    minimum_setting = None
    settings = 0
    cohort_states = 0
    results = []
    for m in [1, 2, 4, 8, 16]:
        for R in sorted({1, min(2, m), max(1, m-1), m}):
            rows, cohort_error = states(m, R)
            max_cohort_error = max(max_cohort_error, cohort_error)
            cohort_states += len(rows)
            cols = np.array(rows, dtype=float)
            n, M, z = (cols[:, j].astype(int) for j in range(3))
            for a in grid:
                p_n = binom.pmf(n, m, a)
                for b in grid:
                    theta = a+(1-a)*b
                    weights = p_n*binom.pmf(z, M, b)*cols[:, 3]
                    mass = weights.sum()
                    mean = np.dot(weights, cols[:, 4])
                    variance = np.dot(weights, (cols[:, 4]-theta)**2)
                    coverage = weights[(cols[:, 5] <= theta+1e-14) &
                                       (theta <= cols[:, 6]+1e-14)].sum()
                    masses = np.arange(m+1)
                    extra = np.dot(binom.pmf(masses, m, 1-a),
                                   masses*np.maximum(masses-R, 0))
                    theory_var = theta*(1-theta)/m+b*(1-b)*extra/(m*m*R)
                    max_mass_error = max(max_mass_error, abs(mass-1))
                    max_mean_error = max(max_mean_error, abs(mean-theta))
                    max_var_error = max(max_var_error, abs(variance-theory_var))
                    if coverage < minimum_coverage:
                        minimum_coverage = float(coverage)
                        minimum_setting = dict(m=m, R=R, a=a, b=b)
                    if coverage < .95-1e-12:
                        raise AssertionError((m, R, a, b, coverage))
                    settings += 1
            results.append(dict(m=m, R=R, enumerated_states=len(rows)))
    assert max_mass_error < 1e-12
    assert max_mean_error < 1e-12
    assert max_var_error < 1e-12
    assert max_cohort_error < 1e-12
    return dict(parameter_settings=settings, full_cohort_selection_states=cohort_states,
                designs=results, max_probability_mass_error=max_mass_error,
                max_unbiasedness_error=max_mean_error, max_variance_error=max_var_error,
                max_conditional_cohort_mean_error=max_cohort_error,
                minimum_grid_coverage=minimum_coverage,
                minimum_grid_coverage_setting=minimum_setting)


def validate_first_hit():
    max_screen_error = max_completion_error = max_full_error = 0.
    settings = 0
    for d in range(1, 10):
        for s in range(d+1):
            configurations = list(combinations(range(1, d+1), s))
            first = np.array([min(c) if c else d+1 for c in configurations])
            full_cost = np.minimum(first, d)
            full_theory = d if s == 0 else (d+1)/(s+1)
            max_full_error = max(max_full_error, abs(full_cost.mean()-full_theory))
            for k in range(1, d+1):
                screen = np.minimum(first, k)
                tau = sum(comb(d-s, t)/comb(d, t) if t <= d-s else 0.
                          for t in range(k))
                max_screen_error = max(max_screen_error, abs(screen.mean()-tau))
                negatives = first > k
                if negatives.any():
                    additional = full_cost[negatives]-k
                    u = d-k if s == 0 else (d-k+1)/(s+1)
                    max_completion_error = max(max_completion_error,
                                               abs(additional.mean()-u))
                settings += 1
    assert max(max_screen_error, max_completion_error, max_full_error) < 1e-12
    return dict(settings=settings, max_screen_error=max_screen_error,
                max_completion_error=max_completion_error, max_full_error=max_full_error)


if __name__ == '__main__':
    result = dict(method='full cohort multinomial types and hypergeometric selection',
                  estimator=validate_estimator(), first_hit=validate_first_hit(),
                  caveat='Grid coverage is implementation validation; uniform coverage follows from proof.')
    out = ROOT/'completion_results'
    out.mkdir(exist_ok=True)
    (out/'theory_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
