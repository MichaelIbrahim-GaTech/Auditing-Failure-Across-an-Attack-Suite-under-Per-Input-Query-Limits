"""Fixed-budget screening followed by uniform completion of negative screens.

All randomness here resamples a frozen finite outcome population. No model is
queried by this module. Uniform attack permutations are simulated through their
exact first-success distribution.
"""
from functools import lru_cache
import numpy as np
from scipy.special import betaincinv
from scipy.stats import binom


@lru_cache(maxsize=200000)
def cp_bounds(x, n, error=.05, one_sided=False):
    """Equal-tail CP interval, or an upper CP bound if one_sided=True."""
    x, n = int(x), int(n)
    if n < 0 or x < 0 or x > n or not 0 < error < 1:
        raise ValueError("Invalid binomial count or error probability")
    if n == 0:
        return 0., 1.
    tail = error if one_sided else error / 2
    lower = 0. if x == 0 or one_sided else float(betaincinv(x, n - x + 1, tail))
    upper = 1. if x == n else float(betaincinv(x + 1, n - x, 1 - tail))
    return lower, upper


def completion_interval(n_detected, m, n_completed_positive, n_completed,
                        R, delta=.05, k=None, d=None, one_sided=False):
    M = int(m - n_detected)
    x, r = int(n_completed_positive), int(n_completed)
    if m < 1 or R < 1 or M < 0 or M > m or r != min(R, M) or x < 0 or x > r:
        raise ValueError("Invalid two-stage audit counts")
    # The full-suite reduction is valid only when completion was deterministic
    # in the design. Realizing M <= R with R < m does NOT trigger this branch.
    if R >= m or (k is not None and d is not None and k == d):
        return cp_bounds(n_detected + x, m, delta, one_sided)
    al, au = cp_bounds(n_detected, m, delta / 2, one_sided)
    bl, bu = cp_bounds(x, r, delta / 2, one_sided)
    return al + (1 - al) * bl, au + (1 - au) * bu


def first_hit_distribution(mu):
    """P(J=j), j=1,...,d; sentinel J=d+1 represents no success."""
    mu = np.asarray(mu, dtype=float)
    d = len(mu) - 1
    if d < 1 or np.min(mu) < 0 or not np.isclose(mu.sum(), 1):
        raise ValueError("Invalid success-count distribution")
    s = np.arange(d + 1)
    remaining = np.ones(d + 1)
    probs = np.zeros(d + 1)
    for j in range(1, d + 1):
        left = d - j + 1
        probs[j - 1] = np.dot(mu, remaining * s / left)
        remaining *= np.maximum(0., (left - s) / left)
    probs[d] = mu[0]
    assert np.min(probs) >= 0 and np.isclose(probs.sum(), 1, atol=1e-13)
    return probs / probs.sum()


def allocation(B, d, k, screen_fraction):
    if not 1 <= k < d or not 0 < screen_fraction < 1:
        raise ValueError("Require 1 <= k < d and a screen fraction in (0,1)")
    m = int(np.floor(screen_fraction * B / k))
    R = min(m, int((B - m * k) // (d - k)))
    if m < 1 or R < 1:
        raise ValueError("Budget too small for both stages")
    return m, R


def draw_first_hits(rng, first_probs, m):
    return rng.choice(len(first_probs), size=m, p=first_probs) + 1


def completion_trial(rng, first_probs, m, R, k, delta=.05):
    d = len(first_probs) - 1
    J = draw_first_hits(rng, first_probs, m)
    negative = np.flatnonzero(J > k)
    M = len(negative)
    detected = m - M
    r = min(R, M)
    selected = rng.choice(negative, size=r, replace=False) if r else np.empty(0, dtype=int)
    x = int(np.sum(J[selected] <= d))
    point = detected / m + (M / m) * (x / r if r else 0.)
    lower, upper = completion_interval(detected, m, x, r, R, delta, k, d)
    _, cert_upper = completion_interval(detected, m, x, r, R, delta, k, d, True)
    calls = int(np.minimum(J, k).sum() + (np.minimum(J[selected], d) - k).sum())
    assert calls <= m * k + R * (d - k)
    return dict(point=point, raw_point=point, lower=lower, upper=upper,
                cert_upper=cert_upper, calls=calls, m=m, R=R,
                screen_negatives=M, completed=r, completed_positive=x)


def full_trial(rng, first_probs, n, delta=.05):
    d = len(first_probs) - 1
    J = draw_first_hits(rng, first_probs, n)
    x = int(np.sum(J <= d))
    lower, upper = cp_bounds(x, n, delta)
    _, cert_upper = cp_bounds(x, n, delta, True)
    return dict(point=x / n, raw_point=x / n, lower=lower, upper=upper,
                cert_upper=cert_upper, calls=int(np.minimum(J, d).sum()),
                m=n, R=0, screen_negatives=n - x, completed=n, completed_positive=x)


def detection_trial(rng, first_probs, m, k, delta=.05):
    d = len(first_probs) - 1
    J = draw_first_hits(rng, first_probs, m)
    x = int(np.sum(J <= k))
    al, au = cp_bounds(x, m, delta)
    _, one_upper = cp_bounds(x, m, delta, True)
    lower, upper = al, min(1., au * d / k)
    point = (lower + upper) / 2
    return dict(point=point, raw_point=point, lower=lower, upper=upper,
                cert_upper=min(1., one_upper * d / k), calls=int(np.minimum(J, k).sum()),
                m=m, R=0, screen_negatives=m - x, completed=0, completed_positive=0)


def exact_completion_moments(first_probs, m, R, k):
    d = len(first_probs) - 1
    theta = float(1 - first_probs[d])
    a = float(first_probs[:k].sum())
    q = float(max(0., 1 - a))
    b = float((theta - a) / q) if q > 1e-14 else 0.
    b = float(np.clip(b, 0, 1))
    M = np.arange(m + 1)
    mass = binom.pmf(M, m, q)
    excess = np.sum(mass * M * np.maximum(M - R, 0))
    variance = theta * (1 - theta) / m + b * (1 - b) * excess / (m * m * R)
    direct = (1 - b) ** 2 * a * (1 - a) / m
    direct += b * (1 - b) / (m * m) * np.sum(mass[1:] * M[1:] ** 2 / np.minimum(R, M[1:]))
    positions = np.arange(1, d + 2)
    screen_cost = float(np.dot(first_probs, np.minimum(positions, k)))
    conditional_finish = float(np.dot(first_probs[k:], np.minimum(positions[k:], d) - k) / q) if q > 1e-14 else 0.
    mean_calls = m * screen_cost + float(np.dot(mass, np.minimum(R, M))) * conditional_finish
    assert abs(variance - direct) < 1e-10
    return dict(theta=theta, detection_probability=a, residual_probability=b,
                expected_point=theta, exact_bias=0., exact_variance=float(variance),
                exact_mse=float(variance), expected_calls=float(mean_calls),
                variance_equivalence_residual=float(abs(variance - direct)))
