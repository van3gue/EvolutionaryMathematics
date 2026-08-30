"""
Setup B: Optimizer evolution.

What's evolved: the optimizer itself (objective smoothing, step schedule,
handling of tied maximizing shifts, projection). The heights it starts
from are FIXED (our warm start) - only the minimizer logic should change
across generations. This file is self-contained (see setup_a/initial.py
for why: it gets copied to an isolated folder per generation).
"""

import os
import numpy as np
from scipy.signal import correlate


def _worst_case_overlap(h: np.ndarray) -> float:
    n = h.shape[0]
    w = 2.0 / n
    g = 1.0 - h
    corr = correlate(h, g, mode="full", method="fft")
    lags = np.arange(-(n - 1), n)
    mask = lags != 0
    return float(np.max(corr[mask]) * w)


def _load_starting_heights(n):
    """Fixed starting point for this setup - resample the n=600 warm start
    to whatever resolution n is being used."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    warm_start_path = os.path.join(
        os.path.dirname(this_dir), "warm_start_n600.npy"
    )
    h0 = np.load(warm_start_path)
    if len(h0) == n:
        return h0
    # Resample via nearest-neighbor on the step grid.
    idx = np.floor(np.linspace(0, len(h0), n, endpoint=False)).astype(int)
    idx = np.clip(idx, 0, len(h0) - 1)
    return h0[idx]


def _project(h, n):
    """Project onto the feasible set: 0<=h<=1, sum(h)*w == 1."""
    w = 2.0 / n
    h = np.clip(h, 0.0, 1.0)
    total = np.sum(h) * w
    if total > 1e-9:
        h = np.clip(h * (1.0 / total), 0.0, 1.0)
    return h


# EVOLVE-BLOCK-START
"""
Evolve this part: the optimizer. The starting heights are fixed (loaded
above) - your job is to write a better minimize_overlap(h0, n) function.

The current baseline is deliberately naive: plain subgradient descent that
only looks at the SINGLE most-violating shift each step. This struggles
near the optimum because multiple shifts tend to tie for the maximum
there, so the "gradient" direction keeps jumping between them (an
active-set problem). Consider: smoothing the max with log-sum-exp or a
p-norm so it's differentiable everywhere, tracking a small set of
near-tied active shifts instead of just one, adaptive step sizes, or a
proper constrained solver (SLSQP/L-BFGS-B) on the smoothed objective.
"""


def minimize_overlap(h0, n):
    h = h0.copy()
    w = 2.0 / n
    g = 1.0 - h

    step = 0.01
    for _ in range(300):
        g = 1.0 - h
        corr = correlate(h, g, mode="full", method="fft")
        lags = np.arange(-(n - 1), n)
        mask = lags != 0
        j_star_idx = np.argmax(corr[mask])
        j_star = lags[mask][j_star_idx]

        # Subgradient of w * sum_i h[i]*(1-h[i+j_star]) w.r.t. h, considering
        # only this single most-violating shift.
        grad = np.zeros(n)
        if j_star > 0:
            valid = np.arange(0, n - j_star)
            grad[valid] += (1.0 - h[valid + j_star]) * w
            grad[valid + j_star] += -h[valid] * w
        elif j_star < 0:
            jj = -j_star
            valid = np.arange(0, n - jj)
            grad[valid + jj] += (1.0 - h[valid]) * w
            grad[valid] += -h[valid + jj] * w

        h = h - step * grad
        h = _project(h, n)

    return h


# EVOLVE-BLOCK-END


def run_setup_b():
    n = 2048
    h0 = _load_starting_heights(n)
    h = minimize_overlap(h0, n)
    h = _project(h, n)
    c_max = _worst_case_overlap(h)
    return h, c_max, n
