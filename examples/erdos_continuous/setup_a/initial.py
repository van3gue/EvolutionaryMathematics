"""
Setup A: SOTA refinement.

What's evolved: the heights array (starting point) AND the local polisher
that refines it. This file must be fully self-contained (no importing
sibling files like common.py) because ShinkaEvolve copies this file into
an isolated folder for each generation - only evaluate.py, which stays in
its original location, can safely import shared helpers.
"""

import os
import numpy as np
from scipy.optimize import minimize
from scipy.signal import correlate


def _worst_case_overlap(h: np.ndarray) -> float:
    """Self-contained copy of the correlation objective (see common.py for
    the fully documented version - kept in sync manually since this file
    must stand alone)."""
    n = h.shape[0]
    w = 2.0 / n
    g = 1.0 - h
    corr = correlate(h, g, mode="full", method="fft")
    lags = np.arange(-(n - 1), n)
    mask = lags != 0
    return float(np.max(corr[mask]) * w)


# EVOLVE-BLOCK-START
"""Evolve this part: the starting construction and/or the polish routine."""


def load_starting_heights(n):
    """
    Load our best available starting construction. This is a SELF-DERIVED
    warm start (multi-restart local optimization, see make_warm_start.py),
    NOT the literal published record - we could not recover Haugland's or
    AlphaEvolve's actual numeric heights from the source papers.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    warm_start_path = os.path.join(
        os.path.dirname(this_dir), "warm_start_n600.npy"
    )
    if os.path.exists(warm_start_path) and n == 600:
        return np.load(warm_start_path)

    # Fallback for other n: a smooth structured guess.
    w = 2.0 / n
    x = (np.arange(n) + 0.5) * w
    h = 0.5 + 0.3 * np.sin(2 * np.pi * x / 2.0)
    return np.clip(h, 0.0, 1.0)


def polish(h0, n):
    """
    Local refinement via SLSQP directly on the true (non-smoothed) max.
    This is deliberately simple - Setup B is where a smarter, smoothed
    objective and better handling of tied maximizing shifts belongs.
    Evolution here should focus on things like: better initial guesses,
    multiple restarts kept within the per-eval time budget, tweaks to
    solver tolerances, or simple post-processing of the polished result.
    """
    w = 2.0 / n

    def objective(h_free):
        return _worst_case_overlap(np.clip(h_free, 0.0, 1.0))

    cons = [{"type": "eq", "fun": lambda h_free: np.sum(h_free) * w - 1.0}]
    bounds = [(0.0, 1.0)] * n

    res = minimize(
        objective, h0, method="SLSQP", bounds=bounds, constraints=cons,
        options={"maxiter": 150, "ftol": 1e-12},
    )
    return np.clip(res.x, 0.0, 1.0)


def construct_heights():
    n = 600
    h0 = load_starting_heights(n)
    h = polish(h0, n)
    return h, n


# EVOLVE-BLOCK-END


def run_setup_a():
    """Fixed entry point. Normalizes and reports the final construction."""
    h, n = construct_heights()
    w = 2.0 / n
    h = np.clip(h, 0.0, 1.0)
    total = np.sum(h) * w
    if total > 1e-9:
        h = np.clip(h / total, 0.0, 1.0)
    c_max = _worst_case_overlap(h)
    return h, c_max, n
