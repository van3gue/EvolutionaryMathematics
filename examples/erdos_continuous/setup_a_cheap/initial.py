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

# The n=600 warm start is read from a file whose path comes from the
# ERDOS_WARM_START environment variable (set by cheap_run/run_cheap.ps1). It is
# deliberately NOT embedded in this source: an embedded copy is shown to the
# model in every prompt (and again for every inspiration program), which costs
# real money. A missing file fails loudly - never fall back silently to a worse
# start. Kept OUTSIDE the evolve block so mutations cannot break it.
def _warm_start_n600() -> np.ndarray:
    path = os.environ.get("ERDOS_WARM_START", "")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            "ERDOS_WARM_START must point to warm_start_n600.npy "
            f"(got {path!r}). Launch through cheap_run/run_cheap.ps1."
        )
    h0 = np.load(path).astype(np.float64)
    if h0.shape != (600,):
        raise RuntimeError(f"warm start has wrong shape {h0.shape}")
    return h0


# EVOLVE-BLOCK-START
"""Evolve this part: the starting construction and/or the polish routine."""


def load_starting_heights(n):
    """
    Load our best available starting construction. This is a SELF-DERIVED
    warm start (multi-restart local optimization, see make_warm_start.py),
    NOT the literal published record - we could not recover Haugland's or
    AlphaEvolve's actual numeric heights from the source papers.
    """
    h0 = _warm_start_n600()
    if n == len(h0):
        return h0
    # Other resolutions: interpolate the embedded warm start.
    old_x = (np.arange(len(h0)) + 0.5) * (2.0 / len(h0))
    new_x = (np.arange(n) + 0.5) * (2.0 / n)
    return np.clip(np.interp(new_x, old_x, h0), 0.0, 1.0)


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
