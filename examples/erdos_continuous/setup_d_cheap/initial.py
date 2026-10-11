"""
Setup D: Multi-resolution ladder (single rung).

What's evolved: a heights array + polisher, same spirit as Setup A, but
this file represents ONE rung of the coarse-to-fine ladder (n=600 -> 2048
-> 8192 -> 16384). Which rung is active is controlled by two environment
variables, set by the orchestration script (run_ladder.py) that chains
four separate ShinkaEvolve runs together, interpolating the winner of
each rung up to seed the next:

    ERDOS_D_N_STEPS      - resolution for this rung (default 600)
    ERDOS_D_WARM_START    - path to a .npy warm-start file for this rung
                            (default: falls back to the shared n=600 seed)

Self-contained (no importing common.py - see setup_a/initial.py for why).
"""

import os
import numpy as np
from scipy.optimize import minimize
from scipy.signal import correlate


def _worst_case_overlap(h: np.ndarray) -> float:
    n = h.shape[0]
    w = 2.0 / n
    g = 1.0 - h
    corr = correlate(h, g, mode="full", method="fft")
    lags = np.arange(-(n - 1), n)
    mask = lags != 0
    return float(np.max(corr[mask]) * w)


def _project(h, n):
    w = 2.0 / n
    h = np.clip(h, 0.0, 1.0)
    total = np.sum(h) * w
    if total > 1e-9:
        h = np.clip(h * (1.0 / total), 0.0, 1.0)
    return h


def _rung_config():
    n = int(os.environ.get("ERDOS_D_N_STEPS", "600"))
    warm_start = os.environ.get("ERDOS_D_WARM_START", None)
    return n, warm_start

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
"""
Evolve this part: given this rung's warm start, refine it. Since the
resolution changes rung to rung, prefer approaches that are robust to n
rather than hardcoding assumptions tied to one specific resolution.
"""


def load_starting_heights(n, warm_start_path):
    if warm_start_path:
        # An explicit per-rung warm start was requested: fail loudly if it is
        # missing rather than silently starting from something worse.
        if not os.path.exists(warm_start_path):
            raise FileNotFoundError(f"ERDOS_D_WARM_START not found: {warm_start_path}")
        h0 = np.load(warm_start_path)
    else:
        h0 = _warm_start_n600()

    if len(h0) == n:
        return h0
    # Interpolate to this rung's resolution.
    old_x = (np.arange(len(h0)) + 0.5) * (2.0 / len(h0))
    new_x = (np.arange(n) + 0.5) * (2.0 / n)
    return np.clip(np.interp(new_x, old_x, h0), 0.0, 1.0)


def polish(h0, n):
    w = 2.0 / n

    def objective(h_free):
        return _worst_case_overlap(np.clip(h_free, 0.0, 1.0))

    cons = [{"type": "eq", "fun": lambda h_free: np.sum(h_free) * w - 1.0}]
    bounds = [(0.0, 1.0)] * n

    res = minimize(
        objective, h0, method="SLSQP", bounds=bounds, constraints=cons,
        options={"maxiter": 100, "ftol": 1e-12},
    )
    return np.clip(res.x, 0.0, 1.0)


def construct_heights():
    n, warm_start_path = _rung_config()
    h0 = load_starting_heights(n, warm_start_path)
    h = polish(h0, n)
    return h, n


# EVOLVE-BLOCK-END


def run_setup_d():
    h, n = construct_heights()
    h = _project(np.asarray(h, dtype=float), n)
    c_max = _worst_case_overlap(h)
    return h, c_max, n
