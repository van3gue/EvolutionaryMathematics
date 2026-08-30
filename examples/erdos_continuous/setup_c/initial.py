"""
Setup C: Structure search.

What's evolved: the PARAMETRIZATION itself - how the step heights are
generated from a compact set of parameters (basis functions, symmetry
assumptions, etc.) - not just direct numerical tweaks to a heights array.
The number of steps n is also free to choose (256-4096), evolved along
with everything else. Goal: land in a genuinely different structural
"basin" rather than just polishing the same shape everyone else starts
from.

Starting point: NOTE - we do not have Haugland's or AlphaEvolve's actual
published heights (couldn't recover the raw numbers from the source
papers), so this seed uses a distinct, self-designed Fourier-basis
parametrization instead of a raw array, which is at least structurally
different from Setup A/B's direct-array approach.

Self-contained (no importing common.py - see setup_a/initial.py for why).
"""

import numpy as np
from scipy.signal import correlate


def _worst_case_overlap(h: np.ndarray):
    n = h.shape[0]
    w = 2.0 / n
    g = 1.0 - h
    corr = correlate(h, g, mode="full", method="fft")
    lags = np.arange(-(n - 1), n)
    mask = lags != 0
    c_max = float(np.max(corr[mask]) * w)
    # Secondary diagnostic: how many shifts are "tied" near the max -
    # useful context for understanding why local optimizers struggle here.
    near_max = corr[mask] >= (np.max(corr[mask]) - 1e-9 * max(1.0, np.max(corr[mask])))
    num_tied = int(np.sum(near_max))
    return c_max, num_tied


def _project(h, n):
    w = 2.0 / n
    h = np.clip(h, 0.0, 1.0)
    total = np.sum(h) * w
    if total > 1e-9:
        h = np.clip(h * (1.0 / total), 0.0, 1.0)
    return h


# EVOLVE-BLOCK-START
"""
Evolve this part: how the construction is PARAMETRIZED and built, and
what resolution n it uses. Ideas to try: symmetric or antisymmetric basis
functions (e.g. f(x) and f(2-x) related by a fixed rule), a small number
of Fourier harmonics thresholded into steps, spline control points
converted to a step function, or piecewise-defined regions with a few
free parameters each. The point is to explore genuinely different SHAPES
of solution, not just perturb one array.
"""


def choose_resolution():
    """Pick a step count. Free to evolve within [256, 4096]."""
    return 1024


def build_from_parameters(n):
    """
    Baseline structural seed: a smooth Fourier-harmonic construction,
    symmetrized about the domain midpoint (x=1), then clipped to [0,1]
    and renormalized. Deliberately different in spirit from a raw
    optimized array - evolution should replace the actual basis/logic,
    not just the numeric coefficients.
    """
    w = 2.0 / n
    x = (np.arange(n) + 0.5) * w  # step midpoints on [0, 2]

    # A few harmonics, symmetric about x=1.
    coeffs = [0.5, 0.25, -0.1, 0.05]
    h = np.zeros(n)
    for k, a_k in enumerate(coeffs):
        h += a_k * np.cos(k * np.pi * (x - 1.0))

    h = _project(h, n)
    return h


def construct_heights():
    n = choose_resolution()
    h = build_from_parameters(n)
    return h, n


# EVOLVE-BLOCK-END


def run_setup_c():
    h, n = construct_heights()
    h = _project(np.asarray(h, dtype=float), n)
    c_max, num_tied = _worst_case_overlap(h)
    return h, c_max, n, num_tied
