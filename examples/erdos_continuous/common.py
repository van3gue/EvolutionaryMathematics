"""
Shared evaluator core for the continuous (Swinnerton-Dyer) reformulation of
Erdos' minimum overlap problem.

Problem (as stated in the reference spec):
    Minimize, over step functions f on [0, 2] with 0 <= f <= 1 and
    integral(f) = 1, the quantity

        C(f) = max_{k != 0} Integral f(x) * (1 - f(x+k)) dx

    where the integrand is understood to be zero wherever x or x+k falls
    outside [0, 2] (there is no "A" or "B" mass outside the interval).

Discretization:
    f is represented as n step heights h[0..n-1], each covering a width
    w = 2/n on [0, 2]. Because f is piecewise constant, the correlation
    Integral f(x) g(x+k) dx (g = 1 - f) is a *piecewise-linear* function of
    the continuous shift k, with breakpoints exactly at shifts that are
    integer multiples of the step width w. A piecewise-linear function's
    global max over an interval always occurs at one of its breakpoints,
    so checking every integer-grid shift j = 1..n-1 (in both directions)
    is mathematically exact, not an approximation - matching the "no
    subsampling" instruction in the spec.

Score convention: combined_score = -C_max (higher is better, so beating
the published bound of ~0.380871 means combined_score > -0.380871).
"""

from typing import Tuple
import numpy as np
from scipy.signal import correlate


def worst_case_overlap(h: np.ndarray) -> float:
    """
    Compute C_max = max over nonzero grid shifts j of
    w * sum_i h[i] * (1 - h[i+j])  (both shift directions),
    for a step-function height array h on [0, 2] with n = len(h) steps.
    """
    n = h.shape[0]
    w = 2.0 / n
    g = 1.0 - h

    # Cross-correlation of h and g at all integer lags via FFT.
    # correlate(h, g, mode='full')[m] corresponds to lag j = m - (n-1),
    # i.e. sum_i h[i] * g[i + j] over valid i (numpy/scipy convention).
    corr = correlate(h, g, mode="full", method="fft")
    lags = np.arange(-(n - 1), n)

    # Exclude lag 0 (k=0 is not a valid "overlap shift" - it's the trivial
    # self-comparison, not one of the nonzero shifts the problem considers).
    mask = lags != 0
    c_max = float(np.max(corr[mask]) * w)
    return c_max


def validate_heights(h: np.ndarray, tol: float = 1e-6) -> Tuple[bool, str]:
    """Check h is a legal step-function representation of f."""
    if not isinstance(h, np.ndarray):
        return False, f"Expected numpy array, got {type(h)}"
    if h.ndim != 1 or h.shape[0] < 2:
        return False, f"Expected 1D array with >=2 steps, got shape {h.shape}"
    if not np.all(np.isfinite(h)):
        return False, "Non-finite values in heights array."
    if np.any(h < -tol) or np.any(h > 1 + tol):
        return False, f"Heights out of [0,1] range: min={h.min()}, max={h.max()}"

    n = h.shape[0]
    w = 2.0 / n
    integral = float(np.sum(h) * w)
    if abs(integral - 1.0) > 1e-4:
        return False, f"Integral(f) = {integral:.6f}, expected 1.0 (tol 1e-4)."

    return True, "Valid step function."


def clip_and_normalize(h: np.ndarray) -> np.ndarray:
    """Project a heights array onto the feasible set: 0<=h<=1, integral=1."""
    n = h.shape[0]
    w = 2.0 / n
    h = np.clip(h, 0.0, 1.0)
    total = np.sum(h) * w
    if total > 1e-9:
        h = h * (1.0 / total)
        h = np.clip(h, 0.0, 1.0)
    return h
