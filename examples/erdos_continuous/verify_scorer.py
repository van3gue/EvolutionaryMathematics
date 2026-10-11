"""
Independent verification of the Erdos continuous-overlap scorer (common.py).

Run:  python examples/erdos_continuous/verify_scorer.py

Every claim made about the scorer in PAPER_NOTES.md is checked here, so the
notes can be regenerated and trusted. Checks:

  1. Hand-computed cases (tiny n, worked on paper).
  2. FFT scorer vs. a plain-loop implementation that shares no code with it.
  3. Piecewise-linearity: the overlap at a NON-grid shift equals the linear
     interpolation of its two neighbouring grid shifts (so the max over all
     real shifts is attained on the grid).
  4. Refinement invariance: re-expressing the same function on a finer grid
     does not change the score.
  5. Shift-zero caveat: the scorer excludes lag 0, but by continuity the
     supremum over real k != 0 approaches C(0). We report C(0) next to the
     score so it is visible whether this could ever matter.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from examples.erdos_continuous.common import (  # noqa: E402
    clip_and_normalize,
    validate_heights,
    worst_case_overlap,
)

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def grid_overlaps(h):
    """Plain-loop overlap C_j = w * sum_i h[i]*(1-h[i+j]) for every lag j."""
    n = len(h)
    w = 2.0 / n
    g = 1.0 - h
    out = {}
    for j in range(-(n - 1), n):
        if j >= 0:
            out[j] = float(np.dot(h[: n - j], g[j:]) * w)
        else:
            out[j] = float(np.dot(h[-j:], g[: n + j]) * w)
    return out


def brute_force_max(h):
    c = grid_overlaps(h)
    return max(v for j, v in c.items() if j != 0)


# ---- 1. hand-computed cases -------------------------------------------------
# f = 1 on [0,1), 0 on [1,2): shifting right by 1 puts all of A on all of B -> 1.
check("hand case [1,0] -> 1.0", abs(worst_case_overlap(np.array([1.0, 0.0])) - 1.0) < 1e-12)
check("hand case [0,1] -> 1.0", abs(worst_case_overlap(np.array([0.0, 1.0])) - 1.0) < 1e-12)
# n=4, h=[1,0,0,1], w=0.5: worked by hand, worst shifts (+-1, +-2) each give 1*w = 0.5.
check("hand case [1,0,0,1] -> 0.5", abs(worst_case_overlap(np.array([1.0, 0.0, 0.0, 1.0])) - 0.5) < 1e-12)

# ---- 2. FFT vs independent loops --------------------------------------------
warm = np.load(os.path.join(HERE, "warm_start_n600.npy")).astype(np.float64)
ok, msg = validate_heights(warm)
check("warm start is a legal step function", ok, msg)

d = abs(worst_case_overlap(warm) - brute_force_max(warm))
check("FFT == plain loops on warm start (n=600)", d < 1e-12, f"|diff| = {d:.2e}")

rng = np.random.default_rng(0)
worst = 0.0
for _ in range(25):
    n = int(rng.integers(5, 120))
    h = clip_and_normalize(rng.random(n))
    worst = max(worst, abs(worst_case_overlap(h) - brute_force_max(h)))
check("FFT == plain loops on 25 random arrays", worst < 1e-12, f"max |diff| = {worst:.2e}")

# ---- 3. piecewise-linearity at non-grid shifts ------------------------------
r = 4  # split every step into r sub-steps -> shifts that are multiples of w/r
fine = np.repeat(warm, r)
c_coarse = grid_overlaps(warm)
c_fine = grid_overlaps(fine)  # lag m on the fine grid = shift m*(w/r)
max_err = 0.0
for j in range(-(len(warm) - 2), len(warm) - 1):
    for t in range(1, r):
        theta = t / r
        interp = (1 - theta) * c_coarse[j] + theta * c_coarse[j + 1]
        max_err = max(max_err, abs(c_fine[r * j + t] - interp))
check(
    "overlap at fractional shifts == linear interpolation of grid shifts",
    max_err < 1e-12,
    f"max |error| over all shifts/fractions = {max_err:.2e}",
)

# ---- 4. refinement invariance ------------------------------------------------
for name, arr in [("warm start", warm)] + [
    (f"random #{i}", clip_and_normalize(rng.random(40))) for i in range(3)
]:
    coarse = worst_case_overlap(arr)
    refined = worst_case_overlap(np.repeat(arr, 7))
    check(f"refine x7 leaves score unchanged ({name})", abs(coarse - refined) < 1e-9,
          f"coarse={coarse:.8f} refined={refined:.8f}")

# ---- 5. shift-zero caveat -----------------------------------------------------
n = len(warm)
c0 = float(np.sum(warm * (1 - warm)) * (2.0 / n))
cmax = worst_case_overlap(warm)
check("C(0) is below the reported score on the warm start (lag-0 exclusion is harmless here)",
      c0 <= cmax, f"C(0) = {c0:.6f}  <  C_max = {cmax:.6f}")

print()
print(f"warm start score C_max = {cmax:.6f}")
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
print("All checks passed.")
