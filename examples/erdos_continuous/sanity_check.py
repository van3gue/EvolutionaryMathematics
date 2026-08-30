"""
Standalone sanity check for common.py, run BEFORE wiring anything into
ShinkaEvolve or spending any API budget.

Test 1: hand-derivable case (n=2, h=[1,0]) - checked by hand, should give
        C_max exactly 1.0.
Test 2: does local optimization from a random start actually converge
        toward the known ballpark (~0.38-0.42), or is something about the
        objective/constraints wrong at a fundamental level?
"""

import numpy as np
from scipy.optimize import minimize
from common import worst_case_overlap, validate_heights, clip_and_normalize

# --- Test 1: hand-derived case ---
h = np.array([1.0, 0.0])
ok, msg = validate_heights(h)
c = worst_case_overlap(h)
print(f"Test 1 (n=2, h=[1,0]): valid={ok} ({msg}), C_max={c:.6f} (expected 1.000000)")
assert ok and abs(c - 1.0) < 1e-9, "Hand-derived sanity check FAILED"

# --- Test 2: does optimization approach the known ballpark? ---
n = 64
w = 2.0 / n
rng = np.random.default_rng(0)


def objective(h_free):
    h = clip_and_normalize(h_free)
    return worst_case_overlap(h)


# A few random restarts + local polish with SLSQP-style bounded optimization.
best_c = None
best_h = None
for trial in range(5):
    h0 = rng.uniform(0.0, 1.0, size=n)
    h0 = clip_and_normalize(h0)

    def obj(h_free):
        h = np.clip(h_free, 0.0, 1.0)
        return worst_case_overlap(h)

    cons = [{"type": "eq", "fun": lambda h_free: np.sum(h_free) * w - 1.0}]
    bounds = [(0.0, 1.0)] * n

    res = minimize(
        obj,
        h0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 200, "ftol": 1e-10},
    )
    c = obj(res.x)
    ok, msg = validate_heights(np.clip(res.x, 0, 1))
    print(f"Trial {trial}: C_max={c:.6f}, valid={ok} ({msg})")
    if best_c is None or c < best_c:
        best_c = c
        best_h = res.x

print(f"\nBest C_max found via quick local optimization (n={n}): {best_c:.6f}")
print("Known published bound (n up to 8192, years of refinement): ~0.380871")
print(
    "If best_c lands roughly in the 0.38-0.45 range, the objective/constraints "
    "are behaving sensibly. If it's near 0 or near 1, something is wrong."
)
