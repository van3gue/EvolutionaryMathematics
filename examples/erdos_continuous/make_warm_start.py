"""
Generates a solid (but self-derived, NOT the literal literature record)
warm-start construction at n=600 steps, for Setup A to start from.

This is NOT Haugland's/AlphaEvolve's actual published heights - those
weren't recoverable from the source PDFs. This is our own multi-restart
local optimization, used as a reasonable stand-in starting point.
"""

import time
import numpy as np
from scipy.optimize import minimize
from common import worst_case_overlap, validate_heights, clip_and_normalize

n = 600
w = 2.0 / n
rng = np.random.default_rng(42)

best_c = None
best_h = None
start = time.time()

for trial in range(8):
    h0 = clip_and_normalize(rng.uniform(0.0, 1.0, size=n))

    def obj(h_free):
        return worst_case_overlap(np.clip(h_free, 0.0, 1.0))

    cons = [{"type": "eq", "fun": lambda h_free: np.sum(h_free) * w - 1.0}]
    bounds = [(0.0, 1.0)] * n

    res = minimize(
        obj, h0, method="SLSQP", bounds=bounds, constraints=cons,
        options={"maxiter": 300, "ftol": 1e-12},
    )
    h_final = clip_and_normalize(res.x)
    c = worst_case_overlap(h_final)
    ok, msg = validate_heights(h_final)
    print(f"Trial {trial}: C_max={c:.6f}, valid={ok}, elapsed={time.time()-start:.1f}s")
    if ok and (best_c is None or c < best_c):
        best_c = c
        best_h = h_final

print(f"\nBest warm-start C_max at n={n}: {best_c:.6f}")
np.save("warm_start_n600.npy", best_h)
print("Saved to warm_start_n600.npy")
