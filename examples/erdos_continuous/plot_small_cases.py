"""
Brute-force the minimal overlap for every small case, then draw it in 3D.

For a universe {1..N} and a set A of size s (B is everything left over, so
|B| = N - s):
    overlap at shift k  =  number of pairs (a, b), a in A, b in B, with a - b = k
    score of a split    =  the WORST overlap over all shifts k
    z(N, s)             =  the best (smallest) score over every possible choice of A

This is the Erdos minimum overlap problem with the "equal halves" rule
relaxed, so odd universe sizes and unequal sets are allowed.
"""
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

MAX_N = 10


def worst_overlap(A, B):
    counts = {}
    for a in A:
        for b in B:
            k = a - b
            counts[k] = counts.get(k, 0) + 1
    return max(counts.values())


def minimal_overlap(N, s):
    universe = range(1, N + 1)
    best = None
    for A in combinations(universe, s):
        B = [x for x in universe if x not in A]
        score = worst_overlap(A, B)
        if best is None or score < best:
            best = score
    return best


# ---- compute ----
results = {}
for N in range(2, MAX_N + 1):
    for s in range(1, N):  # s=0 or s=N leaves one team empty -> overlap 0, uninteresting
        results[(N, s)] = minimal_overlap(N, s)

print("rows = universe size N, columns = size of set A (s)")
print("N\\s " + " ".join(f"{s:>2}" for s in range(1, MAX_N)))
for N in range(2, MAX_N + 1):
    row = [f"{results[(N, s)]:>2}" if (N, s) in results else " ." for s in range(1, MAX_N)]
    print(f"{N:>3} " + " ".join(row))

# ---- draw ----
xs = np.array([k[0] for k in results])
ys = np.array([k[1] for k in results])
zs = np.array([results[k] for k in results])

norm = Normalize(vmin=0, vmax=zs.max())
colors = cm.Blues(0.25 + 0.7 * norm(zs))  # one hue, light -> dark, never pure white

fig = plt.figure(figsize=(11, 8.5))
ax = fig.add_subplot(111, projection="3d")
ax.bar3d(xs - 0.4, ys - 0.4, np.zeros_like(zs), 0.8, 0.8, zs,
         color=colors, edgecolor="white", linewidth=0.4, shade=True)

ax.set_xlabel("Universe size N  (numbers 1..N)", labelpad=12)
ax.set_ylabel("Size of set A  (the other team gets the rest)", labelpad=12)
ax.set_zlabel("Minimal overlap", labelpad=8)
ax.set_xticks(range(2, MAX_N + 1))
ax.set_yticks(range(1, MAX_N))
ax.set_zticks(range(0, int(zs.max()) + 1))
ax.set_title("Minimal overlap for every universe size up to 10", pad=18, fontsize=14)
ax.view_init(elev=47, azim=-52)

plt.tight_layout()
plt.savefig("small_cases_3d.png", dpi=160)
print("Saved small_cases_3d.png")
