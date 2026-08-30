"""
Visualize a step-function construction for the Erdos minimum overlap
problem: the shape of f(x) itself, and the overlap curve C(k) across all
shifts, with the worst-case shift highlighted (that peak IS the score).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import correlate

h = np.load("warm_start_n600.npy")
n = h.shape[0]
w = 2.0 / n
x_edges = np.linspace(0, 2, n + 1)

g = 1.0 - h
corr = correlate(h, g, mode="full", method="fft") * w
lags = np.arange(-(n - 1), n) * w  # convert lag (in steps) to continuous k

mask = lags != 0
c_max = np.max(corr[mask])
k_star = lags[mask][np.argmax(corr[mask])]

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

ax = axes[0]
ax.fill_between(x_edges[:-1], 0, h, step="post", color="#4C72B0", alpha=0.85, label="f(x) — set A's density")
ax.fill_between(x_edges[:-1], h, 1, step="post", color="#DD8452", alpha=0.6, label="1-f(x) — set B's density")
ax.set_xlim(0, 2)
ax.set_ylim(0, 1)
ax.set_xlabel("x")
ax.set_ylabel("density")
ax.set_title(f"The step function construction (n={n} steps)")
ax.legend(loc="upper right")

ax = axes[1]
ax.plot(lags, corr, color="#4C72B0", linewidth=1.5)
ax.axvline(k_star, color="#C44E52", linestyle="--", linewidth=1.5)
ax.scatter([k_star], [c_max], color="#C44E52", zorder=5, s=60)
ax.annotate(
    f"worst shift k={k_star:.3f}\nC_max={c_max:.4f}",
    xy=(k_star, c_max), xytext=(20, -20), textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="#C44E52"),
    fontsize=10, color="#C44E52",
)
ax.axhline(0.380871, color="gray", linestyle=":", linewidth=1, label="published record (0.380871)")
ax.set_xlabel("shift k")
ax.set_ylabel("overlap C(k)")
ax.set_title("Overlap across every possible shift — the peak is the score being minimized")
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("overlap_visualization.png", dpi=150)
print(f"Saved. C_max = {c_max:.6f} at k = {k_star:.4f}")
