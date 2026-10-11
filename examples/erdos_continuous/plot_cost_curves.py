"""
Best score vs. cumulative API spend for each Erdos setup.

Different setups had different budget caps ($5/$6/$6/$3), so comparing where
each one ENDED is not a fair comparison. Plotting best-so-far score against
dollars spent puts them on a common axis: at any given spend, who was ahead?

Reads the run databases READ-ONLY, so it is safe to run while runs are live.

Caveat (also stated on the figure): cost is summed over stored programs
(api_costs + embed_cost + novelty_cost + meta_cost). Proposals that failed
before producing a stored program are not in the database, so the true spend
can be somewhat higher than the x-axis shows.
"""
import glob
import json
import os
import sqlite3
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = r"C:\ShinkaEvolve\results"
RUN_STAMP = "2026.10.10194443"  # the run set that used the fixed warm start
RECORD = 0.380871

# Categorical slots 1-4 of the reference palette (validated, light surface).
SETUPS = [
    ("A", "A: refine best-known", "#2a78d6"),
    ("B", "B: evolve optimizer", "#eb6834"),
    ("C", "C: structure search", "#1baf7a"),
    ("D", "D: resolution ladder (rung 1)", "#eda100"),
]


def load_curve(letter):
    folder = glob.glob(os.path.join(RESULTS, f"erdos_setup_{letter.lower()}", f"{RUN_STAMP}_*"))
    if not folder:
        return None
    db = glob.glob(os.path.join(folder[0], "*.sqlite"))[0]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "select timestamp, correct, combined_score, metadata from programs order by timestamp"
    ).fetchall()
    con.close()

    cost, best = 0.0, None
    xs, ys = [], []
    n_valid = 0
    for ts, correct, score, meta in rows:
        m = json.loads(meta) if meta else {}
        if m.get("_is_island_copy"):
            continue
        cost += sum(float(m.get(k) or 0.0) for k in ("api_costs", "embed_cost", "novelty_cost", "meta_cost"))
        if correct:
            n_valid += 1
            c_max = -float(score)
            best = c_max if best is None else min(best, c_max)
        if best is not None:
            xs.append(cost)
            ys.append(best)
    return xs, ys, len(rows), n_valid


fig, ax = plt.subplots(figsize=(9, 5.5))
summary = []
for letter, label, color in SETUPS:
    res = load_curve(letter)
    if res is None:
        continue
    xs, ys, n_prog, n_valid = res
    ax.step(xs, ys, where="post", color=color, linewidth=2, label=label)
    ax.plot(xs[-1], ys[-1], "o", color=color, markersize=7, markeredgecolor="white", markeredgewidth=1.5)
    summary.append((letter, xs[-1], ys[-1], n_prog, n_valid))

ax.axhline(RECORD, color="#52514e", linestyle=":", linewidth=1.2)
ax.text(0.99, RECORD, "published record 0.380871", transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=9, color="#52514e")

ax.set_ylim(0.3800, 0.3900)
ax.set_xlim(left=0)
ax.invert_yaxis()  # lower is better, so "up" reads as "better"
ax.set_xlabel("Cumulative API spend on stored programs ($)")
ax.set_ylabel("Best score so far, C_max  (lower is better)")
ax.set_title("Best score vs. money spent")
ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=9)

c_note = [s for s in summary if s[0] == "C"]
if c_note and c_note[0][2] > 0.3900:
    ax.text(0.01, 0.02, f"Setup C is off-scale at {c_note[0][2]:.4f} (starts furthest away)",
            transform=ax.transAxes, fontsize=9, color="#52514e")

fig.text(0.01, 0.005,
         "Cost counts stored programs only; failed proposals are excluded, so true spend is somewhat higher. "
         "Single run per setup.", fontsize=7.5, color="#52514e", ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost_curves.png")
fig.savefig(out, dpi=160)

print("setup | spend($) | best C_max | programs | valid")
for letter, spend, best, n_prog, n_valid in summary:
    print(f"  {letter}   | {spend:7.3f}  | {best:.6f}  | {n_prog:>3}      | {n_valid}")
print("Saved", out)
