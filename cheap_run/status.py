"""
Progress / cost / health check for the cost-reduced runs. Read-only, so it is
safe to run while experiments are live.

    .venv\\Scripts\\python.exe cheap_run\\status.py
"""
import glob
import json
import os
import sqlite3
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD = 0.380871
CAPS = {"A": 2.5, "B": 2.0, "C": 1.0, "D": 1.5}
STALL_MIN = 15  # minutes without a new program before we warn


def newest_run(letter):
    runs = glob.glob(os.path.join(REPO, "results", f"erdos_setup_{letter.lower()}", "*_cheap"))
    return max(runs, key=os.path.getmtime) if runs else None


def recent_rejections(letter):
    """Count credit/auth rejections in the last ~200 lines of the newest log."""
    logs = sorted(glob.glob(os.path.join(REPO, "cheap_run", "logs", f"erdos_setup_{letter.lower()}_cheap_*.log*")),
                  key=os.path.getmtime)
    bad = ok = 0
    for path in logs[-2:]:
        try:
            lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()[-200:]
        except OSError:
            continue
        for ln in lines:
            if "v1/messages" in ln and "HTTP/1.1 200" in ln:
                ok += 1
            elif "v1/messages" in ln and ("HTTP/1.1 400" in ln or "HTTP/1.1 401" in ln):
                bad += 1
    return ok, bad


print(f"{'setup':<6}{'programs':>9}{'valid':>7}{'seed':>10}{'best':>10}{'gain':>11}{'gap->record':>13}{'spend':>9}{'cap':>6}  state")
for letter in "ABCD":
    run = newest_run(letter)
    if not run:
        continue
    dbs = glob.glob(os.path.join(run, "*.sqlite"))
    if not dbs:
        print(f"{letter:<6} (starting - no database yet)")
        continue
    con = sqlite3.connect(f"file:{dbs[0]}?mode=ro", uri=True)
    rows = con.execute("select timestamp, correct, combined_score, metadata from programs order by timestamp").fetchall()
    con.close()
    cost, best, seed, valid, last_ts, n = 0.0, None, None, 0, 0.0, 0
    for ts, ok, score, meta in rows:
        m = json.loads(meta) if meta else {}
        if m.get("_is_island_copy"):
            continue
        n += 1
        last_ts = max(last_ts, ts)
        cost += sum(float(m.get(k) or 0) for k in ("api_costs", "embed_cost", "novelty_cost", "meta_cost"))
        if ok:
            valid += 1
            v = -float(score)
            seed = v if seed is None else seed
            best = v if best is None else min(best, v)
    age = (time.time() - last_ts) / 60 if last_ts else 0
    ok200, bad = recent_rejections(letter)
    if bad and not ok200:
        state = "STALLED: API is rejecting calls (credit/key) - run preflight.py"
    elif age > STALL_MIN:
        state = f"quiet for {age:.0f} min (long evaluation, or stalled)"
    else:
        state = "running"
    if best is None:
        print(f"{letter:<6}{n:>9}{valid:>7}  {state}")
        continue
    print(f"{letter:<6}{n:>9}{valid:>7}{seed:>10.6f}{best:>10.6f}{seed - best:>+11.6f}{best - RECORD:>+13.6f}"
          f"{cost:>9.2f}{CAPS[letter]:>6.1f}  {state}")

print("\nspend = stored programs only (failed proposals are not counted, so real spend is somewhat higher)")
print("lower best is better; record to beat = 0.380871")
