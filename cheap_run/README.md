# cheap_run: the cost-reduced experiments

Everything needed to run the four Erdős setups at much lower cost. Run it on
this laptop (or any Windows machine, see "Another computer" below).

> **Running on your own computer does not remove the API cost.** The credit
> pays for the AI model's answers, which come from Anthropic's servers. This
> folder makes each run *cheaper*; it does not make it free.

## Quick start (this laptop)

Open PowerShell in the repo folder (`C:\ShinkaEvolve`):

```powershell
# 0. Make sure no old runs are still retrying in the background
.\cheap_run\stop_cheap.ps1                       # only touches *_cheap runs

# 1. Run the two best-value setups (A and D, at most $4 combined)
.\cheap_run\run_cheap.ps1

# 2. Check progress, score and spend whenever you like
.venv\Scripts\python.exe cheap_run\status.py

# 3. Stop everything (results are kept)
.\cheap_run\stop_cheap.ps1
```

Want more setups? `-Setups A,B,D` adds B; `-Setups A,B,C,D` runs all four.

## What it costs

| Setup | What it tests | Spending cap | In the default run? |
|---|---|---|---|
| **A** | Refine the best known solution | $2.50 | yes |
| **D** | Resolution ladder, rung 1 | $1.50 | yes |
| **B** | Evolve the optimiser | $2.00 | no (`-Setups A,B,D`) |
| **C** | Search over structures | $1.00 | no (made no progress in the first runs) |

The default run (A + D) cannot spend more than about **$4**; all four cannot
spend more than about **$7**. Spend can slightly exceed a cap, because the cap
is checked after each call. Add credit in that range at
console.anthropic.com -> Plans & Billing.

## What was changed to make it cheaper

Measured from the first run set's token counts:

| Change | Why | Effect |
|---|---|---|
| Warm start is read from a file (path in `ERDOS_WARM_START`), not pasted into the program | The pasted copy was about half of every prompt and was repeated for each example program shown, an estimated 11-18% of proposal cost | Seed files shrank from ~10.5k to ~4k characters; scores identical (A 0.382561, B 0.384200, D 0.382714) |
| Opus 5 removed (B: replaced by Haiku 4.5; C: dropped) | Opus cost ~$0.47 per call vs ~$0.14 for Sonnet; 3 calls were ~1/3 of proposal spend | B is now a Sonnet + Haiku ensemble |
| Lower spending caps | So a small top-up is enough | see table above |

**Not changed on purpose:** the output-token cap (16,384). Most of the cost is
model output, but lowering the cap risks truncating answers mid-reasoning,
which would waste the money already spent. That's untested, so it's left alone.

## Safety features

- **Preflight:** `run_cheap.ps1` makes one tiny request first and **refuses to
  start** if your key is invalid or your credit is empty. Before this existed,
  a run with no credit looked healthy while doing nothing.
- **Loud failure:** if the warm-start file can't be found, the program raises
  an error instead of silently starting from a worse guess (the bug that
  invalidated the first runs).
- **`status.py` warns** when a run is stalled by API rejections.

## Comparability warning (for the paper)

This is a **second run set** with changed models (B, C) and changed prompts
(A, B, D). Its results are **not directly comparable** with the first set; the
paper should report it as a separate, cost-reduced configuration, and not mix
the two. Each setup still gets one seed, so differences remain descriptive.

## Files

| File | Purpose |
|---|---|
| `run_cheap.ps1` | Launch setups (default A and D) |
| `status.py` | Progress, best score, spend, health. Read-only |
| `stop_cheap.ps1` | Stop `*_cheap` runs only |
| `preflight.py` | Key/credit check used by the launcher |
| `.env.example` | Template for the two keys |

The configs live where ShinkaEvolve expects them:
`shinka/configs/variant/erdos_setup_{a,b,c,d}_cheap.yaml`,
`shinka/configs/task/erdos_setup_{a,b,d}_cheap.yaml`, and seeds in
`examples/erdos_continuous/setup_{a,b,d}_cheap/initial.py`.

## Another computer (Windows)

Needs Python 3.12 and git, then:

```powershell
git clone https://github.com/van3gue/EvolutionaryMathematics.git
cd EvolutionaryMathematics
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
copy cheap_run\.env.example .env      # then put your two keys in .env
.\cheap_run\run_cheap.ps1
```

Keep the repository's folder structure intact: the launcher finds everything
relative to itself.
