# Erdős minimum overlap: controlled follow-up experiments

**Status: planning and implementation scaffold only. No experiments or tests were run when this package was created.** No result in this folder is evidence that a method works. This is **not** a byte-for-byte reproduction of the paper's ShinkaEvolve implementation.

## 1. Research question

Can LLM-guided optimization find better feasible step functions for the continuous Erdős minimum overlap problem than a comparably budgeted conventional optimizer? Among four LLM-guided avenues, which appears most promising *under a common scoring protocol*?

For heights `f[0],...,f[n-1]` on `[0,2]` of common bin width `2/n`:

- `0 <= f_i <= 1`, and `sum(f_i) = n/2`, i.e. integral of `f` is 1.
- Set `g = 1-f` inside `[0,2]` and 0 outside.
- For shift `k`, let `C_f(k) = integral f(x)g(x+k) dx`.
- Minimize `C_max(f) = max_k C_f(k)`; **lower is better**.
- The grid-aligned evaluator uses `np.correlate(f,1-f,'full') * 2/n` and includes **shift 0**. Because the cross-correlation of equal-width step functions is piecewise linear between grid-aligned shifts, taking the maximum over every aligned shift yields the continuous maximum.
- Validate the mathematical score with independent loop-based evaluation; exact projection to the feasibility constraints is performed before evaluation.

**Published comparison in the draft:** a published upper bound of `0.380871`, and a best reported experimental result of `0.38175`, starting from `0.38256`. These values are historical context **not** a baseline reproduced by this package.

## 2. Six planned setups: four AI avenues + two controls

| Setup | Treatment | Proposed algorithm in this scaffold | Main hypothesis |
|:---:|---|---|---|
| **A** | AI step-height proposal + polish | LLM proposes frequency/phase/amplitude for a height perturbation; optional SLSQP polish | Modifying heights and polishing improves a fixed starting point |
| **B** | AI optimizer proposal | LLM proposes an **optimizer recipe** (SLSQP/Powell, restart count, sigma, maxiter) and runner executes it | AI can select better optimization procedures |
| **C** | AI parameterization proposal | LLM proposes a truncated Fourier parameterization and perturbation | Changing the representation improves search |
| **D** | AI multiresolution proposal | LLM selects one level of a coarse-to-fine ladder, perturbation and polish settings | Resolution changes can improve a search trajectory |
| **E** | Non-AI random restarts | Fixed random-restart search with conventional SLSQP | Baseline: does AI outperform a simple restart strategy? |
| **F** | Non-AI warm-start polish | A single SLSQP polish from the identical seed | Baseline: how much improvement comes from conventional polishing alone? |

**Important fidelity caveat.** A–D are lightweight *proxies* for the four conceptual avenues in the draft, not the actual original ShinkaEvolve island/archive/bandit evolution. In particular B evolves a **declarative optimizer recipe**, not arbitrary executable optimizer source code. D currently chooses a rung each proposal and resamples to the final resolution; it does **not** implement a full persistent four-rung pipeline. To claim reproduction of the paper, replace these proxy methods with wrappers around the actual `examples/erdos_continuous` implementation and original seeds. These proxy results should be labeled a new study.

## 3. Priority and research design

**Priority 1 — verify the scorer.** Run the provided unit tests *before* any scientific run. This will be a future action; nothing has been executed here. Review the paper's lag-zero and normalization caveats before comparing scores.

**Priority 2 — establish the non-AI control.** First run F and E with multiple seeds at the same final resolution as A–D. Record objective evaluations, wall-clock time and final score. This tells you whether a conventional method already matches the benefits attributed to AI.

**Priority 3 — compare A–D** starting with the **exact same initial vector for each seed and resolution**. Use at least five seeds as an exploratory target (more for a publishable claim), the same evaluation cap, same final scorer, and comparable permitted optimizer operations. Report medians, all individual run results, and variability.

**Priority 4 — cost and robustness.** Record model and token counts. Use a comparable monetary ceiling or report performance as a function of cumulative spend. Also report CPU time and objective function evaluations: equal objective evaluations are **not** equal wall time or equal dollar cost. A dollar cap is only estimated when token prices are provided; overshoot is possible on the last call. For rigorous equal-budget claims, add hard infrastructure tracking and matching wall-time/compute cost outside this scaffold.

**Priority 5 — ablations.** Compare AI with no-polish variants and a matched conventional optimizer, preserving the same starting heights. If A beats F, check whether it comes from AI's perturbations rather than additional conventional optimizer evaluations.

### Pre-registered comparisons

1. **Primary:** A vs F and A vs E (same seeds, final resolution, scoring, predeclared budget).
2. **Secondary:** B vs F; C vs F; D vs E, with explicit matching caveats.
3. **Descriptive:** distributions of all six final scores, improvements from seed, budget-consumption curves, cost per feasible improvement.
4. **Never assert statistical significance** from five exploratory seeds or from unbalanced runs. Use paired seed-by-seed differences and confidence intervals only after a suitable larger design is chosen.

### Critical confounders to address in the *final* research implementation

- A–D may call polish more often or use different optimizers than controls; reserve an equivalent compute budget and consider a **no-LLM recipe-search** baseline for B.
- D should eventually run a full multistage ladder, not merely change a level each iteration.
- In the original draft C used a structurally different and much weaker starting seed; the new scaffold deliberately controls the seed for all methods.
- Historical paper runs were stopped by credit exhaustion and their stored-program spending excludes some costs. Do not reuse the historical ranking as an equal-budget ranking.
- For baseline reproduction, load the actual published starting heights; the included symmetric seed is **synthetic and exploratory**, not the original `n=600` warm start.
- Logs must record error/crash/invalid-proposal rates, not just successful samples. The runner writes rejected JSON proposals and error statuses.

## 4. Repository map

```text
.
├── README.md                          This plan
├── pyproject.toml                     Dependencies and CLI entry point
├── .env.example                       Optional LLM credentials and model
├── configs/
│   ├── pilot.json                     64 bins, exploratory settings
│   └── paper_resolution.json          600 bins, not a reproduction
├── src/erdos_experiments/
│   ├── cli.py                          Experiment runner; saves scores and heights
│   ├── scorer.py                       Exact grid-lag scorer + independent reference
│   ├── optimizers.py                   SLSQP / Powell + random-restart controls
│   ├── llm.py                          OpenAI-compatible JSON proposal client
│   ├── strategies.py                   A, B, C, D, E, F search definitions
│   └── report.py                       Descriptive tables and chart after runs
├── tests/test_scorer.py               Unexecuted scorer tests
└── outputs/                           Empty until deliberate execution
```

## 5. Setup — commands for later, **not executed**

Create an isolated Python environment and install dependencies:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -e .
pip install pytest
```

For AI setups A–D, provide an API credential and model accessible to your account; the client uses the OpenAI-compatible `/chat/completions` endpoint and parses plain JSON content. The endpoint/model must actually support the request; some providers have different reasoning or JSON interfaces and may need an adapter.

```bash
# Copy .env.example to .env and load its values using your preferred shell.
# The code reads environment variables, not .env files automatically.
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_MODEL="A_MODEL_YOU_CAN_ACCESS"
# Optionally: export OPENAI_BASE_URL="https://api.openai.com/v1"
```

**Before initiating paid calls**, configure `model_input_usd_per_million` and `model_output_usd_per_million` from your provider's **current** published pricing; set `max_ai_spend_usd` to your intended per-run limit. API calls incur actual provider charges. The provided configs set prices and money cap to `null`, so they do **not** enforce a dollar ceiling by default; they still cap per-run AI calls.

### Commands for later execution

```bash
# Preflight preview, no files/no optimization/no API requests:
erdos-exp --config configs/pilot.json --setups A B C D E F --dry-run

# Unit test (future manual action):
pytest -q

# Run only the two non-AI controls, 64-bin pilot (future manual action):
erdos-exp --config configs/pilot.json --setups E F

# AI experiments, 64-bin pilot (future manual action; paid API requests):
erdos-exp --config configs/pilot.json --setups A B C D

# Full resolution (future manual action; computationally expensive):
erdos-exp --config configs/paper_resolution.json --setups A B C D E F

# Summarize results ONLY after runs (future manual action):
python -m erdos_experiments.report --results outputs/pilot/results.csv --output outputs/pilot/report
```

**No command above has been run.** The `report` program does no experiment launching. Repeating the same experiment command **appends duplicate rows**; use fresh output directories per campaign.

## 6. Expected outputs (generated only after execution)

- `results.csv`: setup, seed, resolution, initial/final score, improvement, distance to published benchmark, evaluation count, AI call count, optional estimated token cost, elapsed wall time, status, error.
- `{setup}_seed{seed}_start.npy` and `_best.npy`: initial and final step heights.
- `{setup}_seed{seed}_proposals.json`: model responses, per-proposal scores, rejection reasons, token usage when returned by provider.
- `summary.md` and `scores.png`: descriptive cross-setup statistics and a figure (created by `report.py`).

## 7. Scientific acceptance checks

- [ ] Scorer agrees with independent loop evaluation for constant, binary and random candidate arrays.
- [ ] Shift zero is included, and candidates have exactly the intended integral and bounds.
- [ ] The same seed maps to identical starting heights across all setups.
- [ ] Failures are logged and counted; experiments that fail cannot be silently accepted as successful.
- [ ] All A–F results are evaluated by the **same exact** final scorer.
- [ ] Budgets/costs are logged and comparisons honestly account for unequal spending.
- [ ] The final manuscript distinguishes synthetic scaffold results from original ShinkaEvolve program-search results.
- [ ] An adequately controlled pure classical optimizer and (ideally) random-recipe selection are included.
- [ ] Results are presented with seed-level variability rather than one best run alone.

## 8. What to replace in a six-page revision

| Existing material | Replacement at similar length |
|---|---|
| Existing unbalanced results table | Paired-seed A–F results, mean/median and objective evaluation count |
| Existing cost plot | Best-score-vs-budget plot with complete actual API spend |
| One verification table detail | Short verification statement and link to executable tests |
| Extended infrastructure incident narrative | One-paragraph note on initial invalid runs, with full incident in repository |
| Preliminary conclusion | Answer to AI vs classical baseline, with uncertainty and limitations |

Retain the exact scorer justification and provenance of the starting solution. **Do not change the reported result or make claims until new experiments have actually been run and reviewed.**

## 9. Follow-on engineering improvements

The package is a research planning starting point, **not production-grade experiment infrastructure**. The scorer is exact at the specified uniform resolution but comparison to a continuous published construction still requires validating discretization and reconstructing the original warm start. Replace the lightweight LLM API adapter with whichever model/provider and ShinkaEvolve implementation you actually intend to study; add checkpointing, full program archive tracking, a persistent D ladder, a complete cost ledger, and independent reproducibility verification before claiming publication-quality conclusions. The default SLSQP 600-dimensional runs may be very expensive because numerical gradients can require many function evaluations.
