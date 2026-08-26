# Propfirm Convex Payoff Analysis

Monte Carlo framework that models a prop-firm evaluation account as a
**first-passage problem**: reach the profit target before the drawdown
barrier hits. The objective is not P&L — it's maximizing **P(pass)** within
a realistic time frame, by choosing the stop width, risk-per-trade, and R:R
geometry that best survives each firm's specific barrier structure, for a
real strategy (`kernel_bos`, a kernel-regression + break-of-structure entry
on NQ) backed by real 1-minute OHLC data.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt        # Windows
# or: .venv/bin/pip install -r requirements.txt        # macOS/Linux

# Open in Jupyter / VS Code and run top to bottom:
"Master Analysis/analysis_notebook_kernel_bos (points-based stop, 3D risk geometry).ipynb"
```

That notebook is the current, actively-maintained entry point. Its own Knobs
cell and section markdowns carry the up-to-date rationale for every setting
(IS/OOS dates, risk grid, patience ceilings, etc.) — treat this README as
the map, the notebook as the live source of truth for current numbers.

## Project structure

```
Rules/                          core library (import target for every notebook)
  Firms.py                      $50K rule configs for 7 eval paths across 5 firms
  Engine.py                     vectorized Monte Carlo engine (trailing/static DD, DLL, consistency, min-days)
  Strategy.py                   SyntheticStrategy (WR/RR) and EmpiricalStrategy (bootstrap real R-ledgers)
  Backtest_points.py            entry-only strategy + OHLC -> R-multiple ledger, FIXED-POINTS stop sizing
  Analysis_points.py            3-D risk-geometry search: (stop points) x (risk%) x (R:R) -> pass rate
  MultiEval.py                  deterministic N-concurrent-account replay, restart-on-breach (see below)
  Funded.py                     funded-stage (post-eval) payout modeling — separate from the eval-stage work above
  Backtest.py                   ATR-stop-sized backtest -- kept because Master Analysis/funded_analysis.ipynb
                                 still imports it; NOT used by the active points-based pipeline
  strategies/                   entry-signal strategies (kernel_bos.py is the one actively used)
    time_window.py              wraps any strategy to only count entries inside a wall-clock window
    superseded/                 earlier/abandoned strategy variants, kept for reference
  superseded/                   README.md explains what's here and why (ATR-era Analysis/Robustness/Run/edge_diagnostic)
  data/                         NQ.ohlcv-1m.parquet (gitignored — see databento/ below to build/refresh it)

Master Analysis/
  analysis_notebook_kernel_bos (points-based stop, 3D risk geometry).ipynb   <- THE live notebook
  funded_analysis.ipynb          funded-stage exploration (Rules/Funded.py + Rules/Backtest.py)

Markdown Rules/                 the actual firm rulebooks (.md) Firms.py's configs were derived from

databento/
  build_continuous.py           DataBento .dbn/.csv.zst -> Rules/data/*.parquet pipeline (see below)
```

## The core idea: risk geometry as a 3-D search

Position size (contracts) is a free variable — any stop width in points can
be sized to the same dollar risk. What actually matters is which trades
win/lose/scratch at a given stop width, which changes with R:R. So the
search is genuinely 3-dimensional:

**`(stop width in points)` × `(risk % per trade)` × `(R:R)` → pass rate**

`Analysis_points.risk_geometry_3d()` builds one `Entries` ledger per stop
width, then Monte Carlo–simulates every (risk%, R:R) cell against a firm's
actual rules (`Engine.simulate()`), using `EmpiricalStrategy`'s block
bootstrap of *whole historical trading days* — so day-to-day trade
clustering (and therefore trailing-DD/DLL/consistency exposure) is real,
not a synthetic constant rate.

**Pass rate alone isn't the objective — speed matters too.** A 98%-pass
combo that only fires 0.4x/day can still take ~9 months to actually reach
in real trading. `MAX_DAYS_TO_PASS` turns "how many trading days would this
realistically take" into a hard constraint on cell selection, using
`SimResult.mean_days_to_pass` straight from the same Monte Carlo (not a
post-hoc estimate).

**The risk-per-trade grid is built as round fractions of the profit
target** (`risk = target / N` for whole N, e.g. N=5 → "5 wins to pass"),
not a flat %-of-balance ramp — so every grid point is directly interpretable
against the firm's actual dollar constraints, not an arbitrary percentage
that happens to produce an odd number of losses-to-bust.

### Why an ATR-based stop was abandoned

The original pipeline (`Backtest.py`/`Analysis.py`, now in `Rules/superseded/`)
sized the stop as a multiple of ATR. That blends "how volatile was the
market when this trade fired" together with "how much room does the stop
give it" — e.g. switching the underlying bars from RTH-only to a full 24h
feed changes ATR (and therefore the stop, and every downstream number) even
though nothing about the strategy's actual point-based risk changed.
Sweeping stop width directly in points sidesteps that entirely.

## In-sample / out-of-sample discipline

Everything that *selects* a geometry or a parameter combo runs on
`bars_is` only. `bars_oos` — an explicit calendar boundary (`IS_END`/
`OOS_START` in the Knobs cell, not a fraction of days, so it doesn't
silently slide as the data file grows) — is used *only* to measure the
already-chosen winner afterward, never to re-optimize. This is the real
check against overfitting/selection bias that Monte Carlo path-averaging
alone can't catch (path-averaging protects against *which order* a fixed
set of historical trades comes in, not against that historical ledger
itself — or the search across many combos that picked it — being a lucky
sample).

**A real lesson learned the hard way this project:** a first attempt at a
short (~2 month) OOS window produced an apparent collapse (46.6% IS → 6.7%
OOS). A monthly edge breakdown showed that specific bad month was not
unique — comparable bad months sat quietly inside the in-sample period the
whole time, just diluted by ~20 other months around them. Widening the OOS
window (to ~3.5 months) recovered a much more stable, still-somewhat-lower
read. **Lesson: a short OOS window can't distinguish "one bad month" from
"the edge died" — check the monthly breakdown before concluding either
way, and don't trust an OOS number from a small trade count on its own.**

**A second, consistent lesson:** the parameter-optimization search (Stage
A/B over `kernel_bos`'s own knobs — `h`/`r`/`x_0`/`bos_lookback`/
`slope_len`/`slope_min`) has overfit on *every grid shape tried so far* —
large in-sample pass rates that collapsed badly out-of-sample. Treat
`BEST_OVERRIDES` as a standing cautionary result, not a recommendation,
until a search demonstrably holds up OOS. The manual-verification section
(5a/5b) checks the active, hand-set `OVERRIDES` (the live TradingView panel
values) against real signals/trades, deliberately *not* `BEST_OVERRIDES`.

## Multi-eval simulation: what happens with several concurrent accounts

`MultiEval.multi_account_replay()` answers a different question than the
Monte Carlo above. `risk_geometry_3d()`/`Engine.simulate()` ask "what's the
probability ONE account passes, averaged over many resampled hypothetical
futures." This asks: "if I actually run `N_ACCOUNTS` funded evals
*concurrently* off the same real signal stream — one trade per account per
day, and the instant one breaches, a fresh account immediately takes its
slot — what would have ACTUALLY happened over real history?"

- **One trade per account per day**, because a single account can only take
  one signal at a time anyway (`one_position` gating) — busy days that
  would otherwise waste extra signals instead spread them across other
  concurrent accounts, and no account stacks two same-day losses into one
  trailing-drawdown hit.
- **Restart-on-breach**: a slot is a concurrency count, not a cap on total
  accounts tried — it can burn through several accounts over the window if
  it keeps breaching (mirrors actually buying another eval and continuing).
  PASS and TIMEOUT do *not* restart a slot; only a rule breach does.
- **`MultiEvalResult.history`** tallies every account any slot ever ran —
  not just the final state of the live slots — so `n_passed`/`n_blown`/
  `n_timeout` reflect the whole window.
- The notebook runs this both on `bars_oos` (the genuine held-out read) and
  on the full sample (bigger attempt counts, but **not** a held-out test —
  it re-includes the very data the geometry/combo was selected on, and
  every full-sample number in the notebook is measurably more optimistic
  than its OOS counterpart as a direct result. Read the two side by side,
  never the full-sample number alone).
- **A finding worth knowing before trusting any single "best" cell**: the
  single-account IS ranking and the multi-eval ranking do not agree, and
  sometimes disagree sharply — a cell that ranks #1 by undiluted
  single-account pass rate can be the *worst* multi-eval performer, because
  `risk_geometry_3d()`'s ranking has no visibility into how a cell tolerates
  only getting a fraction of its signals once diluted across concurrent
  accounts. `MAX_DAYS_TO_PASS` (single-account selection pace) and
  `MULTI_EVAL_MAX_DAYS` (multi-account patience) are deliberately separate
  knobs for exactly this reason.

## Data pipeline

`data/NQ.ohlcv-1m.parquet` (gitignored — not in this repo) is the 1-minute
continuous NQ series everything above runs on. To build or refresh it:

1. Get a DataBento batch download (schema=`ohlcv-1m`, `stype_in=parent`,
   e.g. `symbols=['NQ.FUT']`) — a `.dbn`/`.dbn.zst` file, in its own
   subfolder is fine.
2. Drop it anywhere under `databento/` (gitignored except the script itself).
3. `pip install databento zstandard` (also in `requirements.txt`), then
   `python databento/build_continuous.py`.

It decodes the batch file, splits by root ticker (`NQM6` → `NQ`), and
upserts into `Rules/data/<TICKER>.ohlcv-1m.parquet` — duplicate timestamps
keep the *new* value, so re-running with a corrected/extended download also
patches past bars, not just appends.

## Firm rules

`Markdown Rules/*.md` are the actual firm rulebooks `Rules/Firms.py`'s
configs were derived from (Apex, Lucid, MFFU, Topstep, Tradeify). Where a
rule doc was ambiguous about trailing-vs-static drawdown or hard-vs-soft
daily-loss behavior, `Firms.py` marks the choice `# VERIFY` — these are the
levers that most dominate pass rates, so confirm them against the current
rulebook before trusting a number for a firm you haven't checked recently.

## Open items / known limitations

- Section 4's parameter search (`BEST_OVERRIDES`) has not yet produced a
  combo that survives out-of-sample — see above.
- The multi-eval simulation only restarts a slot on a rule *breach*; whether
  TIMEOUT should also restart, and whether `MULTI_EVAL_MAX_DAYS` should
  differ from its current value, are open design questions, not settled.
- OOS windows throughout are still short relative to the in-sample period —
  every OOS/multi-eval read in this repo should be treated as a limited but
  useful robustness signal, not a fully-powered statistical test.
- `Funded.py` (post-eval, funded-stage payout modeling) has no current
  notebook wiring it into the points-based pipeline — `funded_analysis.ipynb`
  is the old ATR-based exploration of it.
