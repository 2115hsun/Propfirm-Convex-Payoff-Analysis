# Superseded (ATR-based pipeline)

These four modules belonged to the original ATR-stop risk-geometry pipeline
(`k_stop_atr`-sized stops via `Backtest.py`/`Analysis.py`). That approach was
replaced by the **fixed-points stop** pipeline (`Backtest_points.py` /
`Analysis_points.py`, one directory up) — see the main README for why: the
short version is that position size is a free variable, so the ATR multiple
was blending "how volatile was the market" together with "how wide is the
stop" in a way that made stop-width sweeps unreliable.

- `Analysis.py` — 2-D risk%/R:R heatmap + equity-path plotting for an
  ATR-sized `Entries` ledger.
- `Robustness.py` — rolling-window regime-robustness scoring (picks the
  geometry with the highest *floor* across windows, not highest mean).
- `Run.py` — CLI entry point (`python Run.py <firm>`) for the ATR pipeline.
- `edge_diagnostic.py` — quick measured-edge diagnostic script.

**Kept for reference, not wired to run from here.** All four still do bare
`from Backtest import ...` / `from Analysis import ...`, which only resolves
if `Rules/` (not `Rules/superseded/`) is on `sys.path` — the convention every
notebook uses (`os.chdir` + `sys.path.insert` to the directory containing
`Firms.py`). `Backtest.py` itself stays in `Rules/` (not archived here)
because `Master Analysis/funded_analysis.ipynb` still imports it directly.
If you want to resurrect any of these, either move the file back up to
`Rules/` or add `Rules/superseded/` to `sys.path` alongside `Rules/`.
