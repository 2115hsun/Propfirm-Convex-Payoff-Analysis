"""
Analysis_points.py -- 3-D risk-geometry search for a FIXED-POINTS stop:
(stop distance in points) x (risk % per trade) x (R:R) -> pass rate.

Companion to Backtest_points.py (no ATR anywhere in either file). Where
Analysis.py's risk_rr_heatmap() sweeps risk%/R:R against a SINGLE
already-built Entries ledger (stop width was decided once, upstream, via
k_stop_atr), this module adds stop width as its own axis: a different
stop_points value changes which trades actually hit stop vs. target, so it
needs its own backtest_entries() call per value -- not just a different R:R
read off one ledger, the way risk%/R:R can be.

Core outputs:
  risk_geometry_3d(firm, strategy, bars, stop_grid, risk_pcts, rr_grid)
      -> full 3-D pass-rate/speed/WR grids + the single best cell
  plot_risk_geometry_3d(firm, g)
      -> 3-D bubble chart (stop pts x R:R x risk%, color+size = pass rate)
  summary_table_3d(g, top_n=15)
      -> flat DataFrame of the top cells, for a plain-text readout next to the chart
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

from Strategy import EmpiricalStrategy
from Engine import simulate
from Backtest import backtest_entries, ledger_for_rr, measured_winrate

_HEAT = LinearSegmentedColormap.from_list(
    "rag", ["#8a1f1f", "#c0392b", "#e67e22", "#c9b02a", "#4a9a3a", "#1e7a2e"]

)
def risk_geometry_3d(firm, strategy, bars, stop_grid, risk_pcts, rr_grid,
                     max_hold=120, n_paths=2000, max_trades=300, seed=0,
                     min_trades=25, max_days_to_pass=None):
    """Full 3-D sweep: builds ONE Entries ledger per stop_points value, then
    runs optimal_geometry-style Monte Carlo at every (stop, risk%, R:R) cell.

    `speed_grid` (mean_trades_to_pass, from the Monte Carlo) says nothing
    about WALL-CLOCK time on its own -- a combo needing only 70 trades to
    pass still takes 9 months to actually reach if it only fires 0.4x/day.
    `days_grid` is `SimResult.mean_days_to_pass` -- ALSO straight from the
    same Monte Carlo, not a post-hoc estimate: EmpiricalStrategy is always
    built with day_ids here, so simulate() block-bootstraps WHOLE historical
    trading days (real day-to-day trade-count clustering, not a synthetic
    constant rate) and tracks each simulated path's own day count to passing,
    averaged over all n_paths. So both the "how many trades" and "how many
    days" numbers are simulation outputs, averaged across many resampled
    paths -- not one lucky historical run. Pass `max_days_to_pass` to make
    that a real constraint on which cell counts as 'best': among cells
    estimated to pass within that many trading days, the highest pass_pct
    wins -- not the single highest pass_pct on the whole grid regardless of
    how long it'd take to get there. If NO cell meets the ceiling,
    `best['met_days_ceiling']` comes back False and the unconstrained
    highest-pass_pct cell is used instead (so the function never silently
    fails) -- check that flag.

    CAVEAT this does NOT cover: the Monte Carlo only resamples from ONE
    fixed historical R-ledger per cell -- it averages away PATH-level luck
    (which random order the same trades come in), but if that ledger itself
    is a small/lucky historical sample (e.g. a param combo with few total
    trades), every resampled path inherits that same bias. Sweeping many
    (stop, risk%, R:R) cells x many indicator-param combos and picking the
    best-looking one is a multiple-comparisons search on top of that --
    the winner is systematically more likely to be a combo whose historical
    sample happened to look good, not necessarily the truest edge. Guard
    against this with an in-sample/out-of-sample split (pick params on one
    period, verify pass_pct/days_to_pass on a held-out period), not just a
    bigger n_paths.

    Returns a dict with pass_grid/speed_grid/days_grid (shape
    len(stop_grid) x len(risk_pcts) x len(rr_grid)), wr_grid (shape
    len(stop_grid) x len(rr_grid) -- win rate doesn't depend on risk%),
    trades_grid/trades_per_day (per stop width), the flat 'best' cell, and
    the per-stop Entries ledgers (entries_by_stop) so callers can reuse them
    (e.g. for a 2-D risk_rr_heatmap at just the winning stop width) without
    rebuilding."""
    stop_grid = list(stop_grid); risk_pcts = list(risk_pcts); rr_grid = list(rr_grid)
    ns, nr, nk = len(stop_grid), len(risk_pcts), len(rr_grid)
    pass_grid = np.full((ns, nr, nk), np.nan)
    speed_grid = np.full((ns, nr, nk), np.nan)
    days_grid = np.full((ns, nr, nk), np.nan)   # SimResult.mean_days_to_pass -- see docstring caveat
    wr_grid = np.full((ns, nk), np.nan)
    trades_grid = np.zeros(ns, dtype=int)
    trades_per_day = np.zeros(ns, dtype=float)   # measured (not simulated) -- display/diagnostic only
    entries_by_stop = {}

    for si, sp in enumerate(stop_grid):
        E = backtest_entries(strategy, bars, stop_points=sp, max_hold=max_hold)
        entries_by_stop[sp] = E
        trades_grid[si] = len(E.day_ids)
        trades_per_day[si] = len(E.day_ids) / max(E.n_days, 1)
        if len(E.day_ids) < min_trades:
            continue   # too few trades at this stop width to trust -- leave this slab NaN
        for ki, rr in enumerate(rr_grid):
            R, days = ledger_for_rr(E, rr)
            strat = EmpiricalStrategy(R, name=f"{sp:g}pt {rr:g}R", day_ids=days,
                                      mae_R=getattr(E, "mae_before", None))
            wr_grid[si, ki] = measured_winrate(E, rr)
            for ri, rp in enumerate(risk_pcts):
                res = simulate(strat, firm, firm.start_balance * rp / 100.0,
                               n_paths=n_paths, max_trades=max_trades, seed=seed)
                pass_grid[si, ri, ki] = res.pass_rate * 100
                speed_grid[si, ri, ki] = res.mean_trades_to_pass
                days_grid[si, ri, ki] = res.mean_days_to_pass   # simulated, block-bootstrapped by real day -- not speed_grid/trades_per_day

    if not np.isfinite(pass_grid).any():
        raise ValueError("No (stop, risk%, R:R) cell cleared min_trades -- "
                         "widen stop_grid or lower min_trades.")

    met_ceiling = True
    candidate_grid = pass_grid
    if max_days_to_pass is not None:
        within = np.isfinite(days_grid) & (days_grid <= max_days_to_pass)
        if within.any():
            candidate_grid = np.where(within, pass_grid, np.nan)
        else:
            met_ceiling = False   # nothing qualifies -- fall back to the unconstrained grid below

    bi = np.unravel_index(np.nanargmax(candidate_grid), candidate_grid.shape)
    best = dict(stop_points=float(stop_grid[bi[0]]), risk_pct=float(risk_pcts[bi[1]]),
                rr=float(rr_grid[bi[2]]), pass_pct=float(pass_grid[bi]),
                speed=float(speed_grid[bi]), wr=float(wr_grid[bi[0], bi[2]]),
                trades=int(trades_grid[bi[0]]), trades_per_day=float(trades_per_day[bi[0]]),
                days_to_pass=float(days_grid[bi]), met_days_ceiling=met_ceiling)
    return dict(pass_grid=pass_grid, speed_grid=speed_grid, days_grid=days_grid, wr_grid=wr_grid,
                trades_grid=trades_grid, trades_per_day=trades_per_day,
                stop_grid=np.asarray(stop_grid, float),
                risk_pcts=np.asarray(risk_pcts, float), rr_grid=np.asarray(rr_grid, float),
                best=best, entries_by_stop=entries_by_stop)


def oos_check(firm, strategy, bars_oos, stop_points, rr, risk_pct,
             max_hold=120, n_paths=2000, max_trades=300, seed=0, min_trades=10):
    """Evaluate ONE already-chosen (stop_points, rr, risk_pct) cell for
    `strategy` on held-out data -- no new argmax, no re-optimization, just a
    direct measurement of how the exact combo already picked on in-sample
    data performs on data it never saw.

    This is the check risk_geometry_3d()'s Monte Carlo averaging alone can't
    give you (see its docstring caveat): averaging over many resampled paths
    protects against PATH-level luck in one fixed historical ledger, but says
    nothing about whether that ledger's own history -- or the search over
    many combos that picked it -- was itself a lucky sample. Compare this
    result's pass_pct/days_to_pass against the in-sample `best` dict; a large
    drop is the actual overfitting signal, not anything the in-sample number
    alone can tell you."""
    E = backtest_entries(strategy, bars_oos, stop_points=stop_points, max_hold=max_hold)
    n_trades = len(E.day_ids)
    if n_trades < min_trades:
        return dict(trades=n_trades, pass_pct=float("nan"), days_to_pass=float("nan"),
                    speed=float("nan"), wr=float("nan"),
                    warning=f"only {n_trades} OOS trades (< min_trades={min_trades}) -- "
                            f"too few to trust; widen OOS_FRACTION or treat as inconclusive")
    R, days = ledger_for_rr(E, rr)
    strat_oos = EmpiricalStrategy(R, name=f"OOS {stop_points:g}pt {rr:g}R", day_ids=days,
                                  mae_R=getattr(E, "mae_before", None))
    res = simulate(strat_oos, firm, firm.start_balance * risk_pct / 100.0,
                   n_paths=n_paths, max_trades=max_trades, seed=seed)
    return dict(trades=n_trades, pass_pct=res.pass_rate * 100, days_to_pass=res.mean_days_to_pass,
                speed=res.mean_trades_to_pass, wr=measured_winrate(E, rr), warning=None)


def plot_risk_geometry_3d(firm, g, size_range=(30, 260), savepath=None):
    """3-D bubble chart: x=stop distance (points), y=R:R (target/stop),
    z=risk per trade (%); color AND bubble size both encode pass rate
    (redundant encoding -- a 3-D scatter can't show a per-cell number the way
    the 2-D heatmap does, so size stands in for it). The single best cell is
    marked with a star."""
    pass_grid, stop_grid, risk_pcts, rr_grid = (
        g["pass_grid"], g["stop_grid"], g["risk_pcts"], g["rr_grid"])
    best = g["best"]

    SP, RP, RR = np.meshgrid(stop_grid, risk_pcts, rr_grid, indexing="ij")
    sp_f, rp_f, rr_f, pg_f = SP.ravel(), RP.ravel(), RR.ravel(), pass_grid.ravel()
    valid = np.isfinite(pg_f)
    sp_f, rp_f, rr_f, pg_f = sp_f[valid], rp_f[valid], rr_f[valid], pg_f[valid]

    lo, hi = size_range
    sizes = lo + (hi - lo) * (pg_f / 100.0)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(sp_f, rr_f, rp_f, c=pg_f, cmap=_HEAT, vmin=0, vmax=100,
                    s=sizes, edgecolors="black", linewidths=0.4, alpha=0.85, depthshade=True)
    ax.scatter([best["stop_points"]], [best["rr"]], [best["risk_pct"]],
              marker="*", s=900, c="#00e5ff", edgecolors="black", linewidths=1.2,
              label="best", zorder=10)

    ax.set_xlabel("Stop distance (points)")
    ax.set_ylabel("R:R (target/stop)")
    ax.set_zlabel("Risk per trade (%)")
    tgt_pts = best["stop_points"] * best["rr"]
    days_note = (f" in ~{best['days_to_pass']:.0f} trading days" if "days_to_pass" in best
                 and np.isfinite(best["days_to_pass"]) else "")
    ceiling_note = "" if best.get("met_days_ceiling", True) else "  [!] NO cell met the days ceiling"
    ax.set_title(f"Pass-rate surface -- {firm.name}\n"
                 f"BEST: {best['stop_points']:g}pt stop / {tgt_pts:g}pt tgt "
                 f"({best['rr']:g}R) @ {best['risk_pct']:.2f}% -> "
                 f"{best['pass_pct']:.0f}% pass{days_note} (WR {best['wr']:.0%}){ceiling_note}",
                 fontweight="bold")
    ax.legend(loc="upper left")
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label("Pass rate (%)")
    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig


def plot_top_cells(g, top_n=10, max_days_to_pass=None, savepath=None):
    """Ranked horizontal bar chart of the top cells by pass rate -- a plain-
    English complement to the 3-D bubble chart, which can't show exact
    per-cell numbers (that's what summary_table_3d()'s text table is for;
    this is the same data, but readable as a shape, not a grid of digits).

    Bar LENGTH = pass rate (what it's ranked by). Bar COLOR = days-to-pass
    (green=fast, red=slow, using the same ramp/direction pass rate uses
    elsewhere -- "green always means good") -- so the pass-rate/speed
    trade-off this session's runs are optimizing for is visible in one
    glance instead of a table column you have to read number by number."""
    df = summary_table_3d(g, top_n=top_n, max_days_to_pass=max_days_to_pass)
    df = df[np.isfinite(df["days_to_pass"])]
    if df.empty:
        raise ValueError("No cells with a finite days_to_pass to plot -- "
                         "widen max_days_to_pass or top_n.")
    df = df.iloc[::-1].reset_index(drop=True)   # barh draws bottom-up; reverse so #1 lands on top

    labels = [f"{r.stop_pts:g}pt / {r.rr:g}R @ {r.risk_pct:.2f}%" for r in df.itertuples()]
    days, pass_pct, wr = df["days_to_pass"].to_numpy(), df["pass_pct"].to_numpy(), df["wr"].to_numpy()

    dmax = float(days.max())
    norm = plt.Normalize(vmin=0, vmax=max(dmax, 1e-9))
    cmap_fast_green = _HEAT.reversed()   # low days (fast) -> green end; high days (slow) -> red end
    colors = cmap_fast_green(norm(days))

    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(df) + 1.5))
    bars = ax.barh(labels, pass_pct, color=colors, edgecolor="black", linewidth=0.4, height=0.62, zorder=3)

    for bar, p, d, w in zip(bars, pass_pct, days, wr):
        ax.text(bar.get_width() + max(pass_pct) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%  ({d:.0f}d, WR {w:.0%})", va="center", ha="left", fontsize=9)

    ax.set_xlim(0, max(pass_pct) * 1.30)
    ax.set_xlabel("Pass rate (%)")
    title_suffix = f"  (<= {max_days_to_pass:g} trading days to pass)" if max_days_to_pass is not None else ""
    ax.set_title(f"Top {len(df)} cells{title_suffix}", fontweight="bold", loc="left")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    sm = plt.cm.ScalarMappable(cmap=cmap_fast_green, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label("Days to pass (green = faster)")

    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig


def summary_table_3d(g, top_n=15, max_days_to_pass=None) -> pd.DataFrame:
    """Flat table of the top cells across the whole 3-D grid, for a
    plain-text readout alongside the chart (bubble size/color alone can be
    hard to read exact numbers off of).

    Pass `max_days_to_pass` to filter OUT cells that would take longer than
    that many trading days to pass (same constraint risk_geometry_3d()'s
    `best` cell uses) before ranking by pass_pct -- otherwise the top of
    this table can be dominated by high-pass/low-frequency cells that take
    months to actually reach in real trading."""
    stop_grid, risk_pcts, rr_grid = g["stop_grid"], g["risk_pcts"], g["rr_grid"]
    pass_grid, speed_grid, wr_grid = g["pass_grid"], g["speed_grid"], g["wr_grid"]
    days_grid = g.get("days_grid")
    trades_per_day = g.get("trades_per_day")
    rows = []
    for si, sp in enumerate(stop_grid):
        for ki, rr in enumerate(rr_grid):
            for ri, rp in enumerate(risk_pcts):
                p = pass_grid[si, ri, ki]
                if not np.isfinite(p):
                    continue
                d2p = float(days_grid[si, ri, ki]) if days_grid is not None else np.nan
                rows.append(dict(stop_pts=sp, target_pts=sp * rr, rr=rr, risk_pct=rp,
                                 pass_pct=p, speed=speed_grid[si, ri, ki],
                                 days_to_pass=d2p,
                                 trades_per_day=float(trades_per_day[si]) if trades_per_day is not None else np.nan,
                                 wr=wr_grid[si, ki], trades=int(g["trades_grid"][si])))
    df = pd.DataFrame(rows)
    if max_days_to_pass is not None and "days_to_pass" in df:
        kept = df[df["days_to_pass"] <= max_days_to_pass]
        if len(kept):   # only apply the filter if it doesn't wipe out every row
            df = kept
    df = df.sort_values("pass_pct", ascending=False)
    return df.head(top_n).reset_index(drop=True)