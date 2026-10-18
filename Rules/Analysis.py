"""
Analysis.py: 3-D risk-geometry search for a fixed-points stop:
(stop distance in points) x (risk % per trade) x (R:R) -> pass rate.

Stop width is its own axis here. A different stop_points value changes which
trades hit the stop or the target, so each value needs its own
backtest_entries() call; risk% and R:R can be read off one ledger.

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
    "rag", ["#8a1f1f", "#c0392b", "#e67e22", "#c9b02a", "#4a9a3a", "#1e7a2e"])


def risk_geometry_3d(firm, strategy, bars, stop_grid, risk_pcts, rr_grid,
                     max_hold=120, n_paths=2000, max_trades=300, seed=0,
                     min_trades=25, max_days_to_pass=None):
    """Full 3-D sweep: one Entries ledger per stop_points value, then a Monte
    Carlo run at every (stop, risk%, R:R) cell.

    speed_grid is mean trades to pass. It says nothing about calendar time:
    a combo that passes in 70 trades still takes months if it fires 0.4x/day.
    days_grid is SimResult.mean_days_to_pass from the same Monte Carlo.
    EmpiricalStrategy gets day_ids here, so simulate() resamples whole
    historical days (keeping real trade-count clustering) and counts each
    path's days to passing.

    max_days_to_pass restricts 'best' to cells that pass within that many
    trading days on average. If no cell qualifies, best['met_days_ceiling']
    is False and the unconstrained top cell is returned instead.

    Caveat: each cell resamples one fixed historical ledger. That averages
    out the luck of trade order, but if the ledger itself is a small or lucky
    sample, every path inherits the bias. Picking the best of many cells and
    param combos adds a multiple-comparisons bias on top. A larger n_paths
    doesn't fix either; check the pick on held-out data (oos_check).

    Returns a dict with pass_grid/speed_grid/days_grid (shape
    len(stop_grid) x len(risk_pcts) x len(rr_grid)), wr_grid (shape
    len(stop_grid) x len(rr_grid), since win rate doesn't depend on risk%),
    trades_grid/trades_per_day per stop width, the 'best' cell, and the
    per-stop ledgers (entries_by_stop) so callers can reuse them."""
    stop_grid = list(stop_grid); risk_pcts = list(risk_pcts); rr_grid = list(rr_grid)
    ns, nr, nk = len(stop_grid), len(risk_pcts), len(rr_grid)
    pass_grid = np.full((ns, nr, nk), np.nan)
    speed_grid = np.full((ns, nr, nk), np.nan)
    days_grid = np.full((ns, nr, nk), np.nan)   # SimResult.mean_days_to_pass
    wr_grid = np.full((ns, nk), np.nan)
    trades_grid = np.zeros(ns, dtype=int)
    trades_per_day = np.zeros(ns, dtype=float)   # measured, not simulated; display only
    entries_by_stop = {}

    for si, sp in enumerate(stop_grid):
        E = backtest_entries(strategy, bars, stop_points=sp, max_hold=max_hold)
        entries_by_stop[sp] = E
        trades_grid[si] = len(E.day_ids)
        trades_per_day[si] = len(E.day_ids) / max(E.n_days, 1)
        if len(E.day_ids) < min_trades:
            continue   # too few trades at this stop width; leave this slab NaN
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
                days_grid[si, ri, ki] = res.mean_days_to_pass

    if not np.isfinite(pass_grid).any():
        raise ValueError("No (stop, risk%, R:R) cell cleared min_trades; "
                         "widen stop_grid or lower min_trades.")

    met_ceiling = True
    candidate_grid = pass_grid
    if max_days_to_pass is not None:
        within = np.isfinite(days_grid) & (days_grid <= max_days_to_pass)
        if within.any():
            candidate_grid = np.where(within, pass_grid, np.nan)
        else:
            met_ceiling = False   # nothing qualifies; fall back to the unconstrained grid

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
    """Measure one already-chosen (stop_points, rr, risk_pct) cell on
    held-out data, with no re-optimization.

    Compare pass_pct/days_to_pass here with the in-sample `best` dict. A large
    drop means the in-sample pick was overfit, which risk_geometry_3d() can't
    detect on its own (see its caveat)."""
    E = backtest_entries(strategy, bars_oos, stop_points=stop_points, max_hold=max_hold)
    n_trades = len(E.day_ids)
    if n_trades < min_trades:
        return dict(trades=n_trades, pass_pct=float("nan"), days_to_pass=float("nan"),
                    speed=float("nan"), wr=float("nan"),
                    warning=f"only {n_trades} OOS trades (< min_trades={min_trades}); "
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
    z=risk per trade (%). Color and bubble size both encode pass rate, since
    a 3-D scatter can't label each cell. The best cell is marked with a star."""
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
    ax.set_title(f"Pass-rate surface: {firm.name}\n"
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
    """Horizontal bar chart of the top cells by pass rate (the same rows as
    summary_table_3d()).

    Bar length is pass rate, which sets the ranking. Bar color is days to
    pass, on the same ramp as elsewhere so green means good (fast). That puts
    the pass-rate vs speed trade-off in one picture."""
    df = summary_table_3d(g, top_n=top_n, max_days_to_pass=max_days_to_pass)
    df = df[np.isfinite(df["days_to_pass"])]
    if df.empty:
        raise ValueError("No cells with a finite days_to_pass to plot; "
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
    """Flat table of the top cells across the 3-D grid, for exact numbers
    next to the chart.

    max_days_to_pass drops cells slower than that many trading days before
    ranking by pass_pct (the same rule as risk_geometry_3d()'s `best`).
    Without it, slow high-pass cells that take months can fill the top."""
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