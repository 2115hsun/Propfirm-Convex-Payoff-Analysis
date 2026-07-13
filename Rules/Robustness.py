"""
robustness.py — regime-robust risk-geometry selection.

Runs the risk x R:R grid over rolling windows, collapses each cell across windows
to a distribution (mean/min/p25/std), and picks the geometry with the highest
FLOOR (worst-case), not the highest mean. Then validates the choice out-of-sample.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from Backtest import backtest_entries
from Analysis import optimal_geometry

_HEAT = LinearSegmentedColormap.from_list(
    "rag", ["#8a1f1f", "#c0392b", "#e67e22", "#c9b02a", "#4a9a3a", "#1e7a2e"])


def rolling_windows(bars, months=2, step_months=1):
    """List of (label, bars_slice) for rolling calendar windows."""
    start = bars.index.min().normalize()
    end = bars.index.max().normalize()
    out = []
    cur = pd.Timestamp(start.year, start.month, 1, tz=bars.index.tz)
    while cur <= end:
        w_end = cur + pd.DateOffset(months=months)
        sl = bars.loc[cur:w_end]
        if len(sl) > 0:
            out.append((cur.strftime("%Y-%m"), sl))
        cur = cur + pd.DateOffset(months=step_months)
    return out


def window_pass_grids(strategy_factory, bars, firm, risk_pcts, rr_grid,
                      k_stop_atr=3, max_hold=60, months=2, step_months=1,
                      n_paths=800, max_trades=250, min_trades=25, seed=0):
    """Returns grids [n_windows, n_risk, n_rr] of pass rates + window labels + trade counts."""
    wins = rolling_windows(bars, months, step_months)
    grids, labels, counts = [], [], []
    for label, sl in wins:
        E = backtest_entries(strategy_factory(), sl, k_stop_atr=k_stop_atr, max_hold=max_hold)
        if len(E.day_ids) < min_trades:
            continue
        g = optimal_geometry(firm, E, risk_pcts=risk_pcts, rr_grid=rr_grid,
                             n_paths=n_paths, max_trades=max_trades, seed=seed)
        grids.append(g["pass_grid"]); labels.append(label); counts.append(len(E.day_ids))
    return np.array(grids), labels, counts


def robust_stats(grids):
    """Collapse [windows, risk, rr] -> per-cell mean/min/p25/std across windows."""
    return {
        "mean": grids.mean(axis=0), "min": grids.min(axis=0),
        "p25": np.percentile(grids, 25, axis=0), "std": grids.std(axis=0),
    }


def pick_robust(stats, risk_pcts, rr_grid, floor="p25"):
    fg = stats[floor]
    bi, bj = np.unravel_index(np.argmax(fg), fg.shape)
    return {"risk_pct": float(risk_pcts[bi]), "rr": rr_grid[bj],
            "floor": float(fg[bi, bj]), "mean": float(stats["mean"][bi, bj]),
            "min": float(stats["min"][bi, bj]), "i": bi, "j": bj}


def plot_mean_vs_floor(stats, risk_pcts, rr_grid, floor="p25", savepath=None):
    fig, axes = plt.subplots(1, 2, figsize=(2.0 * len(rr_grid) + 4, 0.5 * len(risk_pcts) + 2))
    for ax, key, title in [(axes[0], "mean", "MEAN pass rate (optimistic)"),
                           (axes[1], floor, f"FLOOR = {floor} pass rate (robust)")]:
        grid = stats[key]
        im = ax.imshow(grid, cmap=_HEAT, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(rr_grid))); ax.set_xticklabels([f"{r:g}R" for r in rr_grid])
        ax.set_yticks(range(len(risk_pcts))); ax.set_yticklabels([f"{p:.2f}%" for p in risk_pcts], fontsize=8)
        ax.set_xlabel("R:R"); ax.set_ylabel("Risk per trade")
        for i in range(len(risk_pcts)):
            for j in range(len(rr_grid)):
                ax.text(j, i, f"{grid[i,j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if grid[i,j] < 55 else "black")
        bi, bj = np.unravel_index(np.argmax(grid), grid.shape)
        ax.add_patch(plt.Rectangle((bj-.5, bi-.5), 1, 1, fill=False, ec="cyan", lw=2.5))
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Pass rate (%)", fraction=0.025)
    fig.suptitle("Optimize the FLOOR, not the mean  (cyan = best cell of each)", fontweight="bold")
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig


def validate_oos(strategy_factory, oos_bars, firm, cell, risk_pcts, rr_grid,
                 is_grids, k_stop_atr=3, max_hold=60, months=2, step_months=1,
                 n_paths=1500, max_trades=250, min_trades=25, savepath=None):
    """Run the chosen cell on OOS rolling windows; compare to its IS distribution."""
    from Strategy import EmpiricalStrategy
    from Backtest import ledger_for_rr
    from Engine import simulate
    i, j = cell["i"], cell["j"]
    is_dist = is_grids[:, i, j]

    oos_dist, oos_labels = [], []
    for label, sl in rolling_windows(oos_bars, months, step_months):
        E = backtest_entries(strategy_factory(), sl, k_stop_atr=k_stop_atr, max_hold=max_hold)
        if len(E.day_ids) < min_trades:
            continue
        R, days = ledger_for_rr(E, cell["rr"])
        emp = EmpiricalStrategy(R, day_ids=days)
        res = simulate(emp, firm, firm.start_balance * cell["risk_pct"] / 100,
                       n_paths=n_paths, max_trades=max_trades)
        oos_dist.append(res.pass_rate * 100); oos_labels.append(label)
    oos_dist = np.array(oos_dist)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(np.zeros_like(is_dist) + np.random.uniform(-.06, .06, len(is_dist)), is_dist,
               s=40, alpha=.6, color="#2e86de", label=f"In-sample windows (n={len(is_dist)})")
    ax.scatter(np.ones_like(oos_dist) + np.random.uniform(-.06, .06, len(oos_dist)), oos_dist,
               s=60, alpha=.8, color="#e67e22", label=f"OOS windows (n={len(oos_dist)})")
    for x, d, c in [(0, is_dist, "#2e86de"), (1, oos_dist, "#e67e22")]:
        ax.hlines(np.median(d), x-.2, x+.2, color=c, lw=2)
        ax.hlines(np.percentile(d, 25), x-.15, x+.15, color=c, lw=1, ls="--")
    ax.axhline(cell["floor"], color="grey", ls=":", label=f"IS floor ({cell['floor']:.0f}%)")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["2021-2024 (IS)", "2025-2026 (OOS)"])
    ax.set_ylabel("Pass rate per window (%)"); ax.set_ylim(0, 100)
    ax.set_title(f"OOS validation of robust cell: {cell['rr']:g}R @ {cell['risk_pct']:.2f}% risk\n"
                 f"(solid = median, dashed = p25)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig, is_dist, oos_dist


# ---------------------------------------------------------------------------
# Consistency / frequency summary  ("how often does each geometry clear a bar")
# ---------------------------------------------------------------------------
def consistency(grids, threshold=60):
    """Percent of windows in which each cell passed >= threshold."""
    return (grids >= threshold).mean(axis=0) * 100.0


def summary_table(is_grids, oos_grids, risk_pcts, rr_grid, start_balance=50000,
                  threshold=60):
    import pandas as pd
    is_con = consistency(is_grids, threshold)
    oos_con = consistency(oos_grids, threshold)
    rows = []
    for i, rp in enumerate(risk_pcts):
        for j, rr in enumerate(rr_grid):
            rows.append({
                "RR": rr, "risk_%": rp, "risk_$": start_balance * rp / 100,
                "IS_mean": is_grids.mean(0)[i, j],
                "IS_floor_p25": np.percentile(is_grids, 25, 0)[i, j],
                "IS_worst": is_grids.min(0)[i, j],
                f"IS_%win>={threshold}": is_con[i, j],
                "OOS_mean": oos_grids.mean(0)[i, j],
                f"OOS_%win>={threshold}": oos_con[i, j],
            })
    df = pd.DataFrame(rows)
    # robustness score = the WORSE of the two consistencies (reward reliable in BOTH)
    df["robust_score"] = df[[f"IS_%win>={threshold}", f"OOS_%win>={threshold}"]].min(axis=1)
    return df.sort_values("robust_score", ascending=False).reset_index(drop=True)


def recommend_range(is_grids, oos_grids, risk_pcts, rr_grid, start_balance=50000,
                    threshold=60, bar=60):
    """The region of cells reliable in BOTH IS and OOS (>= bar% of windows clear
    the threshold). Returns the RR / risk / $ ranges spanning that region."""
    both = np.minimum(consistency(is_grids, threshold), consistency(oos_grids, threshold))
    while bar > 20 and not (both >= bar).any():
        bar -= 5
    ii, jj = np.where(both >= bar)
    if ii.size == 0:
        return None
    rp = np.array(risk_pcts)
    return {
        "bar": bar, "threshold": threshold, "n_cells": int(ii.size),
        "rr_lo": min(rr_grid[j] for j in jj), "rr_hi": max(rr_grid[j] for j in jj),
        "risk_lo": float(rp[ii].min()), "risk_hi": float(rp[ii].max()),
        "dollar_lo": float(rp[ii].min() * start_balance / 100),
        "dollar_hi": float(rp[ii].max() * start_balance / 100),
        "mask": both >= bar, "both": both,
    }


def plot_summary(is_grids, oos_grids, risk_pcts, rr_grid, start_balance=50000,
                 threshold=60, savepath=None):
    is_con = consistency(is_grids, threshold)
    oos_con = consistency(oos_grids, threshold)
    rec = recommend_range(is_grids, oos_grids, risk_pcts, rr_grid, start_balance, threshold)
    ylabels = [f"{p:.2f}%\n(${start_balance*p/100:,.0f})" for p in risk_pcts]

    fig, axes = plt.subplots(1, 2, figsize=(2.1 * len(rr_grid) + 4, 0.6 * len(risk_pcts) + 2.5))
    for ax, con, title in [(axes[0], is_con, "IN-SAMPLE 2021-2024"),
                           (axes[1], oos_con, "OUT-OF-SAMPLE 2025-2026")]:
        im = ax.imshow(con, cmap=_HEAT, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(rr_grid))); ax.set_xticklabels([f"{r:g}R" for r in rr_grid])
        ax.set_yticks(range(len(risk_pcts))); ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xlabel("R:R"); ax.set_ylabel("Risk per trade")
        for i in range(len(risk_pcts)):
            for j in range(len(rr_grid)):
                ax.text(j, i, f"{con[i,j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if con[i, j] < 55 else "black")
        if rec:
            for (i, j) in zip(*np.where(rec["mask"])):
                ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, fill=False, ec="cyan", lw=2))
        ax.set_title(f"{title}\n% of windows passing >= {threshold}%", fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), label=f"% of windows >= {threshold}% pass", fraction=0.025)
    sub = ("Recommended range (reliable in BOTH, cyan): "
           f"{rec['rr_lo']:g}-{rec['rr_hi']:g}R @ {rec['risk_lo']:.2f}-{rec['risk_hi']:.2f}% "
           f"(${rec['dollar_lo']:,.0f}-${rec['dollar_hi']:,.0f}/trade)") if rec else \
          "No range reliable in both IS and OOS at this bar."
    fig.suptitle("Regime-Robust Geometry — consistency across rolling 2-month windows\n" + sub,
                 fontweight="bold", fontsize=11)
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig, rec