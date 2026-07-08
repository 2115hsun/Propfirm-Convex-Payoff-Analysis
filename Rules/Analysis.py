"""
Analysis.py — discover optimal risk geometry per firm (nothing predefined).

Core outputs:
  measured_wr_curve(entries)                 -> WR vs R:R, measured from data
  find_optimal_risk(strategy, firm)          -> pass-maximizing risk % for a firm
  plot_risk_curve(strategy, firm)            -> pass rate & speed vs risk
  risk_rr_heatmap(firm, entries)             -> Risk x R:R pass-rate grid (data)
  optimal_geometry(firm, entries)            -> the single best (R:R, risk%) cell
  optimal_risk_by_firm(strategy, firms)      -> cross-firm best pass & its risk
  plot_equity_paths / plot_strategy_comparison (kept for demos)
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from Firms import FirmConfig
from Strategy import Strategy, SyntheticStrategy, EmpiricalStrategy
from Engine import simulate
from Backtest import ledger_for_rr, measured_winrate

_HEAT = LinearSegmentedColormap.from_list(
    "rag", ["#8a1f1f", "#c0392b", "#e67e22", "#c9b02a", "#4a9a3a", "#1e7a2e"])


# ---------------------------------------------------------------------------
# Data-measured win rate vs R:R
# ---------------------------------------------------------------------------
def measured_wr_curve(entries, rr_grid, savepath=None):
    wr = [measured_winrate(entries, rr) for rr in rr_grid]
    edge = []
    for rr in rr_grid:
        R, _ = ledger_for_rr(entries, rr)
        edge.append(float(R.mean()))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rr_grid, [w * 100 for w in wr], "-o", color="#2e86de")
    ax.set_xlabel("R:R"); ax.set_ylabel("Measured win rate (%)")
    ax.set_title(f"Win rate vs R:R (measured) — {entries.strategy} on {entries.source}")
    ax.grid(alpha=.3)
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig, dict(zip(rr_grid, wr)), dict(zip(rr_grid, edge))


# ---------------------------------------------------------------------------
# Risk search  (risk % is discovered)
# ---------------------------------------------------------------------------
def sweep_risk(strategy, firm, risk_pcts, n_paths=4_000, max_trades=400, seed=0):
    pr, sp, fr = [], [], []
    for rp in risk_pcts:
        res = simulate(strategy, firm, firm.start_balance * rp / 100.0,
                       n_paths=n_paths, max_trades=max_trades, seed=seed)
        pr.append(res.pass_rate * 100); sp.append(res.mean_trades_to_pass)
        fr.append(res.fail_rate * 100)
    return np.array(pr), np.array(sp), np.array(fr)


def find_optimal_risk(strategy, firm, risk_pcts=None, n_paths=4_000, max_trades=400, seed=0):
    if risk_pcts is None:
        risk_pcts = np.round(np.arange(0.10, 4.01, 0.10), 2)
    pr, sp, fr = sweep_risk(strategy, firm, risk_pcts, n_paths, max_trades, seed)
    best_pass = pr.max()
    tied = np.flatnonzero(pr >= best_pass - 1e-9)
    speeds = np.where(np.isnan(sp[tied]), np.inf, sp[tied])
    b = tied[int(np.argmin(speeds))]
    return {"risk_pcts": risk_pcts, "pass_pct": pr, "speed": sp, "fail_pct": fr,
            "opt_risk_pct": float(risk_pcts[b]),
            "opt_risk_dollars": float(firm.start_balance * risk_pcts[b] / 100.0),
            "opt_pass_pct": float(pr[b]), "opt_speed": float(sp[b])}


def plot_risk_curve(strategy, firm, risk_pcts=None, n_paths=4_000, max_trades=400,
                    seed=0, savepath=None):
    r = find_optimal_risk(strategy, firm, risk_pcts, n_paths, max_trades, seed)
    rp, pr, sp = r["risk_pcts"], r["pass_pct"], r["speed"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rp, pr, "-o", color="#2e86de", ms=3, label="Pass rate")
    ax.axvline(r["opt_risk_pct"], color="#e67e22", ls="--",
               label=f"optimum {r['opt_risk_pct']:.2f}%")
    ax.scatter([r["opt_risk_pct"]], [r["opt_pass_pct"]], color="#e67e22", zorder=5)
    ax.annotate(f"{r['opt_pass_pct']:.1f}% pass\n@ {r['opt_risk_pct']:.2f}% "
                f"(${r['opt_risk_dollars']:.0f})", (r["opt_risk_pct"], r["opt_pass_pct"]),
                textcoords="offset points", xytext=(10, -30), fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.set_xlabel("Risk per trade (% of account)"); ax.set_ylabel("Pass rate (%)", color="#2e86de")
    ax.set_ylim(0, 100)
    ax2 = ax.twinx(); ax2.plot(rp, sp, color="#16a085", alpha=.6)
    ax2.set_ylabel("Avg trades to pass", color="#16a085")
    ax.set_title(f"Risk optimization — {firm.name}  [{strategy.name}]", fontweight="bold")
    ax.legend(loc="lower left", fontsize=8); fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig, r


# ---------------------------------------------------------------------------
# Risk x R:R heatmap  (data-driven; the joint geometry optimizer)
# ---------------------------------------------------------------------------
def _grid(firm, entries, risk_pcts, rr_grid, n_paths, max_trades, seed):
    wr = {rr: measured_winrate(entries, rr) for rr in rr_grid}
    pass_grid = np.zeros((len(risk_pcts), len(rr_grid)))
    speed_grid = np.full_like(pass_grid, np.nan)
    for j, rr in enumerate(rr_grid):
        R, days = ledger_for_rr(entries, rr)
        strat = EmpiricalStrategy(R, name=f"{rr}R", day_ids=days)
        pr, sp, _ = sweep_risk(strat, firm, risk_pcts, n_paths, max_trades, seed)
        pass_grid[:, j] = pr; speed_grid[:, j] = sp
    return pass_grid, speed_grid, wr


def optimal_geometry(firm, entries, risk_pcts=None, rr_grid=None,
                     n_paths=2_000, max_trades=300, seed=0):
    """Return the single best (R:R, risk%) cell for this firm + strategy."""
    if risk_pcts is None:
        risk_pcts = np.round(np.arange(0.25, 3.51, 0.25), 2)
    if rr_grid is None:
        rr_grid = [0.5, 1.0, 1.5, 2.0, 3.0]
    pg, sg, wr = _grid(firm, entries, risk_pcts, rr_grid, n_paths, max_trades, seed)
    bi, bj = np.unravel_index(np.nanargmax(pg), pg.shape)
    return {"opt_rr": rr_grid[bj], "opt_risk_pct": float(risk_pcts[bi]),
            "opt_pass_pct": float(pg[bi, bj]), "opt_speed": float(sg[bi, bj]),
            "opt_wr": wr[rr_grid[bj]],
            "pass_grid": pg, "speed_grid": sg, "risk_pcts": risk_pcts, "rr_grid": rr_grid}


def risk_rr_heatmap(firm, entries, risk_pcts=None, rr_grid=None,
                    n_paths=2_000, max_trades=300, seed=0, savepath=None):
    g = optimal_geometry(firm, entries, risk_pcts, rr_grid, n_paths, max_trades, seed)
    pg, sg, rps, rrs = g["pass_grid"], g["speed_grid"], g["risk_pcts"], g["rr_grid"]
    wr = {rr: measured_winrate(entries, rr) for rr in rrs}
    fig, ax = plt.subplots(figsize=(1.6 * len(rrs) + 2, 0.5 * len(rps) + 2))
    im = ax.imshow(pg, cmap=_HEAT, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(rrs)))
    ax.set_xticklabels([f"{rr:g}R\n{wr[rr]:.0%} WR" for rr in rrs])
    ax.set_yticks(range(len(rps)))
    ax.set_yticklabels([f"{p:.2f}%\n(${firm.start_balance*p/100:.0f})" for p in rps], fontsize=8)
    ax.set_xlabel("R:R  (win rate MEASURED from data)"); ax.set_ylabel("Risk per trade")
    for i in range(len(rps)):
        for j in range(len(rrs)):
            v = pg[i, j]; s = "" if np.isnan(sg[i, j]) else f"\n{sg[i,j]:.0f} trd"
            ax.text(j, i, f"{v:.0f}%{s}", ha="center", va="center", fontsize=7,
                    color="white" if v < 55 else "black")
    fig.colorbar(im, ax=ax, label="Pass rate (%)")
    ax.set_title(f"Optimal risk geometry — {firm.name}\nbest: {g['opt_rr']:g}R @ "
                 f"{g['opt_risk_pct']:.2f}% risk = {g['opt_pass_pct']:.0f}% pass "
                 f"(~{g['opt_speed']:.0f} trades)", fontweight="bold", fontsize=10)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig, g


# ---------------------------------------------------------------------------
# Cross-firm
# ---------------------------------------------------------------------------
def optimal_risk_by_firm(strategy, firm_getter, firm_keys, risk_pcts=None,
                         n_paths=3_000, max_trades=300, seed=0, savepath=None):
    results = {k: find_optimal_risk(strategy, firm_getter(k), risk_pcts,
                                    n_paths, max_trades, seed) for k in firm_keys}
    names = [firm_getter(k).name for k in firm_keys]
    opt_r = [results[k]["opt_risk_pct"] for k in firm_keys]
    opt_p = [results[k]["opt_pass_pct"] for k in firm_keys]
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(14, 0.7 * len(names) + 2))
    ax[0].barh(y, opt_p, color="#2e86de")
    for yi, v, rr in zip(y, opt_p, opt_r):
        ax[0].text(v + 0.5, yi, f"{v:.1f}%  @ {rr:.2f}% risk", va="center",
                   fontsize=9, fontweight="bold")
    ax[0].set_yticks(y); ax[0].set_yticklabels(names); ax[0].set_xlim(0, 100)
    ax[0].set_xlabel("Best achievable pass rate (%)")
    ax[0].set_title(f"Optimal risk & pass rate by firm\n[{strategy.name}]")
    ax[1].barh(y, opt_r, color="#e67e22")
    for yi, v in zip(y, opt_r):
        ax[1].text(v + 0.02, yi, f"{v:.2f}%", va="center", fontsize=9)
    ax[1].set_yticks(y); ax[1].set_yticklabels([])
    ax[1].set_xlabel("Optimal risk per trade (%)"); ax[1].set_title("Discovered sizing")
    fig.suptitle("MC-discovered risk per firm (no predefined risk %)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return results, fig


# ---------------------------------------------------------------------------
# Demos kept from before
# ---------------------------------------------------------------------------
def plot_equity_paths(firm, strategy, risk_per_trade, n_show=120, max_trades=300,
                      seed=0, savepath=None):
    res = simulate(strategy, firm, risk_per_trade, n_paths=n_show,
                   max_trades=max_trades, seed=seed, record_paths=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for p in range(n_show):
        color = {1: "#2e8b57", -1: "#c0392b", 0: "#999999"}[res.outcome[p]]
        ax.plot(res.equity_paths[p], color=color, alpha=.35, lw=.8)
    ax.axhline(firm.target_equity, color="#1e7a2e", lw=1.5)
    ax.axhline(firm.initial_floor, color="#8a1f1f", lw=1.5)
    ax.axhline(firm.start_balance, color="k", ls="--", lw=.8, alpha=.6)
    ax.set_xlabel("Trade #"); ax.set_ylabel("Equity ($)")
    ax.set_title(f"Equity paths — {firm.name} [{strategy.name}] "
                 f"risk ${risk_per_trade:.0f}\npass {res.pass_rate:.1%} / "
                 f"fail {res.fail_rate:.1%} / ongoing {res.ongoing_rate:.1%}")
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig