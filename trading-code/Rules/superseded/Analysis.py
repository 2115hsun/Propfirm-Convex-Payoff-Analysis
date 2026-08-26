"""
Analysis.py — discover optimal risk geometry per firm (nothing predefined).
Sizing kwargs (sizing/comfort_room/min_risk_frac) are threaded through every
simulate() call so heatmaps and curves can use room-scaled sizing too.

Core outputs:
  measured_wr_curve(entries)                 -> WR vs R:R, measured from data
  find_optimal_risk(strategy, firm)          -> pass-maximizing risk % for a firm
  plot_risk_curve(strategy, firm)            -> pass rate & speed vs risk
  risk_rr_heatmap(firm, entries)             -> Risk x R:R pass-rate grid (data)
  optimal_geometry(firm, entries)            -> the single best (R:R, risk%) cell
  optimal_risk_by_firm(strategy, firms)      -> cross-firm best pass & its risk
  days_to_pass_by_firm(strategy, firms)      -> days-to-pass distribution by firm
  plot_equity_paths                          -> sample equity paths (demo)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from Strategy import EmpiricalStrategy
from Engine import simulate
from Backtest import ledger_for_rr, measured_winrate

_HEAT = LinearSegmentedColormap.from_list(
    "rag", ["#8a1f1f", "#c0392b", "#e67e22", "#c9b02a", "#4a9a3a", "#1e7a2e"])


def measured_wr_curve(entries, rr_grid, savepath=None):
    wr = [measured_winrate(entries, rr) for rr in rr_grid]
    edge = []
    for rr in rr_grid:
        R, _ = ledger_for_rr(entries, rr); edge.append(float(R.mean()))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rr_grid, [w*100 for w in wr], "-o", color="#2e86de")
    ax.set_xlabel("R:R"); ax.set_ylabel("Measured win rate (%)"); ax.grid(alpha=.3)
    return fig, dict(zip(rr_grid, wr)), dict(zip(rr_grid, edge))


# --- sizing kwargs flow through every simulate() call below ---
def sweep_risk(strategy, firm, risk_pcts, n_paths=4000, max_trades=400, seed=0,
               sizing="flat", comfort_room=None, min_risk_frac=0.10):
    pr, sp, fr = [], [], []
    for rp in risk_pcts:
        res = simulate(strategy, firm, firm.start_balance*rp/100.0,
                       n_paths=n_paths, max_trades=max_trades, seed=seed,
                       sizing=sizing, comfort_room=comfort_room, min_risk_frac=min_risk_frac)
        pr.append(res.pass_rate*100); sp.append(res.mean_trades_to_pass); fr.append(res.fail_rate*100)
    return np.array(pr), np.array(sp), np.array(fr)


def _grid(firm, entries, risk_pcts, rr_grid, n_paths, max_trades, seed,
          sizing="flat", comfort_room=None, min_risk_frac=0.10):
    wr = {rr: measured_winrate(entries, rr) for rr in rr_grid}
    pass_grid = np.zeros((len(risk_pcts), len(rr_grid)))
    speed_grid = np.full_like(pass_grid, np.nan)
    for j, rr in enumerate(rr_grid):
        R, days = ledger_for_rr(entries, rr)
        strat = EmpiricalStrategy(R, name=f"{rr}R", day_ids=days)
        pr, sp, _ = sweep_risk(strat, firm, risk_pcts, n_paths, max_trades, seed,
                               sizing=sizing, comfort_room=comfort_room, min_risk_frac=min_risk_frac)
        pass_grid[:, j] = pr; speed_grid[:, j] = sp
    return pass_grid, speed_grid, wr


def optimal_geometry(firm, entries, risk_pcts=None, rr_grid=None, n_paths=2000,
                     max_trades=300, seed=0, sizing="flat", comfort_room=None, min_risk_frac=0.10):
    if risk_pcts is None: risk_pcts = np.round(np.arange(0.25, 3.51, 0.25), 2)
    if rr_grid is None: rr_grid = [0.5, 1.0, 1.5, 2.0, 3.0]
    pg, sg, wr = _grid(firm, entries, risk_pcts, rr_grid, n_paths, max_trades, seed,
                       sizing=sizing, comfort_room=comfort_room, min_risk_frac=min_risk_frac)
    bi, bj = np.unravel_index(np.nanargmax(pg), pg.shape)
    return {"opt_rr": rr_grid[bj], "opt_risk_pct": float(risk_pcts[bi]),
            "opt_pass_pct": float(pg[bi, bj]), "opt_speed": float(sg[bi, bj]),
            "opt_wr": wr[rr_grid[bj]], "pass_grid": pg, "speed_grid": sg,
            "risk_pcts": risk_pcts, "rr_grid": rr_grid}


def risk_rr_heatmap(firm, entries, risk_pcts=None, rr_grid=None, n_paths=2000,
                    max_trades=300, seed=0, savepath=None,
                    sizing="flat", comfort_room=None, min_risk_frac=0.10):
    g = optimal_geometry(firm, entries, risk_pcts, rr_grid, n_paths, max_trades, seed,
                         sizing=sizing, comfort_room=comfort_room, min_risk_frac=min_risk_frac)
    pg, sg, rps, rrs = g["pass_grid"], g["speed_grid"], g["risk_pcts"], g["rr_grid"]
    wr = {rr: measured_winrate(entries, rr) for rr in rrs}
    fig, ax = plt.subplots(figsize=(1.6*len(rrs)+2, 0.5*len(rps)+2))
    im = ax.imshow(pg, cmap=_HEAT, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(rrs))); ax.set_xticklabels([f"{rr:g}R\n{wr[rr]:.0%} WR" for rr in rrs])
    ax.set_yticks(range(len(rps)))
    ax.set_yticklabels([f"{p:.2f}%\n(${firm.start_balance*p/100:.0f})" for p in rps], fontsize=8)
    ax.set_xlabel("R:R  (win rate MEASURED from data)"); ax.set_ylabel("Risk per trade")
    for i in range(len(rps)):
        for j in range(len(rrs)):
            v = pg[i, j]; s = "" if np.isnan(sg[i, j]) else f"\n{sg[i,j]:.0f} trd"
            ax.text(j, i, f"{v:.0f}%{s}", ha="center", va="center", fontsize=7,
                    color="white" if v < 55 else "black")
    fig.colorbar(im, ax=ax, label="Pass rate (%)")
    mode = "flat" if sizing == "flat" else f"room-scaled (comfort=${comfort_room})"
    ax.set_title(f"Optimal risk geometry — {firm.name} [{mode}]\nbest: {g['opt_rr']:g}R @ "
                 f"{g['opt_risk_pct']:.2f}% risk = {g['opt_pass_pct']:.0f}% pass "
                 f"(~{g['opt_speed']:.0f} trades)", fontweight="bold", fontsize=10)
    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig, g


def _eval_pass(strategy, firm, risk_pct, n_paths, max_trades, seeds):
    ps = [simulate(strategy, firm, risk_per_trade=firm.start_balance*risk_pct/100.0,
                   n_paths=n_paths, max_trades=max_trades, seed=s).pass_rate*100 for s in seeds]
    ps = np.array(ps); return ps.mean(), ps.std(ddof=1)/np.sqrt(len(ps))


def find_optimal_risk(strategy, firm, risk_pcts=None, n_paths=4000, max_trades=400,
                      seed=0, seeds=None, tie_sd=1.0,
                      sizing="flat", comfort_room=None, min_risk_frac=0.10):
    if risk_pcts is None: risk_pcts = np.round(np.arange(0.10, 4.01, 0.10), 2)
    seeds = range(seed, seed+5) if seeds is None else seeds
    prs, sps, frs = [], [], []
    for s in seeds:
        pr_s, sp_s, fr_s = sweep_risk(strategy, firm, risk_pcts, n_paths, max_trades, s,
                                      sizing=sizing, comfort_room=comfort_room, min_risk_frac=min_risk_frac)
        prs.append(pr_s); sps.append(sp_s); frs.append(fr_s)
    prs = np.array(prs); pr = prs.mean(axis=0); pr_se = prs.std(axis=0, ddof=1)/np.sqrt(len(seeds))
    sp = np.nanmean(np.array(sps), axis=0); fr = np.array(frs).mean(axis=0)
    best = int(pr.argmax()); cutoff = pr[best]-tie_sd*pr_se[best]; tied = np.flatnonzero(pr >= cutoff)
    speeds = np.where(np.isnan(sp[tied]), np.inf, sp[tied]); b = tied[int(np.argmin(speeds))]
    return {"risk_pcts": risk_pcts, "pass_pct": pr, "pass_se": pr_se, "speed": sp, "fail_pct": fr,
            "tied_idx": tied, "tied_risk_pcts": risk_pcts[tied].tolist(),
            "opt_risk_pct": float(risk_pcts[b]), "opt_risk_dollars": float(firm.start_balance*risk_pcts[b]/100.0),
            "opt_pass_pct": float(pr[b]), "opt_speed": float(sp[b])}


def plot_risk_curve(strategy, firm, risk_pcts=None, n_paths=4000, max_trades=400, seed=0,
                    savepath=None, sizing="flat", comfort_room=None, min_risk_frac=0.10):
    r = find_optimal_risk(strategy, firm, risk_pcts, n_paths, max_trades, seed,
                          sizing=sizing, comfort_room=comfort_room, min_risk_frac=min_risk_frac)
    rp, pr, sp = r["risk_pcts"], r["pass_pct"], r["speed"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rp, pr, "-o", color="#2e86de", ms=3, label="Pass rate")
    ax.axvline(r["opt_risk_pct"], color="#e67e22", ls="--", label=f"optimum {r['opt_risk_pct']:.2f}%")
    ax.scatter([r["opt_risk_pct"]],[r["opt_pass_pct"]], color="#e67e22", zorder=5)
    ax.set_xlabel("Risk per trade (%)"); ax.set_ylabel("Pass rate (%)", color="#2e86de"); ax.set_ylim(0,100)
    ax2 = ax.twinx(); ax2.plot(rp, sp, color="#16a085", alpha=.6); ax2.set_ylabel("Avg trades to pass", color="#16a085")
    ax.set_title(f"Risk optimization — {firm.name} [{strategy.name}]", fontweight="bold")
    ax.legend(loc="lower left", fontsize=8); fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig, r


def optimal_risk_by_firm(strategy, firm_getter, firm_keys, risk_pcts=None, n_paths=3000,
                         max_trades=300, seed=0, savepath=None,
                         sizing="flat", comfort_room=None, min_risk_frac=0.10):
    results = {k: find_optimal_risk(strategy, firm_getter(k), risk_pcts, n_paths, max_trades, seed,
               sizing=sizing, comfort_room=comfort_room, min_risk_frac=min_risk_frac) for k in firm_keys}
    names = [firm_getter(k).name for k in firm_keys]
    opt_r = [results[k]["opt_risk_pct"] for k in firm_keys]; opt_p = [results[k]["opt_pass_pct"] for k in firm_keys]
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(14, 0.7*len(names)+2))
    ax[0].barh(y, opt_p, color="#2e86de")
    for yi, v, rr in zip(y, opt_p, opt_r):
        ax[0].text(v+0.5, yi, f"{v:.1f}%  @ {rr:.2f}% risk", va="center", fontsize=9, fontweight="bold")
    ax[0].set_yticks(y); ax[0].set_yticklabels(names); ax[0].set_xlim(0,100)
    ax[0].set_xlabel("Best achievable pass rate (%)"); ax[0].set_title(f"Optimal risk & pass by firm\n[{strategy.name}]")
    ax[1].barh(y, opt_r, color="#e67e22")
    for yi, v in zip(y, opt_r): ax[1].text(v+0.02, yi, f"{v:.2f}%", va="center", fontsize=9)
    ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].set_xlabel("Optimal risk per trade (%)")
    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return results, fig


def days_to_pass_by_firm(strategy, firm_getter, firm_keys, risk_pcts=None, n_paths=3000,
                         max_trades=300, seed=0, dist_paths=20000, min_passers=5, savepath=None,
                         sizing="flat", comfort_room=None, min_risk_frac=0.10):
    results = {k: find_optimal_risk(strategy, firm_getter(k), risk_pcts, n_paths, max_trades, seed,
               sizing=sizing, comfort_room=comfort_room, min_risk_frac=min_risk_frac) for k in firm_keys}
    firms = [firm_getter(k) for k in firm_keys]; names = [f.name for f in firms]
    boxstats, medians, passrates, ns = [], [], [], []
    for k, f in zip(firm_keys, firms):
        rpct = results[k]["opt_risk_pct"]
        sim = simulate(strategy, f, risk_per_trade=rpct/100.0*f.start_balance, n_paths=dist_paths,
                       max_trades=max_trades, seed=seed, sizing=sizing,
                       comfort_room=comfort_room, min_risk_frac=min_risk_frac)
        d = sim.days_to_pass[sim.outcome == 1].astype(float)
        passrates.append(100.0*float(np.mean(sim.outcome == 1))); ns.append(int(d.size))
        if d.size >= min_passers:
            p10,q1,med,q3,p90 = np.percentile(d,[10,25,50,75,90])
        elif d.size >= 1:
            med=float(np.median(d)); q1=q3=med; p10,p90=float(d.min()),float(d.max())
        else: med=q1=q3=p10=p90=np.nan
        medians.append(med); boxstats.append(dict(med=med,q1=q1,q3=q3,whislo=p10,whishi=p90,fliers=[]))
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(14, 0.7*len(names)+2))
    valid = [i for i,s in enumerate(boxstats) if np.isfinite(s["med"])]
    if valid:
        ax[0].bxp([boxstats[i] for i in valid], positions=[y[i] for i in valid], vert=False,
                  showfliers=False, widths=0.55, patch_artist=True,
                  boxprops=dict(facecolor="#2e86de", alpha=0.65), medianprops=dict(color="black", linewidth=2))
    for i in range(len(names)):
        if np.isfinite(medians[i]):
            ax[0].text(boxstats[i]["whishi"]+0.4, y[i], f"med {medians[i]:.0f}d  (n={ns[i]:,})", va="center", fontsize=8, fontweight="bold")
    ax[0].set_yticks(y); ax[0].set_yticklabels(names); ax[0].set_xlim(left=0)
    ax[0].set_xlabel("Days to pass (box=IQR, whiskers=p10-p90)"); ax[0].set_title(f"Days-to-pass by firm\n[{strategy.name}]")
    ax[1].barh(y, passrates, color="#e67e22")
    for yi, v in zip(y, passrates): ax[1].text(v+0.5, yi, f"{v:.1f}%", va="center", fontsize=9)
    ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].set_xlim(0,100); ax[1].set_xlabel("Pass rate (%)")
    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=130, bbox_inches="tight")
    # DataFrame, not a plain dict -- every caller does summary.to_string(...)
    summary = pd.DataFrame({"names": names, "medians": medians,
                            "passrates": passrates, "n_passers": ns})
    return results, fig, summary


def plot_equity_paths(firm, strategy, risk_per_trade, n_show=120, max_trades=300, seed=0,
                      savepath=None, sizing="flat", comfort_room=None, min_risk_frac=0.10):
    res = simulate(strategy, firm, risk_per_trade, n_paths=n_show, max_trades=max_trades,
                   seed=seed, record_paths=True, sizing=sizing, comfort_room=comfort_room, min_risk_frac=min_risk_frac)
    fig, ax = plt.subplots(figsize=(9, 5))
    for p in range(n_show):
        color = {1:"#2e8b57", -1:"#c0392b", 0:"#999999"}[res.outcome[p]]
        ax.plot(res.equity_paths[p], color=color, alpha=.35, lw=.8)
    ax.axhline(firm.target_equity, color="#1e7a2e", lw=1.5); ax.axhline(firm.initial_floor, color="#8a1f1f", lw=1.5)
    ax.axhline(firm.start_balance, color="k", ls="--", lw=.8, alpha=.6)
    ax.set_xlabel("Trade #"); ax.set_ylabel("Equity ($)")
    ax.set_title(f"Equity paths — {firm.name} [{strategy.name}] risk ${risk_per_trade:.0f}")
    if savepath: fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig

