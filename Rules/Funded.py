"""
Funded.py — Monte Carlo of the FUNDED (post-eval) stage as an income process.

Where Engine.simulate models the eval as a first-passage problem (reach target
ONCE), the funded stage is a renewal-reward process:

    trade -> accumulate profit -> satisfy a PAYOUT GATE
          -> withdraw (funds LEAVE the account) -> reset cycle counters -> repeat

The only absorbing event is breaching the trailing drawdown, which ends ALL
future income. So the objective is not P(pass) but the DISTRIBUTION of total
dollars withdrawn over a fixed horizon (one or two months of trading days),
and its downside (p25 / P(blow-up)) — because higher risk keeps paying until
it wipes the account and every future payout with it.

Reuses the upstream stack unchanged:
  * strategy.draw(N, T, rng) -> (R, new_day)   (whole-day resampling => real
    winning-day distribution, real trades/day; nothing about days assumed)
  * DrawdownType / DLLBehavior from Firms.py

FUNDED configs below are derived from your rule .md files. Anything the markdown
left ambiguous is flagged `# VERIFY` — those choices move pass/income a lot.
The 50%-per-payout rule you asked for is the DEFAULT (payout_pct=0.50); it is a
parameter, not baked in, so you can sweep it or swap in each firm's real caps.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from Firms import DrawdownType, DLLBehavior
from Strategy import Strategy


# ---------------------------------------------------------------------------
# Funded-stage rule config  (one per firm/path; move into Firms.py when happy)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FundedConfig:
    key: str
    name: str
    start_balance: float                     # 50_000 (Apex/Lucid/Tradeify) or 0 (MFFU/Topstep)
    max_drawdown: float = 2_000.0
    dd_type: DrawdownType = DrawdownType.EOD_TRAIL
    trail_lock: float | None = None          # equity level trailing floor locks at

    # --- withdrawal gate -----------------------------------------------------
    withdraw_buffer: float = 52_100.0        # only equity ABOVE this is withdrawable ("safety net")
    min_daily_profit: float = 150.0          # a day counts as "winning" if day_pnl >= this
    winning_days_required: int = 5           # winning days needed to unlock a payout
    min_days_required: int = 1               # distinct trading days (consistency-path firms)
    consistency_pct: float | None = None     # best day <= pct * cycle profit at payout (None = off)
    min_cycle_profit: float = 0.0            # min profit accrued since last payout to request (Lucid $500)

    # --- payout mechanics ----------------------------------------------------
    payout_pct: float = 0.50                 # fraction of the withdrawable base taken per payout
    payout_base_is_balance: bool = False     # base = (equity-buffer) profit [default] or full balance
    payout_cap: float | None = None          # hard $ cap per request (None = uncapped by dollars)
    profit_split: float = 0.90               # fraction the TRADER keeps (Apex=1.00, others 0.80-0.90)
    reset_floor_on_payout: bool = False      # Topstep/MFFU: MLL re-anchors to equity after a payout
    max_payouts: int | None = None           # Apex EOD PA caps at 6

    @property
    def floor0(self) -> float:
        return self.start_balance - self.max_drawdown


# 50K funded configs. VERIFY every trailing-lock and split against the live docs.
FUNDED: dict[str, FundedConfig] = {

    "apex_eod": FundedConfig(
        key="apex_eod", name="Apex PA (EOD)", start_balance=50_000,
        dd_type=DrawdownType.EOD_TRAIL, trail_lock=52_100,  # VERIFY: DD stops trailing at DDmax+$100
        withdraw_buffer=52_100, min_daily_profit=250, winning_days_required=5,
        consistency_pct=0.50, profit_split=1.00, payout_cap=3_000, max_payouts=6,
    ),
    "apex_intraday": FundedConfig(
        key="apex_intraday", name="Apex PA (Intraday)", start_balance=50_000,
        dd_type=DrawdownType.INTRADAY_TRAIL, trail_lock=52_100,  # VERIFY
        withdraw_buffer=52_100, min_daily_profit=200, winning_days_required=5,
        consistency_pct=0.50, profit_split=1.00, payout_cap=3_000, max_payouts=6,
    ),
    "lucid": FundedConfig(
        key="lucid", name="Lucid (funded)", start_balance=50_000,
        dd_type=DrawdownType.EOD_TRAIL, trail_lock=52_100,  # VERIFY (docs unclear trail vs static)
        withdraw_buffer=52_100, min_daily_profit=0, winning_days_required=1,  # cycle-based, not winning-days
        min_cycle_profit=500, consistency_pct=0.40, profit_split=0.90, payout_cap=2_500,
    ),
    "tradeify_growth": FundedConfig(
        key="tradeify_growth", name="Tradeify Growth (funded)", start_balance=50_000,
        dd_type=DrawdownType.EOD_TRAIL, trail_lock=52_100,  # VERIFY
        withdraw_buffer=53_000, min_daily_profit=150, winning_days_required=5,
        consistency_pct=0.35, profit_split=0.90, payout_cap=3_000,
    ),
    "mffu_rapid": FundedConfig(
        key="mffu_rapid", name="MFFU Rapid (funded)", start_balance=0,
        dd_type=DrawdownType.INTRADAY_TRAIL, trail_lock=100,  # locks at +$100 above the $0 start
        withdraw_buffer=2_100, min_daily_profit=0, winning_days_required=1,  # daily, no consistency
        consistency_pct=None, profit_split=0.90, reset_floor_on_payout=True,
    ),
    "topstep_std": FundedConfig(
        key="topstep_std", name="Topstep XFA (Standard)", start_balance=0,
        dd_type=DrawdownType.EOD_TRAIL, trail_lock=None,  # trailing MLL; VERIFY EOD vs intraday
        withdraw_buffer=0, min_daily_profit=150, winning_days_required=5,
        consistency_pct=None, profit_split=0.90, payout_cap=5_000,
        reset_floor_on_payout=True,  # MLL resets to $0 after each payout
    ),
    "topstep_cons": FundedConfig(
        key="topstep_cons", name="Topstep XFA (Consistency)", start_balance=0,
        dd_type=DrawdownType.EOD_TRAIL, trail_lock=None,
        withdraw_buffer=0, min_daily_profit=0.01, winning_days_required=3,  # 3 days with a trade
        consistency_pct=0.40, profit_split=0.90, payout_cap=6_000,
        reset_floor_on_payout=True,
    ),
}


def get_funded(key: str) -> FundedConfig:
    if key not in FUNDED:
        raise KeyError(f"Unknown funded config '{key}'. Options: {list(FUNDED)}")
    return FUNDED[key]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class FundedResult:
    withdrawn: np.ndarray      # net $ received by trader over the horizon (after split)
    n_payouts: np.ndarray      # payouts taken per path
    blew_up: np.ndarray        # bool: hit drawdown before horizon end
    blow_day: np.ndarray       # day index of blow-up (-1 if survived)
    end_equity: np.ndarray     # account equity at horizon end
    horizon_days: int

    @property
    def blowup_rate(self):  return float(np.mean(self.blew_up))
    @property
    def mean_withdrawn(self): return float(np.mean(self.withdrawn))
    @property
    def median_withdrawn(self): return float(np.median(self.withdrawn))
    @property
    def p25_withdrawn(self): return float(np.percentile(self.withdrawn, 25))
    @property
    def any_payout_rate(self): return float(np.mean(self.n_payouts > 0))

    def describe(self):
        return (f"{self.name if hasattr(self,'name') else ''}"
                f"E[$]={self.mean_withdrawn:,.0f}  med=${self.median_withdrawn:,.0f}  "
                f"p25=${self.p25_withdrawn:,.0f}  blow-up={self.blowup_rate:.1%}  "
                f"≥1 payout={self.any_payout_rate:.1%}  "
                f"avg payouts={self.n_payouts.mean():.2f}")


# ---------------------------------------------------------------------------
# The funded simulation  (renewal-reward over a fixed horizon)
# ---------------------------------------------------------------------------
def simulate_funded(strategy: Strategy, fc: FundedConfig, risk_per_trade: float,
                    horizon_days: int = 42, n_paths: int = 5_000,
                    max_trades: int = 1_200, seed: int = 0,
                    greedy_payout: bool = True) -> FundedResult:
    """Simulate N funded accounts for `horizon_days` trading days.

    greedy_payout=True withdraws the moment the gate is satisfied (income-max
    policy). Set False to model 'let it ride and withdraw once at the end'.

    horizon_days: ~21 = one month, ~42 = two months (trading days).
    Trades/day are EMERGENT from the strategy's whole-day resampling, so pick
    max_trades comfortably above horizon_days * (avg trades/day).
    """
    rng = np.random.default_rng(seed)
    N, T = n_paths, max_trades
    R, new_day = strategy.draw(N, T, rng)
    if new_day is None:
        raise ValueError("Funded model needs day-aware draws. Build the "
                         "EmpiricalStrategy with day_ids= from ledger_for_rr().")
    pnl = R * risk_per_trade

    start = float(fc.start_balance)   # float so equity arrays don't inherit int dtype
    dd = float(fc.max_drawdown)
    lock = fc.trail_lock

    equity   = np.full(N, start)
    eod_peak = np.full(N, start)
    peak     = np.full(N, start)
    dd_level = np.full(N, start - dd)
    day_pnl  = np.zeros(N)

    # cycle counters (reset after each payout)
    win_days  = np.zeros(N, dtype=int)     # winning days since last payout
    days_seen = np.zeros(N, dtype=int)     # distinct trading days since last payout
    best_day  = np.full(N, -np.inf)        # best single day in cycle (consistency numerator)
    cyc_profit = np.zeros(N)               # profit accrued since last payout

    withdrawn = np.zeros(N)
    n_payouts = np.zeros(N, dtype=int)
    day_index = np.zeros(N, dtype=int)
    blew_up   = np.zeros(N, dtype=bool)
    blow_day  = np.full(N, -1, dtype=int)
    alive     = np.ones(N, dtype=bool)

    is_intraday = fc.dd_type == DrawdownType.INTRADAY_TRAIL
    is_eod      = fc.dd_type == DrawdownType.EOD_TRAIL

    for t in range(T):
        live = alive
        equity[live] += pnl[live, t]
        day_pnl[live] += pnl[live, t]

        # intraday trailing floor updates on every bar
        if is_intraday:
            np.maximum(peak, equity, out=peak, where=live)
            nl = peak - dd
            if lock is not None:
                nl = np.minimum(nl, lock)
            np.maximum(dd_level, nl, out=dd_level, where=live)
            breach = live & (equity <= dd_level + 1e-9)
            blew_up[breach] = True; blow_day[breach] = day_index[breach]; alive[breach] = False

        # day boundary?
        natural_end = new_day[:, t + 1] if t + 1 < T else np.ones(N, bool)
        day_end = alive & natural_end
        if day_end.any():
            # 1) EOD trailing floor + breach check
            if is_eod:
                np.maximum(eod_peak, equity, out=eod_peak, where=day_end)
                nl = eod_peak - dd
                if lock is not None:
                    nl = np.minimum(nl, lock)
                np.maximum(dd_level, nl, out=dd_level, where=day_end)
                breach = day_end & (equity <= dd_level + 1e-9)
                blew_up[breach] = True; blow_day[breach] = day_index[breach]; alive[breach] = False
                day_end = day_end & alive

            # 2) cycle accounting
            np.maximum(best_day, day_pnl, out=best_day, where=day_end)
            cyc_profit[day_end] += day_pnl[day_end]
            days_seen[day_end] += 1
            win_days[day_end & (day_pnl >= fc.min_daily_profit)] += 1
            day_index[day_end] += 1

            # 3) payout gate
            cons_ok = np.ones(N, bool)
            if fc.consistency_pct is not None:
                cons_ok = best_day <= fc.consistency_pct * np.maximum(cyc_profit, 1e-9) + 1e-9
            gate = (day_end
                    & (win_days >= fc.winning_days_required)
                    & (days_seen >= fc.min_days_required)
                    & (equity - fc.withdraw_buffer > 1e-9)
                    & (cyc_profit >= fc.min_cycle_profit - 1e-9)
                    & cons_ok)
            if fc.max_payouts is not None:
                gate &= n_payouts < fc.max_payouts

            if greedy_payout and gate.any():
                base = equity - fc.withdraw_buffer
                if fc.payout_base_is_balance:
                    base = equity
                amt = fc.payout_pct * base
                amt = np.minimum(amt, base)                      # never exceed profit-above-buffer
                if fc.payout_cap is not None:
                    amt = np.minimum(amt, fc.payout_cap)
                amt = np.where(gate, np.maximum(amt, 0.0), 0.0)

                withdrawn[gate] += amt[gate] * fc.profit_split   # trader keeps split
                equity[gate]    -= amt[gate]                     # funds leave the account
                n_payouts[gate] += 1

                # re-anchor trailing floor for firms that reset MLL on payout
                if fc.reset_floor_on_payout:
                    eod_peak[gate] = equity[gate]; peak[gate] = equity[gate]
                    nl_new = equity[gate] - dd
                    if lock is not None:
                        nl_new = np.minimum(nl_new, lock)
                    dd_level[gate] = nl_new

                # reset cycle counters
                win_days[gate] = 0; days_seen[gate] = 0
                best_day[gate] = -np.inf; cyc_profit[gate] = 0.0

            # 4) close the day
            day_pnl[day_end | (~alive & natural_end)] = 0.0
            day_pnl[natural_end] = 0.0

        # stop at horizon
        reached_horizon = day_index >= horizon_days
        alive &= ~reached_horizon
        if not alive.any():
            break

    # end-of-horizon: non-greedy takes its one payout here (best-effort, ungated by winning days)
    if not greedy_payout:
        base = np.maximum(equity - fc.withdraw_buffer, 0.0)
        amt = fc.payout_pct * base
        if fc.payout_cap is not None:
            amt = np.minimum(amt, fc.payout_cap)
        withdrawn += amt * fc.profit_split

    res = FundedResult(withdrawn, n_payouts, blew_up, blow_day, equity, horizon_days)
    res.name = f"{fc.name} @ ${risk_per_trade:.0f}/trade | "
    return res


# ---------------------------------------------------------------------------
# Risk x R:R payout-geometry optimizer  (maximize the DOWNSIDE, not the mean)
# ---------------------------------------------------------------------------
def payout_geometry(fc: FundedConfig, entries, risk_pcts=None, rr_grid=None,
                    horizon_days=42, n_paths=3_000, max_trades=1_200, seed=0,
                    objective="p25"):
    """Sweep risk% x R:R; score each cell by a downside stat of $ withdrawn.

    entries: a Backtest.Entries object (scorable at any rr via ledger_for_rr).
    objective: 'p25' (robust), 'median', or 'mean' of net dollars withdrawn.
    Returns grids + the single best cell.  Mirrors Analysis.optimal_geometry.
    """
    from Rules.strategies.superseded.Backtest import ledger_for_rr
    from Strategy import EmpiricalStrategy
    if risk_pcts is None:
        risk_pcts = np.round(np.arange(0.25, 2.51, 0.25), 2)
    if rr_grid is None:
        rr_grid = [1.0, 1.5, 2.0, 3.0]

    ref = fc.start_balance if fc.start_balance > 0 else 50_000  # $0-start firms size off 50K
    score = np.zeros((len(risk_pcts), len(rr_grid)))
    blow  = np.zeros_like(score)
    meang = np.zeros_like(score)
    for j, rr in enumerate(rr_grid):
        R, days = ledger_for_rr(entries, rr)
        strat = EmpiricalStrategy(R, name=f"{rr}R", day_ids=days)
        for i, rp in enumerate(risk_pcts):
            res = simulate_funded(strat, fc, ref * rp / 100.0, horizon_days=horizon_days,
                                  n_paths=n_paths, max_trades=max_trades, seed=seed)
            meang[i, j] = res.mean_withdrawn
            blow[i, j]  = res.blowup_rate * 100
            score[i, j] = {"p25": res.p25_withdrawn, "median": res.median_withdrawn,
                           "mean": res.mean_withdrawn}[objective]
    bi, bj = np.unravel_index(np.argmax(score), score.shape)
    return {"score_grid": score, "mean_grid": meang, "blowup_grid": blow,
            "risk_pcts": risk_pcts, "rr_grid": rr_grid, "objective": objective,
            "opt_rr": rr_grid[bj], "opt_risk_pct": float(risk_pcts[bi]),
            "opt_score": float(score[bi, bj]), "opt_mean": float(meang[bi, bj]),
            "opt_blowup_pct": float(blow[bi, bj])}


# ---------------------------------------------------------------------------
# Plotting  (styled like Analysis.risk_rr_heatmap — green = good)
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

_HEAT = LinearSegmentedColormap.from_list(
    "rag", ["#8a1f1f", "#c0392b", "#e67e22", "#c9b02a", "#4a9a3a", "#1e7a2e"])
_HEAT_R = LinearSegmentedColormap.from_list(         # reversed: green = LOW (good for blow-up)
    "gar", ["#1e7a2e", "#4a9a3a", "#c9b02a", "#e67e22", "#c0392b", "#8a1f1f"])


def plot_payout_heatmap(grid, fc, savepath=None):
    """Dual heatmap: LEFT = downside $ withdrawn (max), RIGHT = blow-up % (min)."""
    score, blow = grid["score_grid"], grid["blowup_grid"]
    rps, rrs, obj = grid["risk_pcts"], grid["rr_grid"], grid["objective"]
    ref = fc.start_balance if fc.start_balance > 0 else 50_000
    ylab = [f"{p:.2f}%\n(${ref*p/100:,.0f})" for p in rps]

    fig, axes = plt.subplots(1, 2, figsize=(2.4 * len(rrs) + 4, 0.6 * len(rps) + 2.5))
    im0 = axes[0].imshow(score, cmap=_HEAT, aspect="auto")
    axes[0].set_title(f"{obj} of net $ withdrawn  (higher = better)", fontsize=10)
    for i in range(len(rps)):
        for j in range(len(rrs)):
            axes[0].text(j, i, f"${score[i,j]:,.0f}", ha="center", va="center", fontsize=7)
    bi, bj = np.unravel_index(np.argmax(score), score.shape)
    axes[0].add_patch(plt.Rectangle((bj-.5, bi-.5), 1, 1, fill=False, ec="cyan", lw=2.5))
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    vmax = max(20.0, float(blow.max()))
    im1 = axes[1].imshow(blow, cmap=_HEAT_R, vmin=0, vmax=vmax, aspect="auto")
    axes[1].set_title("Blow-up rate over horizon (%)  (lower = better)", fontsize=10)
    for i in range(len(rps)):
        for j in range(len(rrs)):
            axes[1].text(j, i, f"{blow[i,j]:.0f}%", ha="center", va="center", fontsize=7,
                         color="white" if blow[i,j] > 12 else "black")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    for ax in axes:
        ax.set_xticks(range(len(rrs))); ax.set_xticklabels([f"{r:g}R" for r in rrs])
        ax.set_yticks(range(len(rps))); ax.set_yticklabels(ylab, fontsize=8)
        ax.set_xlabel("R:R"); ax.set_ylabel("Risk per trade")
    fig.suptitle(f"Funded payout geometry — {fc.name}   "
                 f"(cyan = best {obj}: {grid['opt_rr']:g}R @ {grid['opt_risk_pct']:.2f}% "
                 f"-> ${grid['opt_score']:,.0f}, {grid['opt_blowup_pct']:.0f}% blow-up)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig


def plot_withdrawn_dist(res, title=None, savepath=None):
    """Histogram of net $ withdrawn across paths, with mean/median/p25 marked."""
    w = res.withdrawn
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(w, bins=60, color="#2e86de", alpha=.75)
    for val, c, lab in [(res.mean_withdrawn, "#e67e22", "mean"),
                        (res.median_withdrawn, "#16a085", "median"),
                        (res.p25_withdrawn, "#8a1f1f", "p25")]:
        ax.axvline(val, color=c, ls="--", lw=1.6, label=f"{lab} ${val:,.0f}")
    ax.set_xlabel(f"Net $ withdrawn over {res.horizon_days} trading days")
    ax.set_ylabel("Paths")
    ax.set_title(title or f"Withdrawal distribution  "
                 f"(blow-up {res.blowup_rate:.1%}, ≥1 payout {res.any_payout_rate:.1%}, "
                 f"avg {res.n_payouts.mean():.2f} payouts)", fontweight="bold")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig


def compare_firms(entries, firm_keys, rr, risk_pct, horizon_days=42,
                  n_paths=4_000, max_trades=1_200, seed=0, savepath=None):
    """Same geometry across firms: bars of mean/p25 $ withdrawn + blow-up rate."""
    from Rules.strategies.superseded.Backtest import ledger_for_rr
    from Strategy import EmpiricalStrategy
    R, days = ledger_for_rr(entries, rr)
    strat = EmpiricalStrategy(R, day_ids=days)
    rows = []
    for k in firm_keys:
        fc = get_funded(k)
        ref = fc.start_balance if fc.start_balance > 0 else 50_000
        res = simulate_funded(strat, fc, ref * risk_pct / 100.0, horizon_days=horizon_days,
                              n_paths=n_paths, max_trades=max_trades, seed=seed)
        rows.append((fc.name, res.mean_withdrawn, res.p25_withdrawn, res.blowup_rate * 100))

    names = [r[0] for r in rows]
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(14, 0.7 * len(names) + 2))
    ax[0].barh(y + 0.18, [r[1] for r in rows], height=0.34, color="#2e86de", label="mean $")
    ax[0].barh(y - 0.18, [r[2] for r in rows], height=0.34, color="#16a085", label="p25 $")
    for yi, r in zip(y, rows):
        ax[0].text(r[1], yi + 0.18, f" ${r[1]:,.0f}", va="center", fontsize=8)
        ax[0].text(r[2], yi - 0.18, f" ${r[2]:,.0f}", va="center", fontsize=8)
    ax[0].set_yticks(y); ax[0].set_yticklabels(names)
    ax[0].set_xlabel(f"Net $ withdrawn over {horizon_days}d"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"Payout by firm @ {rr:g}R, {risk_pct:.2f}% risk")

    ax[1].barh(y, [r[3] for r in rows], color="#c0392b")
    for yi, r in zip(y, rows):
        ax[1].text(r[3], yi, f" {r[3]:.0f}%", va="center", fontsize=8)
    ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].set_xlim(left=0)
    ax[1].set_xlabel("Blow-up rate (%)"); ax[1].set_title("Risk of ruin over horizon")
    fig.suptitle("Funded income vs blow-up across firms", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig, rows