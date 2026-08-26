"""
MultiEval.py -- deterministic multi-account replay: given ONE real (already-
built) Entries ledger, simulate running N_ACCOUNTS eval accounts CONCURRENTLY
off the SAME signal stream, with the constraint that each account takes AT
MOST ONE trade per calendar trading day.

Why this is a different question from Engine.simulate(): Engine.simulate()
answers "what's the pass probability of ONE account trading this edge,
averaged over many resampled hypothetical futures" -- a statistical read on
a single account. This module answers a complementary, concrete question:
"if I actually buy N_ACCOUNTS evals and trade them all off the SAME real
signal stream, one trade per account per day, what would ACTUALLY have
happened over this specific historical period" -- a baseline to compare
real multi-account trading against, not a resampled distribution. It also
directly models the motivation for running several evals at once: a day
with multiple signals no longer stacks every one of them into a single
account's day (and therefore a single account's trailing-drawdown/DLL
exposure) -- it spreads them across separate accounts instead.

Trade-distribution rule: entries are walked in chronological order. Within
a calendar day, each new signal is assigned to the next LIVE account (not
yet passed/failed) that hasn't already traded today, following a rotating
pointer that carries over across days (so allocation stays fair over time,
not reset to account 0 every morning). If every live account has already
traded today, any further signal that day is skipped entirely -- mirrors
manually trading N_ACCOUNTS accounts and running out of hands, rather than
queuing the overflow to a later slot.

Each account only advances its OWN clock (equity/peak/day_index/etc -- the
same state machine Engine.simulate() runs per Monte Carlo path, just scalar
here instead of vectorized across N paths) on days it actually took a
trade. Because the one-trade/day cap means a trade IS that account's whole
trading day, every trade doubles as that account's day-end: there's no
separate "day boundary" event the way there is in Engine.simulate(), where
a single account can take several trades before its day closes.

Account replacement (restart_on_fail): a firm-rule BREACH (FAIL) is not the
end of the simulation for that SLOT -- a real trader who blows an eval buys
another one and keeps going. When a slot's account fails, a brand-new
account (fresh equity/drawdown floor/day counter, no memory of the old
one's history) immediately takes its place -- eligible for the very next
signal that slot's rotation turn comes up, even later the SAME calendar
day, since it has no trades of its own yet today. PASS and TIMEOUT do NOT
restart the slot (a passed account graduates and a timed-out one is simply
abandoned, both permanently vacate that slot for the rest of the window --
restart_on_fail's name is deliberately specific to breaches only). Every
attempt any slot ever makes -- however it ends -- is recorded in
MultiEvalResult.history, so a slot that blows through several accounts over
the OOS window shows up as several rows, and n_passed/n_blown/n_timeout
are tallies over ALL of them, not just the N_ACCOUNTS live at the end.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from Firms import FirmConfig, DrawdownType, DLLBehavior
from Backtest import Entries, ledger_for_rr


@dataclass
class AccountResult:
    idx: int                                 # slot index (0..n_accounts-1)
    attempt: int                             # 1-based attempt count for THIS slot (>1 means it's been restarted before)
    status: str                              # "PASS" / "FAIL" / "TIMEOUT" / "ONGOING"
    trades_taken: int
    trading_days: int                        # == day_index -- distinct days THIS attempt traded
    resolved_at: "pd.Timestamp | None"       # entry_time of the trade that resolved it (None if ONGOING)
    final_equity: float
    equity_curve: list                       # [(entry_time, equity_after_trade), ...] -- THIS attempt only, resets to start_balance on restart


@dataclass
class MultiEvalResult:
    history: list                            # list[AccountResult] -- EVERY attempt any slot ever made, in resolution order
    n_accounts: int                          # concurrent slots
    entries_used: int                        # entries actually assigned to some account
    entries_skipped: int                     # overflow signals dropped (every live slot already traded that day, or all slots permanently vacated)
    max_days: "int | None" = None            # the days-to-pass ceiling this replay was run with (None = no ceiling)
    restart_on_fail: bool = True  
    restart_on_pass: bool = True             # whether a breached slot was immediately replaced with a fresh account

    @property
    def current(self) -> list:
        """The most recent attempt per slot -- what's occupying it right now
        (ONGOING if still live, or a frozen PASS/TIMEOUT that was never
        replaced, since only FAIL restarts)."""
        last = {}
        for a in self.history:
            last[a.idx] = a
        return [last[i] for i in range(self.n_accounts) if i in last]

    @property
    def n_attempts(self) -> int: return len(self.history)
    @property
    def n_passed(self) -> int: return sum(a.status == "PASS" for a in self.history)
    @property
    def n_blown(self) -> int: return sum(a.status == "FAIL" for a in self.history)
    @property
    def n_timeout(self) -> int: return sum(a.status == "TIMEOUT" for a in self.history)
    @property
    def n_ongoing(self) -> int: return sum(a.status == "ONGOING" for a in self.history)
    @property
    def pass_rate(self) -> float:
        """PASSED / (PASSED + BLOWN) -- excludes TIMEOUT/ONGOING attempts,
        which didn't break any rule and aren't a 'loss' the way a breach is;
        counting them as failures would understate how often a clean shot
        actually converts. Returns 0.0 if no attempt has resolved either way."""
        resolved = self.n_passed + self.n_blown
        return self.n_passed / resolved if resolved else 0.0

    def summary(self) -> str:
        ceiling_note = f", max_days={self.max_days}" if self.max_days is not None else ""
        restart_note = "restart-on-breach ON" if self.restart_on_fail else "no restart"
        return (f"{self.n_accounts} concurrent slots ({restart_note}{ceiling_note}): "
                f"{self.n_attempts} total account attempts -> {self.n_passed} PASSED, "
                f"{self.n_blown} BLOWN, {self.n_timeout} TIMEOUT, {self.n_ongoing} still ONGOING | "
                f"{self.entries_used} entries used, {self.entries_skipped} skipped (overflow)")


def multi_account_replay(entries: Entries, firm: FirmConfig, risk_pct: float, rr: float,
                         n_accounts: int = 3, max_days: "int | None" = None,
                         restart_on_fail: bool = True, restart_on_pass: bool = True) -> MultiEvalResult:
    
    """Deterministic, single-pass replay -- see module docstring for the
    trade-distribution rule, the restart-on-breach mechanic, and why every
    trade doubles as its account's day-end.

    `max_days`: the SAME "how many trading days is this realistically
    allowed to take" patience ceiling Section 1's `risk_geometry_3d(...,
    max_days_to_pass=...)` uses. An attempt whose `day_index` (==
    `trading_days`) reaches `max_days` without having already passed on
    that same trade is marked "TIMEOUT" and permanently vacates its slot
    (TIMEOUT does not restart, even with restart_on_fail=True -- giving up
    on a slow account isn't the same event as it breaching a rule). None
    (default) means no ceiling.

    `restart_on_fail`: when True (default), a slot whose account FAILS
    (breaches a firm rule) is immediately reset to a brand-new account and
    keeps trading through the rest of the data -- models "buy another eval
    and keep going" rather than abandoning that slot. Set False to freeze a
    slot the moment it breaches (the old, pre-restart behavior).

    `entries` must have been built with backtest_entries(..., detail=True)
    (needs entry_time for chronological/day-boundary ordering and for the
    equity-curve timestamps). Scores each entry with ledger_for_rr()'s exact
    win/loss/scratch rule at `rr` -- the same function every other pass-rate
    number in this project is built from, so this stays consistent with
    Analysis_points.risk_geometry_3d()'s and Engine.simulate()'s grading."""

    if entries.entry_time is None:
        raise ValueError("multi_account_replay() needs entries built with "
                         "backtest_entries(..., detail=True) -- entry_time missing.")

    R, day_ids = ledger_for_rr(entries, rr)
    risk_per_trade = firm.start_balance * risk_pct / 100.0

    is_intraday = firm.dd_type == DrawdownType.INTRADAY_TRAIL
    is_eod_trail = firm.dd_type == DrawdownType.EOD_TRAIL
    has_dll = firm.daily_loss_limit is not None and firm.dll_behavior != DLLBehavior.NONE

    n = n_accounts
    equity = np.full(n, firm.start_balance)
    peak = np.full(n, firm.start_balance)
    eod_peak = np.full(n, firm.start_balance)
    dd_level = np.full(n, firm.start_balance - firm.max_drawdown)
    day_pnl = np.zeros(n)
    best_day = np.full(n, -np.inf)
    day_index = np.zeros(n, dtype=int)
    trades_taken = np.zeros(n, dtype=int)
    status = np.array(["ONGOING"] * n, dtype=object)
    attempt_num = np.ones(n, dtype=int)              # 1-based -- which account THIS slot is currently on
    curves = [[] for _ in range(n)]
    last_traded_day = np.full(n, -1, dtype=int)      # day_id of THIS attempt's most recent trade

    history: list[AccountResult] = []

    def _reset_slot(i: int) -> None:
        """Wipe slot i back to a brand-new account -- no memory of the
        failed one's equity, drawdown floor, or trade history."""
        equity[i] = firm.start_balance
        peak[i] = firm.start_balance
        eod_peak[i] = firm.start_balance
        dd_level[i] = firm.start_balance - firm.max_drawdown
        day_pnl[i] = 0.0
        best_day[i] = -np.inf
        day_index[i] = 0
        trades_taken[i] = 0
        status[i] = "ONGOING"
        curves[i] = []
        last_traded_day[i] = -1

    def _finalize(i: int, new_status: str, ts) -> None:
        status[i] = new_status
        history.append(AccountResult(i, int(attempt_num[i]), new_status, int(trades_taken[i]),
                                     int(day_index[i]), ts, float(equity[i]), curves[i]))
        _restart = ((new_status == "FAIL" and restart_on_fail) or
                    (new_status in ("PASS", "TIMEOUT") and restart_on_pass))
        if _restart:
            attempt_num[i] += 1
            _reset_slot(i)   # immediately live again -- eligible for the very next signal

    pointer = 0
    entries_used = entries_skipped = 0

    for k in range(len(R)):
        day = int(day_ids[k])

        # Next LIVE slot, in rotating order, that hasn't traded today.
        acc = None
        for _ in range(n):
            cand = pointer % n
            pointer += 1
            if status[cand] == "ONGOING" and last_traded_day[cand] != day:
                acc = cand
                break
        if acc is None:
            entries_skipped += 1
            continue

        i = acc
        entries_used += 1
        pnl = R[k] * risk_per_trade
        equity[i] += pnl
        day_pnl[i] += pnl
        trades_taken[i] += 1
        last_traded_day[i] = day
        curves[i].append((entries.entry_time[k], float(equity[i])))

        # --- intraday-trailing / static: checked immediately, same as Engine.simulate ---
        if is_intraday:
            peak[i] = max(peak[i], equity[i])
            nl = peak[i] - firm.max_drawdown
            if firm.trail_lock is not None:
                nl = min(nl, firm.trail_lock)
            dd_level[i] = max(dd_level[i], nl)
        if firm.dd_type in (DrawdownType.STATIC, DrawdownType.INTRADAY_TRAIL) \
                and equity[i] <= dd_level[i] + 1e-9:
            _finalize(i, "FAIL", entries.entry_time[k])
            continue

        # --- daily loss limit ---
        if has_dll and day_pnl[i] <= -firm.daily_loss_limit + 1e-9 \
                and firm.dll_behavior == DLLBehavior.HARD_FAIL:
            _finalize(i, "FAIL", entries.entry_time[k])
            continue
        # SOFT_LOCK needs no extra handling: the one-trade/day cap already
        # means no further trade could happen today regardless.

        # --- this trade IS the account's day-end (one trade/day cap) ---
        best_day[i] = max(best_day[i], day_pnl[i])
        if is_eod_trail:
            eod_peak[i] = max(eod_peak[i], equity[i])
            nl = eod_peak[i] - firm.max_drawdown
            if firm.trail_lock is not None:
                nl = min(nl, firm.trail_lock)
            dd_level[i] = max(dd_level[i], nl)
        if firm.dd_type in (DrawdownType.EOD_TRAIL, DrawdownType.STATIC) \
                and equity[i] <= dd_level[i] + 1e-9:
            _finalize(i, "FAIL", entries.entry_time[k])
            continue

        day_index[i] += 1
        total_profit = equity[i] - firm.start_balance
        cons_ok = True
        if firm.consistency_pct is not None:
            cons_ok = best_day[i] <= firm.consistency_pct * total_profit + 1e-9
        if (equity[i] >= firm.target_equity - 1e-9 and day_index[i] >= firm.min_days and cons_ok):
            _finalize(i, "PASS", entries.entry_time[k])
        elif max_days is not None and day_index[i] >= max_days:
            _finalize(i, "TIMEOUT", entries.entry_time[k])

        # This trade WAS the account's whole day (one-trade/day cap) -- reset
        # day_pnl now so the next trade THIS ATTEMPT takes (a different day,
        # by construction) starts its own day_pnl from zero. Safe unconditionally:
        # a FAIL never reaches this line (it `continue`s above, and if restarted,
        # _reset_slot() already zeroed day_pnl for the new attempt); PASS/TIMEOUT
        # freeze the slot, so the reset is a harmless no-op for them.
        day_pnl[i] = 0.0

    # Whatever's still occupying each slot at the end of the data -- record it
    # too (as ONGOING) so `history` is a complete account of every attempt.
    for i in range(n):
        if status[i] == "ONGOING":
            history.append(AccountResult(i, int(attempt_num[i]), "ONGOING", int(trades_taken[i]),
                                         int(day_index[i]), None, float(equity[i]), curves[i]))

    return MultiEvalResult(history, n, entries_used, entries_skipped,
                           max_days=max_days, restart_on_fail=restart_on_fail,
                           restart_on_pass=restart_on_pass)


def plot_multi_eval(result: MultiEvalResult, firm: FirmConfig, savepath=None):
    """One equity curve per ATTEMPT (not per slot -- a slot that's been
    restarted several times draws several separate curves, each starting
    back at `firm.start_balance`), colored by outcome; linestyle cycles by
    slot so repeated restarts of the same slot are still visually groupable.
    Dashed lines mark the profit target and the STARTING drawdown floor
    (illustrative only -- the real floor moves for trailing-DD firms)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = {"PASS": "#1e7a2e", "FAIL": "#c0392b", "TIMEOUT": "#e67e22", "ONGOING": "#7f8c8d"}
    linestyles = ["-", "--", ":", "-."]   # cycles if n_accounts > 4
    for a in result.history:
        if not a.equity_curve:
            continue
        times = [t for t, _ in a.equity_curve]
        eq = [e for _, e in a.equity_curve]
        ax.plot(times, eq, color=colors[a.status], alpha=0.85, lw=1.5,
               linestyle=linestyles[a.idx % len(linestyles)],
               label=f"slot {a.idx} #{a.attempt}: {a.status} ({a.trades_taken}t)")
        ax.scatter([times[-1]], [eq[-1]], color=colors[a.status], s=40, zorder=5)
    ax.axhline(firm.target_equity, color="#1e7a2e", ls="--", lw=1, alpha=0.5, label="target")
    ax.axhline(firm.initial_floor, color="#c0392b", ls="--", lw=1, alpha=0.5, label="starting floor")
    ax.axhline(firm.start_balance, color="grey", ls=":", lw=1, alpha=0.5)
    ax.set_ylabel("Equity ($)"); ax.set_xlabel("Entry time")
    ax.set_title(f"Multi-eval replay -- {firm.name}, {result.n_accounts} concurrent slots\n{result.summary()}",
                fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig
