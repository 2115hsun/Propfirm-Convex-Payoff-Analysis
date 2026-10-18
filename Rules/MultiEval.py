"""
MultiEval.py: deterministic multi-account replay. Given one real Entries
ledger, run N eval accounts at the same time off the same signal stream,
each taking at most one trade per trading day.

Engine.simulate() gives the pass probability of one account averaged over
many resampled futures. This module replays what would have happened to N
real accounts over this specific period. It also shows why running several
evals helps: on a day with several signals, they go to separate accounts
instead of stacking into one account's drawdown and DLL.

Allocation: entries are walked in order. Each signal goes to the next live
account that hasn't traded today, following a rotating pointer that carries
over between days so no account is always first. Once every live account
has traded today, the day's remaining signals are skipped, not queued.

Each account runs the same state machine as one Engine.simulate() path,
scalar instead of vectorized, and only advances on days it trades. With one
trade per day, every trade is also that account's day-end.

Restarts: when an attempt ends, the slot can be refilled with a brand-new
account (fresh equity, floor and day count) that is eligible for the next
signal, even later the same day. restart_on_fail controls this for FAIL;
restart_on_pass controls it for PASS and TIMEOUT. Both default to True.
Every attempt is recorded in MultiEvalResult.history, so n_passed, n_blown
and n_timeout count all attempts, not just the accounts live at the end.
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
    attempt: int                             # 1-based attempt count for this slot (>1 = restarted before)
    status: str                              # "PASS" / "FAIL" / "TIMEOUT" / "ONGOING"
    trades_taken: int
    trading_days: int                        # == day_index, distinct days this attempt traded
    resolved_at: "pd.Timestamp | None"       # entry_time of the trade that resolved it (None if ONGOING)
    final_equity: float
    equity_curve: list                       # [(entry_time, equity_after_trade), ...] for this attempt only


@dataclass
class MultiEvalResult:
    history: list                            # list[AccountResult], every attempt by any slot, in resolution order
    n_accounts: int                          # concurrent slots
    entries_used: int                        # entries actually assigned to some account
    entries_skipped: int                     # overflow signals dropped (every live slot already traded that day, or all slots permanently vacated)
    max_days: "int | None" = None            # the days-to-pass ceiling this replay was run with (None = no ceiling)
    restart_on_fail: bool = True             # FAIL refills the slot with a fresh account
    restart_on_pass: bool = True             # PASS / TIMEOUT refill the slot with a fresh account

    @property
    def current(self) -> list:
        """The most recent attempt per slot: ONGOING if still live, or the
        final result of an attempt that wasn't restarted."""
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
        """PASSED / (PASSED + BLOWN). TIMEOUT and ONGOING attempts are left
        out because they broke no rule. Returns 0.0 if nothing has resolved."""
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
    """Single-pass replay; the module docstring covers allocation and restarts.

    max_days: the same patience ceiling as risk_geometry_3d(max_days_to_pass=).
    An attempt that reaches max_days trading days without passing on that
    trade is marked TIMEOUT. None (default) means no ceiling.

    restart_on_fail: refill a slot with a fresh account after a FAIL, i.e.
    buy another eval and keep going. False freezes the slot after a breach.

    restart_on_pass: the same for PASS and TIMEOUT.

    entries must come from backtest_entries(..., detail=True), since
    entry_time drives the day ordering and equity-curve timestamps. Each entry
    is scored with ledger_for_rr() at rr, the same grading Engine.simulate()
    and risk_geometry_3d() use."""

    if entries.entry_time is None:
        raise ValueError("multi_account_replay() needs entries built with "
                         "backtest_entries(..., detail=True) -- entry_time missing.")

    R, day_ids = ledger_for_rr(entries, rr)
    # intra-trade adverse excursion (R, positive) per trade; None on old ledgers
    mae = getattr(entries, 'mae_before', None)
    stopped_arr = getattr(entries, 'stopped', None)
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
        """Reset slot i to a brand-new account."""
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
            _reset_slot(i)   # live again for the next signal

    pointer = 0
    entries_used = entries_skipped = 0

    for k in range(len(R)):
        day = int(day_ids[k])

        # next live slot, in rotating order, that hasn't traded today
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

        # Intra-trade breach, for non-stopped trades only: a winner or timeout
        # that dipped through the floor mid-trade is liquidated on the spot.
        # Stopped trades go through the normal -1R path below; the ledger caps
        # their loss at 1R, so their excursion can't breach any deeper.
        is_stopped_k = (stopped_arr is not None and bool(stopped_arr[k]))
        if (mae is not None and not is_stopped_k
                and np.isfinite(mae[k]) and mae[k] > 0):
            intra_low = equity[i] - min(mae[k], 1.0) * risk_per_trade
            if intra_low <= dd_level[i] + 1e-9:
                trades_taken[i] += 1
                last_traded_day[i] = day
                curves[i].append((entries.entry_time[k], float(intra_low)))
                _finalize(i, "FAIL", entries.entry_time[k])
                continue

        pnl = R[k] * risk_per_trade
        equity[i] += pnl
        day_pnl[i] += pnl
        trades_taken[i] += 1
        last_traded_day[i] = day
        curves[i].append((entries.entry_time[k], float(equity[i])))

        # intraday-trailing / static floors are checked right away, as in Engine.simulate
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

        # daily loss limit
        if has_dll and day_pnl[i] <= -firm.daily_loss_limit + 1e-9 \
                and firm.dll_behavior == DLLBehavior.HARD_FAIL:
            _finalize(i, "FAIL", entries.entry_time[k])
            continue
        # SOFT_LOCK needs no extra handling: the one-trade/day cap already
        # means no further trade could happen today regardless.

        # with one trade per day, this trade is also the account's day-end
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

        # This trade was the whole day, so the next one starts day_pnl at zero.
        # A FAIL never reaches here (it `continue`s above), and a restarted
        # slot was already zeroed by _reset_slot().
        day_pnl[i] = 0.0

    # record attempts still live at the end as ONGOING, so history is complete
    for i in range(n):
        if status[i] == "ONGOING":
            history.append(AccountResult(i, int(attempt_num[i]), "ONGOING", int(trades_taken[i]),
                                         int(day_index[i]), None, float(equity[i]), curves[i]))

    return MultiEvalResult(history, n, entries_used, entries_skipped,
                           max_days=max_days, restart_on_fail=restart_on_fail,
                           restart_on_pass=restart_on_pass)


def plot_multi_eval(result: MultiEvalResult, firm: FirmConfig, savepath=None):
    """One equity curve per attempt, colored by outcome. A restarted slot
    draws several curves, each starting at firm.start_balance; linestyle
    follows the slot so its attempts can be grouped by eye. Dashed lines mark
    the profit target and the starting floor (the real floor moves for
    trailing-DD firms)."""
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
    ax.set_title(f"Multi-eval replay: {firm.name}, {result.n_accounts} concurrent slots\n{result.summary()}",
                fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig