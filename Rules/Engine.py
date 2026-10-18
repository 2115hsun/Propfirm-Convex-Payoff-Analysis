"""
Engine.py: vectorized Monte Carlo of prop-firm challenge paths.

Simulates N accounts in parallel, one trade at a time. Trades are grouped into
days because the day boundary drives EOD-trailing drawdown, daily loss limits,
the consistency rule and min-days.

Day boundaries come from the strategy when possible:
  * strategy.draw() returns new_day  -> real, variable trades/day
  * new_day is None                  -> fixed `trades_per_day` fallback

risk_per_trade is a dollar amount (a full -1R loss = -risk_per_trade). Callers
sweep it; the engine doesn't pick it.

Sizing modes:
  sizing="flat"         -> constant $ risk per trade.
  sizing="room_scaled"  -> risk scales with the drawdown room left. Full base
                           risk while room >= comfort_room, then linearly less
                           as equity nears the floor, never below
                           min_risk_frac * base.

Outcome codes: +1 PASS, -1 FAIL, 0 ONGOING.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from Firms import FirmConfig, DrawdownType, DLLBehavior
from Strategy import Strategy


@dataclass
class SimResult:
    outcome: np.ndarray
    trades_to_pass: np.ndarray
    days_to_pass: np.ndarray
    equity_paths: np.ndarray | None

    @property
    def pass_rate(self):    return float(np.mean(self.outcome == 1))
    @property
    def fail_rate(self):    return float(np.mean(self.outcome == -1))
    @property
    def ongoing_rate(self): return float(np.mean(self.outcome == 0))
    @property
    def mean_trades_to_pass(self):
        w = self.outcome == 1
        return float(self.trades_to_pass[w].mean()) if w.any() else float("nan")
    @property
    def mean_days_to_pass(self):
        w = self.outcome == 1
        return float(self.days_to_pass[w].mean()) if w.any() else float("nan")


def simulate(strategy: Strategy, firm: FirmConfig, risk_per_trade: float,
             n_paths=5_000, max_trades=400, trades_per_day=3, seed=0,
             record_paths=False,
             sizing="flat", comfort_room=None, min_risk_frac=0.10):
    rng = np.random.default_rng(seed)
    N, T = n_paths, max_trades
    R, new_day, mae = strategy.draw(N, T, rng)   # mae: adverse excursion (R>0) or None

    flat = (sizing == "flat")
    if flat:
        pnl_all = R * risk_per_trade          # precompute (constant risk)
    if comfort_room is None:                  # default: half the drawdown budget
        comfort_room = firm.max_drawdown * 0.5
    comfort_room = max(float(comfort_room), 1e-9)

    start = firm.start_balance
    target_eq = firm.target_equity
    dd = firm.max_drawdown
    lock = firm.trail_lock

    equity   = np.full(N, start, dtype=float)
    peak     = np.full(N, start, dtype=float)
    eod_peak = np.full(N, start, dtype=float)
    dd_level = np.full(N, start - dd, dtype=float)
    day_pnl  = np.zeros(N)
    best_day = np.full(N, -np.inf)
    trades_in_day = np.zeros(N, dtype=int)
    day_index = np.zeros(N, dtype=int)
    status = np.zeros(N, dtype=int)
    ttp = np.full(N, -1, dtype=int)
    dtp = np.full(N, -1, dtype=int)
    paths = np.empty((N, T)) if record_paths else None

    is_intraday = firm.dd_type == DrawdownType.INTRADAY_TRAIL
    is_eod_trail = firm.dd_type == DrawdownType.EOD_TRAIL
    has_dll = firm.daily_loss_limit is not None and firm.dll_behavior != DLLBehavior.NONE
    emergent = new_day is not None

    for t in range(T):
        live = status == 0

        # per-trade P&L (flat or room-scaled)
        if flat:
            pnl_t = pnl_all[:, t]
        else:
            room = equity - dd_level                       # $ to the floor, right now
            scale = np.clip(room / comfort_room, min_risk_frac, 1.0)
            pnl_t = R[:, t] * (risk_per_trade * scale)

        # Intra-trade breach: a dip through the floor liquidates the account even
        # if the trade later recovers.
        if mae is not None:
            if flat:
                risk_now = risk_per_trade
            else:
                room = equity - dd_level
                risk_now = risk_per_trade * np.clip(room / comfort_room, min_risk_frac, 1.0)
            # A -1R trade was stopped, so its loss is capped at 1R; cap its
            # excursion at 1R too, so a gap past the stop can't breach deeper.
            # Non-stopped trades use their actual mae.
            stopped_t = R[:, t] <= -1.0 + 1e-9
            mae_eff = np.where(stopped_t, np.minimum(mae[:, t], 1.0), mae[:, t])
            intra_low = equity - mae_eff * risk_now
            mae_breach = live & (mae_eff > 0) & (intra_low <= dd_level + 1e-9)
            status[mae_breach] = -1
            ttp[mae_breach] = -1
            dtp[mae_breach] = -1
            live = status == 0

        equity[live] += pnl_t[live]
        day_pnl[live] += pnl_t[live]
        trades_in_day[live] += 1
        if record_paths:
            paths[:, t] = equity

        if is_intraday:
            np.maximum(peak, equity, out=peak, where=live)
            nl = peak - dd
            if lock is not None:
                nl = np.minimum(nl, lock)
            np.maximum(dd_level, nl, out=dd_level, where=live)
        if firm.dd_type in (DrawdownType.STATIC, DrawdownType.INTRADAY_TRAIL):
            status[live & (equity <= dd_level + 1e-9)] = -1

        forced_end = np.zeros(N, dtype=bool)
        if has_dll:
            hit = (status == 0) & (day_pnl <= -firm.daily_loss_limit + 1e-9)
            if firm.dll_behavior == DLLBehavior.HARD_FAIL:
                status[hit] = -1
            else:
                forced_end |= hit

        if emergent:
            natural_end = new_day[:, t + 1] if t + 1 < T else np.ones(N, bool)
        else:
            natural_end = trades_in_day >= trades_per_day
        day_end = (status == 0) & (natural_end | forced_end)

        if day_end.any():
            np.maximum(best_day, day_pnl, out=best_day, where=day_end)
            if is_eod_trail:
                np.maximum(eod_peak, equity, out=eod_peak, where=day_end)
                nl = eod_peak - dd
                if lock is not None:
                    nl = np.minimum(nl, lock)
                np.maximum(dd_level, nl, out=dd_level, where=day_end)
            if firm.dd_type in (DrawdownType.EOD_TRAIL, DrawdownType.STATIC):
                eod_breach = day_end & (equity <= dd_level + 1e-9)
                status[eod_breach] = -1
                day_end = day_end & (status == 0)

            day_index[day_end] += 1
            total_profit = equity - start
            cons_ok = np.ones(N, bool)
            if firm.consistency_pct is not None:
                cons_ok = best_day <= firm.consistency_pct * total_profit + 1e-9
            passed = (day_end & (equity >= target_eq - 1e-9)
                      & (day_index >= firm.min_days) & cons_ok)
            status[passed] = 1
            ttp[passed] = t + 1
            dtp[passed] = day_index[passed]

            trades_in_day[day_end] = 0
            day_pnl[day_end] = 0.0

        if (status != 0).all():
            if record_paths:
                paths[:, t + 1:] = equity[:, None]
            break

    return SimResult(status, ttp, dtp, paths)
