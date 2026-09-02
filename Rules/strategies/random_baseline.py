"""
strategies/random_baseline.py — TRUE zero-edge control (FAST version).

RandomEntryBaseline: places entries at RANDOM bars within the session window,
with RANDOM direction. This is the pure "beat variance / barrier geometry" floor:
no timing edge, no direction edge.

FAST: the eligible pool is defined DIRECTLY from the session window
(sess_start..sess_end on every trading day), so it does NOT re-run the (expensive)
inner strategy on every seed. Pass either:
  * n_entries  = fixed number of random entries per run (matches your real count), OR
  * per_day    = average entries/day (count = per_day * n_trading_days), OR
  * inner      = a strategy whose entry COUNT is matched (run ONCE, cached by caller).

Direction is random, so MFE/MAE are recomputed per trade by backtest_entries()'s
barrier logic (this wrapper only supplies idx+direction) -> fully consistent with
the real ledger's mechanics.

Usage (fast):
    z = RandomEntryBaseline(bars_session=("09:30","11:00"), n_entries=n_real, seed=s)
    E = backtest_entries(z, bars, stop_points=..., detail=True, one_position=False)
"""
from __future__ import annotations
import datetime as dt
import numpy as np


class RandomEntryBaseline:
    wants_bars = True

    def __init__(self, inner=None, seed=0, n_entries=None,
                 sess_start="09:30", sess_end="11:00", per_day=None):
        self.inner = inner
        self._seed = int(seed)
        self._n = n_entries
        self._per_day = per_day
        self.s0 = dt.time.fromisoformat(sess_start)
        self.s1 = dt.time.fromisoformat(sess_end)
        self.name = (getattr(inner, "name", "rand") if inner else "rand") + f"_random{seed}"

    def _eligible_pool(self, bars):
        times = bars.index
        tod = np.array([self.s0 <= t <= self.s1 for t in times.time])
        pool = np.flatnonzero(tod)
        return pool[(pool > 30) & (pool < len(bars) - 5)]

    def entries(self, bars):
        pool = self._eligible_pool(bars)
        if pool.size == 0:
            return np.array([], int), np.array([], int)

        # decide how many entries to draw
        if self._n is not None:
            k = int(self._n)
        elif self._per_day is not None:
            n_days = np.unique(bars.index[pool].normalize()).size
            k = int(round(self._per_day * n_days))
        elif self.inner is not None:
            ri, _ = self.inner.entries(bars)       # slow path: only if nothing cached
            k = len(ri)
        else:
            raise ValueError("provide n_entries, per_day, or inner")

        k = min(k, pool.size)
        if k <= 0:
            return np.array([], int), np.array([], int)
        rng = np.random.default_rng(self._seed)
        idx = np.sort(rng.choice(pool, size=k, replace=False))
        direction = rng.choice([1, -1], size=k).astype(int)
        return idx, direction


def get_strategy(**kw):
    return RandomEntryBaseline(**kw)