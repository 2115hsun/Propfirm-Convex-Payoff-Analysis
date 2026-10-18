"""
strategies/random_baseline.py: zero-edge control.

RandomEntryBaseline places entries at random bars inside the session window,
with random direction: no timing edge and no direction edge. It shows what
the barrier geometry alone produces.

The eligible bars come straight from the session window (sess_start..sess_end
each day), so the inner strategy doesn't have to be re-run for every seed.
Set the entry count with one of:
  * n_entries  = fixed number of random entries per run (e.g. the real count)
  * per_day    = average entries/day (count = per_day * n_trading_days)
  * inner      = a strategy whose entry count is matched (slow: runs it once)

This wrapper only supplies idx and direction; backtest_entries() computes
MFE/MAE the same way as for the real strategy.

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