"""
Strategy.py: the Monte Carlo resampling layer.

The engine calls strategy.draw(n_paths, n_trades, rng) and gets:
    R        : ndarray[n_paths, n_trades]  per-trade R-multiples
    new_day  : ndarray[n_paths, n_trades] bool, or None
               True marks the first trade of a trading day, so trades per
               day come from the data (whole days are resampled). If None,
               the engine falls back to its trades_per_day argument.
    mae      : ndarray[n_paths, n_trades] adverse excursion in R, or None

  * EmpiricalStrategy(returns_R, day_ids=...)  <- from Backtest.ledger_for_rr(...)
    Pass day_ids so whole days are resampled.
  * SyntheticStrategy(win_rate, rr)            <- for abstract geometry demos only.
"""
from __future__ import annotations
import numpy as np
import datetime as dt

class Timewindow:
    """Wrap an entry strategy and keep only entries between start and end."""
    wants_bars = True
    def __init__(self, inner, start="09:30", end="16:00"):
        self.inner = inner
        self.start = dt.time.fromisoformat(start)
        self.end = dt.time.fromisoformat(end)
        self.name = getattr(inner, "name", f"{inner.__class__.__name__} {start}-{end}")
    def entries(self, bars):
        idx, direction = self.inner.entries(bars)
        idx = np.asarray(idx); direction = np.asarray(direction)
        times = bars.index[idx].time
        keep = np.array([self.start <= t <= self.end for t in times], dtype=bool)
        return idx[keep], direction[keep]


class Strategy:
    name: str = "strategy"

    def draw(self, n_paths, n_trades, rng):
        raise NotImplementedError

    @property
    def expectancy_R(self) -> float:
        raise NotImplementedError


class SyntheticStrategy(Strategy):
    def __init__(self, win_rate, rr, name=None):
        assert 0 < win_rate < 1
        self.win_rate = win_rate; self.rr = rr
        self.name = name or f"WR={win_rate:.0%}, RR={rr:g}"

    def draw(self, n_paths, n_trades, rng):
        wins = rng.random((n_paths, n_trades)) < self.win_rate
        return np.where(wins, self.rr, -1.0), None, None

    @property
    def expectancy_R(self):
        return self.win_rate * self.rr - (1 - self.win_rate)

    @staticmethod
    def from_edge(rr, edge_R=0.0, name=None):
        wr = min(max((edge_R + 1.0) / (rr + 1.0), 1e-4), 1 - 1e-4)
        return SyntheticStrategy(wr, rr, name or f"RR={rr:g} (WR={wr:.1%})")


class EmpiricalStrategy(Strategy):
    """Bootstrap a real R-ledger. With day_ids, whole days are resampled, so
    trades per day and their clustering come from the data."""

    def __init__(self, returns_R, name="empirical", day_ids=None, mae_R=None):
        self.returns_R = np.asarray(returns_R, float)
        assert self.returns_R.ndim == 1 and self.returns_R.size > 0
        self.name = name
        # per-trade adverse excursion (R, positive), paired 1:1 with returns_R.
        # None -> MAE breach disabled (old ledgers); the engine then skips it.
        self.mae_R = None if mae_R is None else np.asarray(mae_R, float)
        if self.mae_R is not None:
            assert self.mae_R.shape == self.returns_R.shape, "mae_R must pair with returns_R"
        self._days = None
        self._days_mae = None
        self.avg_trades_per_day = None
        if day_ids is not None:
            day_ids = np.asarray(day_ids)
            change = np.flatnonzero(day_ids[1:] != day_ids[:-1]) + 1
            self._days = np.split(self.returns_R, change)
            if self.mae_R is not None:
                self._days_mae = np.split(self.mae_R, change)
            self.avg_trades_per_day = float(np.mean([len(d) for d in self._days]))

    def _bootstrap(self, n_paths, n_trades, rng):
        idx = rng.integers(0, self.returns_R.size, size=(n_paths, n_trades))
        R = self.returns_R[idx]
        M = self.mae_R[idx] if self.mae_R is not None else None
        return R, M

    def _draw_days(self, n_paths, n_trades, rng):
        days = self._days
        n_days = len(days)
        lens = np.array([len(d) for d in days])
        mean_len = max(lens.mean(), 1e-9)
        kmax = int(np.ceil(n_trades / mean_len * 1.8)) + 8
        di = rng.integers(0, n_days, size=(n_paths, kmax))
        clen = np.cumsum(lens[di], axis=1)
        days_mae = self._days_mae
        R = np.empty((n_paths, n_trades)); new_day = np.zeros((n_paths, n_trades), bool)
        M = np.empty((n_paths, n_trades)) if days_mae is not None else None
        for p in range(n_paths):
            k = min(int(np.searchsorted(clen[p], n_trades) + 1), kmax)
            seq = np.concatenate([days[d] for d in di[p, :k]])
            if days_mae is not None:
                seqm = np.concatenate([days_mae[d] for d in di[p, :k]])
            if seq.size < n_trades:
                extra = rng.integers(0, n_days, kmax)
                seq = np.concatenate([seq] + [days[d] for d in extra])
                if days_mae is not None:
                    seqm = np.concatenate([seqm] + [days_mae[d] for d in extra])
            R[p] = seq[:n_trades]
            if days_mae is not None:
                M[p] = seqm[:n_trades]
            marks = np.concatenate(([0], clen[p, :k - 1])).astype(int)
            new_day[p, marks[marks < n_trades]] = True
        new_day[:, 0] = True
        return R, new_day, M

    def draw(self, n_paths, n_trades, rng):
        if self._days is None:
            R, M = self._bootstrap(n_paths, n_trades, rng)
            return R, None, M
        return self._draw_days(n_paths, n_trades, rng)

    @property
    def expectancy_R(self):
        return float(self.returns_R.mean())