"""
Strategy.py — the Monte Carlo resampling layer.

The engine calls strategy.draw(n_paths, n_trades, rng) and gets:
    R        : ndarray[n_paths, n_trades]  per-trade R-multiples
    new_day  : ndarray[n_paths, n_trades] bool, or None
               True marks the first trade of a trading day. This makes
               trades-per-day EMERGENT from data (whole days are resampled).
               If None, the engine falls back to its trades_per_day argument.

  * EmpiricalStrategy(returns_R, day_ids=...)  <- from Backtest.ledger_for_rr(...)
    Pass day_ids so whole days are resampled (nothing about trades/day assumed).
  * SyntheticStrategy(win_rate, rr)            <- for abstract geometry demos only.
"""
from __future__ import annotations
import numpy as np


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
        return np.where(wins, self.rr, -1.0), None

    @property
    def expectancy_R(self):
        return self.win_rate * self.rr - (1 - self.win_rate)

    @staticmethod
    def from_edge(rr, edge_R=0.0, name=None):
        wr = min(max((edge_R + 1.0) / (rr + 1.0), 1e-4), 1 - 1e-4)
        return SyntheticStrategy(wr, rr, name or f"RR={rr:g} (WR={wr:.1%})")


class EmpiricalStrategy(Strategy):
    """Bootstrap a real R-ledger. With day_ids, resamples WHOLE days so
    trades-per-day (and their clustering) come straight from the data."""

    def __init__(self, returns_R, name="empirical", day_ids=None):
        self.returns_R = np.asarray(returns_R, float)
        assert self.returns_R.ndim == 1 and self.returns_R.size > 0
        self.name = name
        self._days = None
        self.avg_trades_per_day = None
        if day_ids is not None:
            day_ids = np.asarray(day_ids)
            change = np.flatnonzero(day_ids[1:] != day_ids[:-1]) + 1
            self._days = np.split(self.returns_R, change)
            self.avg_trades_per_day = float(np.mean([len(d) for d in self._days]))

    def _bootstrap(self, n_paths, n_trades, rng):
        idx = rng.integers(0, self.returns_R.size, size=(n_paths, n_trades))
        return self.returns_R[idx]

    def _draw_days(self, n_paths, n_trades, rng):
        days = self._days
        n_days = len(days)
        lens = np.array([len(d) for d in days])
        mean_len = max(lens.mean(), 1e-9)
        kmax = int(np.ceil(n_trades / mean_len * 1.8)) + 8
        di = rng.integers(0, n_days, size=(n_paths, kmax))
        clen = np.cumsum(lens[di], axis=1)
        R = np.empty((n_paths, n_trades)); new_day = np.zeros((n_paths, n_trades), bool)
        for p in range(n_paths):
            k = min(int(np.searchsorted(clen[p], n_trades) + 1), kmax)
            seq = np.concatenate([days[d] for d in di[p, :k]])
            if seq.size < n_trades:
                seq = np.concatenate([seq] + [days[d] for d in rng.integers(0, n_days, kmax)])
            R[p] = seq[:n_trades]
            marks = np.concatenate(([0], clen[p, :k - 1])).astype(int)
            new_day[p, marks[marks < n_trades]] = True
        new_day[:, 0] = True
        return R, new_day

    def draw(self, n_paths, n_trades, rng):
        if self._days is None:
            return self._bootstrap(n_paths, n_trades, rng), None
        return self._draw_days(n_paths, n_trades, rng)

    @property
    def expectancy_R(self):
        return float(self.returns_R.mean())