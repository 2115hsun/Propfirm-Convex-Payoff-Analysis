"""
strategies/ema_cross.py — EMA crossover ENTRY, self-contained.

  long  when the fast EMA crosses ABOVE the slow EMA
  short when the fast EMA crosses BELOW the slow EMA

Entry only — the exit is the R:R bracket applied by Backtest.py. Price-only, so
it uses the default contract (wants_bars = False): entries(high, low, close).

Optional filters (all causal; set to None/0 to disable):
  trend_ema   only take crosses on the correct side of a longer EMA (regime)
  adx_min     require ADX >= this (skip chop)
  reverse     fade the cross instead of following it

Tunable params (sweep these):
  fast, slow  the two crossover EMAs (slow > fast)
  trend_ema   longer-term regime filter
  adx_min     trend-strength gate
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategies.base import EntryStrategy


def _ema(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


def _adx(high, low, close, period=14):
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    up, dn = h.diff(), -l.diff()
    plus_dm = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()


def ema_cross_entries(high, low, close, fast=9, slow=21, trend_ema=None,
                      adx_min=0.0, adx_p=14, reverse=False):
    c = np.asarray(close, float)
    f, s = _ema(c, fast), _ema(c, slow)
    diff = f - s
    up = np.zeros(c.size, bool); up[1:] = (diff[1:] > 0) & (diff[:-1] <= 0)
    dn = np.zeros(c.size, bool); dn[1:] = (diff[1:] < 0) & (diff[:-1] >= 0)

    long_e, short_e = up.copy(), dn.copy()

    if trend_ema:
        te = _ema(c, trend_ema)
        long_e &= c > te
        short_e &= c < te

    if adx_min and adx_min > 0:
        adx = _adx(high, low, c, adx_p)
        gate = adx >= adx_min
        long_e &= gate; short_e &= gate

    idx = np.flatnonzero(long_e | short_e)
    direction = np.where(long_e[idx], 1, -1)
    if reverse:
        direction = -direction
    return idx, direction


class EMACross(EntryStrategy):
    name = "ema_cross"
    wants_bars = False

    def __init__(self, fast=9, slow=21, trend_ema=None, adx_min=0.0,
                 adx_p=14, reverse=False):
        self.fast = fast; self.slow = slow; self.trend_ema = trend_ema
        self.adx_min = adx_min; self.adx_p = adx_p; self.reverse = reverse

    def entries(self, high, low, close):
        return ema_cross_entries(high, low, close, self.fast, self.slow,
                                 self.trend_ema, self.adx_min, self.adx_p, self.reverse)


def get_strategy():
    return EMACross(fast=9, slow=21, trend_ema=200, adx_min=0.0)