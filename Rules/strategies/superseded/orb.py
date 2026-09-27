"""
strategies/orb.py — Opening Range Breakout ENTRY, self-contained.

Reproduces the entry from your orb.py without config/indicators/base:
  opening range = high/low of the first `or_minutes` after the 09:30 ET open
  long  when close crosses ABOVE the range high (after the range is set)
  short when close crosses BELOW the range low
  gated by ADX >= adx_min, and optionally only within a session window.

Entry only — exit is the R:R bracket in Backtest.py. ML grader omitted.

REQUIRES a DatetimeIndex in Eastern Time. load_ohlc(..., tz="America/New_York")
(the default) already does this for Databento/build_continuous data.

Tunable params (sweep these; see notebook guidance):
  or_minutes     length of the opening range (15, 30, 60 ...)
  open_min       session open in ET minutes (570 = 09:30)
  close_min      stop taking breakouts after this ET minute (None = whole session)
  adx_min        require ADX >= this (regime gate; 0 = off)
  one_per_session keep only the first breakout each day
  reverse        fade the breakout instead of following it
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategies.base import EntryStrategy


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


def opening_range(index, high, low, or_minutes=15, open_min=570):
    """Per-bar opening-range high/low (NaN until the range is complete)."""
    tmin = (index.hour * 60 + index.minute).to_numpy()
    day_key = (index.year * 10000 + index.month * 100 + index.day).to_numpy()
    in_win = (tmin >= open_min) & (tmin < open_min + or_minutes)

    hi = pd.Series(np.where(in_win, high, np.nan))
    lo = pd.Series(np.where(in_win, low, np.nan))
    oh_per = hi.groupby(day_key).max()
    ol_per = lo.groupby(day_key).min()
    oh = pd.Series(day_key).map(oh_per).to_numpy()
    ol = pd.Series(day_key).map(ol_per).to_numpy()

    active = tmin >= open_min + or_minutes           # range known only after it closes
    oh = np.where(active, oh, np.nan)
    ol = np.where(active, ol, np.nan)
    return oh, ol, tmin, day_key


def orb_entries(bars, or_minutes=15, open_min=570, close_min=None,
                adx_p=14, adx_min=20.0, one_per_session=True, reverse=False):
    index = bars.index
    high = bars["high"].to_numpy(float); low = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)

    oh, ol, tmin, day_key = opening_range(index, high, low, or_minutes, open_min)
    cprev = np.empty_like(c); cprev[0] = c[0]; cprev[1:] = c[:-1]
    ohp = np.empty_like(oh); ohp[0] = np.nan; ohp[1:] = oh[:-1]
    olp = np.empty_like(ol); olp[0] = np.nan; olp[1:] = ol[:-1]

    long_e = np.isfinite(oh) & np.isfinite(ohp) & (cprev <= ohp) & (c > oh)
    short_e = np.isfinite(ol) & np.isfinite(olp) & (cprev >= olp) & (c < ol)

    if close_min is not None:
        late = tmin >= close_min
        long_e &= ~late; short_e &= ~late

    if adx_min and adx_min > 0:
        adx = _adx(high, low, c, adx_p)
        gate = adx >= adx_min
        long_e &= gate; short_e &= gate

    idx = np.flatnonzero(long_e | short_e)
    direction = np.where(long_e[idx], 1, -1)

    if one_per_session and idx.size:
        keys = day_key[idx]
        seen = set(); keep = []
        for j, k in enumerate(keys):
            if k not in seen:
                seen.add(k); keep.append(j)
        idx = idx[keep]; direction = direction[keep]

    if reverse:
        direction = -direction
    return idx, direction


class ORB(EntryStrategy):
    name = "orb"
    wants_bars = True

    def __init__(self, or_minutes=15, open_min=570, close_min=None,
                 adx_p=14, adx_min=20.0, one_per_session=True, reverse=False):
        self.or_minutes = or_minutes; self.open_min = open_min
        self.close_min = close_min; self.adx_p = adx_p; self.adx_min = adx_min
        self.one_per_session = one_per_session; self.reverse = reverse

    def entries(self, bars):
        return orb_entries(bars, or_minutes=self.or_minutes, open_min=self.open_min,
                           close_min=self.close_min, adx_p=self.adx_p,
                           adx_min=self.adx_min, one_per_session=self.one_per_session,
                           reverse=self.reverse)


def get_strategy():
    return ORB(or_minutes=15, adx_min=20.0, one_per_session=True)