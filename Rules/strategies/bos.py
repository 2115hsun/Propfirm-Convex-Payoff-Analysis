"""
strategies/bos.py — Break of Structure (BOS) ENTRY logic, self-contained.

Raw BOS on 1-min index futures fires on every micro-break, most of which are
noise. This version keeps the same core entry (close crosses a CONFIRMED swing)
but adds causal, tunable quality filters that target trend-aligned, decisive
breaks — the ones with follow-through — and skips marginal pokes and chop.

All filters use only past/current data (no look-ahead). Every filter is
optional (set its param to None/0 to disable) so you can sweep them.

Levers, biggest first:
  swing_k        larger = more significant structure, fewer/cleaner breaks
  trend_ema      only long above / short below this EMA (regime alignment)
  min_break_atr  break must clear the level by this many ATRs (decisiveness)
  adx_min        require trend strength >= this ADX (skip chop)
  cooldown       min bars between entries (avoid clustering one move)

Remember: the exit is the R:R bracket in Backtest.py, and stop width
(k_stop_atr) + timeframe are set there — they matter as much as these filters.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategies.base import EntryStrategy


# ---------------------------------------------------------------------------
# Indicators (causal; pandas ewm with adjust=False = Wilder smoothing)
# ---------------------------------------------------------------------------
def _ema(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


def _atr(high, low, close, period=14):
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()


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


# ---------------------------------------------------------------------------
# Causal fractal swings (unchanged core)
# ---------------------------------------------------------------------------
def causal_swings(high, low, k=2):
    high = np.asarray(high, float); low = np.asarray(low, float)
    n = high.size
    sh_cond = np.ones(n, bool); sl_cond = np.ones(n, bool)
    for o in range(1, k + 1):
        lh = np.zeros(n, bool); lh[o:] = high[o:] > high[:-o]
        rh = np.zeros(n, bool); rh[:-o] = high[:-o] > high[o:]
        sh_cond &= lh & rh
        ll = np.zeros(n, bool); ll[o:] = low[o:] < low[:-o]
        rl = np.zeros(n, bool); rl[:-o] = low[:-o] < low[o:]
        sl_cond &= ll & rl
    sh_cond[:k] = sh_cond[-k:] = False
    sl_cond[:k] = sl_cond[-k:] = False

    def _levels(cond, price):
        idx = np.flatnonzero(cond)
        lvl = np.full(n, np.nan)
        if idx.size:
            confirm = idx + k
            pos = np.searchsorted(confirm, np.arange(n), side="right") - 1
            v = pos >= 0
            lvl[v] = price[idx[pos[v]]]
        return lvl

    return _levels(sh_cond, high), _levels(sl_cond, low)


# ---------------------------------------------------------------------------
# Entries with filters
# ---------------------------------------------------------------------------
def bos_entries(high, low, close, swing_k=8, trend_ema=200, min_break_atr=0.10,
                adx_min=0.0, cooldown=0, atr_period=14):
    high = np.asarray(high, float); low = np.asarray(low, float)
    c = np.asarray(close, float)

    sh, sl = causal_swings(high, low, swing_k)
    cprev = np.empty_like(c); cprev[0] = c[0]; cprev[1:] = c[:-1]
    shp = np.empty_like(sh); shp[0] = np.nan; shp[1:] = sh[:-1]
    slp = np.empty_like(sl); slp[0] = np.nan; slp[1:] = sl[:-1]

    long_e = np.isfinite(sh) & np.isfinite(shp) & (c > sh) & (cprev <= shp)
    short_e = np.isfinite(sl) & np.isfinite(slp) & (c < sl) & (cprev >= slp)

    atr = _atr(high, low, c, atr_period)

    # break strength: must clear the broken level by min_break_atr * ATR
    if min_break_atr and min_break_atr > 0:
        with np.errstate(invalid="ignore"):
            long_e &= (c - sh) >= min_break_atr * atr
            short_e &= (sl - c) >= min_break_atr * atr

    # trend regime: trade breaks only in the direction of the longer EMA
    if trend_ema:
        ema = _ema(c, trend_ema)
        long_e &= c > ema
        short_e &= c < ema

    # trend strength: skip chop
    if adx_min and adx_min > 0:
        adx = _adx(high, low, c, atr_period)
        strong = adx >= adx_min
        long_e &= strong; short_e &= strong

    idx = np.flatnonzero(long_e | short_e)
    direction = np.where(long_e[idx], 1, -1)

    # cooldown: enforce a minimum bar gap between entries
    if cooldown and cooldown > 0 and idx.size:
        keep = [0]
        last = idx[0]
        for j in range(1, idx.size):
            if idx[j] - last >= cooldown:
                keep.append(j); last = idx[j]
        idx = idx[keep]; direction = direction[keep]

    return idx, direction


class BOS(EntryStrategy):
    name = "bos"

    def __init__(self, swing_k=8, trend_ema=200, min_break_atr=0.10,
                 adx_min=0.0, cooldown=0, atr_period=14):
        self.swing_k = swing_k
        self.trend_ema = trend_ema
        self.min_break_atr = min_break_atr
        self.adx_min = adx_min
        self.cooldown = cooldown
        self.atr_period = atr_period

    def entries(self, high, low, close):
        return bos_entries(high, low, close, swing_k=self.swing_k,
                           trend_ema=self.trend_ema, min_break_atr=self.min_break_atr,
                           adx_min=self.adx_min, cooldown=self.cooldown,
                           atr_period=self.atr_period)


def get_strategy():
    # Tuned defaults for index futures; sweep these (see notebook guidance).
    return BOS(swing_k=8, trend_ema=200, min_break_atr=0.10, adx_min=0.0, cooldown=0)