"""
strategies/kernel_bos.py — kernel DIRECTION + break-of-structure (BOS), aligned.

Entry (3-min, enter at close):
  1. kernel direction : NW-RQK kernel is rising (long bias) / falling (short bias),
                        computed PER SESSION so it never blends across overnight
                        gaps (rth_only data has them).
  2. BOS trigger      : price breaks the recent micro-structure.
        bearish BOS (short): close breaks BELOW the lowest low of the prior
                             `bos_lookback` (2-3) candles.
        bullish BOS (long) : close breaks ABOVE the highest high of the prior
                             `bos_lookback` candles.
  3. alignment        : the BOS direction MUST match the kernel direction, else
                        no entry. (bull BOS + rising kernel = long; mirror short.)

No ER/FVG/limit-fill. Enters at the bar's close -> returns (idx, direction), so
Backtest.py brackets each entry with k_stop_atr*ATR as usual.

Optional gates (default off, so the base is pure direction+BOS):
  require_flip : also require a FRESH kernel flip within `agree` bars (not just
                 current direction). Off = any bar where direction agrees.
  er_min       : efficiency-ratio chop gate (0 = off).

Interface: wants_bars=True; entries(bars)->(idx, dir); name; get_strategy(**over).
"""
from __future__ import annotations
import numpy as np


def _rqk(close, h, r, x_0):
    L = int(x_0) + 1
    i = np.arange(L + 1, dtype=float)
    w = (1.0 + (i * i) / (2.0 * r * h * h)) ** (-r)
    w /= w.sum()
    out = np.convolve(close, w)[:close.size]
    out[:L] = np.nan
    return out


def _efficiency_ratio(close, length):
    close = np.asarray(close, float)
    d = np.abs(np.diff(close))
    csum = np.concatenate([[0.0], np.cumsum(d)])
    er = np.zeros(close.size)
    path = csum[length:] - csum[:-length]
    net = np.abs(close[length:] - close[:-length])
    er[length:] = np.where(path > 0, net / path, 0.0)
    return er


def _atr(high, low, close, period=14):
    h, l, c = np.asarray(high, float), np.asarray(low, float), np.asarray(close, float)
    prev = np.empty_like(c); prev[0] = c[0]; prev[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    out = np.full_like(tr, np.nan)
    if tr.size >= period:
        out[period - 1] = tr[:period].mean()
        a = 1.0 / period
        for i in range(period, tr.size):
            out[i] = out[i - 1] * (1 - a) + tr[i] * a
    return out


def _recent_true(mask, lookback):
    conv = np.convolve(mask.astype(float), np.ones(lookback + 1))[:mask.size]
    return conv > 0


def _recent_flip(rising, agree):
    n = rising.size
    flip = np.zeros(n, bool)
    flip[1:] = rising[1:] & ~rising[:-1]
    return _recent_true(flip, agree)


class KernelBOS:
    name = "kernel_bos"
    wants_bars = True

    def __init__(self, h=8.0, r=8.0, x_0=25, bos_lookback=3,
                 slope_len=3, slope_min=0.0, atr_len=14,
                 require_flip=False, agree=3, er_len=10, er_min=0.0,
                 allow_long=True, allow_short=True):
        self.h = float(h); self.r = float(r); self.x_0 = int(x_0)
        self.bos_lookback = int(bos_lookback)
        self.slope_len = int(slope_len); self.slope_min = float(slope_min); self.atr_len = int(atr_len)
        self.require_flip = bool(require_flip); self.agree = int(agree)
        self.er_len = int(er_len); self.er_min = float(er_min)
        self.allow_long = bool(allow_long); self.allow_short = bool(allow_short)

    def entries(self, bars):
        close = bars["close"].to_numpy(float)
        high = bars["high"].to_numpy(float); low = bars["low"].to_numpy(float)
        n = close.size

        # --- PER-SESSION kernel (no overnight-gap blending) ---
        k = np.full(n, np.nan)
        dates = bars.index.normalize().to_numpy()
        _, starts = np.unique(dates, return_index=True)
        bounds = list(np.sort(starts)) + [n]
        for s, e in zip(bounds[:-1], bounds[1:]):
            k[s:e] = _rqk(close[s:e], self.h, self.r, self.x_0)

        rising = np.zeros(n, bool); falling = np.zeros(n, bool)
        rising[1:] = k[1:] > k[:-1]; falling[1:] = k[1:] < k[:-1]
        if self.require_flip:
            up_dir = _recent_flip(rising, self.agree)
            dn_dir = _recent_flip(falling, self.agree)
        else:
            up_dir = rising; dn_dir = falling

        # --- SLOPE CONVICTION GATE: kernel must be turning with conviction, not
        #     flat-and-wiggling (a barely-rising flat kernel IS chop). ATR-normalized
        #     so slope_min is one threshold across volatility regimes. 0 = off. ---
        if self.slope_min > 0:
            a_slope = _atr(high, low, close, self.atr_len)
            slope = np.full(n, np.nan)
            if self.slope_len < n:
                slope[self.slope_len:] = k[self.slope_len:] - k[:-self.slope_len]
            with np.errstate(invalid="ignore"):
                slope_atr = slope / a_slope                 # kernel change per window, in ATR
            up_dir = up_dir & (slope_atr >= self.slope_min)
            dn_dir = dn_dir & (slope_atr <= -self.slope_min)

        # --- BOS: break of the prior bos_lookback candles' structure (PER SESSION) ---
        lb = self.bos_lookback
        prior_low = (bars.groupby(bars.index.normalize())["low"]
                     .transform(lambda s: s.shift(1).rolling(lb).min()).to_numpy())
        prior_high = (bars.groupby(bars.index.normalize())["high"]
                      .transform(lambda s: s.shift(1).rolling(lb).max()).to_numpy())
        with np.errstate(invalid="ignore"):
            bear_bos = close < prior_low        # broke below recent lows
            bull_bos = close > prior_high       # broke above recent highs

        # --- optional chop gate ---
        if self.er_min > 0:
            regime = _efficiency_ratio(close, self.er_len) >= self.er_min
        else:
            regime = np.ones(n, bool)

        finite = np.isfinite(k)
        long_ok = finite & up_dir & bull_bos & regime & self.allow_long
        short_ok = finite & dn_dir & bear_bos & regime & self.allow_short

        idx = np.flatnonzero(long_ok | short_ok)
        direction = np.where(long_ok[idx], 1, -1).astype(int)
        return idx, direction


DEFAULTS = dict(h=8.0, r=8.0, x_0=25, bos_lookback=3, slope_len=3, slope_min=0.0, atr_len=14, require_flip=False, agree=3,
                er_len=10, er_min=0.0, allow_long=True, allow_short=True)


def get_strategy(**overrides):
    return KernelBOS(**{**DEFAULTS, **overrides})