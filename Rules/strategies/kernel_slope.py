"""
strategies/kernel_slope.py — pure trend-following kernel entry: NW-RQK kernel
DIRECTION + ATR-normalized SLOPE conviction, with a LATCH (one signal per kernel
leg). No BOS, no efficiency-ratio gate.

Entry (long; short mirrors):
  1. kernel direction : NW-RQK kernel rising (long) / falling (short), computed
                        PER SESSION (reset_daily) so it never blends across
                        overnight gaps.
  2. FLIP             : the bar a new leg starts --
                            flip_up[i] = rising[i]  and not rising[i-1]
                            flip_dn[i] = falling[i] and not falling[i-1]
  3. SLOPE conviction : ATR-normalized kernel change over slope_len bars must
                        clear slope_min (in ATR units): |k - k[slope_len]| / ATR
                        >= slope_min. slope_min=0 disables. Same slope_atr the
                        kernel_bos strategy uses, so thresholds are comparable.
  4. LATCH            : at most one signal per leg. Once a leg fires, no more
                        until the NEXT flip in that direction re-arms it.

NOTE ON ATR: ATR here normalizes the SLOPE CONVICTION only -- it is NOT the
stop. The stop stays pure points (Backtest.py's stop_points / STOP_GRID). So
this is "ATR-normalized slope + point-based stop", which is consistent: ATR
never touches the stop, it only makes the conviction threshold scale-free
(the natural 0.1-0.3 range) across volatility regimes.

Faithful to kernel_bos.py's _rqk, _atr, per-session kernel, slope_atr, and the
vectorized latch -- with BOS and ER removed. Interface: wants_bars=True;
entries(bars)->(idx, dir); name; get_strategy(**over).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _rqk(close, h, r, x_0):
    L = int(x_0) + 1
    i = np.arange(L + 1, dtype=float)
    w = (1.0 + (i * i) / (2.0 * r * h * h)) ** (-r)
    w /= w.sum()
    out = np.convolve(close, w)[:close.size]
    out[:L] = np.nan
    return out


def _atr(high, low, close, period=14):
    """Wilder ATR, vectorized via pandas ewm -- identical to kernel_bos._atr."""
    h, l, c = np.asarray(high, float), np.asarray(low, float), np.asarray(close, float)
    prev = np.empty_like(c); prev[0] = c[0]; prev[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    out = np.full_like(tr, np.nan)
    n = tr.size
    if n >= period:
        seed = tr[:period].mean()
        out[period - 1] = seed
        if n > period:
            tail = np.concatenate(([seed], tr[period:]))
            out[period:] = pd.Series(tail).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()[1:]
    return out


def _compute(bars, h, r, x_0, slope_len, slope_min, atr_len,
             reset_daily, allow_long, allow_short):
    close = bars["close"].to_numpy(float)
    high = bars["high"].to_numpy(float); low = bars["low"].to_numpy(float)
    n = close.size

    # per-session kernel (no overnight-gap blending)
    k = np.full(n, np.nan)
    if reset_daily:
        dates = bars.index.normalize().to_numpy(dtype="datetime64[ns]")
        _, starts = np.unique(dates, return_index=True)
        bounds = list(np.sort(starts)) + [n]
        for s, e in zip(bounds[:-1], bounds[1:]):
            k[s:e] = _rqk(close[s:e], h, r, x_0)
    else:
        k = _rqk(close, h, r, x_0)

    rising = np.zeros(n, bool); falling = np.zeros(n, bool)
    rising[1:] = k[1:] > k[:-1]
    falling[1:] = k[1:] < k[:-1]

    flip_up = np.zeros(n, bool); flip_dn = np.zeros(n, bool)
    flip_up[1:] = rising[1:] & ~rising[:-1]
    flip_dn[1:] = falling[1:] & ~falling[:-1]

    # ATR-normalized slope conviction (same quantity as kernel_bos)
    a = _atr(high, low, close, atr_len)
    slope = np.full(n, np.nan)
    if slope_len < n:
        slope[slope_len:] = k[slope_len:] - k[:-slope_len]
    with np.errstate(invalid="ignore"):
        slope_atr = slope / a
    if slope_min > 0:
        slope_ok_long = np.isfinite(slope_atr) & (slope_atr >= slope_min)
        slope_ok_short = np.isfinite(slope_atr) & (slope_atr <= -slope_min)
    else:
        slope_ok_long = np.ones(n, bool)
        slope_ok_short = np.ones(n, bool)

    finite = np.isfinite(k)
    # candidate = currently in-direction AND slope conviction (no BOS, no ER)
    cand_long = finite & rising & slope_ok_long & allow_long
    cand_short = finite & falling & slope_ok_short & allow_short

    # LATCH: one signal per leg (first qualifying bar between consecutive flips)
    long_ok = np.zeros(n, bool)
    ci = np.flatnonzero(cand_long)
    if ci.size:
        leg = np.cumsum(flip_up)[ci]
        _, first = np.unique(leg, return_index=True)
        long_ok[ci[first]] = True

    short_ok = np.zeros(n, bool)
    ci = np.flatnonzero(cand_short)
    if ci.size:
        leg = np.cumsum(flip_dn)[ci]
        _, first = np.unique(leg, return_index=True)
        short_ok[ci[first]] = True

    idx = np.flatnonzero(long_ok | short_ok)
    direction = np.where(long_ok[idx], 1, -1).astype(int)
    return dict(close=close, k=k, rising=rising, falling=falling,
                flip_up=flip_up, flip_dn=flip_dn, slope_atr=slope_atr,
                cand_long=cand_long, cand_short=cand_short,
                long_ok=long_ok, short_ok=short_ok, idx=idx, direction=direction)


class KernelSlope:
    name = "kernel_slope"
    wants_bars = True

    def __init__(self, h=8.0, r=8.0, x_0=8, slope_len=3, slope_min=0.3, atr_len=14,
                 reset_daily=True, allow_long=True, allow_short=True):
        self.h = float(h); self.r = float(r); self.x_0 = int(x_0)
        self.slope_len = int(slope_len); self.slope_min = float(slope_min)
        self.atr_len = int(atr_len); self.reset_daily = bool(reset_daily)
        self.allow_long = bool(allow_long); self.allow_short = bool(allow_short)

    def _args(self):
        return dict(h=self.h, r=self.r, x_0=self.x_0, slope_len=self.slope_len,
                    slope_min=self.slope_min, atr_len=self.atr_len,
                    reset_daily=self.reset_daily, allow_long=self.allow_long,
                    allow_short=self.allow_short)

    def entries(self, bars):
        d = _compute(bars, **self._args())
        return d["idx"], d["direction"]

    def debug(self, bars):
        d = _compute(bars, **self._args())
        return pd.DataFrame({
            "close": d["close"], "k": d["k"], "rising": d["rising"], "falling": d["falling"],
            "flip_up": d["flip_up"], "flip_dn": d["flip_dn"], "slope_atr": d["slope_atr"],
            "cand_long": d["cand_long"], "cand_short": d["cand_short"],
            "long_ok": d["long_ok"], "short_ok": d["short_ok"],
        }, index=bars.index)


DEFAULTS = dict(h=8.0, r=8.0, x_0=8, slope_len=3, slope_min=0.3, atr_len=14,
                reset_daily=True, allow_long=True, allow_short=True)


def get_strategy(**overrides):
    return KernelSlope(**{**DEFAULTS, **overrides})