"""
strategies/kernel_turn.py — your kernel_mtf_fvg with ONLY the FVG system removed.

Everything else is byte-for-byte identical to your proven strategy: the NW-RQK
kernel, the efficiency-ratio chop gate (UNSIGNED), the rolling `agree` flip
window, and the same defaults. The multi-timeframe FVG confirmation is deleted.

This is exactly `kernel_mtf_fvg` with min_tf_agree=0 (FVG condition always true),
so it reproduces your baseline. Nothing new added — optimize from here.

Entry (long; short mirrors):
    1. turn   : 1-min kernel flipped UP within the last `agree` bars
    2. regime : efficiency ratio >= er_min                (not chop)

Exits are NOT here — Backtest.py brackets each entry with k_stop_atr*ATR.
Interface: wants_bars=True; entries(bars)->(idx, dir); name; get_strategy(**over).
"""
from __future__ import annotations
import numpy as np


def _efficiency_ratio(close, length):
    close = np.asarray(close, float)
    d = np.abs(np.diff(close))
    csum = np.concatenate([[0.0], np.cumsum(d)])
    er = np.zeros(close.size)
    path = csum[length:] - csum[:-length]
    net = np.abs(close[length:] - close[:-length])
    er[length:] = np.where(path > 0, net / path, 0.0)
    return er


def _rqk(close, h, r, x_0):
    L = int(x_0) + 1
    i = np.arange(L + 1, dtype=float)
    w = (1.0 + (i * i) / (2.0 * r * h * h)) ** (-r)
    w /= w.sum()
    out = np.convolve(close, w)[:close.size]
    out[:L] = np.nan
    return out


def _recent_true(mask, lookback):
    conv = np.convolve(mask.astype(float), np.ones(lookback + 1))[:mask.size]
    return conv > 0


def _recent_flip(rising, agree):
    n = rising.size
    flip = np.zeros(n, bool)
    flip[1:] = rising[1:] & ~rising[:-1]
    return _recent_true(flip, agree)


class KernelTurn:
    name = "kernel"
    wants_bars = True

    def __init__(self, h=8.0, r=8.0, x_0=25, er_len=10, er_min=0.3, agree=1,
                 allow_long=True, allow_short=True):
        self.h = float(h); self.r = float(r); self.x_0 = int(x_0)
        self.er_len = int(er_len); self.er_min = float(er_min); self.agree = int(agree)
        self.allow_long = bool(allow_long); self.allow_short = bool(allow_short)

    def entries(self, bars):
        close = bars["close"].to_numpy(float)
        n = close.size
        k = _rqk(close, self.h, self.r, self.x_0)
        er = _efficiency_ratio(close, self.er_len)

        rising = np.zeros(n, bool); falling = np.zeros(n, bool)
        rising[1:] = k[1:] > k[:-1]; falling[1:] = k[1:] < k[:-1]
        flip_up = _recent_flip(rising, self.agree)
        flip_dn = _recent_flip(falling, self.agree)

        finite = np.isfinite(k)
        regime = er >= self.er_min

        long_ok = finite & flip_up & regime & self.allow_long
        short_ok = finite & flip_dn & regime & self.allow_short

        idx = np.flatnonzero(long_ok | short_ok)
        direction = np.where(long_ok[idx], 1, -1).astype(int)
        return idx, direction


DEFAULTS = dict(h=8.0, r=8.0, x_0=25, er_len=10, er_min=0.3, agree=3,
                allow_long=True, allow_short=True)


def get_strategy(**overrides):
    return KernelTurn(**{**DEFAULTS, **overrides})