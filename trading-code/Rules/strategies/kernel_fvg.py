"""
strategies/kernel_mtf_fvg.py — trend-change entry confirmed by MULTI-TIMEFRAME
FVG impulse. Directional + anti-chop, plus a "real move" screen: the 1-min
kernel turn only counts if the move gapped out at least `min_tf_agree` of the
higher timeframes {2,3,4,5}-min in the SAME direction.

Thesis (impulse-confirmed trend chasing):
    A genuine trend change is violent enough to leave a fair-value gap on several
    timeframes at once. A fake 3-minute pop that immediately reverses does not.
    So requiring a FRESH FVG in the trade direction on >=2 of the higher TFs
    filters out the reverse-in-3-minutes noise, and you enter on the 1-min
    kernel direction change with that impulse confirmed.

NOTE: uses REGULAR FVGs (impulse), not inverted ones. Direction of the FVG must
match the trade: LONG needs BULLISH gaps, SHORT needs BEARISH gaps.

    Bullish FVG at candle i (on a TF): low[i] > high[i-2]
    Bearish FVG at candle i:           high[i] < low[i-2]

Entry (long; short mirrors), ALL on the 1-min entry bar:
    1. turn    : 1-min kernel flipped UP within the last `agree` bars
    2. regime  : efficiency ratio >= er_min                       (not chop)
    3. impulse : >= min_tf_agree of {fvg_tfs} formed a BULLISH FVG within the
                 last `fvg_lookback` 1-min bars                    (multi-TF force)

Exits are NOT here — Backtest.py brackets each entry with k_stop_atr*ATR.
Interface: wants_bars=True; entries(bars)->(idx, dir); name; get_strategy(**over).
Base bars MUST be 1-min (direction is 1-min; higher TFs resampled internally).

Knobs (pre-register a SMALL grid; validate OOS — do NOT hand-fit):
    h, r, x_0        NW-RQK kernel (1-min direction)
    er_len, er_min   chop gate (0 => off)
    fvg_tfs          higher timeframes to check, minutes            (default 2,3,4,5)
    min_tf_agree     how many of them must gap in-direction         (default 2)
    fvg_lookback     how recent the gap must be, in 1-min bars      (default 15)
    agree            slack (bars) between the kernel turn and now
    allow_long, allow_short
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
    """Causal rolling-any: True if `mask` was True within the last `lookback` bars."""
    conv = np.convolve(mask.astype(float), np.ones(lookback + 1))[:mask.size]
    return conv > 0


def _recent_flip(rising, agree):
    n = rising.size
    flip = np.zeros(n, bool)
    flip[1:] = rising[1:] & ~rising[:-1]
    return _recent_true(flip, agree)


class KernelMTFFVG:
    name = "kernel_mtf_fvg"
    wants_bars = True

    def __init__(self, h=8.0, r=8.0, x_0=25, er_len=10, er_min=0.3,
                 fvg_tfs=(2, 3, 4, 5), min_tf_agree=0, fvg_lookback=15, agree=1,
                 allow_long=True, allow_short=True):
        self.h = float(h); self.r = float(r); self.x_0 = int(x_0)
        self.er_len = int(er_len); self.er_min = float(er_min)
        self.fvg_tfs = tuple(int(x) for x in fvg_tfs)
        self.min_tf_agree = int(min_tf_agree); self.fvg_lookback = int(fvg_lookback)
        self.agree = int(agree)
        self.allow_long = bool(allow_long); self.allow_short = bool(allow_short)

    def _fvg_formed_masks(self, bars):
        """For each TF, two 1-min-length bool arrays: bull/bear FVG FORMED at bar."""
        n = len(bars); base_ts = bars.index
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        bull_masks, bear_masks = [], []
        for tf in self.fvg_tfs:
            f = bars.resample(f"{tf}min").agg(agg).dropna()
            hi = f["high"].to_numpy(float); lo = f["low"].to_numpy(float); ts = f.index
            bull = np.zeros(n, bool); bear = np.zeros(n, bool)
            for i in range(2, len(f)):
                a = int(base_ts.searchsorted(ts[i]))   # 1-min bar when 3rd TF candle closes
                if a >= n:
                    continue
                if lo[i] > hi[i - 2]:
                    bull[a] = True
                elif hi[i] < lo[i - 2]:
                    bear[a] = True
            bull_masks.append(bull); bear_masks.append(bear)
        return bull_masks, bear_masks

    def entries(self, bars):
        close = bars["close"].to_numpy(float)
        n = close.size
        k = _rqk(close, self.h, self.r, self.x_0)
        er = _efficiency_ratio(close, self.er_len)
        rising = np.zeros(n, bool); falling = np.zeros(n, bool)
        rising[1:] = k[1:] > k[:-1]; falling[1:] = k[1:] < k[:-1]

        flip_up = _recent_flip(rising, self.agree)
        flip_dn = _recent_flip(falling, self.agree)

        bull_masks, bear_masks = self._fvg_formed_masks(bars)
        # count TFs with a fresh in-direction FVG within lookback of each bar
        bull_count = np.zeros(n, int); bear_count = np.zeros(n, int)
        for bm in bull_masks:
            bull_count += _recent_true(bm, self.fvg_lookback).astype(int)
        for bm in bear_masks:
            bear_count += _recent_true(bm, self.fvg_lookback).astype(int)

        finite = np.isfinite(k)
        regime = er >= self.er_min
        long_ok = (finite & flip_up & regime & (bull_count >= self.min_tf_agree)
                   & self.allow_long)
        short_ok = (finite & flip_dn & regime & (bear_count >= self.min_tf_agree)
                    & self.allow_short)

        idx = np.flatnonzero(long_ok | short_ok)
        direction = np.where(long_ok[idx], 1, -1).astype(int)
        return idx, direction


DEFAULTS = dict(
    h=8.0, r=8.0, x_0=25, er_len=10, er_min=0.3,
    fvg_tfs=(2, 3, 4, 5), min_tf_agree=2, fvg_lookback=15, agree=3,
    allow_long=True, allow_short=True,
)


def get_strategy(**overrides):
    """load_strategy('kernel_mtf_fvg') calls this. Override for sweeps, e.g.
    get_strategy(min_tf_agree=3, er_min=0.4, fvg_lookback=10)."""
    return KernelMTFFVG(**{**DEFAULTS, **overrides})