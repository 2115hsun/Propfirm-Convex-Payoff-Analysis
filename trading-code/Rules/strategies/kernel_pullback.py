"""
strategies/kernel_pullback.py — kernel-flip PULLBACK LIMIT entry.

The idea (3-min RTH):
  1. On a candle close, detect a KERNEL DIRECTION FLIP (pure — no ER/ADX/Hurst).
  2. SLOPE GUARD (anti-chop): the kernel must be turning with conviction — its
     slope over `slope_len` bars, in ATR units, must exceed `slope_min` in the
     flip direction. A kernel that flips but stays flat IS chop, so this rejects
     the 3-flips-in-5-candles whipsaws. (slope_min=None disables it.)
  3. COOLDOWN: no new setup within `cooldown` bars of the last armed flip, so a
     flip-cluster can't arm several overlapping limits.
  4. Arm a LIMIT AT THE KERNEL and wait the next `wait_bars` (1-3) candles for
     price to RETRACE and touch the kernel (intrabar — not waiting for a close).
       long : low[j]  <= kernel[j]   -> filled at kernel[j]
       short: high[j] >= kernel[j]   -> filled at kernel[j]
     The limit drifts WITH the kernel each bar (fill price = kernel at the touch
     bar). First touch fills; if no touch by candle f+wait_bars, the setup dies.
  5. One entry per flip.

Returns (idx, direction, fill_price) — idx is the TOUCH bar, fill_price is the
kernel value there. Backtest.py measures the stop/target bracket from that fill
price (not the close), which is the whole point of a limit entry.
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
    """Wilder ATR. Vectorized via pandas ewm -- Wilder smoothing IS a
    first-order IIR filter (out[i] = out[i-1]*(1-a) + tr[i]*a), which is
    exactly what ewm(adjust=False) computes in C, so seed the recursion with
    the SMA of the first `period` true ranges (Wilder's own convention) and
    let ewm carry it forward. Numerically identical to the old bar-by-bar
    Python loop (verified to full float precision), just far faster --
    that loop was the dominant cost in large parameter sweeps."""
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


class KernelPullback:
    name = "kernel_pullback"
    wants_bars = True

    def __init__(self, h=8.0, r=8.0, x_0=8, wait_bars=3,
                 slope_len=3, slope_min=0.10, cooldown=3, atr_len=14,
                 allow_long=True, allow_short=True):
        self.h = float(h); self.r = float(r); self.x_0 = int(x_0)
        self.wait_bars = int(wait_bars)
        self.slope_len = int(slope_len); self.slope_min = slope_min
        self.cooldown = int(cooldown); self.atr_len = int(atr_len)
        self.allow_long = bool(allow_long); self.allow_short = bool(allow_short)

    def entries(self, bars):
        close = bars["close"].to_numpy(float)
        high = bars["high"].to_numpy(float); low = bars["low"].to_numpy(float)
        n = close.size
        # PER-SESSION kernel: rth_only=True leaves overnight gaps between sessions,
        # so a single smoother would blend yesterday's closes with today's price at
        # each open (kernel detaches from price by hundreds of pts). Reset per day.
        k = np.full(n, np.nan)
        # dtype=datetime64[ns]: plain .to_numpy() on a tz-aware index boxes
        # every timestamp as a Python object (very slow at 1-min scale over
        # multi-year data, and makes np.unique below ~40x slower too).
        # Session boundaries come out identical either way since normalize()
        # already snapped to local midnight before this point.
        dates = bars.index.normalize().to_numpy(dtype="datetime64[ns]")
        _, starts = np.unique(dates, return_index=True)
        bounds = list(np.sort(starts)) + [n]
        for s, e in zip(bounds[:-1], bounds[1:]):
            k[s:e] = _rqk(close[s:e], self.h, self.r, self.x_0)
        a = _atr(high, low, close, self.atr_len)

        rising = np.zeros(n, bool)
        rising[1:] = k[1:] > k[:-1]
        flip_up = np.zeros(n, bool); flip_dn = np.zeros(n, bool)
        flip_up[1:] = rising[1:] & ~rising[:-1]
        flip_dn[1:] = (~rising[1:]) & rising[:-1]

        slope = np.full(n, np.nan)                       # kernel change over slope_len bars
        if self.slope_len < n:
            slope[self.slope_len:] = k[self.slope_len:] - k[:-self.slope_len]
        with np.errstate(invalid="ignore"):
            slope_atr = slope / a                        # in ATR units

        flips = sorted([(f, 1) for f in np.flatnonzero(flip_up)] +
                       [(f, -1) for f in np.flatnonzero(flip_dn)])
        out_idx, out_dir, out_fill = [], [], []
        last_arm = -10**9
        for f, d in flips:
            if f - last_arm < self.cooldown:             # cooldown since last armed flip
                continue
            if not (np.isfinite(k[f]) and np.isfinite(slope_atr[f])):
                continue
            if self.slope_min is not None:               # slope guard (anti-chop)
                if d == 1 and slope_atr[f] < self.slope_min:
                    continue
                if d == -1 and slope_atr[f] > -self.slope_min:
                    continue
            last_arm = f                                 # armed (whether or not it fills)
            if (d == 1 and not self.allow_long) or (d == -1 and not self.allow_short):
                continue
            for j in range(f + 1, min(f + 1 + self.wait_bars, n)):   # wait for retrace
                if not np.isfinite(k[j]):
                    continue
                if d == 1 and low[j] <= k[j]:
                    out_idx.append(j); out_dir.append(1); out_fill.append(k[j]); break
                if d == -1 and high[j] >= k[j]:
                    out_idx.append(j); out_dir.append(-1); out_fill.append(k[j]); break

        if not out_idx:
            return np.array([], int), np.array([], int), np.array([], float)
        idx = np.array(out_idx, int); direction = np.array(out_dir, int); fill = np.array(out_fill, float)
        order = np.argsort(idx, kind="stable")
        idx, direction, fill = idx[order], direction[order], fill[order]
        uniq, keep = np.unique(idx, return_index=True)    # one entry per bar
        return idx[keep], direction[keep], fill[keep]


DEFAULTS = dict(h=8.0, r=8.0, x_0=8, wait_bars=3, slope_len=3, slope_min=0.10,
                cooldown=3, atr_len=14, allow_long=True, allow_short=True)


def get_strategy(**overrides):
    return KernelPullback(**{**DEFAULTS, **overrides})