"""
strategies/kernel_bos.py — kernel direction + break-of-structure (BOS), with a
LATCH: at most ONE signal per kernel leg. Ported to match the reference Pine
Script ("Kernel + BOS" indicator) exactly.

Entry (long; short mirrors):
  1. kernel direction : NW-RQK kernel rising (long bias) / falling (short
                        bias), computed PER SESSION by default
                        (reset_daily=True) so it never blends across
                        overnight gaps.
  2. FLIP             : marks the start of a new "leg" --
                            flip_up[i] = rising[i]  and not rising[i-1]
                            flip_dn[i] = falling[i] and     rising[i-1]
                        (asymmetric by design -- flip_dn specifically
                        requires the PRIOR bar to have been rising, not
                        merely "not falling". This matches the Pine source
                        literally; it is not a symmetrized approximation.)
  3. LATCH            : at most one signal per leg. Once a signal fires, no
                        more until the NEXT flip in that direction re-arms
                        it. This is what the previous version of this file
                        was missing -- it fired on every bar where
                        direction+BOS held, so one sustained trending leg
                        with several fresh BOS bars produced several
                        signals instead of one. The latch state is NOT
                        reset at session boundaries (matches the Pine
                        `var bool firedLong/firedShort`, which persists for
                        the whole script run) -- only the kernel itself
                        resets per session.
  4. BOS              : close breaks above the highest high (long) / below
                        the lowest low (short) of the `bos_lookback` bars
                        strictly BEFORE the current bar.
  5. conviction       : optional ATR-normalized slope filter, slope_min=0
                        (the Pine default) disables it.

No efficiency-ratio/chop gate, no `require_flip`/`agree` lookback window --
neither exists in the reference Pine script. The old `agree`-bars window was
an approximation of "only signal near a flip"; the latch now does that
exactly (and correctly), so both params are gone.

Interface: wants_bars=True; entries(bars)->(idx, dir); name; get_strategy(**over).
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
    """Wilder ATR, vectorized via pandas ewm (mathematically identical to the
    classic bar-by-bar recursion -- see kernel_pullback.py's _atr)."""
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


def _compute(bars, h, r, x_0, bos_lookback, slope_len, slope_min, atr_len,
             reset_daily, allow_long, allow_short):
    """All of KernelBOS's per-bar arithmetic, as one dict of arrays. The SOLE
    source of truth for signal generation -- KernelBOS.entries() (the live
    path) and KernelBOS.debug() (the bar-by-bar introspection table, for
    lining up against a reference Pine chart) both call this and nothing
    else, so debug() can never silently drift out of sync with entries()."""
    close = bars["close"].to_numpy(float)
    high = bars["high"].to_numpy(float); low = bars["low"].to_numpy(float)
    n = close.size

    # --- kernel: PER-SESSION by default (no overnight-gap blending) ---
    k = np.full(n, np.nan)
    if reset_daily:
        dates = bars.index.normalize().to_numpy(dtype="datetime64[ns]")
        _, starts = np.unique(dates, return_index=True)
        bounds = list(np.sort(starts)) + [n]
        for s, e in zip(bounds[:-1], bounds[1:]):
            k[s:e] = _rqk(close[s:e], h, r, x_0)
    else:
        k = _rqk(close, h, r, x_0)

    # direction (NaN-safe: comparisons against NaN are False in numpy, so
    # warmup / session-boundary bars naturally get rising=falling=False)
    rising = np.zeros(n, bool); falling = np.zeros(n, bool)
    rising[1:] = k[1:] > k[:-1]
    falling[1:] = k[1:] < k[:-1]

    # flips mark the start of a new leg (see module docstring re: the
    # intentional flip_dn asymmetry -- matches the Pine source exactly)
    flip_up = np.zeros(n, bool); flip_dn = np.zeros(n, bool)
    flip_up[1:] = rising[1:] & ~rising[:-1]
    flip_dn[1:] = falling[1:] & rising[:-1]

    # --- BOS: break of the bos_lookback bars strictly BEFORE this bar,
    # reset every session (calendar day) -- vectorized equivalent of
    # `groupby(date)["high"].transform(lambda s: s.shift(1).rolling(lb).max())`
    # (~80x faster: one rolling pass over the full array instead of one
    # per session via groupby+lambda, verified bit-for-bit identical). A
    # window is only "full" (valid) once >= lb prior bars exist IN THE
    # SAME session, which is exactly what bars_since_start >= lb checks.
    lb = bos_lookback
    dates_bos = bars.index.normalize().to_numpy(dtype="datetime64[ns]")
    _, day_starts = np.unique(dates_bos, return_index=True)
    bounds_bos = np.concatenate([np.sort(day_starts), [n]])
    session_start_pos = np.repeat(bounds_bos[:-1], np.diff(bounds_bos))
    bars_since_start = np.arange(n) - session_start_pos

    shifted_high = np.empty(n); shifted_high[0] = np.nan; shifted_high[1:] = high[:-1]
    shifted_low = np.empty(n); shifted_low[0] = np.nan; shifted_low[1:] = low[:-1]
    roll_max = pd.Series(shifted_high).rolling(lb, min_periods=lb).max().to_numpy()
    roll_min = pd.Series(shifted_low).rolling(lb, min_periods=lb).min().to_numpy()
    window_full = bars_since_start >= lb
    prior_high = np.where(window_full, roll_max, np.nan)
    prior_low = np.where(window_full, roll_min, np.nan)
    with np.errstate(invalid="ignore"):
        bull_bos = close > prior_high
        bear_bos = close < prior_low

    # --- optional ATR-normalized slope (conviction) filter, off by default ---
    a = _atr(high, low, close, atr_len)          # always computed -- debug() wants
    slope = np.full(n, np.nan)                    # to show it even when the filter's off
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
    cand_long = finite & rising & bull_bos & slope_ok_long & allow_long
    cand_short = finite & falling & bear_bos & slope_ok_short & allow_short

    # --- LATCH, vectorized: segment bars by cumulative flip count (all
    # bars from one flip to just before the next share a "leg id"), then
    # keep only the FIRST candidate bar in each leg. cand_idx is already
    # bar-ordered and flip-cumsum is non-decreasing, so "first occurrence
    # of each leg id" == "first qualifying bar in that leg", chronologically. ---
    long_ok = np.zeros(n, bool)
    cand_idx = np.flatnonzero(cand_long)
    if cand_idx.size:
        leg_id = np.cumsum(flip_up)[cand_idx]
        _, first_pos = np.unique(leg_id, return_index=True)
        long_ok[cand_idx[first_pos]] = True

    short_ok = np.zeros(n, bool)
    cand_idx = np.flatnonzero(cand_short)
    if cand_idx.size:
        leg_id = np.cumsum(flip_dn)[cand_idx]
        _, first_pos = np.unique(leg_id, return_index=True)
        short_ok[cand_idx[first_pos]] = True

    idx = np.flatnonzero(long_ok | short_ok)
    direction = np.where(long_ok[idx], 1, -1).astype(int)

    return dict(close=close, high=high, low=low, k=k, rising=rising, falling=falling,
                flip_up=flip_up, flip_dn=flip_dn, prior_high=prior_high, prior_low=prior_low,
                bull_bos=bull_bos, bear_bos=bear_bos, slope_atr=slope_atr,
                slope_ok_long=slope_ok_long, slope_ok_short=slope_ok_short,
                cand_long=cand_long, cand_short=cand_short,
                long_ok=long_ok, short_ok=short_ok, idx=idx, direction=direction)


class KernelBOS:
    name = "kernel_bos"
    wants_bars = True

    def __init__(self, h=8.0, r=8.0, x_0=8, bos_lookback=3,
                 slope_len=3, slope_min=0.0, atr_len=14,
                 reset_daily=True, allow_long=True, allow_short=True):
        self.h = float(h); self.r = float(r); self.x_0 = int(x_0)
        self.bos_lookback = int(bos_lookback)
        self.slope_len = int(slope_len); self.slope_min = float(slope_min)
        self.atr_len = int(atr_len)
        self.reset_daily = bool(reset_daily)
        self.allow_long = bool(allow_long); self.allow_short = bool(allow_short)

    def _args(self):
        return dict(h=self.h, r=self.r, x_0=self.x_0, bos_lookback=self.bos_lookback,
                    slope_len=self.slope_len, slope_min=self.slope_min, atr_len=self.atr_len,
                    reset_daily=self.reset_daily, allow_long=self.allow_long,
                    allow_short=self.allow_short)

    def entries(self, bars):
        d = _compute(bars, **self._args())
        return d["idx"], d["direction"]

    def debug(self, bars):
        """Bar-by-bar table of every intermediate value entries() computes --
        kernel value, rising/falling, flip_up/flip_dn, BOS levels/crosses,
        slope (ATR-normalized, shown even when slope_min=0 disables the
        filter), candidate flags, and the final latch outcome (fired/
        direction). Same _compute() call entries() uses, so this is
        guaranteed to match the live signals exactly -- line it up bar-by-bar
        against your TradingView chart to find exactly where (and on what
        condition) the two diverge."""
        d = _compute(bars, **self._args())
        fired = d["long_ok"] | d["short_ok"]
        direction = np.where(d["long_ok"], 1, np.where(d["short_ok"], -1, 0))
        return pd.DataFrame({
            "close": d["close"], "k": d["k"],
            "rising": d["rising"], "falling": d["falling"],
            "flip_up": d["flip_up"], "flip_dn": d["flip_dn"],
            "prior_high": d["prior_high"], "prior_low": d["prior_low"],
            "bull_bos": d["bull_bos"], "bear_bos": d["bear_bos"],
            "slope_atr": d["slope_atr"],
            "slope_ok_long": d["slope_ok_long"], "slope_ok_short": d["slope_ok_short"],
            "cand_long": d["cand_long"], "cand_short": d["cand_short"],
            "fired": fired, "direction": direction,
        }, index=bars.index)


DEFAULTS = dict(h=10.0, r=8.0, x_0=14, bos_lookback=6, slope_len=3, slope_min=0.4,
                atr_len=14, reset_daily=True, allow_long=True, allow_short=True)


def get_strategy(**overrides):
    return KernelBOS(**{**DEFAULTS, **overrides})
