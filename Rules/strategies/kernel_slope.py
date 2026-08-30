"""
strategies/kernel_slope.py — pure trend-following kernel entry: NW-RQK kernel
DIRECTION + ATR-normalized SLOPE conviction, with a LATCH (one signal per kernel
leg). No BOS, no efficiency-ratio gate.

Kernel SOURCE is selectable via `src` (default "close", matching the TradingView
default). Options: "close", "hl2", "hlc3", "ohlc4". Changing src is a DIFFERENT
variant -- re-run IS/OOS and match your Pine `src`.

Entry (long; short mirrors):
  1. kernel direction : NW-RQK kernel rising (long)/falling (short), PER SESSION.
  2. FLIP             : bar a new leg starts (rising & not rising[-1]).
  3. SLOPE conviction : |k - k[slope_len]| / ATR >= slope_min (ATR units).
  4. LATCH            : one signal per leg.

OPTIONAL NEXT-CANDLE BODY GATE (gate_next_body=True, default OFF):
  A signal on bar i is only taken if bar i+1's body is NOT >= next_body_mult x
  bar i's body (and, if next_body_min_pts>0, only when i+1's body also exceeds
  that absolute point floor -- so tiny signal candles don't trip the ratio).
  CAUSAL: when the gate is on, the ENTRY IS DELAYED to bar i+1 -- you only act
  after the checked candle has closed, so no lookahead. Direction is still the
  signal bar's direction; only the execution bar moves forward by one.

Interface: wants_bars=True; entries(bars)->(idx, dir); name; get_strategy(**over).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _src_series(bars, kind):
    o = bars["open"].to_numpy(float); h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float);  c = bars["close"].to_numpy(float)
    if kind == "close": return c
    if kind == "hl2":   return (h + l) / 2.0
    if kind == "hlc3":  return (h + l + c) / 3.0
    if kind == "ohlc4": return (o + h + l + c) / 4.0
    raise ValueError(f"unknown src {kind!r} (use close/hl2/hlc3/ohlc4)")


def _rqk(price, h, r, x_0):
    L = int(x_0) + 1
    i = np.arange(L + 1, dtype=float)
    w = (1.0 + (i * i) / (2.0 * r * h * h)) ** (-r)
    w /= w.sum()
    out = np.convolve(price, w)[:price.size]
    out[:L] = np.nan
    return out


def _atr(high, low, close, period=14):
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
             reset_daily, allow_long, allow_short, src,
             gate_next_body, next_body_mult, next_body_min_pts):
    price = _src_series(bars, src)
    open_ = bars["open"].to_numpy(float)
    high = bars["high"].to_numpy(float); low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    n = close.size

    k = np.full(n, np.nan)
    if reset_daily:
        dates = bars.index.normalize().to_numpy(dtype="datetime64[ns]")
        _, starts = np.unique(dates, return_index=True)
        bounds = list(np.sort(starts)) + [n]
        for s, e in zip(bounds[:-1], bounds[1:]):
            k[s:e] = _rqk(price[s:e], h, r, x_0)
    else:
        k = _rqk(price, h, r, x_0)

    rising = np.zeros(n, bool); falling = np.zeros(n, bool)
    rising[1:] = k[1:] > k[:-1]
    falling[1:] = k[1:] < k[:-1]
    flip_up = np.zeros(n, bool); flip_dn = np.zeros(n, bool)
    flip_up[1:] = rising[1:] & ~rising[:-1]
    flip_dn[1:] = falling[1:] & ~falling[:-1]

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
    cand_long = finite & rising & slope_ok_long & allow_long
    cand_short = finite & falling & slope_ok_short & allow_short

    # LATCH: one signal per leg
    long_ok = np.zeros(n, bool)
    ci = np.flatnonzero(cand_long)
    if ci.size:
        leg = np.cumsum(flip_up)[ci]; _, first = np.unique(leg, return_index=True)
        long_ok[ci[first]] = True
    short_ok = np.zeros(n, bool)
    ci = np.flatnonzero(cand_short)
    if ci.size:
        leg = np.cumsum(flip_dn)[ci]; _, first = np.unique(leg, return_index=True)
        short_ok[ci[first]] = True

    sig_bar = np.flatnonzero(long_ok | short_ok)          # bars where a signal fired

    if not gate_next_body:
        idx = sig_bar
        direction = np.where(long_ok[idx], 1, -1).astype(int)
    else:
        # CAUSAL next-candle body gate: inspect bar i+1, ENTER on i+1 if it passes.
        body = np.abs(close - open_)
        keep_idx = []          # execution bars (i+1)
        keep_dir = []          # direction from signal bar i
        for i in sig_bar:
            j = i + 1
            if j >= n:
                continue                                  # no next bar -> can't validate
            reject = (body[i] > 0 and body[j] >= next_body_mult * body[i]
                      and (next_body_min_pts <= 0 or body[j] >= next_body_min_pts))
            if reject:
                continue                                  # violent next candle -> skip
            keep_idx.append(j)                            # execute on the checked bar
            keep_dir.append(1 if long_ok[i] else -1)
        idx = np.array(keep_idx, int)
        direction = np.array(keep_dir, int)

    return dict(close=close, k=k, rising=rising, falling=falling,
                flip_up=flip_up, flip_dn=flip_dn, slope_atr=slope_atr,
                cand_long=cand_long, cand_short=cand_short,
                long_ok=long_ok, short_ok=short_ok, idx=idx, direction=direction)


class KernelSlope:
    name = "kernel_slope"
    wants_bars = True

    def __init__(self, h=8.0, r=8.0, x_0=8, slope_len=3, slope_min=0.3, atr_len=14,
                 reset_daily=True, allow_long=True, allow_short=True, src="close",
                 gate_next_body=False, next_body_mult=1.5, next_body_min_pts=0.0):
        self.h = float(h); self.r = float(r); self.x_0 = int(x_0)
        self.slope_len = int(slope_len); self.slope_min = float(slope_min)
        self.atr_len = int(atr_len); self.reset_daily = bool(reset_daily)
        self.allow_long = bool(allow_long); self.allow_short = bool(allow_short)
        self.src = str(src)
        self.gate_next_body = bool(gate_next_body)
        self.next_body_mult = float(next_body_mult)
        self.next_body_min_pts = float(next_body_min_pts)

    def _args(self):
        return dict(h=self.h, r=self.r, x_0=self.x_0, slope_len=self.slope_len,
                    slope_min=self.slope_min, atr_len=self.atr_len,
                    reset_daily=self.reset_daily, allow_long=self.allow_long,
                    allow_short=self.allow_short, src=self.src,
                    gate_next_body=self.gate_next_body,
                    next_body_mult=self.next_body_mult,
                    next_body_min_pts=self.next_body_min_pts)

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
                reset_daily=True, allow_long=True, allow_short=True, src="close",
                gate_next_body=True, next_body_mult=1.5, next_body_min_pts=0.0)


def get_strategy(**overrides):
    return KernelSlope(**{**DEFAULTS, **overrides})