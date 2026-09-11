"""
strategies/dual_kernel.py — dual-kernel CROSSOVER entry (matches the Pine
'Dual-Kernel Crossover + displacement' indicator exactly).

TRIGGER: fast kernel crosses the slow kernel (ta.crossover). No cross -> no entry.
GATES (all optional, applied on top of the cross):
  - slope confirmation: slopeF >= slope_min_f (dir), slopeS >= slope_min_s.
      NOTE: the slow slope is often AGAINST the cross AT the cross bar, so keep
      slope_min_s low/0 or it kills entries.
  - displacement (Rule 1, default ON): the cross candle's body must clear the
      prior candle's wick in the trade direction, and body >= min_body_pts.
      long : body_top > prior_high ; short: body_bot < prior_low.
  - cross quality (default 'none'):
      'force'    -> |slopeF - slopeS| >= force_min AT the cross (cross angle).
      'post_sep' -> separation |kF-kS|/ATR >= sep_conf, sep_wait bars AFTER the
                    cross (delays the entry that many bars).
  - session gate (sess_start/sess_end), cooldown (min_gap_bars).

Kernels computed continuously on the FULL bar series (no reset).
Interface: wants_bars=True; entries(bars)->(idx,dir); name; get_strategy(**over).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import datetime as _dt


def _rqk(price, h, x0, r):
    L = int(x0) + 1
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
    out = np.full_like(tr, np.nan); n = tr.size
    if n >= period:
        seed = tr[:period].mean(); out[period - 1] = seed
        if n > period:
            tail = np.concatenate(([seed], tr[period:]))
            out[period:] = pd.Series(tail).ewm(alpha=1.0/period, adjust=False).mean().to_numpy()[1:]
    return out


def _compute(bars, hF, x0F, rF, hS, x0S, rS, atr_len,
             slope_len_f, slope_min_f, slope_len_s, slope_min_s,
             use_disp, min_body_pts, cross_filter, force_min, sep_conf, sep_wait,
             min_gap_bars, sess_start, sess_end, allow_long, allow_short, src):
    o = bars["open"].to_numpy(float); hi = bars["high"].to_numpy(float)
    lo = bars["low"].to_numpy(float);  c = bars["close"].to_numpy(float)
    n = c.size
    price = c  # src=close (matches Pine default)

    kF = _rqk(price, hF, x0F, rF)
    kS = _rqk(price, hS, x0S, rS)
    a  = _atr(hi, lo, c, atr_len)

    with np.errstate(invalid="ignore"):
        slopeF = np.concatenate([np.full(slope_len_f, np.nan), kF[slope_len_f:] - kF[:-slope_len_f]]) / a
        slopeS = np.concatenate([np.full(slope_len_s, np.nan), kS[slope_len_s:] - kS[:-slope_len_s]]) / a

    # crossover (fast vs slow)
    fgs = kF > kS
    bullCross = np.zeros(n, bool); bearCross = np.zeros(n, bool)
    bullCross[1:] =  fgs[1:] & ~fgs[:-1]
    bearCross[1:] = ~fgs[1:] &  fgs[:-1]

    # slope gates (0 = off)
    sFu = (slope_min_f <= 0) | (slopeF >=  slope_min_f)
    sFd = (slope_min_f <= 0) | (slopeF <= -slope_min_f)
    sSu = (slope_min_s <= 0) | (slopeS >=  slope_min_s)
    sSd = (slope_min_s <= 0) | (slopeS <= -slope_min_s)

    # displacement (Rule 1)
    body_top = np.maximum(o, c); body_bot = np.minimum(o, c); body = np.abs(c - o)
    bull = c > o; bear = c < o
    prev_hi = np.concatenate([[np.inf], hi[:-1]]); prev_lo = np.concatenate([[-np.inf], lo[:-1]])
    if use_disp:
        dispL = bull & (body_top > prev_hi) & (body >= min_body_pts)
        dispS = bear & (body_bot < prev_lo) & (body >= min_body_pts)
    else:
        dispL = np.ones(n, bool); dispS = np.ones(n, bool)

    # cross quality
    crossForce = slopeF - slopeS
    if cross_filter == "force":
        forceOkL = crossForce >=  force_min
        forceOkS = crossForce <= -force_min
    else:
        forceOkL = np.ones(n, bool); forceOkS = np.ones(n, bool)

    # post_sep: bars since last cross, confirm separation sep_wait bars after
    bBull = np.full(n, 10**9, int); bBear = np.full(n, 10**9, int)
    cb = cs = 10**9
    for i in range(n):
        cb = 0 if bullCross[i] else cb + 1
        cs = 0 if bearCross[i] else cs + 1
        bBull[i] = cb; bBear[i] = cs
    with np.errstate(invalid="ignore"):
        sepNow = np.abs(kF - kS) / a
    postOkL = (cross_filter == "post_sep") & (bBull == sep_wait) & (sepNow >= sep_conf) & (kF > kS)
    postOkS = (cross_filter == "post_sep") & (bBear == sep_wait) & (sepNow >= sep_conf) & (kF < kS)

    # session
    if sess_start is not None and sess_end is not None:
        s0 = _dt.time.fromisoformat(sess_start); s1 = _dt.time.fromisoformat(sess_end)
        insess = np.array([s0 <= t <= s1 for t in bars.index.time], bool)
    else:
        insess = np.ones(n, bool)

    finite = np.isfinite(kF) & np.isfinite(kS) & np.isfinite(a)

    if cross_filter == "post_sep":
        longSig  = postOkL & sFu & sSu & dispL & bool(allow_long)
        shortSig = postOkS & sFd & sSd & dispS & bool(allow_short)
    else:
        longSig  = bullCross & sFu & sSu & forceOkL & dispL & bool(allow_long)
        shortSig = bearCross & sFd & sSd & forceOkS & dispS & bool(allow_short)
    longSig &= finite & insess; shortSig &= finite & insess

    # cooldown (sequential)
    keep_idx, keep_dir = [], []; last = -10**9
    cand = np.flatnonzero(longSig | shortSig)
    for i in cand:
        if (i - last) < min_gap_bars: continue
        keep_idx.append(i); keep_dir.append(1 if longSig[i] else -1); last = i

    idx = np.array(keep_idx, int); direction = np.array(keep_dir, int)
    return dict(kF=kF, kS=kS, slopeF=slopeF, slopeS=slopeS, crossForce=crossForce,
                sepNow=sepNow, bullCross=bullCross, bearCross=bearCross,
                idx=idx, direction=direction)


class DualKernel:
    name = "dual_kernel"
    wants_bars = True

    def __init__(self, hF=4.0, x0F=1, rF=8.0, hS=12.0, x0S=3, rS=8.0, atr_len=14,
                 slope_len_f=1, slope_min_f=0.0, slope_len_s=3, slope_min_s=0.0,
                 use_disp=True, min_body_pts=0.0, cross_filter="none",
                 force_min=0.2, sep_conf=0.2, sep_wait=2,
                 min_gap_bars=5, sess_start="09:30", sess_end="11:00",
                 allow_long=True, allow_short=True, src="close"):
        for k, v in dict(hF=hF, rF=rF, hS=hS, rS=rS, force_min=force_min, sep_conf=sep_conf,
                         slope_min_f=slope_min_f, slope_min_s=slope_min_s, min_body_pts=min_body_pts).items():
            setattr(self, k, float(v))
        for k, v in dict(x0F=x0F, x0S=x0S, atr_len=atr_len, slope_len_f=slope_len_f,
                         slope_len_s=slope_len_s, sep_wait=sep_wait, min_gap_bars=min_gap_bars).items():
            setattr(self, k, int(v))
        self.use_disp = bool(use_disp); self.allow_long = bool(allow_long); self.allow_short = bool(allow_short)
        self.cross_filter = str(cross_filter); self.sess_start = sess_start; self.sess_end = sess_end
        self.src = str(src)

    def _args(self):
        return dict(hF=self.hF, x0F=self.x0F, rF=self.rF, hS=self.hS, x0S=self.x0S, rS=self.rS,
                    atr_len=self.atr_len, slope_len_f=self.slope_len_f, slope_min_f=self.slope_min_f,
                    slope_len_s=self.slope_len_s, slope_min_s=self.slope_min_s, use_disp=self.use_disp,
                    min_body_pts=self.min_body_pts, cross_filter=self.cross_filter, force_min=self.force_min,
                    sep_conf=self.sep_conf, sep_wait=self.sep_wait, min_gap_bars=self.min_gap_bars,
                    sess_start=self.sess_start, sess_end=self.sess_end, allow_long=self.allow_long,
                    allow_short=self.allow_short, src=self.src)

    def entries(self, bars):
        d = _compute(bars, **self._args()); return d["idx"], d["direction"]

    def debug(self, bars):
        d = _compute(bars, **self._args())
        return pd.DataFrame({"kF": d["kF"], "kS": d["kS"], "slopeF": d["slopeF"], "slopeS": d["slopeS"],
                             "crossForce": d["crossForce"], "sepNow": d["sepNow"],
                             "bullCross": d["bullCross"], "bearCross": d["bearCross"]}, index=bars.index)


DEFAULTS = dict(hF=4.0, x0F=1, rF=8.0, hS=12.0, x0S=3, rS=8.0, atr_len=14,
                slope_len_f=1, slope_min_f=0.0, slope_len_s=3, slope_min_s=0.0,
                use_disp=True, min_body_pts=0.0, cross_filter="none",
                force_min=0.2, sep_conf=0.2, sep_wait=2,
                min_gap_bars=5, sess_start="09:30", sess_end="11:00",
                allow_long=True, allow_short=True, src="close")


def get_strategy(**over):
    return DualKernel(**{**DEFAULTS, **over})