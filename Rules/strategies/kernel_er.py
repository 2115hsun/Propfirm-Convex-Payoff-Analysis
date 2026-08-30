"""
strategies/kernel_regime.py — single-kernel NW-RQK REGIME-TRANSITION entry
(matches the Pine 'NW-RQK Regime + arrows' with confirm_bars=1) plus two
optional DISCRETIONARY candle filters.

ENTRY (base):
  regime = slopePass AND erPass, where
    slopePass = |slope_atr| >= slope_min   (slope_atr = (k - k[slope_len]) / ATR)
    erPass    = |ER| >= er_min
  debounced with confirm_bars consecutive regime bars (default 1),
  fire on the FIRST confirmed regime-ON bar (regime just turned on).
  direction = kernel slope at that bar (dirUp = k > k[1]).

DISCRETIONARY FILTERS (both default OFF; each toggleable):

  filter_clear_prior (Rule 1): the entry candle must (1) have body >= min_body_pts,
    (2) close in the SIGNAL direction (long=bullish, short=bearish), and (3) EITHER
    displace past the prior bar's wick tip in the signal direction, OR have a body
    >= body_frac_prev * the prior body (a big candle qualifies without clearing).

  filter_displacement (Rule 2): entry bar must be a DISPLACEMENT candle:
      2a: close beyond the prior candle's extreme in trade direction
          long : close > prior_high ; short: close < prior_low
      2b: the LEADING wick (direction of travel) < disp_wick_frac * body
          long : upper_wick  < frac * body ; short: lower_wick < frac * body
    (2a overlaps Rule 1's clear; kept separate so each can be toggled.)

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
    raise ValueError(f"unknown src {kind!r}")


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
        seed = tr[:period].mean(); out[period - 1] = seed
        if n > period:
            tail = np.concatenate(([seed], tr[period:]))
            out[period:] = pd.Series(tail).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()[1:]
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


def _compute(bars, h, r, x_0, slope_len, slope_min, atr_len, er_len, er_min,
             confirm_bars, src, allow_long, allow_short,
             filter_clear_prior, filter_displacement, disp_wick_frac, min_body_pts, use_er,
             body_frac_prev, check_next, min_gap_bars, sess_start, sess_end):
    price = _src_series(bars, src)
    o = bars["open"].to_numpy(float); hi = bars["high"].to_numpy(float)
    lo = bars["low"].to_numpy(float); c = bars["close"].to_numpy(float)
    n = c.size

    k = _rqk(price, h, r, x_0)
    a = _atr(hi, lo, c, atr_len)
    er = np.abs(_efficiency_ratio(c, er_len))

    slope = np.full(n, np.nan)
    if slope_len < n:
        slope[slope_len:] = k[slope_len:] - k[:-slope_len]
    with np.errstate(invalid="ignore"):
        slope_atr = slope / a

    dirUp = np.zeros(n, bool); dirUp[1:] = k[1:] > k[:-1]

    slopePass = np.isfinite(slope_atr) & (np.abs(slope_atr) >= slope_min)
    erPass = (er >= er_min) if use_er else np.ones(n, bool)   # ER gate toggleable
    isTrend = np.isfinite(k) & slopePass & erPass

    # debounce: consecutive isTrend bars; regimeOn once run >= confirm_bars
    run = np.zeros(n, int)
    for i in range(1, n):
        run[i] = run[i - 1] + 1 if isTrend[i] else 0
    regimeOn = run >= int(confirm_bars)
    regimeJustOn = np.zeros(n, bool)
    regimeJustOn[1:] = regimeOn[1:] & ~regimeOn[:-1]

    # leg id: increments each time a new regime-ON run starts; -1 when off.
    leg_id = np.full(n, -1, int)
    cur = -1
    for i in range(n):
        if regimeOn[i] and (i == 0 or not regimeOn[i - 1]):
            cur += 1            # new leg starts
        leg_id[i] = cur if regimeOn[i] else -1

    # direction candidates per bar (from kernel slope)
    dir_long  = regimeOn & dirUp & allow_long
    dir_short = regimeOn & ~dirUp & allow_short

    # session gate: only bars whose timestamp is in [sess_start, sess_end] may FIRE.
    # kernel/regime already computed on ALL bars above (continuous, no reset).
    times = bars.index
    if sess_start is not None and sess_end is not None:
        import datetime as _dt
        s0 = _dt.time.fromisoformat(sess_start); s1 = _dt.time.fromisoformat(sess_end)
        tt = times.time
        in_session = np.array([s0 <= t <= s1 for t in tt], dtype=bool)
    else:
        in_session = np.ones(n, bool)

    # ---- discretionary candle filters (causal: use entry bar i and prior i-1) ----
    body = np.abs(c - o)
    upper_wick = hi - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - lo
    bullish = c > o
    bearish = c < o

    def _passes(i, is_long):
        if i < 1:
            return False
        if filter_clear_prior:
            body_top = max(o[i], c[i]); body_bot = min(o[i], c[i])
            if body[i] < min_body_pts:
                return False
            if is_long and not bullish[i]:
                return False
            if (not is_long) and not bearish[i]:
                return False
            prev_body = body[i - 1]
            displ = (body_top > hi[i - 1]) if is_long else (body_bot < lo[i - 1])
            big_enough = (prev_body > 0) and (body[i] >= body_frac_prev * prev_body)
            if not (displ or big_enough):
                return False
        if filter_displacement:
            if is_long and not (c[i] > hi[i - 1]):
                return False
            if (not is_long) and not (c[i] < lo[i - 1]):
                return False
            b = body[i]
            if b <= 0:
                return False
            if is_long and not (upper_wick[i] < disp_wick_frac * b):
                return False
            if (not is_long) and not (lower_wick[i] < disp_wick_frac * b):
                return False
        return True

    # ---- session-open anchors: first in-session bar/day where regime already ON ----
    # (a leg that began BEFORE the session should still be tradeable at the open)
    session_open_anchor = np.zeros(n, bool)
    if sess_start is not None:
        dates = bars.index.normalize().to_numpy()
        # first in-session index for each day
        in_sess_idx = np.flatnonzero(in_session)
        if in_sess_idx.size:
            day_of = dates[in_sess_idx]
            # first occurrence per day
            _, first_pos = np.unique(day_of, return_index=True)
            for fp in first_pos:
                bar = in_sess_idx[fp]
                # only if regime is ON here AND it is NOT already a mid-session transition
                if regimeOn[bar] and not regimeJustOn[bar]:
                    session_open_anchor[bar] = True

    # ---- ENTRY: anchor to the regime TRANSITION bar of each leg ----
    # For each leg, the anchor is its transition bar (regimeJustOn). We fire on the
    # transition bar if it passes Rule 1 AND is in session; else, if check_next is
    # ON, we allow the SINGLE next bar (transition+1) if IT passes and is in session.
    # We do NOT scan the whole leg (that let an early marginal bar steal the entry).
    # A leg fires at most once (latch). Direction is taken from the transition bar.
    keep_idx, keep_dir = [], []
    last_entry = -10**9         # bar index of the last accepted entry (for cooldown)

    anchor_bars = np.flatnonzero(regimeJustOn | session_open_anchor)
    for t in anchor_bars:
        is_long = bool(dir_long[t]) if (dir_long[t] or dir_short[t]) else None
        if is_long is None:
            continue
        # candidate bars for THIS leg: the transition bar, then optionally +1
        cand = [t]
        if check_next:
            cand.append(t + 1)
        fired = False
        for i in cand:
            if i >= n or not in_session[i]:
                continue
            # direction must still match the leg's transition direction and regime on
            if not regimeOn[i]:
                continue
            if is_long and not dir_long[i] and i != t:
                # on the +1 bar, require same direction as the transition
                if dirUp[i] != is_long:
                    continue
            if _passes(i, is_long):
                # cooldown: must be >= min_gap_bars since last accepted entry
                if (i - last_entry) < min_gap_bars:
                    fired = True    # suppressed by cooldown, but leg is 'consumed'
                    break
                keep_idx.append(i); keep_dir.append(1 if is_long else -1)
                last_entry = i
                fired = True
                break
        # if not fired, leg produced no entry (transition+grace both failed/out-of-session)

    order = np.argsort(keep_idx) if keep_idx else []
    idx = np.array([keep_idx[j] for j in order], dtype=int) if keep_idx else np.array([], int)
    direction = np.array([keep_dir[j] for j in order], dtype=int) if keep_dir else np.array([], int)
    return dict(k=k, slope_atr=slope_atr, er=er, isTrend=isTrend,
                regimeOn=regimeOn, regimeJustOn=regimeJustOn, dirUp=dirUp,
                idx=idx, direction=direction)


class KernelRegime:
    name = "kernel_regime"
    wants_bars = True

    def __init__(self, h=8.0, r=8.0, x_0=25, slope_len=2, slope_min=0.3, atr_len=14,
                 er_len=10, er_min=0.30, confirm_bars=1, src="close",
                 allow_long=True, allow_short=True,
                 filter_clear_prior=False, filter_displacement=False, disp_wick_frac=0.20,
                 min_body_pts=15.0, use_er=True, body_frac_prev=0.85,
                 check_next=True, min_gap_bars=0, sess_start=None, sess_end=None):
        self.h = float(h); self.r = float(r); self.x_0 = int(x_0)
        self.slope_len = int(slope_len); self.slope_min = float(slope_min)
        self.atr_len = int(atr_len); self.er_len = int(er_len); self.er_min = float(er_min)
        self.confirm_bars = int(confirm_bars); self.src = str(src)
        self.allow_long = bool(allow_long); self.allow_short = bool(allow_short)
        self.filter_clear_prior = bool(filter_clear_prior)
        self.filter_displacement = bool(filter_displacement)
        self.disp_wick_frac = float(disp_wick_frac)
        self.min_body_pts = float(min_body_pts)
        self.use_er = bool(use_er)
        self.body_frac_prev = float(body_frac_prev)
        self.check_next = bool(check_next)
        self.min_gap_bars = int(min_gap_bars)
        self.sess_start = sess_start
        self.sess_end = sess_end

    def _args(self):
        return dict(h=self.h, r=self.r, x_0=self.x_0, slope_len=self.slope_len,
                    slope_min=self.slope_min, atr_len=self.atr_len, er_len=self.er_len,
                    er_min=self.er_min, confirm_bars=self.confirm_bars, src=self.src,
                    allow_long=self.allow_long, allow_short=self.allow_short,
                    filter_clear_prior=self.filter_clear_prior,
                    filter_displacement=self.filter_displacement,
                    disp_wick_frac=self.disp_wick_frac,
                    min_body_pts=self.min_body_pts, use_er=self.use_er,
                    body_frac_prev=self.body_frac_prev,
                    check_next=self.check_next, min_gap_bars=self.min_gap_bars,
                    sess_start=self.sess_start, sess_end=self.sess_end)

    def entries(self, bars):
        d = _compute(bars, **self._args())
        return d["idx"], d["direction"]

    def debug(self, bars):
        d = _compute(bars, **self._args())
        return pd.DataFrame({"k": d["k"], "slope_atr": d["slope_atr"], "er": d["er"],
                             "isTrend": d["isTrend"], "regimeOn": d["regimeOn"],
                             "regimeJustOn": d["regimeJustOn"], "dirUp": d["dirUp"]},
                            index=bars.index)


DEFAULTS = dict(h=8.0, r=8.0, x_0=25, slope_len=2, slope_min=0.3, atr_len=14,
                er_len=10, er_min=0.30, confirm_bars=1, src="close",
                allow_long=True, allow_short=True,
                filter_clear_prior=False, filter_displacement=False, disp_wick_frac=0.20,
                min_body_pts=15.0, use_er=True, body_frac_prev=0.85,
                check_next=True, min_gap_bars=0, sess_start=None, sess_end=None)


def get_strategy(**overrides):
    return KernelRegime(**{**DEFAULTS, **overrides})