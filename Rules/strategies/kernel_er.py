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
             body_frac_prev, check_next, min_gap_bars, sess_start, sess_end, max_scan_bars,
             use_slope):
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

    # SLOPE gate (toggleable). When use_slope=False: no slope magnitude gate and
    # no slope-based grey; direction falls back to the kernel flip (dirUp).
    if use_slope:
        slopePass  = np.isfinite(slope_atr) & (np.abs(slope_atr) >= slope_min)
        slope_up   = np.isfinite(slope_atr) & (slope_atr >=  slope_min)
        slope_down = np.isfinite(slope_atr) & (slope_atr <= -slope_min)
    else:
        slopePass  = np.isfinite(k)                 # slope never blocks the regime
        slope_up   = np.isfinite(k) & dirUp         # direction from kernel flip
        slope_down = np.isfinite(k) & ~dirUp
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
    # entry direction requires DIRECTIONAL slope conviction (slope pointing the
    # trade's way with magnitude), plus regime on. dirUp alone (kernel rose 1 bar)
    # is NOT enough -- that let weak wiggles fire against the prevailing slope.
    dir_long  = regimeOn & slope_up   & allow_long
    dir_short = regimeOn & slope_down & allow_short

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

    # ---- ENTRY: forward-scan within each leg for the first Rule-1-passing bar ----
    # State machine matching the Pine: on each anchor (regime transition OR a
    # session-open where the regime was already ON), ARM the leg with its
    # direction. While armed AND regime stays ON AND we are in session, fire on
    # the FIRST bar that passes Rule 1 in the leg's direction, then latch the leg
    # (one entry per leg). If the regime turns off (or session ends) before any
    # bar passes, the leg produces no entry. check_next=False restricts firing to
    # the transition bar only (no forward scan). Rule 1 (body>=min_body_pts etc.)
    # prevents a tiny early bar from stealing the entry -- the scan waits for the
    # first REAL candle, which is what the chart shows.
    keep_idx, keep_dir = [], []
    last_entry = -10**9
    armed = False
    leg_is_long = True
    arm_bar = -1                 # bar where the current leg armed (for bounded scan)

    # direction-flip anchor: within a CONTINUOUS regime, a kernel direction reversal
    # (down-leg -> up-leg) is a NEW tradable leg. regimeJustOn misses it because
    # regimeOn never went False across the flip. Detect the first bar the in-regime
    # direction changes.
    dir_sign = np.where(dir_long, 1, np.where(dir_short, -1, 0))
    flip_anchor = np.zeros(n, bool)
    flip_anchor[1:] = regimeOn[1:] & (dir_sign[1:] != 0) & (dir_sign[1:] != dir_sign[:-1])
    anchor = regimeJustOn | session_open_anchor | flip_anchor
    for i in range(n):
        # arm (or RE-arm) on a fresh anchor, capturing the CURRENT direction. A
        # direction-flip anchor re-arms even if we were armed the other way, so a
        # down-leg -> up-leg reversal starts a fresh long leg.
        if anchor[i] and in_session[i] and (dir_long[i] or dir_short[i]):
            armed = True
            leg_is_long = bool(dir_long[i])
            arm_bar = i          # remember where this leg armed
        # disarm if regime ended or we left the session
        if not regimeOn[i] or not in_session[i]:
            armed = False
        if not armed:
            continue
        # if not scanning forward, only the anchor bar itself may fire
        if (not check_next) and not anchor[i]:
            continue
        # BOUNDED SCAN: don't chase a Rule-passing bar more than max_scan_bars past
        # the arm (slope-cross). Beyond that, abandon the leg -- keeps the entry near
        # the slope-cross instead of firing on a distant big candle.
        if check_next and (i - arm_bar) > max_scan_bars:
            armed = False
            continue
        # direction must still match the leg's direction at the firing bar
        cur_dir_ok = dir_long[i] if leg_is_long else dir_short[i]
        if not cur_dir_ok:
            continue
        if _passes(i, leg_is_long):
            if (i - last_entry) < min_gap_bars:
                armed = False           # cooldown suppresses, but leg is consumed
                continue
            keep_idx.append(i); keep_dir.append(1 if leg_is_long else -1)
            last_entry = i
            armed = False               # latch: one entry per leg

    order = np.argsort(keep_idx) if keep_idx else []
    idx = np.array([keep_idx[j] for j in order], dtype=int) if keep_idx else np.array([], int)
    direction = np.array([keep_dir[j] for j in order], dtype=int) if keep_dir else np.array([], int)
    return dict(k=k, slope_atr=slope_atr, slope_up=slope_up, slope_down=slope_down,
                dir_long=dir_long, dir_short=dir_short, er=er, isTrend=isTrend,
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
                 check_next=True, min_gap_bars=0, sess_start=None, sess_end=None,
                 max_scan_bars=1, use_slope=True):
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
        self.max_scan_bars = int(max_scan_bars)
        self.use_slope = bool(use_slope)

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
                    sess_start=self.sess_start, sess_end=self.sess_end,
                    max_scan_bars=self.max_scan_bars, use_slope=self.use_slope)

    def entries(self, bars):
        d = _compute(bars, **self._args())
        return d["idx"], d["direction"]

    def debug(self, bars):
        d = _compute(bars, **self._args())
        return pd.DataFrame({"k": d["k"], "slope_atr": d["slope_atr"],
                             "slope_up": d["slope_up"], "slope_down": d["slope_down"],
                             "dir_long": d["dir_long"], "dir_short": d["dir_short"],
                             "er": d["er"], "isTrend": d["isTrend"],
                             "regimeOn": d["regimeOn"], "regimeJustOn": d["regimeJustOn"],
                             "dirUp": d["dirUp"]},
                            index=bars.index)


DEFAULTS = dict(h=8.0, r=8.0, x_0=25, slope_len=2, slope_min=0.3, atr_len=14,
                er_len=10, er_min=0.30, confirm_bars=1, src="close",
                allow_long=True, allow_short=True,
                filter_clear_prior=False, filter_displacement=False, disp_wick_frac=0.20,
                min_body_pts=15.0, use_er=True, body_frac_prev=0.85,
                check_next=True, min_gap_bars=0, sess_start=None, sess_end=None,
                max_scan_bars=1, use_slope=True)


def get_strategy(**overrides):
    return KernelRegime(**{**DEFAULTS, **overrides})