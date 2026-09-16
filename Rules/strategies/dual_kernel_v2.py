"""
strategies/dual_kernel_v2.py — dual-kernel crossover with:
  - displacement scan window (cross bar / +1 / +2 after cross), entry bar MUST displace
  - body-size + optional close-beyond-prior-wick displacement definition
  - extension filter (skip if the extBars candles BEFORE the entry already ran extPts)
  - ER chop gate (toggle)
  - cross-quality "open speed" (kernels pulling apart fast at the cross)
  - slope confirmation (optional), session, cooldown, one entry per cross (latch)

Matches the Pine 'Dual-Kernel Cross + disp-scan + ext/ER/wick' exactly.
Kernels computed continuously (no reset). Interface: entries(bars)->(idx,dir).
"""
from __future__ import annotations
import numpy as np, pandas as pd, datetime as _dt


def _rqk(price, h, x0, r):
    L = int(x0) + 1
    i = np.arange(L + 1, dtype=float)
    w = (1.0 + (i * i) / (2.0 * r * h * h)) ** (-r); w /= w.sum()
    out = np.convolve(price, w)[:price.size]; out[:L] = np.nan
    return out


def _atr(high, low, close, p=14):
    h, l, c = map(lambda x: np.asarray(x, float), (high, low, close))
    prev = np.empty_like(c); prev[0] = c[0]; prev[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    out = np.full_like(tr, np.nan); n = tr.size
    if n >= p:
        s = tr[:p].mean(); out[p - 1] = s
        if n > p:
            t = np.concatenate(([s], tr[p:]))
            out[p:] = pd.Series(t).ewm(alpha=1.0 / p, adjust=False).mean().to_numpy()[1:]
    return out


def _compute(bars, hF, x0F, rF, hS, x0S, rS, atr_len,
             slope_len_f, slope_min_f, slope_len_s, slope_min_s,
             min_body_pts, use_wick, scan_at, scan_p1, scan_p2,
             use_ext, ext_bars, ext_pts,
             use_er, er_len, er_min,
             use_cq, open_min,
             min_gap_bars, sess_start, sess_end, allow_long, allow_short, src):
    o = bars["open"].to_numpy(float); hi = bars["high"].to_numpy(float)
    lo = bars["low"].to_numpy(float); c = bars["close"].to_numpy(float)
    n = c.size
    kF = _rqk(c, hF, x0F, rF); kS = _rqk(c, hS, x0S, rS)
    a = _atr(hi, lo, c, atr_len)

    with np.errstate(invalid="ignore"):
        slopeF = np.concatenate([np.full(slope_len_f, np.nan), kF[slope_len_f:] - kF[:-slope_len_f]]) / a
        slopeS = np.concatenate([np.full(slope_len_s, np.nan), kS[slope_len_s:] - kS[:-slope_len_s]]) / a
    sFu = (slope_min_f <= 0) | (slopeF >=  slope_min_f)
    sFd = (slope_min_f <= 0) | (slopeF <= -slope_min_f)
    sSu = (slope_min_s <= 0) | (slopeS >=  slope_min_s)
    sSd = (slope_min_s <= 0) | (slopeS <= -slope_min_s)

    # crossover
    fgs = kF > kS
    bullCross = np.zeros(n, bool); bearCross = np.zeros(n, bool)
    bullCross[1:] =  fgs[1:] & ~fgs[:-1]
    bearCross[1:] = ~fgs[1:] &  fgs[:-1]
    # bars since last cross
    bBull = np.full(n, 10**9, int); bBear = np.full(n, 10**9, int)
    cb = cs = 10**9
    for i in range(n):
        cb = 0 if bullCross[i] else cb + 1
        cs = 0 if bearCross[i] else cs + 1
        bBull[i] = cb; bBear[i] = cs

    # displacement of the CURRENT bar (entry bar must displace)
    body = np.abs(c - o); bull = c > o; bear = c < o
    prev_hi = np.concatenate([[np.inf], hi[:-1]]); prev_lo = np.concatenate([[-np.inf], lo[:-1]])
    wickL = np.ones(n, bool) if not use_wick else (c > prev_hi)
    wickS = np.ones(n, bool) if not use_wick else (c < prev_lo)
    dispL = bull & (body >= min_body_pts) & wickL
    dispS = bear & (body >= min_body_pts) & wickS

    # eligibility = current bar within the scan window of a recent cross
    eligL = ((scan_at & (bBull == 0)) | (scan_p1 & (bBull == 1)) | (scan_p2 & (bBull == 2)))
    eligS = ((scan_at & (bBear == 0)) | (scan_p1 & (bBear == 1)) | (scan_p2 & (bBear == 2)))

    # extension: ext_bars candles BEFORE the current bar, same dir, net move >= ext_pts
    if use_ext:
        allBull = np.ones(n, bool); allBear = np.ones(n, bool)
        for k in range(1, ext_bars + 1):
            cb_ = np.concatenate([np.full(k, False), (c[:-k] > o[:-k])]) if k < n else np.zeros(n, bool)
            # shift so index i looks at bar i-k
            shifted_bull = np.full(n, False); shifted_bear = np.full(n, False)
            shifted_bull[k:] = (c[:-k] > o[:-k]); shifted_bear[k:] = (c[:-k] < o[:-k])
            allBull &= shifted_bull; allBear &= shifted_bear
        moveBefore = np.full(n, 0.0)
        if ext_bars + 1 < n:
            moveBefore[ext_bars + 1:] = np.abs(c[1:-ext_bars] - c[:-(ext_bars + 1)])
        extendedUp = allBull & (moveBefore >= ext_pts)
        extendedDown = allBear & (moveBefore >= ext_pts)
    else:
        extendedUp = np.zeros(n, bool); extendedDown = np.zeros(n, bool)

    # ER
    if use_er:
        er = np.zeros(n)
        d = np.abs(np.diff(c)); csum = np.concatenate([[0.0], np.cumsum(d)])
        path = csum[er_len:] - csum[:-er_len]
        net = np.abs(c[er_len:] - c[:-er_len])
        er[er_len:] = np.where(path > 0, net / path, 0.0)
        erOk = er >= er_min
    else:
        erOk = np.ones(n, bool)

    # cross-quality open speed at the cross, carried forward
    with np.errstate(invalid="ignore"):
        gap = np.abs(kF - kS)
        openSpeed = np.concatenate([[np.nan], (gap[1:] - gap[:-1])]) / a
    crossOpenBull = np.full(n, np.nan); crossOpenBear = np.full(n, np.nan)
    ob = os_ = np.nan
    for i in range(n):
        if bullCross[i]: ob = openSpeed[i]
        if bearCross[i]: os_ = openSpeed[i]
        crossOpenBull[i] = ob; crossOpenBear[i] = os_
    if use_cq:
        cqL = np.isfinite(crossOpenBull) & (crossOpenBull >= open_min)
        cqS = np.isfinite(crossOpenBear) & (crossOpenBear >= open_min)
    else:
        cqL = np.ones(n, bool); cqS = np.ones(n, bool)

    # session
    if sess_start is not None and sess_end is not None:
        s0 = _dt.time.fromisoformat(sess_start); s1 = _dt.time.fromisoformat(sess_end)
        insess = np.array([s0 <= t <= s1 for t in bars.index.time], bool)
    else:
        insess = np.ones(n, bool)

    finite = np.isfinite(kF) & np.isfinite(kS) & np.isfinite(a)

    longSig  = eligL & dispL & sFu & sSu & cqL & erOk & ~extendedUp   & bool(allow_long)
    shortSig = eligS & dispS & sFd & sSd & cqS & erOk & ~extendedDown & bool(allow_short)
    longSig  &= finite & insess; shortSig &= finite & insess

    # latch: one entry per cross + cooldown (sequential)
    # track which cross bar produced each candidate; fire first per cross
    keep_idx, keep_dir = [], []
    last = -10**9
    firedBullX = -1; firedBearX = -1
    for i in np.flatnonzero(longSig | shortSig):
        if longSig[i]:
            crossbar = i - bBull[i]
            if firedBullX == crossbar: continue          # already fired this cross
            if (i - last) < min_gap_bars: continue
            keep_idx.append(i); keep_dir.append(1); last = i; firedBullX = crossbar
        else:
            crossbar = i - bBear[i]
            if firedBearX == crossbar: continue
            if (i - last) < min_gap_bars: continue
            keep_idx.append(i); keep_dir.append(-1); last = i; firedBearX = crossbar

    idx = np.array(keep_idx, int); direction = np.array(keep_dir, int)
    return dict(kF=kF, kS=kS, slopeF=slopeF, slopeS=slopeS, bullCross=bullCross, bearCross=bearCross,
                dispL=dispL, dispS=dispS, extendedUp=extendedUp, extendedDown=extendedDown,
                idx=idx, direction=direction)


class DualKernelV2:
    name = "dual_kernel_v2"
    wants_bars = True

    def __init__(self, hF=4.0, x0F=1, rF=8.0, hS=12.0, x0S=3, rS=8.0, atr_len=14,
                 slope_len_f=1, slope_min_f=0.0, slope_len_s=3, slope_min_s=0.0,
                 min_body_pts=10.0, use_wick=True, scan_at=True, scan_p1=True, scan_p2=True,
                 use_ext=True, ext_bars=3, ext_pts=30.0,
                 use_er=False, er_len=10, er_min=0.3,
                 use_cq=False, open_min=0.10,
                 min_gap_bars=5, sess_start="09:30", sess_end="11:00",
                 allow_long=True, allow_short=True, src="close"):
        for k, v in dict(hF=hF, rF=rF, hS=hS, rS=rS, slope_min_f=slope_min_f, slope_min_s=slope_min_s,
                         min_body_pts=min_body_pts, ext_pts=ext_pts, er_min=er_min, open_min=open_min).items():
            setattr(self, k, float(v))
        for k, v in dict(x0F=x0F, x0S=x0S, atr_len=atr_len, slope_len_f=slope_len_f, slope_len_s=slope_len_s,
                         ext_bars=ext_bars, er_len=er_len, min_gap_bars=min_gap_bars).items():
            setattr(self, k, int(v))
        for k, v in dict(use_wick=use_wick, scan_at=scan_at, scan_p1=scan_p1, scan_p2=scan_p2,
                         use_ext=use_ext, use_er=use_er, use_cq=use_cq,
                         allow_long=allow_long, allow_short=allow_short).items():
            setattr(self, k, bool(v))
        self.sess_start = sess_start; self.sess_end = sess_end; self.src = str(src)

    def _args(self):
        return dict(hF=self.hF, x0F=self.x0F, rF=self.rF, hS=self.hS, x0S=self.x0S, rS=self.rS,
                    atr_len=self.atr_len, slope_len_f=self.slope_len_f, slope_min_f=self.slope_min_f,
                    slope_len_s=self.slope_len_s, slope_min_s=self.slope_min_s,
                    min_body_pts=self.min_body_pts, use_wick=self.use_wick,
                    scan_at=self.scan_at, scan_p1=self.scan_p1, scan_p2=self.scan_p2,
                    use_ext=self.use_ext, ext_bars=self.ext_bars, ext_pts=self.ext_pts,
                    use_er=self.use_er, er_len=self.er_len, er_min=self.er_min,
                    use_cq=self.use_cq, open_min=self.open_min,
                    min_gap_bars=self.min_gap_bars, sess_start=self.sess_start, sess_end=self.sess_end,
                    allow_long=self.allow_long, allow_short=self.allow_short, src=self.src)

    def entries(self, bars):
        d = _compute(bars, **self._args()); return d["idx"], d["direction"]

    def debug(self, bars):
        d = _compute(bars, **self._args())
        return pd.DataFrame({"kF": d["kF"], "kS": d["kS"], "slopeF": d["slopeF"], "slopeS": d["slopeS"],
                             "dispL": d["dispL"], "dispS": d["dispS"],
                             "extendedUp": d["extendedUp"], "extendedDown": d["extendedDown"]}, index=bars.index)


DEFAULTS = dict(hF=4.0, x0F=1, rF=8.0, hS=12.0, x0S=3, rS=8.0, atr_len=14,
                slope_len_f=1, slope_min_f=0.0, slope_len_s=3, slope_min_s=0.0,
                min_body_pts=10.0, use_wick=True, scan_at=True, scan_p1=True, scan_p2=True,
                use_ext=True, ext_bars=3, ext_pts=30.0,
                use_er=False, er_len=10, er_min=0.3, use_cq=False, open_min=0.10,
                min_gap_bars=5, sess_start="09:30", sess_end="11:00",
                allow_long=True, allow_short=True, src="close")


def get_strategy(**over):
    return DualKernelV2(**{**DEFAULTS, **over})