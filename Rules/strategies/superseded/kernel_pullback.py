"""
strategies/kernel_pullback.py — trend-pullback entry for the Backtest.py framework.

Thesis (prop-firm-shaped, not PnL-maximizing):
    Use the Nadaraya-Watson Rational-Quadratic kernel as a FAIR-VALUE ANCHOR.
    Its LEVEL is fair value; its SLOPE is the trend. Enter on a pullback that
    RESUMES the trend, near fair value, so the stop is tight and the favorable
    excursion is front-loaded — the shape that lifts pass rate against a
    trailing-drawdown barrier (raises E[R]/sigma[R], not just E[R]).

Entry (long; short mirrors), ALL must hold on the bar:
    1. direction : slow kernel rising                      (yhat1[t] > yhat1[t-1])
    2. trigger   : re-cross of kernel  OR  fast/slow cross (see `trigger`)
    3. conviction: normalized slope >= slope_min           (% rise per bar)
    4. regime    : efficiency ratio >= er_min              (0 disables)
    5. don't-chase: |close - kernel| / ATR <= max_dist_atr (None disables)
    6. one position at a time (both triggers are single-bar events anyway)

Exits are NOT here — Backtest.py brackets each entry with k_stop_atr*ATR (=1R)
and scores the MFE at every candidate R:R. We only emit (idx, direction).

Interface: wants_bars=True; entries(bars)->(idx, dir); name; get_strategy(**over).

Swept knobs (tune in Robustness.py on IS, validate OOS — do NOT hand-fit):
    h, r, x_0          NW-RQK kernel (h=slow window, r=weighting, x_0=window len)
    lag                fast kernel uses window (h - lag)
    trigger            "recross" (pullback) | "cross" (fast/slow) | "either"
    slope_len, slope_min   Gate B (conviction). slope is % rise per bar.
    er_len, er_min         Gate C (regime). er_min=0 => off.
    max_dist_atr           Gate D (don't chase). None => off.
    allow_long, allow_short
"""
from __future__ import annotations
import numpy as np


# --- Wilder ATR (matches Backtest.atr's RMA) --------------------------------
def _wilder_atr(high, low, close, period):
    high = np.asarray(high, float); low = np.asarray(low, float); close = np.asarray(close, float)
    prev = np.empty_like(close); prev[0] = close[0]; prev[1:] = close[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    out = np.full_like(tr, np.nan)
    if tr.size >= period:
        out[period - 1] = tr[:period].mean()
        a = 1.0 / period
        for i in range(period, tr.size):
            out[i] = out[i - 1] * (1 - a) + tr[i] * a
        out[:period - 1] = out[period - 1]
    return out


# --- Efficiency Ratio (Kaufman), |value| in [0,1] ---------------------------
def _efficiency_ratio(close, length):
    close = np.asarray(close, float)
    d = np.abs(np.diff(close))
    csum = np.concatenate([[0.0], np.cumsum(d)])
    er = np.zeros(close.size)
    path = csum[length:] - csum[:-length]
    net = np.abs(close[length:] - close[:-length])
    er[length:] = np.where(path > 0, net / path, 0.0)
    return er


# --- NW-RQK kernel as a causal FIR filter -----------------------------------
# Note the faithful port of jdehorty's quirk: `size = array.size(array.from(src))`
# is 1, so the loop `for i = 0 to size + x_0` sums the most recent (x_0 + 2)
# bars. h and r only shape the RQ weight decay; x_0 sets the window LENGTH.
def _rqk(close, h, r, x_0):
    L = int(x_0) + 1
    i = np.arange(L + 1, dtype=float)
    w = (1.0 + (i * i) / (2.0 * r * h * h)) ** (-r)
    w /= w.sum()
    out = np.convolve(close, w)[:close.size]      # out[t] = sum_i w[i]*close[t-i]
    out[:L] = np.nan                              # warmup (partial window) invalid
    return out


class KernelPullback:
    name = "kernel_pullback"
    wants_bars = True

    def __init__(self, h=8.0, r=8.0, x_0=25, lag=2,
                 trigger="recross", slope_len=5, slope_min=0.0,
                 er_len=10, er_min=0.0, max_dist_atr=1.5,
                 atr_len=14, allow_long=True, allow_short=True):
        self.h = float(h); self.r = float(r); self.x_0 = int(x_0); self.lag = int(lag)
        assert trigger in ("recross", "cross", "either")
        self.trigger = trigger
        self.slope_len = int(slope_len); self.slope_min = float(slope_min)
        self.er_len = int(er_len); self.er_min = float(er_min)
        self.max_dist_atr = None if max_dist_atr is None else float(max_dist_atr)
        self.atr_len = int(atr_len)
        self.allow_long = bool(allow_long); self.allow_short = bool(allow_short)

    def entries(self, bars):
        high = bars["high"].to_numpy(float)
        low = bars["low"].to_numpy(float)
        close = bars["close"].to_numpy(float)
        n = close.size

        k = _rqk(close, self.h, self.r, self.x_0)             # slow kernel (fair value)
        kf = _rqk(close, max(self.h - self.lag, 1e-6), self.r, self.x_0)  # fast kernel
        atr = _wilder_atr(high, low, close, self.atr_len)
        er = _efficiency_ratio(close, self.er_len)

        sl = self.slope_len
        slope = np.full(n, np.nan)                            # % rise per bar of slow kernel
        past = np.empty(n); past[:sl] = np.nan; past[sl:] = k[:-sl]
        with np.errstate(invalid="ignore", divide="ignore"):
            slope[sl:] = (k[sl:] - k[:-sl]) / (k[:-sl] * sl) * 100.0

        dist = np.full(n, np.nan)                             # (close - kernel) / ATR
        good = np.isfinite(atr) & (atr > 0)
        dist[good] = (close[good] - k[good]) / atr[good]

        rising = np.zeros(n, bool); falling = np.zeros(n, bool)
        rising[1:] = k[1:] > k[:-1]
        falling[1:] = k[1:] < k[:-1]

        # triggers (single-bar events)
        rc_up = np.zeros(n, bool); rc_dn = np.zeros(n, bool)
        rc_up[1:] = (close[1:] > k[1:]) & (close[:-1] <= k[:-1])     # price re-cross up
        rc_dn[1:] = (close[1:] < k[1:]) & (close[:-1] >= k[:-1])
        xu = np.zeros(n, bool); xd = np.zeros(n, bool)
        xu[1:] = (kf[1:] > k[1:]) & (kf[:-1] <= k[:-1])              # fast/slow cross up
        xd[1:] = (kf[1:] < k[1:]) & (kf[:-1] >= k[:-1])

        if self.trigger == "recross":
            trig_up, trig_dn = rc_up, rc_dn
        elif self.trigger == "cross":
            trig_up, trig_dn = xu, xd
        else:  # either
            trig_up, trig_dn = (rc_up | xu), (rc_dn | xd)

        finite = np.isfinite(k) & np.isfinite(slope) & np.isfinite(dist)
        conv_up = slope >= self.slope_min
        conv_dn = slope <= -self.slope_min
        regime = er >= self.er_min
        near = np.ones(n, bool) if self.max_dist_atr is None else (np.abs(dist) <= self.max_dist_atr)

        long_ok = finite & rising & trig_up & conv_up & regime & near & self.allow_long
        short_ok = finite & falling & trig_dn & conv_dn & regime & near & self.allow_short

        idx = np.flatnonzero(long_ok | short_ok)
        direction = np.where(long_ok[idx], 1, -1).astype(int)
        return idx, direction


DEFAULTS = dict(
    h=8.0, r=8.0, x_0=25, lag=2, trigger="recross",
    slope_len=5, slope_min=0.0, er_len=10, er_min=0.0,
    max_dist_atr=1.5, atr_len=14, allow_long=True, allow_short=True,
)


def get_strategy(**overrides):
    """load_strategy('kernel_pullback') calls this. Override for sweeps, e.g.
    get_strategy(trigger='cross', slope_min=0.02, er_min=0.3, max_dist_atr=1.0)."""
    return KernelPullback(**{**DEFAULTS, **overrides})