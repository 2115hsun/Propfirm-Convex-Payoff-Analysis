"""
strategies/kernel_trend.py — PURE trend-following entry for the Backtest.py
framework. No fair-value / pullback / distance logic — this is the stripped
"trade the confirmed trend, avoid chop" version.

The kernel is used ONLY as a direction+steepness estimator (its slope), not as
a fair-value level. Chop is avoided with two orthogonal filters:
    slope steepness  (is the trend strong enough)   -> slope_min
    efficiency ratio (is the move clean or churn)   -> er_min          <-- the
    ER gate is the main prop-firm lever: it removes the low-efficiency chop
    where trend entries whipsaw, cutting the serial-loss clusters that a
    trailing-drawdown account is punished by.

Entry (long; short mirrors) fires on a discrete EVENT, not a persistent state:
    trigger="regime": the bar the regime FLIPS to tradeable in the trend
                      direction (rising kernel + both gates pass, having NOT
                      been true last bar). "Enter as the trend is confirmed."
    trigger="cross" : fast kernel crosses the slow kernel in the trend
                      direction while both gates pass (classic MA-cross entry).

Exits are NOT here — Backtest.py brackets each entry with k_stop_atr*ATR (=1R)
and scores the MFE at every candidate R:R. We only emit (idx, direction).

Interface: wants_bars=True; entries(bars)->(idx, dir); name; get_strategy(**over).

NOTE on stops (important, framework-specific): in Backtest.py 1R = k_stop_atr*ATR
regardless of where you enter, so a confirmation entry does NOT widen your stop.
Its cost vs a pullback entry shows up as LOWER favorable-excursion-per-entry
(you enter later, so less move remains) -> lower measured WR at a given R:R.
The benefit is chop avoidance. Which wins is empirical -> race it OOS.

Swept knobs (tune on IS, validate OOS — do NOT hand-fit):
    h, r, x_0, lag     NW-RQK kernel (h=slow window, x_0=length, fast=h-lag)
    trigger            "regime" | "cross"
    slope_len, slope_min   steepness gate (% rise per bar). 0 => off.
    er_len, er_min         efficiency/chop gate. 0 => off.
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


# NW-RQK kernel as a causal FIR filter (faithful port: window length = x_0 + 2;
# h, r only shape the RQ weight decay).
def _rqk(close, h, r, x_0):
    L = int(x_0) + 1
    i = np.arange(L + 1, dtype=float)
    w = (1.0 + (i * i) / (2.0 * r * h * h)) ** (-r)
    w /= w.sum()
    out = np.convolve(close, w)[:close.size]
    out[:L] = np.nan
    return out


def _rising_edge(mask):
    """True only on the bar a boolean turns False->True (a discrete event)."""
    out = np.zeros_like(mask)
    out[1:] = mask[1:] & ~mask[:-1]
    return out


class KernelTrend:
    name = "kernel_trend"
    wants_bars = True

    def __init__(self, h=8.0, r=8.0, x_0=25, lag=2, trigger="regime",
                 slope_len=5, slope_min=0.0, er_len=10, er_min=0.0,
                 allow_long=True, allow_short=True):
        self.h = float(h); self.r = float(r); self.x_0 = int(x_0); self.lag = int(lag)
        assert trigger in ("regime", "cross")
        self.trigger = trigger
        self.slope_len = int(slope_len); self.slope_min = float(slope_min)
        self.er_len = int(er_len); self.er_min = float(er_min)
        self.allow_long = bool(allow_long); self.allow_short = bool(allow_short)

    def entries(self, bars):
        close = bars["close"].to_numpy(float)
        n = close.size

        k = _rqk(close, self.h, self.r, self.x_0)                       # direction/steepness
        kf = _rqk(close, max(self.h - self.lag, 1e-6), self.r, self.x_0)  # fast (for cross)
        er = _efficiency_ratio(close, self.er_len)

        sl = self.slope_len
        slope = np.full(n, np.nan)                                       # % rise per bar
        with np.errstate(invalid="ignore", divide="ignore"):
            slope[sl:] = (k[sl:] - k[:-sl]) / (k[:-sl] * sl) * 100.0

        rising = np.zeros(n, bool); falling = np.zeros(n, bool)
        rising[1:] = k[1:] > k[:-1]
        falling[1:] = k[1:] < k[:-1]

        finite = np.isfinite(k) & np.isfinite(slope)
        conv_up = slope >= self.slope_min                               # steepness gate
        conv_dn = slope <= -self.slope_min
        regime = er >= self.er_min                                      # ER chop gate

        # "may enter" state per side = trend direction + both gates pass
        long_state = finite & rising & conv_up & regime
        short_state = finite & falling & conv_dn & regime

        if self.trigger == "regime":
            trig_up = _rising_edge(long_state)                          # flip INTO tradeable
            trig_dn = _rising_edge(short_state)
        else:  # "cross": fast crosses slow in-trend, gates pass
            xu = np.zeros(n, bool); xd = np.zeros(n, bool)
            xu[1:] = (kf[1:] > k[1:]) & (kf[:-1] <= k[:-1])
            xd[1:] = (kf[1:] < k[1:]) & (kf[:-1] >= k[:-1])
            trig_up = xu & long_state
            trig_dn = xd & short_state

        long_ok = trig_up & self.allow_long
        short_ok = trig_dn & self.allow_short
        idx = np.flatnonzero(long_ok | short_ok)
        direction = np.where(long_ok[idx], 1, -1).astype(int)
        return idx, direction


DEFAULTS = dict(
    h=8.0, r=8.0, x_0=25, lag=2, trigger="regime",
    slope_len=5, slope_min=0.0, er_len=10, er_min=0.0,
    allow_long=True, allow_short=True,
)


def get_strategy(**overrides):
    """load_strategy('kernel_trend') calls this. Override for sweeps, e.g.
    get_strategy(trigger='cross', slope_min=0.02, er_min=0.4)."""
    return KernelTrend(**{**DEFAULTS, **overrides})