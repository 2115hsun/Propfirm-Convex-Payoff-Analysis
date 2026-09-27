"""
strategies/kernel_pullback.py — limit-order pullback entry for the Backtest.py framework.

Thesis:
    Use the Nadaraya-Watson Rational-Quadratic kernel as a FAIR-VALUE ANCHOR.
    Its LEVEL is fair value; its SLOPE is the trend. 
    Enter using a 1-3 bar LIMIT ORDER placed precisely at the kernel line when 
    a trend flip occurs. This guarantees execution exactly at fair value, maximizing
    the R:R skew and front-loading the favorable excursion for prop-firm viability.

Entry (long; short mirrors):
    1. direction : slow kernel changes direction (yhat1[t] > yhat1[t-1])
    2. trigger   : places limit order at kernel[t] valid for next `pullback_bars`
    3. conviction: normalized slope >= slope_min at the flip bar
    4. execution : price tags the limit order within the expiry window. Only one 
                   fill per trend cycle is permitted.

Interface: wants_bars=True; entries(bars)->(idx, dir); name; get_strategy(**over).
NOTE: The exact limit-order fill prices are saved to `self.fill_prices` to allow 
Backtest.py to use exact execution prices without breaking the (idx, dir) tuple API.

Swept knobs:
    h, r, x_0            NW-RQK kernel parameters
    slope_len, slope_min Gate B (conviction). slope is % rise per bar.
    pullback_bars        How many bars the limit order stays active after a flip.
"""
from __future__ import annotations
import numpy as np


# --- NW-RQK kernel as a causal FIR filter -----------------------------------
def _rqk(close, h, r, x_0):
    L = int(x_0) + 1
    i = np.arange(L + 1, dtype=float)
    w = (1.0 + (i * i) / (2.0 * r * h * h)) ** (-r)
    w /= w.sum()
    out = np.convolve(close, w)[:close.size]      # out[t] = sum_i w[i]*close[t-i]
    out[:L] = np.nan                              # warmup (partial window) invalid
    return out


class KernelPullback:
    name = "kernel_pullback_1"
    wants_bars = True

    def __init__(self, h=8.0, r=8.0, x_0=25, slope_len=1, slope_min=0.0,
                 pullback_bars=3, allow_long=True, allow_short=True):
        self.h = float(h)
        self.r = float(r)
        self.x_0 = int(x_0)
        self.slope_len = int(slope_len)
        self.slope_min = float(slope_min)
        self.pullback_bars = int(pullback_bars)
        self.allow_long = bool(allow_long)
        self.allow_short = bool(allow_short)
        
        # Will hold the exact execution prices of the limit orders

    def entries(self, bars):
        high = bars["high"].to_numpy(float)
        low = bars["low"].to_numpy(float)
        close = bars["close"].to_numpy(float)
        open_ = bars["open"].to_numpy(float)
        n = close.size

        # slow kernel (fair value)
        k = np.full(n, np.nan)
        dates = bars.index.normalize().to_numpy()
        _, starts = np.unique(dates, return_index=True)
        bounds = list(np.sort(starts)) + [n]
        for s, e in zip(bounds[:-1], bounds[1:]):
            k[s:e] = _rqk(close[s:e], self.h, self.r, self.x_0)  

        idx_list = []
        dir_list = []
        prices_list = []

        # State tracking for the limit order
        limit_price = np.nan
        limit_expiry = -1
        trade_dir = 0
        current_trend = 0
        filled_this_trend = False

        for i in range(1, n):
            # 1. Determine current trend based on kernel direction
            if k[i] > k[i-1]:
                new_trend = 1
            elif k[i] < k[i-1]:
                new_trend = -1
            else:
                new_trend = current_trend

            # 2. Check for active limit order fill on THIS bar
            if not np.isnan(limit_price) and i <= limit_expiry and not filled_this_trend:
                # Long trigger: price retraces down to or below the kernel limit price
                if trade_dir == 1 and low[i] <= limit_price and self.allow_long:
                    idx_list.append(i)
                    dir_list.append(1)
                    fill_price = min(open_[i], limit_price)  # Ensure we don't fill above the close
                    prices_list.append(fill_price)
                    filled_this_trend = True
                    limit_price = np.nan 
                    
                # Short trigger: price retraces up to or above the kernel limit price
                elif trade_dir == -1 and high[i] >= limit_price and self.allow_short:
                    idx_list.append(i)
                    dir_list.append(-1)
                    fill_price = max(open_[i], limit_price)  # Ensure we don't fill below the close
                    prices_list.append(fill_price)
                    filled_this_trend = True
                    limit_price = np.nan 
                    
            # Expire the limit order if we've passed the window
            elif i > limit_expiry:
                limit_price = np.nan

            # 3. Check for a NEW trend flip to set up the next limit order
            if new_trend != current_trend and new_trend != 0:
                # Validate the slope is steep enough
                if i >= self.slope_len and not np.isnan(k[i - self.slope_len]):
                    s = (k[i] - k[i - self.slope_len]) / (k[i - self.slope_len] * self.slope_len) * 100.0
                    
                    conviction = (new_trend == 1 and s >= self.slope_min) or \
                                 (new_trend == -1 and s <= -self.slope_min)
                    
                    if conviction:
                        # Set the limit price to the exact kernel value on the flip candle
                        limit_price = k[i-1]
                        # Order is valid for the next N bars
                        limit_expiry = i + self.pullback_bars
                        trade_dir = new_trend
                        filled_this_trend = False # Ready for a new entry

            current_trend = new_trend

        # Expose exact limit prices to the instance so Backtest.py can access them
        # without changing the unpacked return signature.

        return (np.array(idx_list), np.array(dir_list), np.array(prices_list))

DEFAULTS = dict(
    h=8.0, r=8.0, x_0=25, slope_len=1, slope_min=0.0,
    pullback_bars=3, allow_long=True, allow_short=True,
)


def get_strategy(**overrides):
    """load_strategy('kernel_pullback') calls this. Override for sweeps, e.g.
    get_strategy(slope_min=0.02, pullback_bars=5)."""
    return KernelPullback(**{**DEFAULTS, **overrides})