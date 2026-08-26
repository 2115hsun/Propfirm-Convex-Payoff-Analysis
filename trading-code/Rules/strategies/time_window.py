"""
strategies/time_window.py — wraps ANY EntryStrategy and restricts its entries
to a fixed local time-of-day window (e.g., 09:30-11:00 ET).

The wrapped strategy runs exactly as it normally would against the FULL bars
(kernel level, BOS, slope, direction, fill price all computed the same way --
nothing about the underlying strategy's logic changes) -- only entries whose
own bar falls outside [start, end) are dropped afterward. This keeps the
wrapped strategy's file itself clean/general (no session-window logic baked
into e.g. kernel_bos.py) and lets any notebook/experiment apply a
first-N-minutes-of-session constraint without touching that file.

`start`/`end` are compared against the bar's LOCAL wall-clock time (whatever
timezone the bars' DatetimeIndex is already in -- typically America/New_York
after Backtest.load_ohlc's tz conversion), start inclusive / end exclusive.

Interface: wants_bars=True; entries(bars)->(idx, dir[, fill]); name.
"""
from __future__ import annotations
import datetime as _dt
import numpy as np


class TimeWindow:
    wants_bars = True

    def __init__(self, strategy, start="09:30", end="11:00"):
        self.strategy = strategy
        self.start = _dt.time.fromisoformat(start)
        self.end = _dt.time.fromisoformat(end)
        self.name = f"{getattr(strategy, 'name', 'strategy')}_{start.replace(':','')}-{end.replace(':','')}"

    def entries(self, bars):
        wants_bars = getattr(self.strategy, "wants_bars", False)
        res = self.strategy.entries(bars) if wants_bars else self.strategy.entries(
            bars["high"].to_numpy(float), bars["low"].to_numpy(float), bars["close"].to_numpy(float))

        if len(res) == 3:
            idx, direction, fill = res
            fill = np.asarray(fill, float)
        else:
            idx, direction = res
            fill = None

        if len(idx) == 0:
            return res

        t = bars.index[idx].time    # array of datetime.time, local wall-clock
        keep = np.fromiter((self.start <= tt < self.end for tt in t), dtype=bool, count=len(t))
        idx = idx[keep]; direction = direction[keep]
        if fill is not None:
            return idx, direction, fill[keep]
        return idx, direction
