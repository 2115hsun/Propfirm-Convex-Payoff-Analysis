"""
strategies/base.py: the entry-only strategy interface.

A strategy says which bars to enter on and in which direction. It has no exit
and no R:R; the backtester brackets each entry for whatever R:R is being
tested.

To add one (SuperTrend, EMA cross, ...), create strategies/<name>.py with a
class that subclasses EntryStrategy and a module-level get_strategy() that
returns an instance.
"""
from __future__ import annotations
import numpy as np


class EntryStrategy:
    name: str = "base"
    wants_bars: bool = False  # True = wants OHLCV bars; False = price-only (close)

    def entries(self, high, low, close):
        """Given 1-D price arrays, return (entry_idx, direction) where
        entry_idx is an int array of bar indices and direction is +1 (long)
        or -1 (short) for each. No look-ahead: signals must be causal."""
        raise NotImplementedError