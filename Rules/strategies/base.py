"""
strategies/base.py — the ENTRY-ONLY strategy interface.

A strategy here answers exactly one question: "on which bars do we enter, and in
which direction?" It defines NO exit and NO R:R — those are supplied by the
backtester as the bracket for whatever R:R the Monte Carlo is currently testing.

To add a new strategy (SuperTrend, EMA cross, ...): create strategies/<name>.py
with a class subclassing EntryStrategy and a module-level get_strategy() that
returns an instance. That's the whole contract.
"""
from __future__ import annotations
import numpy as np


class EntryStrategy:
    name: str = "base"

    def entries(self, high, low, close):
        """Given 1-D price arrays, return (entry_idx, direction) where
        entry_idx is an int array of bar indices and direction is +1 (long)
        or -1 (short) for each. No look-ahead: signals must be causal."""
        raise NotImplementedError