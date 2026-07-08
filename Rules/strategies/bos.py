"""
strategies/bos.py — Break of Structure (BOS) ENTRY logic, self-contained.

Reproduces BosStrategy._fired from your BOS.py without needing your config /
indicators / strategies.base repo:
  long  when close crosses ABOVE the last CONFIRMED swing high
  short when close crosses BELOW the last CONFIRMED swing low

Only the entry is reproduced. Exits are the R:R bracket applied by Backtest.py,
because R:R is exactly what we're solving for. The ML grader (bos_chronos) and
hand-features are intentionally omitted — they don't affect entry timing.
"""
from __future__ import annotations
import numpy as np

from strategies.base import EntryStrategy


def causal_swings(high, low, k=2):
    """Causal fractal swings. Bar j is a swing high if its high strictly exceeds
    the k bars either side; it is KNOWN only k bars later (causal). Returns the
    last confirmed swing-high / swing-low LEVEL as of each bar."""
    high = np.asarray(high, float); low = np.asarray(low, float)
    n = high.size
    sh_cond = np.ones(n, bool); sl_cond = np.ones(n, bool)
    for o in range(1, k + 1):
        lh = np.zeros(n, bool); lh[o:] = high[o:] > high[:-o]
        rh = np.zeros(n, bool); rh[:-o] = high[:-o] > high[o:]
        sh_cond &= lh & rh
        ll = np.zeros(n, bool); ll[o:] = low[o:] < low[:-o]
        rl = np.zeros(n, bool); rl[:-o] = low[:-o] < low[o:]
        sl_cond &= ll & rl
    sh_cond[:k] = sh_cond[-k:] = False
    sl_cond[:k] = sl_cond[-k:] = False

    def _levels(cond, price):
        idx = np.flatnonzero(cond)
        lvl = np.full(n, np.nan)
        if idx.size:
            confirm = idx + k                       # known k bars later
            pos = np.searchsorted(confirm, np.arange(n), side="right") - 1
            v = pos >= 0
            lvl[v] = price[idx[pos[v]]]
        return lvl

    return _levels(sh_cond, high), _levels(sl_cond, low)


def bos_entries(high, low, close, swing_k=2):
    sh, sl = causal_swings(high, low, swing_k)
    c = np.asarray(close, float)
    cprev = np.empty_like(c); cprev[0] = c[0]; cprev[1:] = c[:-1]
    shp = np.empty_like(sh); shp[0] = np.nan; shp[1:] = sh[:-1]
    slp = np.empty_like(sl); slp[0] = np.nan; slp[1:] = sl[:-1]
    long_e = np.isfinite(sh) & np.isfinite(shp) & (c > sh) & (cprev <= shp)
    short_e = np.isfinite(sl) & np.isfinite(slp) & (c < sl) & (cprev >= slp)
    idx = np.flatnonzero(long_e | short_e)
    direction = np.where(long_e[idx], 1, -1)
    return idx, direction


class BOS(EntryStrategy):
    name = "bos"

    def __init__(self, swing_k: int = 2):
        self.swing_k = swing_k

    def entries(self, high, low, close):
        return bos_entries(high, low, close, self.swing_k)


def get_strategy():
    return BOS(swing_k=2)