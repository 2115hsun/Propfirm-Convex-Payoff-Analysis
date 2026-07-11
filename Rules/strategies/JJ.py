"""
strategies/JJ.py
================
AM/PM opening fair-price strategy — ENTRY-ONLY, adapted to the Backtest.py
contract.

Backtest.py owns all exits: each entry gets an ATR-based 1R stop and is scored
across every rr. So this module contributes ONLY entry signals. All the SL/TP,
position-sizing, R:R and PnL logic from the standalone script has been removed;
what remains is the signal logic (continuation, reversion, consolidation
re-anchor) for the AM (09:30) and PM (14:00) sessions.

Contract implemented:
  - get_strategy() -> returns an EntryStrategy instance
  - instance.wants_bars = True   (we need the datetime index / session clock)
  - instance.entries(bars) -> (idx, direction)
        idx       : int positions INTO `bars` where we enter (chronological)
        direction : +1 long / -1 short, same length as idx
  - instance.swing_k : settable (Backtest.build_entries injects it)
  - instance.name    : label for summaries
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# --- Session clock (must match your data's local tz; Backtest converts to ET) ---
PREMARKET_START = "08:30"
PREMARKET_END   = "09:29"
AM_FAIR_TIME    = "09:30"
AM_SESSION_END  = "11:30"
PM_FAIR_TIME    = "14:00"
PM_SESSION_END  = "16:00"
CONTINUATION_WINDOW_MIN = 15

# --- Fair candle ---
FAIR_CANDLE_MINUTES = 1

# --- Strong-move (continuation trigger) ---
STRONG_MOVE_MIN_PTS  = 25.0
PREMARKET_RANGE_MULT = 1.0
REQUIRE_PREMARKET_ALIGNMENT = False

# --- BOS / swing detection (overridden by build_entries' swing_k if passed) ---
SWING_LOOKBACK = 2

# --- Consolidation re-anchor ---
ENABLE_CONSOLIDATION = True
CONSOL_MIN_BARS      = 5
CONSOL_MAX_RANGE_PTS = 15.0

# --- Caps ---
MAX_TRADES_PER_SETUP_PER_SESSION = 10


class JJStrategy:
    name = "JJ"
    wants_bars = True   # entries(bars) receives the full DataFrame

    def __init__(self, swing_k=SWING_LOOKBACK):
        self.swing_k = swing_k

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _confirmed_swings(high, low, k):
        """Causal fractal swings: last CONFIRMED swing high/low as of each bar.
        A swing at bar c is only usable from bar c+k onward (no lookahead)."""
        n = len(high)
        sh = np.full(n, np.nan); sl = np.full(n, np.nan)
        last_h = last_l = np.nan
        for i in range(n):
            c = i - k
            if c - k >= 0:
                wh = high[c - k:c + k + 1]; wl = low[c - k:c + k + 1]
                if high[c] == wh.max() and wh.argmax() == k:
                    last_h = high[c]
                if low[c] == wl.min() and wl.argmin() == k:
                    last_l = low[c]
            sh[i] = last_h; sl[i] = last_l
        return sh, sl

    @staticmethod
    def _premarket_dir_range(day_bars):
        pm = day_bars.between_time(PREMARKET_START, PREMARKET_END)
        if pm.empty:  # e.g. rth_only data has no premarket -> fall back to abs floor
            return 0.0, 0
        rng = pm["high"].max() - pm["low"].min()
        net = pm["close"].iloc[-1] - pm["open"].iloc[0]
        return float(rng), int(np.sign(net))

    # ----------------------------------------------------- one session's signals
    def _session_entries(self, session_bars, fair_time_str, pm_range, pm_dir,
                         pos, out_idx, out_dir):
        if session_bars.empty:
            return
        day0 = session_bars.index[0].normalize()
        start = day0 + pd.Timedelta(fair_time_str + ":00")
        fc_end = start + pd.Timedelta(minutes=FAIR_CANDLE_MINUTES)

        ts = session_bars.index
        o = session_bars["open"].to_numpy(float)
        h = session_bars["high"].to_numpy(float)
        l = session_bars["low"].to_numpy(float)
        c = session_bars["close"].to_numpy(float)

        fc_mask = (ts >= start) & (ts < fc_end)
        if not fc_mask.any():
            return
        fair_open = o[np.flatnonzero(fc_mask)[0]]
        fair_end_idx = int(np.flatnonzero(fc_mask)[-1])

        sh, sl = self._confirmed_swings(h, l, self.swing_k)

        # --- 1) CONTINUATION: strong move within first N minutes ---
        threshold = max(STRONG_MOVE_MIN_PTS, PREMARKET_RANGE_MULT * pm_range)
        win_end = start + pd.Timedelta(minutes=CONTINUATION_WINDOW_MIN)
        initial_dir = 0
        for j in range(len(session_bars)):
            if ts[j] >= win_end:
                break
            up = h[j] - fair_open
            dn = fair_open - l[j]
            if up >= threshold and up >= dn:
                d = 1
            elif dn >= threshold:
                d = -1
            else:
                continue
            if REQUIRE_PREMARKET_ALIGNMENT and pm_dir != 0 and d != pm_dir:
                continue
            initial_dir = d
            out_idx.append(pos[ts[j]]); out_dir.append(d)
            break

        # infer the initial move direction if continuation didn't fire
        if initial_dir == 0:
            for j in range(fair_end_idx + 1, len(session_bars)):
                up = h[j] - fair_open; dn = fair_open - l[j]
                if max(up, dn) >= STRONG_MOVE_MIN_PTS / 2:
                    initial_dir = 1 if up >= dn else -1
                    break

        # --- 2) REVERSION on opposite 1-bar BOS ---
        if initial_dir != 0:
            opp = -initial_dir
            taken = 0
            for j in range(fair_end_idx + 1, len(session_bars)):
                if taken >= MAX_TRADES_PER_SETUP_PER_SESSION:
                    break
                bos = ((opp > 0 and not np.isnan(sh[j]) and c[j] > sh[j]) or
                       (opp < 0 and not np.isnan(sl[j]) and c[j] < sl[j]))
                if bos:
                    out_idx.append(pos[ts[j]]); out_dir.append(opp)
                    taken += 1

        # --- 3) CONSOLIDATION re-anchor -> BOS back toward it ---
        if ENABLE_CONSOLIDATION and initial_dir != 0:
            consol = None
            for s in range(fair_end_idx + 1, len(session_bars) - CONSOL_MIN_BARS):
                seg_h = h[s:s + CONSOL_MIN_BARS]; seg_l = l[s:s + CONSOL_MIN_BARS]
                if seg_h.max() - seg_l.min() <= CONSOL_MAX_RANGE_PTS:
                    consol = (s + CONSOL_MIN_BARS - 1,
                              (seg_h.max() + seg_l.min()) / 2.0)
                    break
            if consol is not None:
                c_end, c_mid = consol
                post_h = h[c_end + 1:]; post_l = l[c_end + 1:]
                continued = (
                    (initial_dir > 0 and post_h.size and
                     post_h.max() - c_mid >= STRONG_MOVE_MIN_PTS) or
                    (initial_dir < 0 and post_l.size and
                     c_mid - post_l.min() >= STRONG_MOVE_MIN_PTS)
                )
                if continued:
                    opp = -initial_dir
                    taken = 0
                    for j in range(c_end + 1, len(session_bars)):
                        if taken >= MAX_TRADES_PER_SETUP_PER_SESSION:
                            break
                        bos = ((opp > 0 and not np.isnan(sh[j]) and c[j] > sh[j]) or
                               (opp < 0 and not np.isnan(sl[j]) and c[j] < sl[j]))
                        if bos:
                            out_idx.append(pos[ts[j]]); out_dir.append(opp)
                            taken += 1

    # --------------------------------------------------------------- public API
    def entries(self, bars: pd.DataFrame):
        pos = {ts: i for i, ts in enumerate(bars.index)}  # timestamp -> row number
        out_idx, out_dir = [], []
        for _, day_bars in bars.groupby(bars.index.normalize()):
            pm_range, pm_dir = self._premarket_dir_range(day_bars)
            am = day_bars.between_time(AM_FAIR_TIME, AM_SESSION_END)
            self._session_entries(am, AM_FAIR_TIME, pm_range, pm_dir,
                                  pos, out_idx, out_dir)
            pm = day_bars.between_time(PM_FAIR_TIME, PM_SESSION_END)
            self._session_entries(pm, PM_FAIR_TIME, pm_range, pm_dir,
                                  pos, out_idx, out_dir)
        if not out_idx:
            return np.array([], dtype=int), np.array([], dtype=int)
        idx = np.array(out_idx, dtype=int)
        direction = np.array(out_dir, dtype=int)
        order = np.argsort(idx, kind="stable")   # chronological for one_position logic
        return idx[order], direction[order]


def get_strategy():
    return JJStrategy()