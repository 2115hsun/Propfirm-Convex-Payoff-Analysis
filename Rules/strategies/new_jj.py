"""
strategies/new_jj.py — AM/PM/Evening/Asia session "fair price" model, ENTRY-ONLY.

Fully-causal. Backtest.py owns exits (ATR 1R stop, scored across every rr).

REVERSION structure detection is a faithful port of LuxAlgo "Market Structure
(Fractal)": strict monotonic fractal swings + a BODY-CLOSE crossover of the last
uncrossed swing, classified BOS (continuation) vs CHoCH (reversal) via an `os`
state. Parameter note: LuxAlgo `length` == 2*swing_k + 1, so swing_k=2 reproduces
LuxAlgo's default length=5 (swing_k=5 == LuxAlgo length=11).

  CONTINUATION (opening-candle rule; premarket ignored)
    direction = opening candle direction (sign of close-open).
    CLEAN open  (max wick <= open_wick_max, 25): enter that direction on the open.
    MESSY open  (max wick >  open_wick_max): wait for the first candle within
      confirm_window_min (10) min that is SAME direction AND max wick <
      confirm_wick_max (10); enter there. Else no continuation.

  REVERSION (LuxAlgo structure -> fade to fair)
    A break = the CLOSE crosses the last uncrossed fractal (body, not wick;
    require_body_close=False uses the wick). Either direction; the fade side is
    set by which side of fair price is on:
        break DOWN a swing low while ABOVE fair -> SHORT to fair
        break UP   a swing high while BELOW fair -> LONG  to fair
    reversion_choch_only=True restricts to CHoCH (reversal) breaks only.
    Room filter: distance back to fair >= rev_room_pts (PM may also use AM open).

Sessions (ET): AM 09:30->11:00 | PM 14:00->15:00 | Evening 18:00->20:00 |
               Asia 20:00->21:45
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategies.base import EntryStrategy


SESSIONS = [
    ("AM",      "09:30", "11:00", "08:30"),
    ("PM",      "14:00", "15:00", "13:00"),
    ("Evening", "18:00", "20:00", "17:00"),
    ("Asia",    "20:00", "21:45", "19:00"),
]


def _fractals_lux(high, low, p, strict_monotonic=False):
    """Fractal swings, fully causal. A swing centered at bar c is confirmed p bars
    later (at c+p). Returns, as of each bar i, the last CONFIRMED swing-high level
    (up[i]) and swing-low level (dn[i]) — nan until one exists.

    Pivot test: the center bar is the strict window extreme (max high / min low
    over the 2p+1 window, uniquely at the center). This is LuxAlgo's
    `high[p] == ta.highest(length)` condition and is robust to tick ties.

    strict_monotonic=True additionally requires highs to rise p bars into the
    pivot and fall p bars out (LuxAlgo's `dh`/`dl` sign-sum). Faithful to the Pine
    source but BRITTLE on tick-quantized 1-min data (one equal tick kills a swing),
    so it is OFF by default; enable only for indicator-parity checks."""
    n = high.size
    up = np.full(n, np.nan); dn = np.full(n, np.nan)
    last_up = last_dn = np.nan
    for i in range(n):
        c = i - p
        if c - p >= 0:
            wh = high[c - p:c + p + 1]; wl = low[c - p:c + p + 1]
            is_hi = high[c] == wh.max() and int(np.argmax(wh)) == p   # unique center max
            is_lo = low[c] == wl.min() and int(np.argmin(wl)) == p
            if strict_monotonic:
                is_hi = is_hi and all(high[c - p + 1 + t] > high[c - p + t] for t in range(p)) \
                              and all(high[c + 1 + t]     < high[c + t]     for t in range(p))
                is_lo = is_lo and all(low[c - p + 1 + t] < low[c - p + t] for t in range(p)) \
                              and all(low[c + 1 + t]      > low[c + t]     for t in range(p))
            if is_hi:
                last_up = high[c]
            if is_lo:
                last_dn = low[c]
        up[i] = last_up; dn[i] = last_dn
    return up, dn


class NewJJ(EntryStrategy):
    name = "new_jj"
    wants_bars = True

    def __init__(self, swing_k=2, bos_tf=None,
                 open_wick_max=25.0, confirm_window_min=10, confirm_wick_max=10.0,
                 rev_room_pts=38.0, rev_room_mode="extension",
                 break_buffer_pts=0.0, require_body_close=True,
                 reversion_choch_only=False, strict_monotonic=False,
                 continuation_sessions=("AM",),
                 reversion_sessions=("AM", "PM", "Evening", "Asia"),
                 premarket_max_pts=None, premarket_min=60,      # legacy / unused
                 cont_trigger_pts=30.0, cont_window_min=None,   # legacy / unused
                 take_continuation=True, take_reversion=True):
        self.swing_k = swing_k                 # == LuxAlgo p ; LuxAlgo length = 2*swing_k+1
        self.bos_tf = bos_tf
        self.open_wick_max = open_wick_max
        self.confirm_window_min = confirm_window_min
        self.confirm_wick_max = confirm_wick_max
        self.rev_room_pts = rev_room_pts
        self.break_buffer_pts = break_buffer_pts
        self.require_body_close = require_body_close
        self.rev_room_mode = rev_room_mode   # 'extension' (how far price ran from fair) or 'residual'
        self.reversion_choch_only = reversion_choch_only
        self.strict_monotonic = strict_monotonic
        self.continuation_sessions = tuple(continuation_sessions)
        self.reversion_sessions = tuple(reversion_sessions)
        self.take_continuation = take_continuation
        self.take_reversion = take_reversion
        self.premarket_max_pts = premarket_max_pts
        self.premarket_min = premarket_min
        self.cont_trigger_pts = cont_trigger_pts
        self.cont_window_min = cont_window_min

    @staticmethod
    def _max_wick(op, hi, lo, cl):
        bh = max(op, cl); bl = min(op, cl)
        return max(hi - bh, bl - lo)

    # ---- LuxAlgo BOS/CHoCH port (body-close crossover of last uncrossed swing) ----
    def _bos(self, sess):
        """Returns (brk_up, brk_dn, choch) over sess rows.
        brk_up/brk_dn: a break of the last uncrossed swing high/low (each fires
        once). choch[j]: that break was a CHoCH (reversal) rather than a BOS
        (continuation), via LuxAlgo's `os` state. Fully causal."""
        p = self.swing_k
        n = len(sess)
        brk_up = np.zeros(n, bool); brk_dn = np.zeros(n, bool); choch = np.zeros(n, bool)
        if n < 2 * p + 2:
            return brk_up, brk_dn, choch
        h = sess["high"].to_numpy(float)
        l = sess["low"].to_numpy(float)
        c = sess["close"].to_numpy(float)
        up_lvl, dn_lvl = _fractals_lux(h, l, p, self.strict_monotonic)
        up_break = c if self.require_body_close else h
        dn_break = c if self.require_body_close else l
        buf = self.break_buffer_pts
        os = 0
        cur_up = np.nan; up_crossed = True
        cur_dn = np.nan; dn_crossed = True
        for j in range(n):
            if np.isfinite(up_lvl[j]) and not (up_lvl[j] == cur_up):   # new swing high -> re-arm
                cur_up = up_lvl[j]; up_crossed = False
            if np.isfinite(dn_lvl[j]) and not (dn_lvl[j] == cur_dn):
                cur_dn = dn_lvl[j]; dn_crossed = False
            if j == 0:
                continue
            if (not up_crossed and np.isfinite(cur_up)
                    and up_break[j - 1] <= cur_up and up_break[j] > cur_up + buf):
                brk_up[j] = True; up_crossed = True
                if os == -1:
                    choch[j] = True
                os = 1
            if (not dn_crossed and np.isfinite(cur_dn)
                    and dn_break[j - 1] >= cur_dn and dn_break[j] < cur_dn - buf):
                brk_dn[j] = True; dn_crossed = True
                if os == 1:
                    choch[j] = True
                os = -1
        return brk_up, brk_dn, choch

    def _session_entries(self, sname, day_bars, fair_time, sess_end, pm_start,
                         am_fair, pos, out_idx, out_dir):
        sess = day_bars.between_time(fair_time, sess_end)
        if sess.empty:
            return
        ts = sess.index
        o = sess["open"].to_numpy(float); h = sess["high"].to_numpy(float)
        l = sess["low"].to_numpy(float); c = sess["close"].to_numpy(float)
        anchor = ts[0].normalize() + pd.Timedelta(fair_time + ":00")
        fair_open = o[0]

        # ---------- CONTINUATION ----------
        if self.take_continuation and sname in self.continuation_sessions:
            d0 = 1 if c[0] > o[0] else (-1 if c[0] < o[0] else 0)
            if d0 != 0:
                if self._max_wick(o[0], h[0], l[0], c[0]) <= self.open_wick_max:
                    out_idx.append(pos[ts[0]]); out_dir.append(d0)
                else:
                    win_end = anchor + pd.Timedelta(minutes=self.confirm_window_min)
                    for j in range(1, len(sess)):
                        if ts[j] >= win_end:
                            break
                        dj = 1 if c[j] > o[j] else (-1 if c[j] < o[j] else 0)
                        if dj == d0 and self._max_wick(o[j], h[j], l[j], c[j]) < self.confirm_wick_max:
                            out_idx.append(pos[ts[j]]); out_dir.append(d0); break

        # ---------- REVERSION ----------
        if self.take_reversion and sname in self.reversion_sessions:
            brk_up, brk_dn, choch = self._bos(sess)
            targets = [fair_open]
            if am_fair is not None:
                targets.append(am_fair)
            ext = self.rev_room_mode == "extension"
            run_hi, run_lo = -np.inf, np.inf        # causal running session extremes
            for j in range(len(sess)):
                run_hi = max(run_hi, h[j]); run_lo = min(run_lo, l[j])
                if self.reversion_choch_only and not choch[j]:
                    continue
                price = c[j]
                # DOWN-break with fair BELOW -> SHORT toward fair (price reverts down to
                # fair after an unfair move up). room = how far price ran ABOVE fair
                # ('extension') or the residual distance at the break ('residual').
                if brk_dn[j]:
                    below = [t for t in targets if t < price]
                    if below:
                        room = (run_hi - max(below)) if ext else (price - max(below))
                        if room >= self.rev_room_pts:
                            out_idx.append(pos[ts[j]]); out_dir.append(-1)
                # UP-break with fair ABOVE -> LONG toward fair (reverts up after an
                # unfair move down). room = how far price ran BELOW fair, or residual.
                if brk_up[j]:
                    above = [t for t in targets if t > price]
                    if above:
                        room = (min(above) - run_lo) if ext else (min(above) - price)
                        if room >= self.rev_room_pts:
                            out_idx.append(pos[ts[j]]); out_dir.append(1)

    def entries(self, bars: pd.DataFrame):
        pos = {ts: i for i, ts in enumerate(bars.index)}
        out_idx, out_dir = [], []
        for _, day_bars in bars.groupby(bars.index.normalize()):
            am = day_bars.between_time("09:30", "11:00")
            am_fair = float(am["open"].iloc[0]) if len(am) else None
            for sname, fair_time, sess_end, pm_start in SESSIONS:
                extra = am_fair if sname == "PM" else None
                self._session_entries(sname, day_bars, fair_time, sess_end, pm_start,
                                      extra, pos, out_idx, out_dir)
        if not out_idx:
            return np.array([], dtype=int), np.array([], dtype=int)
        idx = np.array(out_idx, dtype=int); direction = np.array(out_dir, dtype=int)
        order = np.argsort(idx, kind="stable")
        uniq = {}
        for i, d in zip(idx[order], direction[order]):
            uniq.setdefault((i, d), True)
        keys = np.array(list(uniq.keys()))
        return keys[:, 0].astype(int), keys[:, 1].astype(int)


def get_strategy():
    return NewJJ()