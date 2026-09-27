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
    A candle qualifies iff its DIRECTIONAL wick <= wick_body_ratio_max (0.25)
    * |body|. Directional wick = the wick on the SAME side as the close only
    (top wick for a long, bottom wick for a short) -- that's the one that
    shows price ran further and pulled back before closing (hesitation). The
    opposite-side wick is ignored entirely: a long that dips on a lower wick
    and still closes strong through it is a sign of conviction, not
    something to penalize. This one relative test on the directional wick
    rejects BOTH a hesitant candle (wick dominates its own side) AND a
    trivially small body (little room for any wick to stay under 25% of it).
    CLEAN open  (opening candle itself qualifies): enter that direction on the open.
    MESSY open  (opening candle doesn't qualify): wait for the first qualifying
      candle, SAME direction, within confirm_window_min (5) min; enter there.
      Else no continuation. (Whole setup is therefore confined to the first
      5 minutes of the session open.)

  REVERSION (LuxAlgo structure -> fade to fair)
    A break = the CLOSE crosses the last uncrossed fractal (body, not wick;
    require_body_close=False uses the wick). Either direction; the fade side is
    set by which side of fair price is on:
        break DOWN a swing low while ABOVE fair -> SHORT to fair
        break UP   a swing high while BELOW fair -> LONG  to fair
    reversion_choch_only=False (default): both CHoCH (reversal) and same-
    direction BOS (continuation) breaks fire. A run of successive same-
    direction BOS breaks (e.g. several SHORTs during one selloff) is NOT
    redundant re-signaling -- each is a break of a genuinely different,
    fresh swing pivot (there's a real intervening reversal at the smaller
    degree between them by construction of the fractal swings), which is
    exactly what continuation/structure trading treats as separately
    actionable. one_position (in Backtest.py) already prevents these from
    overlapping into simultaneous trades. Set True to require CHoCH only --
    but note `os` resets to 0 every session (_bos() runs per session), so a
    session's genuinely first break can then never qualify (os can't be
    +-1 yet), which discards real entries, not duplicates.
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
    (up[i]) and swing-low level (dn[i]) — nan until one exists — plus the bar
    INDEX of the pivot candle itself (up_pos[i]/dn_pos[i], -1 until one exists;
    up[i] == high[up_pos[i]] by construction). The position lets a caller point
    back at exactly which candle a level came from, e.g. for debug_entries().

    Pivot test: the center bar is the strict window extreme (max high / min low
    over the 2p+1 window, uniquely at the center). This is LuxAlgo's
    `high[p] == ta.highest(length)` condition and is robust to tick ties.

    strict_monotonic=True additionally requires highs to rise p bars into the
    pivot and fall p bars out (LuxAlgo's `dh`/`dl` sign-sum). Faithful to the Pine
    source but BRITTLE on tick-quantized 1-min data (one equal tick kills a swing),
    so it is OFF by default; enable only for indicator-parity checks."""
    n = high.size
    up = np.full(n, np.nan); dn = np.full(n, np.nan)
    up_pos = np.full(n, -1, dtype=int); dn_pos = np.full(n, -1, dtype=int)
    last_up = last_dn = np.nan
    last_up_pos = last_dn_pos = -1
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
                last_up = high[c]; last_up_pos = c
            if is_lo:
                last_dn = low[c]; last_dn_pos = c
        up[i] = last_up; dn[i] = last_dn
        up_pos[i] = last_up_pos; dn_pos[i] = last_dn_pos
    return up, dn, up_pos, dn_pos


class NewJJ(EntryStrategy):
    name = "new_jj"
    wants_bars = True

    def __init__(self, swing_k=2, bos_tf=None,
                 wick_body_ratio_max=0.25, confirm_window_min=5,
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
        self.wick_body_ratio_max = wick_body_ratio_max   # candle qualifies iff max_wick <= this * |body|
        self.confirm_window_min = confirm_window_min
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
    def _dir_wick(direction, hi, lo, cl):
        """Wick on the SAME side as the close -- the only wick that signals
        hesitation/rejection at the end of the candle. For a long (direction=1,
        close above open) that's the TOP wick, high - close: price ran higher
        and pulled back before the bar closed. For a short it's the BOTTOM
        wick, close - low. The other wick (e.g. a long's lower wick, open -
        low) is NOT penalized -- price dipping and still closing strong
        through it is a sign of conviction, not indecision, so it's ignored
        here entirely (not just under-weighted)."""
        return (hi - cl) if direction > 0 else (cl - lo)

    # ---- LuxAlgo BOS/CHoCH port (body-close crossover of last uncrossed swing) ----
    def _bos(self, sess):
        """Returns (brk_up, brk_dn, choch, up_pivot_pos, dn_pivot_pos) over sess rows.
        brk_up/brk_dn: a break of the last uncrossed swing high/low (each fires
        once). choch[j]: that break was a CHoCH (reversal) rather than a BOS
        (continuation), via LuxAlgo's `os` state. up_pivot_pos[j]/dn_pivot_pos[j]
        (int, -1 if none): the LOCAL index within `sess` of the pivot candle that
        set the level being broken at bar j (only meaningful where brk_up[j]/
        brk_dn[j] is True) -- e.g. sess.index[up_pivot_pos[j]] is the actual
        candle "the break is based on". Fully causal."""
        p = self.swing_k
        n = len(sess)
        brk_up = np.zeros(n, bool); brk_dn = np.zeros(n, bool); choch = np.zeros(n, bool)
        up_pivot_pos = np.full(n, -1, dtype=int); dn_pivot_pos = np.full(n, -1, dtype=int)
        if n < 2 * p + 2:
            return brk_up, brk_dn, choch, up_pivot_pos, dn_pivot_pos
        h = sess["high"].to_numpy(float)
        l = sess["low"].to_numpy(float)
        c = sess["close"].to_numpy(float)
        up_lvl, dn_lvl, up_lvl_pos, dn_lvl_pos = _fractals_lux(h, l, p, self.strict_monotonic)
        up_break = c if self.require_body_close else h
        dn_break = c if self.require_body_close else l
        buf = self.break_buffer_pts
        os = 0
        cur_up = np.nan; up_crossed = True; cur_up_pos = -1
        cur_dn = np.nan; dn_crossed = True; cur_dn_pos = -1
        for j in range(n):
            if np.isfinite(up_lvl[j]) and not (up_lvl[j] == cur_up):   # new swing high -> re-arm
                cur_up = up_lvl[j]; up_crossed = False; cur_up_pos = up_lvl_pos[j]
            if np.isfinite(dn_lvl[j]) and not (dn_lvl[j] == cur_dn):
                cur_dn = dn_lvl[j]; dn_crossed = False; cur_dn_pos = dn_lvl_pos[j]
            if j == 0:
                continue
            if (not up_crossed and np.isfinite(cur_up)
                    and up_break[j - 1] <= cur_up and up_break[j] > cur_up + buf):
                brk_up[j] = True; up_crossed = True; up_pivot_pos[j] = cur_up_pos
                if os == -1:
                    choch[j] = True
                os = 1
            if (not dn_crossed and np.isfinite(cur_dn)
                    and dn_break[j - 1] >= cur_dn and dn_break[j] < cur_dn - buf):
                brk_dn[j] = True; dn_crossed = True; dn_pivot_pos[j] = cur_dn_pos
                if os == 1:
                    choch[j] = True
                os = -1
        return brk_up, brk_dn, choch, up_pivot_pos, dn_pivot_pos

    def _session_entries(self, sname, day_bars, fair_time, sess_end, pm_start,
                         am_fair, pos, out_idx, out_dir, debug=None):
        sess = day_bars.between_time(fair_time, sess_end)
        if sess.empty:
            return
        ts = sess.index
        o = sess["open"].to_numpy(float); h = sess["high"].to_numpy(float)
        l = sess["low"].to_numpy(float); c = sess["close"].to_numpy(float)
        anchor = ts[0].normalize() + pd.Timedelta(fair_time + ":00")
        fair_open = o[0]

        # ---------- CONTINUATION ----------
        # A candle "qualifies" iff its DIRECTIONAL wick (top wick for a long,
        # bottom wick for a short -- see _dir_wick) <= wick_body_ratio_max *
        # |body|. Only the wick on the close's own side is penalized: it's the
        # one that shows price ran further and pulled back before closing
        # (hesitation). The opposite wick is ignored -- e.g. a long that dips
        # on a lower wick and still closes strong through it is a sign of
        # conviction, not a reason to reject the candle.
        if self.take_continuation and sname in self.continuation_sessions:
            body0 = c[0] - o[0]
            d0 = 1 if body0 > 0 else (-1 if body0 < 0 else 0)
            if d0 != 0:
                if self._dir_wick(d0, h[0], l[0], c[0]) <= self.wick_body_ratio_max * abs(body0):
                    out_idx.append(pos[ts[0]]); out_dir.append(d0)
                    if debug is not None:
                        debug.append(dict(time=ts[0], setup="continuation", direction=d0,
                                          session=sname, qualified_at="open",
                                          body=body0, dir_wick=self._dir_wick(d0, h[0], l[0], c[0])))
                else:
                    win_end = anchor + pd.Timedelta(minutes=self.confirm_window_min)
                    for j in range(1, len(sess)):
                        if ts[j] >= win_end:
                            break
                        bodyj = c[j] - o[j]
                        dj = 1 if bodyj > 0 else (-1 if bodyj < 0 else 0)
                        if dj == d0 and self._dir_wick(dj, h[j], l[j], c[j]) <= self.wick_body_ratio_max * abs(bodyj):
                            out_idx.append(pos[ts[j]]); out_dir.append(d0)
                            if debug is not None:
                                debug.append(dict(time=ts[j], setup="continuation", direction=d0,
                                                  session=sname, qualified_at="confirm",
                                                  body=bodyj, dir_wick=self._dir_wick(dj, h[j], l[j], c[j])))
                            break

        # ---------- REVERSION ----------
        if self.take_reversion and sname in self.reversion_sessions:
            brk_up, brk_dn, choch, up_pivot_pos, dn_pivot_pos = self._bos(sess)
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
                            if debug is not None:
                                pp = dn_pivot_pos[j]
                                debug.append(dict(time=ts[j], setup="reversion", direction=-1,
                                                  session=sname, break_price=price, choch=bool(choch[j]),
                                                  pivot_time=ts[pp] if pp >= 0 else pd.NaT,
                                                  pivot_price=(l[pp] if pp >= 0 else np.nan),
                                                  pivot_confirm_time=(ts[pp + self.swing_k]
                                                      if pp >= 0 and pp + self.swing_k < len(ts) else pd.NaT),
                                                  fair_target=max(below), room=room))
                # UP-break with fair ABOVE -> LONG toward fair (reverts up after an
                # unfair move down). room = how far price ran BELOW fair, or residual.
                if brk_up[j]:
                    above = [t for t in targets if t > price]
                    if above:
                        room = (min(above) - run_lo) if ext else (min(above) - price)
                        if room >= self.rev_room_pts:
                            out_idx.append(pos[ts[j]]); out_dir.append(1)
                            if debug is not None:
                                pp = up_pivot_pos[j]
                                debug.append(dict(time=ts[j], setup="reversion", direction=1,
                                                  session=sname, break_price=price, choch=bool(choch[j]),
                                                  pivot_time=ts[pp] if pp >= 0 else pd.NaT,
                                                  pivot_price=(h[pp] if pp >= 0 else np.nan),
                                                  pivot_confirm_time=(ts[pp + self.swing_k]
                                                      if pp >= 0 and pp + self.swing_k < len(ts) else pd.NaT),
                                                  fair_target=min(above), room=room))

    def entries(self, bars: pd.DataFrame, debug=None):
        """debug: optional list -- if passed, every appended entry (continuation
        AND reversion, pre-dedup) also appends a dict of its full context to it.
        See debug_entries() for the DataFrame wrapper; this param exists so
        debug_entries() runs the EXACT same code path as the live entries(),
        guaranteeing the two can never silently drift apart."""
        pos = {ts: i for i, ts in enumerate(bars.index)}
        out_idx, out_dir = [], []
        for _, day_bars in bars.groupby(bars.index.normalize()):
            am = day_bars.between_time("09:30", "11:00")
            am_fair = float(am["open"].iloc[0]) if len(am) else None
            for sname, fair_time, sess_end, pm_start in SESSIONS:
                extra = am_fair if sname == "PM" else None
                self._session_entries(sname, day_bars, fair_time, sess_end, pm_start,
                                      extra, pos, out_idx, out_dir, debug=debug)
        if not out_idx:
            return np.array([], dtype=int), np.array([], dtype=int)
        idx = np.array(out_idx, dtype=int); direction = np.array(out_dir, dtype=int)
        order = np.argsort(idx, kind="stable")
        uniq = {}
        for i, d in zip(idx[order], direction[order]):
            uniq.setdefault((i, d), True)
        keys = np.array(list(uniq.keys()))
        return keys[:, 0].astype(int), keys[:, 1].astype(int)

    def debug_entries(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Every entry entries() emits (pre inter-session dedup), with full
        context for manual verification -- calls entries() itself (see its
        `debug` param) so this can never drift out of sync with the live
        signal path. For reversion rows: pivot_time/pivot_price is the actual
        candle "the break is based on" (the swing high/low), pivot_confirm_time
        is when it became knowable (pivot_time + swing_k bars -- LuxAlgo/this
        port only track a swing once it's confirmed), break_price is the close
        that crossed it, choch marks reversal vs. continuation breaks, and
        fair_target/room are the distance-to-fair filter's inputs. For
        continuation rows: qualified_at is 'open' or 'confirm', body/dir_wick
        are the values wick_body_ratio_max was tested against."""
        rows = []
        self.entries(bars, debug=rows)
        cols = ["time", "setup", "direction", "session", "qualified_at", "body", "dir_wick",
                "break_price", "choch", "pivot_time", "pivot_price", "pivot_confirm_time",
                "fair_target", "room"]
        df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
        return df.set_index("time").sort_index()


def get_strategy():
    return NewJJ()