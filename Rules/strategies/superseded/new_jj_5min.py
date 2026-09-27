"""
strategies/new_jj.py — AM/PM/Evening/Asia session "fair price" model, ENTRY-ONLY.

Clean, fully-causal rebuild (replaces the look-ahead-prone JJ.py). Backtest.py
owns exits (ATR 1R stop, scored across every rr); this file emits entries only.

Two setups per session, each independently toggleable:

  CONTINUATION (momentum, first 15 min only)
    - Gated by a QUIET premarket: skip if the 60 min before the anchor ranged
      more than `premarket_max_pts` (75). Causal — reads only the prior hour.
    - If price moves more than `cont_trigger_pts` (30) from the session fair-open
      within `cont_window_min` (15) minutes, enter in that direction.

  REVERSION (fade back to fair price)
    - Signal: a Break of Structure on a `bos_tf` (5-min) resampled view, detected
      with causal fractal swings (confirmed k bars later).
    - Direction: back toward the session fair-open (already established -> not
      look-ahead).
    - Room filter: take only if the distance from the current price back to the
      target fair price is >= `rev_room_pts` (38). PM may also target the AM
      fair-open, whichever valid target has >= room.

Sessions (ET; requires full-session/globex data):
  AM 09:30->11:00 | PM 14:00->15:00 | Evening 18:00->20:00 | Asia 20:00->21:00

Contract: wants_bars=True, entries(bars)->(idx,direction), get_strategy(),
settable swing_k, name.

NQ points, no ATR normalization (by request).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategies.base import EntryStrategy


# session anchors: (name, fair_time, session_end, premarket_start)
SESSIONS = [
    ("AM",      "09:30", "11:00", "08:30"),
    ("PM",      "14:00", "15:00", "13:00"),
    ("Evening", "18:00", "20:00", "17:00"),
    ("Asia",    "20:00", "21:00", "19:00"),
]


def _causal_swing_levels(high, low, k):
    """Last CONFIRMED swing-high / swing-low level as of each bar (no lookahead).
    A swing centered at bar c is only usable from bar c+k onward."""
    n = high.size
    sh = np.full(n, np.nan); sl = np.full(n, np.nan)
    last_h = last_l = np.nan
    for i in range(n):
        c = i - k                              # candidate center, confirmed now
        if c - k >= 0:
            wh = high[c - k:c + k + 1]; wl = low[c - k:c + k + 1]
            if high[c] == wh.max() and int(np.argmax(wh)) == k:
                last_h = high[c]
            if low[c] == wl.min() and int(np.argmin(wl)) == k:
                last_l = low[c]
        sh[i] = last_h; sl[i] = last_l
    return sh, sl


class NewJJ(EntryStrategy):
    name = "new_jj"
    wants_bars = True

    def __init__(self, swing_k=2, bos_tf="3min",
                 premarket_max_pts=75.0, premarket_min=60,
                 cont_trigger_pts=30.0, cont_window_min=15,
                 rev_room_pts=38.0,
                 take_continuation=True, take_reversion=True):
        self.swing_k = swing_k
        self.bos_tf = bos_tf
        self.premarket_max_pts = premarket_max_pts
        self.premarket_min = premarket_min
        self.cont_trigger_pts = cont_trigger_pts
        self.cont_window_min = cont_window_min
        self.rev_room_pts = rev_room_pts
        self.take_continuation = take_continuation
        self.take_reversion = take_reversion

    # ---- 5-min BOS mapped back onto the base-timeframe bars ----
    def _htf_bos(self, sess):
        """Return two boolean arrays over sess rows: (bos_up, bos_dn) where a
        higher-timeframe (bos_tf) break of the last confirmed swing occurred."""
        n = len(sess)
        bos_up = np.zeros(n, bool); bos_dn = np.zeros(n, bool)
        htf = sess.resample(self.bos_tf).agg(
            {"high": "max", "low": "min", "close": "last"}).dropna()
        if len(htf) < 2 * self.swing_k + 2:
            return bos_up, bos_dn
        h = htf["high"].to_numpy(float); l = htf["low"].to_numpy(float)
        c = htf["close"].to_numpy(float)
        sh, sl = _causal_swing_levels(h, l, self.swing_k)
        up = np.zeros(len(htf), bool); dn = np.zeros(len(htf), bool)
        for j in range(1, len(htf)):
            if np.isfinite(sh[j]) and c[j - 1] <= sh[j] and c[j] > sh[j]:
                up[j] = True
            if np.isfinite(sl[j]) and c[j - 1] >= sl[j] and c[j] < sl[j]:
                dn[j] = True
        # map each htf bar's signal onto the base bar at the htf bar's timestamp
        htf_ts = htf.index
        base_pos = {ts: i for i, ts in enumerate(sess.index)}
        for j in range(len(htf)):
            if up[j] or dn[j]:
                # htf bar closes at its label; find the base bar at/after that close
                p = base_pos.get(htf_ts[j], None)
                if p is None:
                    # nearest base bar within the htf interval
                    loc = sess.index.searchsorted(htf_ts[j])
                    if loc >= n:
                        continue
                    p = loc
                bos_up[p] |= up[j]; bos_dn[p] |= dn[j]
        return bos_up, bos_dn

    def _session_entries(self, day_bars, fair_time, sess_end, pm_start,
                         am_fair, pos, out_idx, out_dir):
        sess = day_bars.between_time(fair_time, sess_end)
        if sess.empty:
            return
        ts = sess.index
        o = sess["open"].to_numpy(float); h = sess["high"].to_numpy(float)
        l = sess["low"].to_numpy(float); c = sess["close"].to_numpy(float)

        anchor = ts[0].normalize() + pd.Timedelta(fair_time + ":00")
        fair_open = o[0]                                   # first bar's open = fair price

        # ---------- CONTINUATION ----------
        if self.take_continuation:
            pm = day_bars.between_time(pm_start, fair_time)
            pm = pm.iloc[:-1] if len(pm) else pm          # exclude the anchor bar itself
            pm_range = (pm["high"].max() - pm["low"].min()) if len(pm) else 0.0
            quiet = (not np.isfinite(pm_range)) or (pm_range <= self.premarket_max_pts)
            if quiet:
                win_end = anchor + pd.Timedelta(minutes=self.cont_window_min)
                for j in range(len(sess)):
                    if ts[j] >= win_end:
                        break
                    up = h[j] - fair_open
                    dn = fair_open - l[j]
                    if up > self.cont_trigger_pts and up >= dn:
                        out_idx.append(pos[ts[j]]); out_dir.append(1); break
                    if dn > self.cont_trigger_pts:
                        out_idx.append(pos[ts[j]]); out_dir.append(-1); break

        # ---------- REVERSION ----------
        if self.take_reversion:
            bos_up, bos_dn = self._htf_bos(sess)
            targets = [fair_open]
            if am_fair is not None:                        # PM: AM open is a 2nd target
                targets.append(am_fair)
            for j in range(len(sess)):
                price = c[j]
                # a DOWN bos above fair -> fade short back toward a target below
                if bos_dn[j]:
                    room = max((price - t) for t in targets if t < price) \
                        if any(t < price for t in targets) else -1
                    if room >= self.rev_room_pts:
                        out_idx.append(pos[ts[j]]); out_dir.append(-1)
                # an UP bos below fair -> fade long back toward a target above
                if bos_up[j]:
                    room = max((t - price) for t in targets if t > price) \
                        if any(t > price for t in targets) else -1
                    if room >= self.rev_room_pts:
                        out_idx.append(pos[ts[j]]); out_dir.append(1)

    def entries(self, bars: pd.DataFrame):
        pos = {ts: i for i, ts in enumerate(bars.index)}
        out_idx, out_dir = [], []
        for _, day_bars in bars.groupby(bars.index.normalize()):
            # AM fair-open (target reused by PM); None if no AM session that day
            am = day_bars.between_time("09:30", "11:00")
            am_fair = float(am["open"].iloc[0]) if len(am) else None
            for sname, fair_time, sess_end, pm_start in SESSIONS:
                extra = am_fair if sname == "PM" else None
                self._session_entries(day_bars, fair_time, sess_end, pm_start,
                                      extra, pos, out_idx, out_dir)
        if not out_idx:
            return np.array([], dtype=int), np.array([], dtype=int)
        idx = np.array(out_idx, dtype=int); direction = np.array(out_dir, dtype=int)
        order = np.argsort(idx, kind="stable")
        # de-dup identical (idx,dir) that both setups might emit on the same bar
        uniq = {}
        for i, d in zip(idx[order], direction[order]):
            uniq.setdefault((i, d), True)
        keys = np.array(list(uniq.keys()))
        return keys[:, 0].astype(int), keys[:, 1].astype(int)


def get_strategy():
    return NewJJ()