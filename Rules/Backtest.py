"""
Backtest.py — turn an entry-only strategy + OHLC data into an R-multiple ledger.

Exits are NOT part of the strategy. Each entry gets a bracket:
    stop distance = k_stop_atr * ATR   (this is 1R, the risk unit)
    target        = rr * stop distance (rr is what we're solving for)
We walk each entry forward ONCE and record the max favorable excursion reached
before the stop was touched. That single pass lets us score EVERY candidate rr:
a trade wins at rr iff it reached rr before being stopped. So the win rate at
each rr is MEASURED from your data, never assumed.

Data: 1-min OHLC parquet (or csv) in the data/ folder. Columns open/high/low/
close (+ a datetime index or column). See data/README.md.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import importlib.util
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def atr(high, low, close, period=14):
    """Wilder ATR (the 1R unit)."""
    high = np.asarray(high, float); low = np.asarray(low, float); close = np.asarray(close, float)
    prev = np.empty_like(close); prev[0] = close[0]; prev[1:] = close[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    out = np.full_like(tr, np.nan)
    if tr.size >= period:
        out[period - 1] = tr[:period].mean()
        a = 1.0 / period
        for i in range(period, tr.size):
            out[i] = out[i - 1] * (1 - a) + tr[i] * a
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_ohlc(path, rth_only=False, timeframe=None, tz="America/New_York"):
    """Load OHLC parquet/csv at any timeframe. Flexible column names.

    Reads the output of build_continuous.py ({TICKER}_{period}.csv) as well as
    parquet. That file's timestamps are UTC (tz-aware); when so, they're
    converted to `tz` so rth_only filters the correct local session. Set
    tz=None to leave timestamps untouched (e.g. for already-local data)."""
    path = path if os.path.isabs(path) else os.path.join(HERE, path)
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    df.columns = [str(c).lower() for c in df.columns]
    dtcol = next((c for c in df.columns
                  if c in ("datetime", "date", "timestamp", "time", "ts_event")), None)
    if dtcol:
        df[dtcol] = pd.to_datetime(df[dtcol]); df = df.set_index(dtcol)
    else:
        df.index = pd.to_datetime(df.index)
    df = df[["open", "high", "low", "close"]].sort_index()
    # UTC (or any tz-aware) -> target zone, so the RTH window is the right session
    if tz is not None and getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(tz)
    if rth_only:  # US equity-index regular session, in `tz` (default ET)
        import datetime as _dt
        t = df.index.time
        df = df[(t >= _dt.time(9, 30)) & (t <= _dt.time(16, 0))]
    if timeframe:
        df = resample(df, timeframe)
    return df


def resample(df, rule="5min"):
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    return df.resample(rule).agg(agg).dropna()


# ---------------------------------------------------------------------------
# Barrier backtest
# ---------------------------------------------------------------------------
@dataclass
class Entries:
    day_ids: np.ndarray       # integer trading-day label per entry (chronological)
    mfe_before: np.ndarray    # max favorable excursion (in R) before stop touched
    stopped: np.ndarray       # bool: stop touched within max_hold
    timeout_R: np.ndarray     # R at exit if neither barrier hit (else nan)
    n_bars: int = 0
    n_days: int = 0
    source: str = "?"
    strategy: str = "?"

    def summary(self):
        tpd = len(self.day_ids) / max(self.n_days, 1)
        return (f"{len(self.day_ids)} entries over {self.n_days} days "
                f"(~{tpd:.2f} trades/day) | {self.strategy} on {self.source}")


def backtest_entries(strategy, bars, atr_p=14, k_stop_atr=1.0, max_hold=480,
                     one_position=True):
    """Walk each entry forward once; record what's needed to score any rr."""
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    n = close.size
    a = atr(high, low, close, atr_p)
    dates = bars.index.normalize().to_numpy()

    if getattr(strategy, "wants_bars", False):
        idx, direction = strategy.entries(bars)          # time/volume-aware (ORB, Keltner)
    else:
        idx, direction = strategy.entries(high, low, close)  # price-only (BOS, EMA)

    day_ids, mfe_before, stopped, timeout_R = [], [], [], []
    day_counter, last_day, busy_until = -1, None, -1
    for e, d in zip(idx, direction):
        if e <= atr_p or e >= n - 2:
            continue
        if one_position and e <= busy_until:
            continue
        stop_dist = k_stop_atr * a[e]
        if not np.isfinite(stop_dist) or stop_dist <= 0:
            continue
        entry = close[e]
        j0, j1 = e + 1, min(e + 1 + max_hold, n)
        h = high[j0:j1]; lo = low[j0:j1]
        if d == 1:
            fav = (h - entry) / stop_dist; adv = (entry - lo) / stop_dist
        else:
            fav = (entry - lo) / stop_dist; adv = (h - entry) / stop_dist
        stops = np.flatnonzero(adv >= 1.0)
        if stops.size:
            sb = int(stops[0])
            mfe = float(fav[:sb].max()) if sb > 0 else 0.0   # conservative: exclude stop bar
            was_stopped, to_R, exit_bar = True, np.nan, j0 + sb
        else:
            mfe = float(fav.max()) if fav.size else 0.0
            was_stopped = False
            to_R = float((close[j1 - 1] - entry) / stop_dist * d)
            exit_bar = j1 - 1
        day = dates[e]
        if day != last_day:
            day_counter += 1; last_day = day
        day_ids.append(day_counter); mfe_before.append(mfe)
        stopped.append(was_stopped); timeout_R.append(to_R)
        busy_until = exit_bar

    return Entries(np.array(day_ids), np.array(mfe_before), np.array(stopped),
                   np.array(timeout_R), n_bars=n, n_days=int(day_counter + 1),
                   strategy=getattr(strategy, "name", "?"))


def ledger_for_rr(entries: Entries, rr: float):
    """Score every entry at a given rr -> (R_array, day_ids). Win iff the trade
    reached rr before its stop; else -1 (stopped) or the scratch/timeout R."""
    win = entries.mfe_before >= rr
    to = np.where(np.isfinite(entries.timeout_R), entries.timeout_R, -1.0)
    to = np.clip(to, -1.0, rr)
    R = np.where(win, rr, np.where(entries.stopped, -1.0, to))
    return R, entries.day_ids


def measured_winrate(entries: Entries, rr: float) -> float:
    return float(np.mean(entries.mfe_before >= rr))


# ---------------------------------------------------------------------------
# Strategy loading  (by name in strategies/, or by file path)
# ---------------------------------------------------------------------------
def load_strategy(name_or_path: str):
    """load_strategy('bos') imports strategies/bos.py; or pass a file path.
    The module must expose get_strategy() returning an EntryStrategy."""
    if os.path.sep in name_or_path or name_or_path.endswith(".py"):
        path = name_or_path if os.path.isabs(name_or_path) else os.path.join(HERE, name_or_path)
        spec = importlib.util.spec_from_file_location("user_strategy", path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    else:
        import importlib
        mod = importlib.import_module(f"strategies.{name_or_path}")
    return mod.get_strategy()


def build_entries(strategy_name, data_path, rth_only=False, timeframe=None,
                  atr_p=14, k_stop_atr=1.0, max_hold=480, swing_k=None,
                  tz="America/New_York"):
    """One call: strategy name/path + data file -> Entries (scorable at any rr)."""
    strat = load_strategy(strategy_name)
    if swing_k is not None and hasattr(strat, "swing_k"):
        strat.swing_k = swing_k
    bars = load_ohlc(data_path, rth_only=rth_only, timeframe=timeframe, tz=tz)
    E = backtest_entries(strat, bars, atr_p=atr_p, k_stop_atr=k_stop_atr, max_hold=max_hold)
    E.source = os.path.basename(data_path)
    return E