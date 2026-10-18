"""
Backtest.py: turn an entry-only strategy + OHLC data into an R-multiple
ledger. The stop (1R) is a fixed number of points, not an ATR multiple.

Why points: contract count is a free variable, so a 20pt stop and a 50pt stop
can carry the same dollar risk. What stop width really changes is which
trades win, lose or scratch. A tighter stop gets touched more often before the
move plays out; a wider one needs a bigger move to reach the same R. An ATR
stop mixes that with how volatile the market was when the trade fired, so
switching from RTH-only bars to a 24h feed changes ATR, the stop, and every
number downstream, even though the strategy didn't change. Sweeping the stop
in points makes the risk-geometry search a 3-D grid over (stop points,
risk % per trade, R:R); see Analysis.py.

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
# Barrier backtest -- stop_points is a FIXED number of points, not ATR-derived
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
    stop_points: float = 0.0  # the fixed stop distance (points) this ledger was built with
    mae_before: np.ndarray = None  # max adverse excursion (R, positive) before the trade resolved
    sizes: "np.ndarray | None" = None   # per-trade size multiplier (None = all 1.0)
    # Filled only by backtest_entries(..., detail=True), so trade_log() can print
    # exact times/prices to check against real fills. None otherwise.
    entry_time: "pd.DatetimeIndex | None" = None
    entry_price: "np.ndarray | None" = None
    direction: "np.ndarray | None" = None
    stop_price: "np.ndarray | None" = None
    exit_time: "pd.DatetimeIndex | None" = None
    bar_duration: "pd.Timedelta | None" = None   # modal bar spacing. entry_time is the bar's
                                                  # start (label='left'); entry_price is that bar's
                                                  # close, known only at entry_time + this

    def summary(self):
        tpd = len(self.day_ids) / max(self.n_days, 1)
        return (f"{len(self.day_ids)} entries over {self.n_days} days "
                f"(~{tpd:.2f} trades/day) | {self.strategy} on {self.source} "
                f"| stop={self.stop_points:g}pt")


def backtest_entries(strategy, bars, stop_points=25.0, max_hold=480,
                     one_position=True, detail=False):
    """Walk each entry forward once; record what's needed to score any rr.

    stop_points: the 1R unit in price points, the same for every entry.
    Analysis.risk_geometry_3d sweeps it as its own axis next to risk% and R:R.

    detail=True also records each entry's timestamp, price, stop and exit
    timestamp, for checking trades one by one with trade_log()."""
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    n = close.size
    stop_dist = float(stop_points)
    # dtype=datetime64[ns]: on a tz-aware index, plain .to_numpy() boxes every
    # timestamp as a Python object (~1s on 500k rows) and slows the day
    # comparisons below. Day boundaries are the same either way, since
    # normalize() already snapped to local midnight.
    dates = bars.index.normalize().to_numpy(dtype="datetime64[ns]")

    res = strategy.entries(bars) if getattr(strategy, "wants_bars", False) \
        else strategy.entries(high, low, close)
    if len(res) == 3:
        # a 3rd return value is the exact fill price for each entry
        idx, direction, fill_prices = res
        fill_prices = np.asarray(fill_prices, float)
    else:
        idx, direction = res
        fill_prices = np.full(len(idx), np.nan)
    sizes_in = np.ones(len(idx), float)

    day_ids, mfe_before, stopped, timeout_R, sizes = [], [], [], [], []
    mae_before = []   # deepest adverse excursion (R, positive) before the trade resolved
    entry_times, entry_prices, directions, stop_prices, exit_times = [], [], [], [], []
    day_counter, last_day, busy_until = -1, None, -1

    for k, (e, d) in enumerate(zip(idx, direction)):
        if e <= 0 or e >= n - 2:
            continue
        if one_position and e <= busy_until:
            continue
        if not np.isfinite(stop_dist) or stop_dist <= 0:
            continue

        # the strategy's own fill price if it gave one, else the bar close
        entry = fill_prices[k] if np.isfinite(fill_prices[k]) else close[e]

        # The window starts at e+1. `entry` is bar e's close, so the position
        # isn't live until that bar finishes, and bar e's high/low (which
        # always straddle its close) must not count as a stop or target touch.
        j0, j1 = e + 1, min(e + 1 + max_hold, n)
        h = high[j0:j1]; lo = low[j0:j1]
        if d == 1:
            fav = (h - entry) / stop_dist
            adv = (entry - lo) / stop_dist
        else:
            fav = (entry - lo) / stop_dist
            adv = (h - entry) / stop_dist

        stops = np.flatnonzero(adv >= 1.0)   # bars where the stop is touched

        if stops.size:
            sb = int(stops[0])
            # conservative: no favorable excursion is credited on the stop bar
            mfe = float(fav[:sb].max()) if sb > 0 else 0.0
            mae = float(adv[:sb + 1].max()) if adv.size else 0.0   # includes the stop touch (=1R)
            was_stopped = True
            to_R = np.nan
            exit_bar = j0 + sb
        else:
            mfe = float(fav.max()) if fav.size else 0.0
            mae = float(adv.max()) if adv.size else 0.0            # deepest dip against us, never reached 1R
            was_stopped = False
            to_R = float((close[j1 - 1] - entry) / stop_dist * d)
            exit_bar = j1 - 1

        day = dates[e]
        if day != last_day:
            day_counter += 1; last_day = day
        day_ids.append(day_counter); mfe_before.append(mfe); mae_before.append(mae)
        stopped.append(was_stopped); timeout_R.append(to_R)
        sizes.append(sizes_in[k])
        if detail:
            entry_times.append(bars.index[e]); entry_prices.append(entry)
            directions.append(d); stop_prices.append(entry - d * stop_dist)
            exit_times.append(bars.index[exit_bar])
        busy_until = exit_bar

    kw = {}
    if detail:
        # modal spacing between bars, e.g. Timedelta("3min") for 3-min bars.
        # trade_log() adds it to the bar-open label to get the fill time.
        diffs = pd.Series(np.diff(bars.index.values))
        bar_dur = diffs.mode().iloc[0] if len(diffs) else pd.Timedelta(0)
        kw = dict(entry_time=pd.DatetimeIndex(entry_times),
                  entry_price=np.array(entry_prices, float),
                  direction=np.array(directions, int),
                  stop_price=np.array(stop_prices, float),
                  exit_time=pd.DatetimeIndex(exit_times),
                  bar_duration=bar_dur)
    return Entries(np.array(day_ids), np.array(mfe_before), np.array(stopped),
                   np.array(timeout_R), n_bars=n, n_days=int(day_counter + 1),
                   mae_before=np.array(mae_before),
                   strategy=getattr(strategy, "name", "?"), stop_points=float(stop_points),
                   sizes=np.array(sizes), **kw)


def ledger_for_rr(entries: Entries, rr: float):
    """Score every entry at a given rr -> (R_array, day_ids). Win iff the trade
    reached rr before its stop; else -1 (stopped) or the scratch/timeout R."""
    win = entries.mfe_before >= rr
    to = np.where(np.isfinite(entries.timeout_R), entries.timeout_R, -1.0)
    to = np.clip(to, -1.0, rr)
    R = np.where(win, rr, np.where(entries.stopped, -1.0, to))
    if getattr(entries, "sizes", None) is not None:      # per-setup position sizing
        R = R * entries.sizes
    return R, entries.day_ids


def measured_winrate(entries: Entries, rr: float) -> float:
    return float(np.mean(entries.mfe_before >= rr))


def trade_log(entries: Entries, rr: float | None = None) -> pd.DataFrame:
    """One row per trade taken (after one_position gating), with timestamp,
    direction and entry/stop price, for checking against real fills.

    Timestamps: the index `time` is the bar's start (pandas labels resample
    bins by their left edge), but `entry`/`stop` are that bar's close, which
    is only fillable at `fill_time` = time + one bar. `exit_bar_open` uses the
    same bar-start convention; the real exit can land anywhere in that bar.

    Needs entries from backtest_entries(..., detail=True). Pass `rr` to add
    the target price and the win/loss/scratch outcome at that R:R; it doesn't
    change which entries are listed."""
    if entries.entry_time is None:
        raise ValueError("trade_log() needs entries built with "
                         "backtest_entries(..., detail=True); timestamps/prices "
                         "were not recorded on this Entries object.")
    stop_pts = np.abs(entries.entry_price - entries.stop_price)
    bar_dur = entries.bar_duration if entries.bar_duration is not None else pd.Timedelta(0)
    df = pd.DataFrame({
        "time": entries.entry_time,                       # bar open; the index, for .loc[] date slicing
        "fill_time": entries.entry_time + bar_dur,         # bar close, when `entry` filled
        "exit_bar_open": entries.exit_time,
        "direction": np.where(entries.direction == 1, "LONG", "SHORT"),
        "entry": entries.entry_price,
        "stop": entries.stop_price,
        "stop_pts": stop_pts,
        "mfe_R": entries.mfe_before,
        "stopped": entries.stopped,
        "timeout_R": entries.timeout_R,
    })
    if rr is not None:
        R, _ = ledger_for_rr(entries, rr)
        df["target"] = entries.entry_price + entries.direction * rr * stop_pts
        df["graded_R"] = R   # the R credited at this rr, as fed to the Monte Carlo;
                              # mfe_R/timeout_R above are diagnostic only
        df["outcome_at_rr"] = np.where(R >= rr - 1e-9, "WIN",
                                np.where(R <= -1 + 1e-9, "LOSS", "SCRATCH"))
    return df.set_index("time")


def signal_log(strategy, bars) -> pd.DataFrame:
    """Every raw signal strategy.entries() flags, with no stop, max_hold or
    one_position gating. This matches what a TradingView indicator plots:
    one marker per qualifying bar.

    Use it to check signal logic against a reference chart. trade_log() can
    miss signals, because one_position skips any signal that fires while an
    earlier trade is still open, and a wide stop keeps trades open longer.

    `time` is the bar's open (same convention as trade_log()); the bar is
    fillable at fill_time = time + one bar. `price` is that bar's close, or
    the strategy's own fill price if entries() returned a 3-tuple."""
    wants_bars = getattr(strategy, "wants_bars", False)
    res = strategy.entries(bars) if wants_bars else strategy.entries(
        bars["high"].to_numpy(float), bars["low"].to_numpy(float), bars["close"].to_numpy(float))
    if len(res) == 3:
        idx, direction, fill = res
        price = np.asarray(fill, float)
    else:
        idx, direction = res
        price = bars["close"].to_numpy(float)[idx]

    diffs = pd.Series(np.diff(bars.index.values))
    bar_dur = diffs.mode().iloc[0] if len(diffs) else pd.Timedelta(0)
    df = pd.DataFrame({
        "time": bars.index[idx],
        "fill_time": bars.index[idx] + bar_dur,
        "direction": np.where(np.asarray(direction) == 1, "LONG", "SHORT"),
        "price": price,
    })
    return df.set_index("time")


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
        mod = importlib.import_module(f"strategies.{name_or_path}")
    return mod.get_strategy()


def build_entries(strategy_name, data_path, rth_only=False, timeframe=None,
                  stop_points=25.0, max_hold=480, swing_k=None,
                  tz="America/New_York"):
    """One call: strategy name/path + data file -> Entries (scorable at any rr)."""
    strat = load_strategy(strategy_name)
    if swing_k is not None and hasattr(strat, "swing_k"):
        strat.swing_k = swing_k
    bars = load_ohlc(data_path, rth_only=rth_only, timeframe=timeframe, tz=tz)
    E = backtest_entries(strat, bars, stop_points=stop_points, max_hold=max_hold)
    E.source = os.path.basename(data_path)
    return E