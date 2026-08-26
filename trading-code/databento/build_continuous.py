"""
Build continuous OHLCV CSVs from DataBento .csv.zst files, and/or merge
DataBento .dbn / .dbn.zst batch downloads straight into project parquet files.

Reads every .csv.zst file in this folder, deduplicates roll-overlap bars,
resamples to RESAMPLE_PERIOD, and writes to csv_output/.

To change timeframe: edit RESAMPLE_PERIOD below — that's the only line you need.
Valid values: '1min', '3min', '5min', '15min', '30min', '1h', '4h', '1D'

DataBento roll handling (applies to both .csv.zst and .dbn inputs):
  - Spread rows (symbol contains '-', e.g. NQM1-NQU1) carry a price diff,
    not a real bar — dropped.
  - When two contracts trade the same minute during a roll, keep the row
    with the highest volume (front month).

.csv.zst output format: datetime, open, high, low, close, volume — a
leftover from an earlier, unrelated project (Colab fine-tuning scripts);
kept working in case it's ever needed again, but NOT what this repo uses.

.dbn / .dbn.zst handling (the pipeline this repo actually uses):
  Batch downloads requested with schema=ohlcv-1m, stype_in=parent (e.g.
  symbols=['NQ.FUT']) decode to a DataFrame with a resolved `symbol` column
  (NQM6, NQU6, NQM6-NQU6 spread rows, etc.) — no filename ticker prefix like
  the .csv.zst files have. Any such file found in this folder (searched
  recursively — DataBento batch downloads land in a job subfolder) is
  decoded, cleaned the same way as the CSVs, split by root ticker
  (NQM6 -> NQ), and upserted directly into Rules/data/<TICKER>.ohlcv-1m.parquet:
  existing rows + new rows are concatenated, duplicate timestamps keep the
  new value (so re-running with a corrected download fixes past bars too),
  and the result is sorted and written back.

Usage:
    pip install zstandard databento     # one-time (also in requirements.txt)
    Drop the new DataBento download (a .dbn/.dbn.zst file, in its own
    subfolder is fine) anywhere under this databento/ directory, then:
    python databento/build_continuous.py
"""

import sys
from pathlib import Path

import pandas as pd

# ── Configuration — only edit this section ─────────────────────────────────────

RESAMPLE_PERIOD = '1min'   # ← change this to switch timeframe (.csv.zst pipeline only)

DATABENTO_DIR = Path(__file__).parent                    # drop new downloads anywhere under here
OUTPUT_DIR    = Path(__file__).parent / 'csv_output'      # .csv.zst pipeline output only (see docstring)

# Where the project's per-ticker 1-min parquet files live. .dbn / .dbn.zst
# files get upserted here directly. Relative to this script's location so it
# works on any machine the repo is cloned to.
PARQUET_DIR = Path(__file__).parent.parent / 'Rules' / 'data'

# DataBento uses 'SIL' for Silver; FFM / INSTRUMENT_MAP uses 'SI'.
# Add any other mismatches here. Applied to both filename-derived tickers
# (.csv.zst) and symbol-derived roots (.dbn).
TICKER_REMAP = {
    'SIL': 'SI',
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def ticker_from_path(path: Path) -> str:
    """Extract ticker from 'NQ-glbx-mdp3-...' filename."""
    raw = path.name.split('-')[0].upper()
    return TICKER_REMAP.get(raw, raw)


def ticker_root_from_symbol(symbol: str) -> str:
    """
    Extract the root ticker from a resolved futures symbol, e.g.
    'NQM6' -> 'NQ', 'ESU6' -> 'ES'. Futures symbols end in a 1-letter
    month code + 1-digit year, so strip the last two characters.
    """
    raw = symbol[:-2].upper()
    return TICKER_REMAP.get(raw, raw)


def clean_ohlcv(df: pd.DataFrame, symbol_col: str = 'symbol') -> pd.DataFrame:
    """
    Shared cleaning steps for a raw DataBento OHLCV frame (from either the
    .csv.zst reader or the .dbn decoder): drop spread rows, drop
    non-positive prices, and keep the highest-volume row per timestamp
    (front month) when a roll causes two contracts to share a minute.

    Expects a DatetimeIndex and the given symbol column; returns just
    open/high/low/close/volume.
    """
    spread_mask = df[symbol_col].str.contains('-', na=False)
    if spread_mask.sum():
        print(f'    Dropped {spread_mask.sum():,} spread rows')
    df = df[~spread_mask].copy()

    bad_price = df['close'] <= 0
    if bad_price.sum():
        print(f'    Dropped {bad_price.sum():,} non-positive price rows')
    df = df[~bad_price].copy()

    before = len(df)
    df = (df.sort_values('volume', ascending=False)
            .pipe(lambda d: d[~d.index.duplicated(keep='first')])
            .sort_index())
    dupes_removed = before - len(df)
    if dupes_removed:
        print(f'    Deduplicated {dupes_removed:,} back-month overlap rows')

    return df[['open', 'high', 'low', 'close', 'volume']]


def load_databento(path: Path) -> pd.DataFrame:
    """
    Read a DataBento OHLCV .csv.zst file and return a clean 1-min DataFrame,
    indexed by tz-aware UTC datetime (open/high/low/close/volume only).
    See clean_ohlcv() for the roll/dedup rules applied.
    """
    print(f'  Reading {path.name} ...')
    df = pd.read_csv(path, compression='zstd')
    df['datetime'] = pd.to_datetime(df['ts_event'], utc=True)
    df = df.set_index('datetime')

    df = clean_ohlcv(df)

    print(f'    {len(df):,} 1-min bars  '
          f'| {df.index[0]}  →  {df.index[-1]}')
    return df


def load_databento_dbn(path: Path) -> dict[str, pd.DataFrame]:
    """
    Read a DataBento .dbn / .dbn.zst batch file (e.g. schema=ohlcv-1m,
    stype_in=parent) and return one clean 1-min DataFrame per root ticker
    found in it, e.g. {'NQ': <df>}.

    Unlike the .csv.zst files, a .dbn batch download can carry a resolved
    `symbol` column (NQM6, NQU6, spread rows, ...) with no ticker in the
    filename, so tickers are derived from the symbols themselves and the
    data is split accordingly (see clean_ohlcv() for the roll/dedup rules).
    """
    import databento as db

    print(f'  Reading {path.name} ...')
    df = db.DBNStore.from_file(path).to_df()
    df.index = df.index.tz_convert('UTC')
    df.index.name = 'datetime'

    roots = df.loc[~df['symbol'].str.contains('-', na=False), 'symbol'].map(ticker_root_from_symbol)
    out = {}
    for root in sorted(roots.unique()):
        # Keep spread rows too (needed by clean_ohlcv's dedup logic) whose
        # legs both belong to this root, e.g. 'NQM6-NQU6' for root 'NQ'.
        sub = df[df['symbol'].apply(lambda s: any(ticker_root_from_symbol(leg) == root
                                                    for leg in s.split('-')))]
        cleaned = clean_ohlcv(sub)
        print(f'    {root}: {len(cleaned):,} 1-min bars  '
              f'| {cleaned.index[0]}  →  {cleaned.index[-1]}')
        out[root] = cleaned

    return out


def update_parquet(new_df: pd.DataFrame, parquet_path: Path) -> None:
    """
    Upsert new_df into an existing per-ticker parquet (or create it):
    concatenate, and on any overlapping timestamp keep the NEW row (so
    re-running with a corrected/extended download also patches past bars).
    Result is sorted by timestamp and written back with volume as int64,
    matching the existing file's schema.
    """
    new_df = new_df.astype({'volume': 'int64'})

    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        before = len(existing)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        added = len(combined) - before
        print(f'  {parquet_path.name}: {before:,} existing + upsert  '
              f'→ {len(combined):,} bars ({added:+,})')
    else:
        combined = new_df.sort_index()
        print(f'  {parquet_path.name}: creating new  → {len(combined):,} bars')

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path)
    print(f'    Range: {combined.index[0]}  →  {combined.index[-1]}')


def resample_ohlcv(df: pd.DataFrame, period: str) -> pd.DataFrame:
    resampled = df.resample(period).agg({
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum',
    }).dropna()
    return resampled


# ── Main ───────────────────────────────────────────────────────────────────────

def process_csv_zst_files():
    """CSV pipeline: every .csv.zst in this folder → resampled CSV in csv_output/.
    Leftover from an unrelated earlier project -- NOT what this repo's data
    comes from; skipped entirely if there are no .csv.zst files present."""
    zst_files = sorted(DATABENTO_DIR.glob('*.csv.zst'))
    if not zst_files:
        print('No .csv.zst files found — skipping CSV pipeline.\n')
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'DataBento dir : {DATABENTO_DIR}')
    print(f'Output dir    : {OUTPUT_DIR}')
    print(f'Resample to   : {RESAMPLE_PERIOD}')
    print(f'Files found   : {len(zst_files)}\n')

    for path in zst_files:
        ticker = ticker_from_path(path)
        print(f'{"="*55}')
        print(f'  {ticker}  ({path.name})')
        print(f'{"="*55}')

        try:
            df_1m = load_databento(path)
        except Exception as e:
            print(f'  ERROR loading {path.name}: {e}')
            continue

        df_out = resample_ohlcv(df_1m, RESAMPLE_PERIOD)
        out_path = OUTPUT_DIR / f'{ticker}_{RESAMPLE_PERIOD}.csv'

        df_out.index.name = 'datetime'
        df_out.to_csv(out_path)

        print(f'  Saved: {out_path.name}  ({len(df_out):,} bars)')
        print(f'  Range: {df_out.index[0]}  →  {df_out.index[-1]}')
        print()


def process_dbn_files():
    """
    .dbn pipeline: every .dbn / .dbn.zst batch download under this folder →
    upserted straight into Rules/data/<TICKER>.ohlcv-1m.parquet. This is the
    pipeline this repo's NQ data actually comes through.
    Searched recursively (DataBento batch downloads land in a job subfolder,
    e.g. databento/glbx-mdp3-....dbn) — unlike the flat .csv.zst glob above.
    If both a .dbn and .dbn.zst exist for the same job, the uncompressed
    .dbn is used and the .zst copy is skipped (same data, avoid double work).
    """
    by_stem = {}
    for path in DATABENTO_DIR.rglob('*.dbn.zst'):
        by_stem[path.with_suffix('')] = path   # strip trailing .zst
    for path in DATABENTO_DIR.rglob('*.dbn'):
        by_stem[path] = path   # uncompressed wins if both present
    dbn_files = sorted(by_stem.values())

    if not dbn_files:
        print('No .dbn / .dbn.zst files found — skipping parquet upsert.\n')
        return

    try:
        import databento  # noqa: F401
    except ImportError:
        print('ERROR: databento not installed.  Run:  pip install databento')
        return

    print(f'Parquet dir   : {PARQUET_DIR}')
    print(f'.dbn files    : {len(dbn_files)}\n')

    for path in dbn_files:
        print(f'{"="*55}')
        print(f'  {path.name}')
        print(f'{"="*55}')

        try:
            by_ticker = load_databento_dbn(path)
        except Exception as e:
            print(f'  ERROR loading {path.name}: {e}')
            continue

        for ticker, df in by_ticker.items():
            update_parquet(df, PARQUET_DIR / f'{ticker}.ohlcv-1m.parquet')
        print()


def main():
    if list(DATABENTO_DIR.glob('*.csv.zst')):
        try:
            import zstandard  # noqa: F401
        except ImportError:
            print('ERROR: zstandard not installed.  Run:  pip install zstandard')
            sys.exit(1)

    process_csv_zst_files()
    process_dbn_files()

    print('Done.')


if __name__ == '__main__':
    main()
