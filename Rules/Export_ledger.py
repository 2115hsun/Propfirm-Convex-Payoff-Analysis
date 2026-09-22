"""export_ledgers.py — replicate the validated eval setup, save per-RR ledgers."""
import pickle
import numpy as np
from Backtest import load_ohlc, backtest_entries, ledger_for_rr, measured_winrate

DATA_PATH = "data/NQ.ohlcv-1m (1).parquet"   # <-- set to your actual DATA_PATH

FIXED_DK = dict(
    hF=4.0, x0F=1, rF=8.0,
    hS=16.0, x0S=3, rS=8.0,
    atr_len=14,
    slope_len_f=1, slope_min_f=0.0,
    slope_len_s=3, slope_min_s=0.0,
    min_body_pts=0,
    use_wick=True,
    scan_at=True, scan_p1=True, scan_p2=True,
    use_ext=False, ext_bars=3, ext_pts=15.0,
    use_er=False, er_len=10, er_min=0.3,
    use_cq=False, open_min=0.10,
    min_gap_bars=2,
    sess_start="09:30", sess_end="11:15",
    allow_long=True, allow_short=True, src="close",
)

STOP, MAXHOLD = 25, 60          # <-- the stop that DEFINES 1R (your validated value)

# build E exactly as your validated block does
strat  = __import__("strategies.dual_kernel_v2", fromlist=["get_strategy"]).get_strategy(**FIXED_DK)
bars   = load_ohlc(DATA_PATH, rth_only=False, timeframe="1min", tz="America/New_York")
window = bars.loc["2026-01-01":]
E = backtest_entries(strat, window, stop_points=STOP, max_hold=MAXHOLD, detail=True, one_position=True)

# score every RR off the SAME E (same entries, same stop) — this IS the RR sweep
RRS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
RETURN_ARRAYS = {rho: ledger_for_rr(E, rho)[0] for rho in RRS}

# verify edges are POSITIVE and match your table before saving
print(f"E: {len(E.day_ids)} trades | stop={STOP} max_hold={MAXHOLD}\n")
all_pos = True
for rho in RRS:
    edge = RETURN_ARRAYS[rho].mean()
    if edge <= 0: all_pos = False
    print(f"  RR={rho:<4}: WR {measured_winrate(E,rho):5.1%} | edge {edge:+.4f}R  {'OK' if edge>0 else '*** NEG ***'}")

if all_pos:
    with open(r"..\Funded\ledgers.pkl", "wb") as f:
        pickle.dump(RETURN_ARRAYS, f)
    print("\nSaved ledgers.pkl (stop=25, 2026 window, one_position=True)")
else:
    print("\nNOT saved — edges negative, E doesn't match validated setup.")