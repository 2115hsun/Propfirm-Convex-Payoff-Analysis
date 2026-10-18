"""Export_ledger.py: rebuild the validated eval setup and save per-RR ledgers."""
import pickle
import numpy as np
from Backtest import load_ohlc, backtest_entries, ledger_for_rr, measured_winrate

DATA_PATH = "data/NQ.ohlcv-1m (1).parquet"

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

STOP, MAXHOLD = 25, 60          # STOP defines 1R (the validated value)

def mae_by_rr(E, bars, rrs, max_hold):
    """Worst dip (R, capped at 1) for each trade, measured only until that RR's exit:
    the target bar if the target is hit before the stop, else the stop bar, else the time limit."""
    high, low = bars["high"].to_numpy(), bars["low"].to_numpy()
    pos = bars.index.get_indexer(E.entry_time)              # entry bar of each trade
    sd = E.stop_points
    out = {rr: np.empty(len(pos)) for rr in rrs}
    for t, e in enumerate(pos):
        d, entry = E.direction[t], E.entry_price[t]
        j0, j1 = e + 1, min(e + 1 + max_hold, len(bars))   # same window as backtest_entries
        h, lo = high[j0:j1], low[j0:j1]
        fav = (h - entry) / sd if d == 1 else (entry - lo) / sd
        adv = (entry - lo) / sd if d == 1 else (h - entry) / sd
        hits = np.flatnonzero(adv >= 1.0)
        sb = int(hits[0]) if hits.size else len(adv)        # stop bar, or "never stopped"
        for rr in rrs:
            tgt = np.flatnonzero(fav[:sb] >= rr)            # target touched before the stop bar
            end = int(tgt[0]) if tgt.size else min(sb, len(adv) - 1)   # exit bar for this RR
            out[rr][t] = min(float(adv[:end + 1].max()), 1.0) if adv.size else 0.0
    return out

# build E the same way as the validated run
strat  = __import__("strategies.dual_kernel_v2", fromlist=["get_strategy"]).get_strategy(**FIXED_DK)
bars   = load_ohlc(DATA_PATH, rth_only=False, timeframe="1min", tz="America/New_York")
window = bars.loc["2026-01-01":]
E = backtest_entries(strat, window, stop_points=STOP, max_hold=MAXHOLD, detail=True, one_position=True)

# the RR sweep: score every RR off the same E (same entries, same stop)
RRS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
RETURN_ARRAYS = {rho: ledger_for_rr(E, rho)[0] for rho in RRS}

# check that every edge is positive before saving
print(f"E: {len(E.day_ids)} trades | stop={STOP} max_hold={MAXHOLD}\n")
all_pos = True
for rho in RRS:
    edge = RETURN_ARRAYS[rho].mean()
    if edge <= 0: all_pos = False
    print(f"  RR={rho:<4}: WR {measured_winrate(E,rho):5.1%} | edge {edge:+.4f}R  {'OK' if edge>0 else '*** NEG ***'}")

# MAE per RR, checked before anything is saved
MAE_BY_RR = mae_by_rr(E, window, RRS, MAXHOLD)
old = np.minimum(E.mae_before, 1.0)
for rr in RRS:
    win = E.mfe_before >= rr
    assert np.all(MAE_BY_RR[rr] <= old + 1e-9),             f"RR {rr}: MAE larger than full-window MAE"
    assert np.allclose(MAE_BY_RR[rr][~win], old[~win]),     f"RR {rr}: non-winners changed"
for a, b in zip(RRS, RRS[1:]):
    assert np.all(MAE_BY_RR[a] <= MAE_BY_RR[b] + 1e-9),     f"MAE at RR {a} exceeds RR {b}"
for rr in RRS:
    print(f"RR {rr:<4}: mean MAE {MAE_BY_RR[rr].mean():.3f}R   (full window {old.mean():.3f}R)")

# save all four files together, only if everything checks out
if all_pos:
    OUT = r"..\Funded\data"
    for name, obj in [("ledgers.pkl", RETURN_ARRAYS), ("mae.pkl", MAE_BY_RR),
                      ("day_ids.pkl", np.asarray(E.day_ids)), ("entry_times.pkl", list(E.entry_time))]:
        with open(rf"{OUT}\{name}", "wb") as f:
            pickle.dump(obj, f)
    print(f"\nSaved ledgers, MAE, day ids and entry times: {len(E.day_ids)} trades over "
          f"{len(set(E.day_ids))} trading days (stop={STOP}, 2026 window, one_position=True)")
else:
    print("\nNothing saved: edges negative, E doesn't match the validated setup.")