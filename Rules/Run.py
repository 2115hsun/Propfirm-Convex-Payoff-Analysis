"""
Run.py — command-line orchestrator.

    python Run.py --strategy bos --data data/SAMPLE_1min.parquet --firm apex_intraday

Discovers the optimal risk geometry for the chosen strategy + data + firm and
saves the figures. No predefined risk %, R:R, or trades/day.

Firm keys: apex_eod, apex_intraday, lucid, tradeify_select,
           tradeify_growth, mffu_rapid, topstep
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")

from Firms import get_firm
from Backtest import build_entries, ledger_for_rr, measured_winrate
from Strategy import EmpiricalStrategy
from Analysis import measured_wr_curve, plot_risk_curve, risk_rr_heatmap

OUT = os.environ.get("OUT_DIR", "outputs")
os.makedirs(OUT, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="bos")
    ap.add_argument("--data", default="data/SAMPLE_1min.parquet")
    ap.add_argument("--firm", default="apex_intraday")
    ap.add_argument("--rth", action="store_true")
    ap.add_argument("--timeframe", default=None)
    ap.add_argument("--swing_k", type=int, default=2)
    ap.add_argument("--k_stop_atr", type=float, default=1.0)
    args = ap.parse_args()

    firm = get_firm(args.firm)
    E = build_entries(args.strategy, args.data, rth_only=args.rth,
                      timeframe=args.timeframe, swing_k=args.swing_k,
                      k_stop_atr=args.k_stop_atr)
    print(E.summary())

    rr_grid = [0.5, 1.0, 1.5, 2.0, 3.0]
    measured_wr_curve(E, rr_grid, savepath=f"{OUT}/wr_by_rr_{args.strategy}.png")

    # heatmap picks the joint-optimal (R:R, risk%) and reports it
    _, g = risk_rr_heatmap(firm, E, savepath=f"{OUT}/heatmap_{firm.key}_{args.strategy}.png")
    print(f"\nOptimal geometry for {firm.name}: {g['opt_rr']:g}R @ "
          f"{g['opt_risk_pct']:.2f}% risk -> {g['opt_pass_pct']:.0f}% pass "
          f"(~{g['opt_speed']:.0f} trades, WR {g['opt_wr']:.0%})")

    # risk curve at the optimal R:R
    R, days = ledger_for_rr(E, g["opt_rr"])
    strat = EmpiricalStrategy(R, name=f"{args.strategy} {g['opt_rr']:g}R", day_ids=days)
    plot_risk_curve(strat, firm, savepath=f"{OUT}/risk_curve_{firm.key}_{args.strategy}.png")
    print(f"Figures saved to {OUT}/")


if __name__ == "__main__":
    main()