"""export_results.py — write the risk sweep and the policy tables to funded_results.json for the dashboard.
Run from the Funded folder:  python export_results.py
"""
import json
import numpy as np
import Postlock as post
import Prelock as pre
import Lifecycle as life
import Sweep
from Config import PLAN
import os

step  = 50 if PLAN.dd_limit <= 2_000 else 150          # keep the sweep to about 20-40 levels
RISKS = list(range(100, int(PLAN.dd_limit) + 1, step)) # up to one loss = the whole drawdown        # $ per R, 2 to 20 MNQ at a 25-point stop

print(f"running the sweep over {len(RISKS)} risk levels...")
sweep = Sweep.run_sweep(RISKS)

print("building the policy tables...")
grids = {}
for r in RISKS:
    V_post, pol_post = post.solve_post(post.Coins, r)
    V_pre, pol_pre = pre.solve_pre_fast(post.Coins, V_post, risk=r)
    grids[r] = dict(
        pre=dict(m=pre.m_grid.tolist(), y=pre.y_grid.tolist(),
                 V=np.round(V_pre, 4).tolist(), policy=pol_pre.astype(float).tolist()),
        post=dict(x=post.grid.tolist(), V=np.round(np.asarray(V_post), 4).tolist(),
                  policy=[float(pol_post[i]) for i in range(len(post.grid))]),
    )

results = dict(
    meta=dict(stop_points=25, trades=len(post.RETURN_ARRAYS[1.0]),
              plan_key=os.environ.get("PLAN", "lucid25flex")),
    eval=dict(fee=life.Eval_fee, pass_rate=life.Eval_pass_rate,
              days_pass=life.Days_pass, days_fail=life.Days_fail),
    funded=dict(payout=life.Payout, split=life.Split, slots=life.N_slots,
                trades_per_day=life.Trades_per_day),
    rows=sweep.to_dict(orient="records"),
    grids=grids,
)
with open("funded_results.json", "w") as f:
    json.dump(results, f)

row = sweep[sweep.risk == 250].iloc[0]
print(f"wrote funded_results.json: {len(RISKS)} risk levels with policy tables")
print(f"check, $250 row: P(pay1) {row.p_pay1:.4f} (expect 0.3733), P(again) {row.p_again:.4f} (expect 0.6401)")