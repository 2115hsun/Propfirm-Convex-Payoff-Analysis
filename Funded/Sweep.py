"""Sweep.py — run the full funded lifecycle at several risk levels and return one table."""
import pandas as pd
import Postlock as post
import Prelock as pre
import Lifecycle as life
from Config import PLAN 

def default_risks(plan=PLAN):
    step = 50 if plan.dd_limit <= 2_000 else 150
    return list(range(100, int(plan.dd_limit) + 1, step))


def run_sweep(risks, coins=None):
    """One row per risk level: probabilities, trades, payouts, cash, days, $/day, ROI."""
    coins = coins or post.Coins
    i_entry = post.cell_index(post.Entry, post.Floor, post.Dx)
    eval_cost, eval_days = life.eval_stage(life.Eval_fee, life.Eval_pass_rate,
                                           life.Days_pass, life.Days_fail,
                                           activation=life.Activation_fee)
    rows = []
    for r in risks:
        V_post_r, pol_post_r = post.solve_post(coins, r)
        T_post_r = post.expected_trades_post(coins, pol_post_r, r)
        V_pre_r, pol_pre_r = pre.solve_pre_fast(coins, V_post_r, risk=r)
        p_lock  = pre.evaluate_policy_pre_fast(pol_pre_r, coins, V_post_r, risk=r, lock_one=True)[0, 0]
        T_first = pre.expected_trades_pre(pol_pre_r, coins, T_post_r, risk=r)[0, 0]
        payouts, cash, fdays = life.funded_stage(
            q=V_pre_r[0, 0], p=V_post_r[i_entry], trades_first=T_first,
            trades_cycle=T_post_r[i_entry], payout=life.Payout, split=life.Split,
            trades_per_day=life.Trades_per_day, max_payouts=life.Max_payouts)
        net, cdays, per_day, fleet, roi = life.cycle_economics(
            eval_cost, eval_days, cash, fdays, life.N_slots)
        rows.append(dict(risk=r, mnq=r // 50, p_lock=p_lock, p_pay1=V_pre_r[0, 0],
                         p_again=V_post_r[i_entry], trades_first=T_first,
                         trades_cycle=T_post_r[i_entry], payouts=payouts, cash=cash,
                         funded_days=fdays, per_day=per_day, fleet_per_day=fleet, roi=roi))
    return pd.DataFrame(rows)