# ==========================================================================================
# Eval Lifecycle Calc
# ==========================================================================================
from Config import PLAN

Eval_fee = PLAN.eval_fee
Activation_fee = PLAN.activation_fee
Payout = PLAN.payout 
Split = PLAN.split
Max_payouts = PLAN.max_payouts 
# Rough estimates from the 5-slot rotation in analysis_notebook_1 (updated 09.15.26),
# "2) Full Scope Analysis": 2026 trades, 25pt stop, RR 1.5, 2% risk, MAE breach on.
# Lucid 25K Flex, LucidPro 50K and Topstep 50K eval rules all give 85 passed / 162 blown
# (34%), mean 3.4 days to pass and 2.0 to fail. Reused for every plan.
Eval_pass_rate = 0.35
Days_pass = 4
Days_fail = 2

def eval_stage(fee, pass_rate, days_pass, days_fail, activation=Activation_fee):
    "Expected cost and days to get ONE funded account"
    attempts = 1 / pass_rate
    fails = attempts - 1
    days = fails * days_fail + days_pass
    cost = attempts * fee + activation

    return cost, days 

# ==========================================================================================
# Funded Lifecycle Calc
# ==========================================================================================

Payout = PLAN.payout 
Trades_per_day = 1

def funded_stage(q, p, trades_first, trades_cycle, payout, split, trades_per_day, max_payouts=None):
    """Expected payouts, cash, and days for one funded account.
    max_payouts=None: no limit. Otherwise the account leaves after that many payouts."""
    if max_payouts is None:
        payouts = q / (1 - p)                                    # unlimited geometric series
        cycles  = payouts                                        # every payout starts a new cycle
    else:
        payouts = q * (1 - p ** max_payouts) / (1 - p)           # series stops after max_payouts terms
        cycles  = q * (1 - p ** (max_payouts - 1)) / (1 - p)     # no new cycle after the last payout
    cash   = payouts * split * payout
    trades = trades_first + cycles * trades_cycle
    days   = trades / trades_per_day
    return payouts, cash, days

# ==========================================================================================
# Cycle Economics 
# ==========================================================================================

N_slots = 5

def cycle_economics(eval_cost, eval_days, cash, funded_days, n_slots):
    net = cash - eval_cost
    days = eval_days + funded_days
    per_day = net / days
    fleet = per_day * n_slots
    roi = net / eval_cost 
    return net, days, per_day, fleet, roi 


if __name__ == "__main__":
    cost, days = eval_stage(Eval_fee, Eval_pass_rate, Days_pass, Days_fail)
    print(f"per funded account: ${cost:.0f} in evals over {days:.1f} trading days")