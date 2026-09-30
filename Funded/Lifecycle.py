# ==========================================================================================
# Eval Lifecycle Calc
# ==========================================================================================

Eval_fee = 60
Eval_pass_rate = 0.35
Days_pass = 4
Days_fail = 2

def eval_stage(fee, pass_rate, days_pass, days_fail):
    "Expected cost and days to get ONE funded account"
    attempts = 1 / pass_rate
    fails = attempts - 1
    days = fails * days_fail + days_pass
    cost = attempts * fee

    return cost, days 

if __name__ == "__main__":
    cost, days = eval_stage(Eval_fee, Eval_pass_rate, Days_pass, Days_fail)
    print(f"per funded account: ${cost:.0f} in evals over {days:.1f} trading days")

# ==========================================================================================
# Funded Lifecycle Calc
# ==========================================================================================

Payout = 1000
Split = 0.9
Trades_per_day = 1

def funded_stage(q, p, trades_first, trades_cycle, payout, split, trades_per_day):
    payouts = q / (1 - p)
    cash = payouts * split * payout
    trades = trades_first + payouts * trades_cycle
    days = trades / trades_per_day
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

