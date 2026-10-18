"""run_postlock.py — checks and tables for the post-lock model in Postlock.py."""
import numpy as np
from Config import PLAN
from Postlock import (Coins, risk, RETURN_ARRAYS,              # data
                      grid, Floor, Target, Entry, Dx,          # rules and grid
                      cell_index, transition, bellman_backup,  # functions
                      solve, solve_post, expected_trades_post)


print(f"ledger n = {len(RETURN_ARRAYS[1.0])} trades")
for rho in sorted(Coins):
    edge = sum(p * r for p, r, *_ in Coins[rho])
    assert abs(sum(p for p, r, *_ in Coins[rho]) - 1.0) < 1e-9
    print(f"RR={rho:<4}: edge={edge:+.4f}R")

V_init = np.zeros(len(grid))     # initial guess: all zeros

V_star, policy = solve(Coins, V_init, Floor, Target, Dx, risk)

# show the value function and optimal RR at a few cushion levels
print("\ncell   equity   V*(prob reach target)   optimal RR")
for i in range(0, len(grid), 5):      # every 5th cell
    print(f"  {i:2d}   ${grid[i]:>6.0f} {V_star[i]:.4f} {policy[i]}")

#====================================================================================
# Expected trades from the lock entry
# ===================================================================================
print("cell  equity   bestRR  bestV   2ndRR  2ndV   gap")
for i in range(len(grid)):
    qs = sorted([(bellman_backup(i, Coins[rho], V_star, Floor, Target, Dx, risk), rho)
                 for rho in Coins], reverse=True)
    (v1,rr1),(v2,rr2) = qs[0], qs[1]
    print(f"  {i:2d}  ${grid[i]:>6.0f}   {rr1:<5} {v1:.4f}  {rr2:<5} {v2:.4f}  {v1-v2:.4f}")

d = transition(Entry, Coins[1.5], Floor, Target, Dx, risk)
print("total prob:", sum(d.values()))

i_entry = cell_index(Entry, Floor, Dx)          
print(f"P(reach ${PLAN.start + Target:,.0f} before ${PLAN.start + Floor:,.0f} | "
      f"locked at ${PLAN.start + Entry:,.0f}) = {V_star[i_entry]:.4f}  best RR {policy[i_entry]}")

assert all(p >= 0 for p in d.values()), f"negative probability: {d}"

print(f"\nP(payout | lock entry) at ${risk}: {V_star[i_entry]:.4f} (compare with Baselines.md)")

# ===================================================================================
# Time layer check 
# ===================================================================================
# A fair coin with one-cell steps has an exact answer.
fair = {1.0: [(0.5, 1.0), (0.5, -1.0)]}
fair_policy = {i: 1.0 for i in range(len(grid))}
T_fair = expected_trades_post(fair, fair_policy, risk=Dx)
i = cell_index(Entry, Floor, Dx)
a = round((Entry - Floor) / Dx)          # steps from the lock entry down to the floor
b = round((Target - Entry) / Dx)         # steps from the lock entry up to the payout
print(f"\nfair coin, expected trades from the lock entry: {T_fair[i]:.2f}   (exact {a * b})")

# The real answer: trades from the lock entry to payout-or-death, at $250, optimal policy.
V_star, policy = solve_post(Coins, risk)
T_real = expected_trades_post(Coins, policy, risk)
print(f"expected trades from the lock entry at ${risk}: {T_real[i]:.1f}")