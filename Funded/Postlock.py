from math import floor

import numpy as np
import pickle

Floor = 100
Target = 2100
Entry = 1100
Dx = 50

grid = np.arange(Floor+Dx, Target, Dx)  # [50, 100, ..., 1950]

V = np.zeros(len(grid))
V_floor = 0.0
V_target = 1.0

with open("ledgers.pkl", "rb") as f:
    RETURN_ARRAYS = pickle.load(f)

risk = 250

def collapse_to_coin(R_array):
    values, counts = np.unique(R_array, return_counts=True)
    return list (zip(counts / counts.sum(), values))  # [(prob, R), ...]

Coins = {rho: collapse_to_coin(R_array) for rho, R_array in RETURN_ARRAYS.items()}

print(f"ledger n = {len(RETURN_ARRAYS[1.0])} trades")
for rho in sorted(Coins):
    edge = sum(p * r for p, r in Coins[rho])
    assert abs(sum(p for p, r in Coins[rho]) - 1.0) < 1e-9
    print(f"RR={rho:<4}: edge={edge:+.4f}R")

#====================================================================================

def cell_index(x_prime, Floor, Dx):
    return floor((x_prime - Floor) / Dx - 1)

def add_to_grid(x_prime, prob, dest, Floor, Target, Dx):
    """Add a transition probability to the appropriate grid states."""
    if x_prime <= Floor:
        dest["floor"] = dest.get("floor", 0.0) + prob
    elif x_prime >= Target:
        dest["target"] = dest.get("target", 0.0) + prob
    else:
        # Case 3: INTERIOR — interpolate x_prime across two grid cells.
        lo = cell_index(x_prime, Floor, Dx)
        x_lo = Floor + (lo + 1) * Dx
        f = (x_prime - x_lo) / Dx

        # lower half of the weight
        if lo < 0:
            dest["floor"] = dest.get("floor", 0.0) + prob * (1 - f)
        else:
            dest[lo] = dest.get(lo, 0.0) + prob * (1 - f)

        # upper half of the weight
        if lo + 1 >= len(grid):
            dest["target"] = dest.get("target", 0.0) + prob * f
        else:
            dest[lo + 1] = dest.get(lo + 1, 0.0) + prob * f

def transition (x, coin, Floor, Target, Dx, risk):
    dest = {}
    for proob, R in coin:
        x_prime = x + R * risk
        add_to_grid(x_prime, proob, dest, Floor, Target, Dx)
    return dest

#====================================================================================

def bellman_backup(x_index, coin, V, Floor, Target, Dx, risk):
    """
    Expected value of taking one trade (with `coin`) from grid cell x_index.
    Returns  sum over destinations of  prob * value(destination).
    """
    x = grid[x_index]                                  # dollar equity of this cell
    dest = transition(x, coin, Floor, Target, Dx, risk)   # {destination: prob}

    expected = 0.0
    for d, prob in dest.items():
        # value of destination d (the three-way lookup):
        if d == "floor":
            v = V_floor
        elif d == "target":
            v = V_target
        else:
            v = V[d]
        expected += prob * v

    return expected

#====================================================================================

def solve(COINS, V_init, Floor, Target, Dx, risk, tol=1e-10, max_sweeps=10000):
    """Value iteration over the equity grid. Returns (V, policy)."""
    V = V_init.copy()                       # start from the initial guess (zeros)
    policy = {i: None for i in range(len(grid))}

    for sweep in range(max_sweeps):
        delta = 0.0
        newV = V.copy()
        for i in range(len(grid)):          # sweep every interior grid cell
            # try every RR, keep the best (max value AND argmax RR)
            q_by_rr = [(bellman_backup(i, COINS[rho], V, Floor, Target, Dx, risk), rho)
                       for rho in COINS]
            best_q, best_rho = max(q_by_rr, key=lambda x: x[0])  # <-- FILL IN: pick the largest
            newV[i]   = best_q                  # <-- FILL IN: optimal value at cell i
            policy[i] = best_rho                  # <-- FILL IN: the RR that achieved it
            delta = max(delta, abs(newV[i] - V[i]))
        V = newV
        if delta < tol:
            print(f"converged after {sweep+1} sweeps")
            break
    return V, policy

#====================================================================================

def solve_post(coins, risk):
    V_init = np.zeros(len(grid))
    return solve(coins, V_init, Floor, Target, Dx, risk)

#====================================================================================

V_init = np.zeros(len(grid))     # initial guess: all zeros

V_star, policy = solve(Coins, V_init, Floor, Target, Dx, risk)

# show the value function and optimal RR at a few cushion levels
print("\ncell   equity   V*(prob reach target)   optimal RR")
for i in range(0, len(grid), 5):      # every 5th cell
    print(f"  {i:2d}   ${grid[i]:>6.0f}        {V_star[i]:.4f}              {policy[i]}")

#====================================================================================

print("cell  equity   bestRR  bestV   2ndRR  2ndV   gap")
for i in range(len(grid)):
    qs = sorted([(bellman_backup(i, Coins[rho], V_star, Floor, Target, Dx, risk), rho)
                 for rho in Coins], reverse=True)
    (v1,rr1),(v2,rr2) = qs[0], qs[1]
    print(f"  {i:2d}  ${grid[i]:>6.0f}   {rr1:<5} {v1:.4f}  {rr2:<5} {v2:.4f}  {v1-v2:.4f}")

#====================================================================================

d = transition(Entry, Coins[1.5], Floor, Target, Dx, risk)
print("total prob:", sum(d.values()))

i_entry = cell_index(Entry, Floor, Dx)          # = 19
print(f"P(reach $27,100 before $25,100 | locked at $26,100) = "
      f"{V_star[i_entry]:.4f}  best RR {policy[i_entry]}")

assert all(p >= 0 for p in d.values()), f"negative probability: {d}"

V_250, _ = solve_post(Coins, 250)
V_150, _ = solve_post(Coins, 150)
i = cell_index(Entry, Floor, Dx)
print(f"\nsolve_post at $250: {V_250[i]:.4f}   (must match 0.6401)")
print(f"solve_post at $150: {V_150[i]:.4f}   (new: post-lock at 3 MNQ)")