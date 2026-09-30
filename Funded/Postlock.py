from math import floor

import numpy as np
import pickle

Floor = 100
Target = 2100
Entry = 1100
Dx = 50

grid = np.arange(Floor+Dx, Target, Dx)  # [50, 100, ..., 1950]

V_floor = 0.0
V_target = 1.0

with open("ledgers.pkl", "rb") as f:
    RETURN_ARRAYS = pickle.load(f)

risk = 250

def collapse_to_coin(R_array):
    values, counts = np.unique(R_array, return_counts=True)
    return list (zip(counts / counts.sum(), values))  # [(prob, R), ...]

Coins = {rho: collapse_to_coin(R_array) for rho, R_array in RETURN_ARRAYS.items()}

for rho in Coins:
    assert abs(sum(p for p, r in Coins[rho]) - 1.0) < 1e-9, f"coin mass != 1 for RR {rho}"

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

def expected_trades_post(coins, policy, risk, tol=1e-9, max_sweeps=100_000):
    """
    Expected number of trades from each post-lock cell until payout or breach,
    when every cell uses the RR in `policy`. Both finish lines count as 0.
    """
    T = np.zeros(len(grid))
    for sweep in range(max_sweeps):
        newT = np.zeros(len(grid))
        for i in range(len(grid)):
            dest = transition(grid[i], coins[policy[i]], Floor, Target, Dx, risk)
            remaining = sum(prob * T[d] for d, prob in dest.items()
                            if d not in ("floor", "target"))
            newT[i] = 1.0 + remaining
        delta = np.abs(newT - T).max()
        T = newT
        if delta < tol:
            break
    return T
