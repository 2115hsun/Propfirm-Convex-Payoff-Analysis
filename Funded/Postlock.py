from math import floor

import numpy as np
import pickle

Floor = 0.0
Target = 2000
Dx = 50

grid = np.arange(Floor+Dx, Target, Dx)  # [50, 100, ..., 1950]

V = np.zeros(len(grid))
V_floor = 0.0
V_target = 1.0

with open("ledgers.pkl", "rb") as f:
    RETURN_ARRAYS = pickle.load(f)

risk = 500

def collapse_to_coin(R_array):
    values, counts = np.unique(R_array, return_counts=True)
    return list (zip(counts / counts.sum(), values))  # [(prob, R), ...]

Coins = {rho: collapse_to_coin(R_array) for rho, R_array in RETURN_ARRAYS.items()}

for rho in sorted(Coins):
    edge = sum(p * r for r, p in Coins[rho])
    assert abs(sum(p for p, r in Coins[rho]) - 1.0) < 1e-9 

#====================================================================================

def cell_index(x_prime, Floor, Dx):
    return int((x_prime - Floor) / Dx - 1)

def add_to_grid(x_prime, prob, dest):
    """Add a transition probability to the appropriate grid states."""
    if x_prime <= Floor:
        dest["floor"] = dest.get("floor", 0.0) + prob
    elif x_prime >= Target:
        dest["target"] = dest.get("target", 0.0) + prob
    else:
        # Case 3: INTERIOR — interpolate x_prime across two grid cells.
        lo = cell_index(x_prime, Floor, Dx)     # index of cell just below
        f = (x_prime - grid[lo]) / Dx            # fraction of the way up the gap
        dest[lo] = dest.get(lo, 0.0) + prob * (1 - f)
        dest[lo + 1] = dest.get(lo + 1, 0.0) + prob * f

def transition (x, coin, Floor, Target, Dx, risk):
    dest = {}
    for proob, R in coin:
        x_prime = x + R * risk
        add_to_grid(x_prime, proob, dest)
    return dest

d = transition(1000, Coins[1.5], Floor, Target, Dx, risk)
print("destinations:", d)
print("total prob:", sum(d.values()), "(must be 1.0)")

#====================================================================================

if d == "floor": V_floor
elif d == "target": V_target
else: V[d]

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