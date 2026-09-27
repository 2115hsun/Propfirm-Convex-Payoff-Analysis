from math import floor
import numpy as np
import pickle

VERBOSE = False
# Setting up Grid for Prelock Value Function =================================================================================

DD_limit = 1000
Lock_peak = 1100
Dx = 50

m_grid = np.arange(0, Lock_peak, Dx)  

y_grid = np.arange(0, DD_limit, Dx)

V_pre = np.zeros((len(m_grid), len(y_grid)))

print("m_grid:", len(m_grid), "values, from", m_grid[0], "to", m_grid[-1])
print("y_grid:", len(y_grid), "values, from", y_grid[0], "to", y_grid[-1])
print("V_pre shape:", V_pre.shape)

# Transition Probabilities and Value Function =================================================================================

def add_to_grid_pre(m, x_prime, prob, dest):
    floor_now = m - DD_limit
    if VERBOSE:
        print(f"  landing x'={x_prime:>6}   peak m={m}   floor={floor_now}", end="   -> ")

    if x_prime <= floor_now:
        if VERBOSE:
            print("BREACH (at or below floor)")
        dest["breach"] = dest.get("breach", 0.0) + prob

    elif x_prime >= Lock_peak:
        if VERBOSE:
            print(f"LOCK (hand off to post-lock at x'={x_prime}, "
                  f"cushion above $25,100 floor = {x_prime - 100})")
        key = ("lock", x_prime)                       # x' kept in the key
        dest[key] = dest.get(key, 0.0) + prob

    elif x_prime < m:                                 # SAME PEAK -> (m' = m, y' = m - x')
        if VERBOSE:
            print(f"SAME PEAK  (m'={m}, y'={m - x_prime})")
        j      = int(int(m/Dx))                    # row of the current peak (exact)
        y_new  = m - x_prime                         # new drawdown
        k_lo   = floor(y_new/Dx)                  # column at or below y_new
        f      = (y_new - y_grid[k_lo]) / Dx          # fraction toward the next column

        key_lo = (j, k_lo)                            # same row, lower column
        dest[key_lo] = dest.get(key_lo, 0.0) + prob * (1 - f)

        if k_lo + 1 >= len(y_grid):                   # past the last column -> drawdown 1000
            key_hi = "breach"         # what fate is that?
        else:
            key_hi = (j, k_lo + 1)
        dest[key_hi] = dest.get(key_hi, 0.0) + prob * f

    else:                                        # NEW PEAK -> (m' = x', y' = 0)
        if VERBOSE:
            print(f"NEW PEAK   (m'={x_prime}, y'=0)")
        j_lo = floor(x_prime/Dx)             # row at or below x'  (floor, Dx)
        f    = (x_prime - m_grid[j_lo]) / Dx     # fraction of the way to the next row

        key_lo = (j_lo, 0)                        # lower row, drawdown column 0
        dest[key_lo] = dest.get(key_lo, 0.0) + prob * (1 - f)

        if j_lo + 1 >= len(m_grid):               # past the last row -> the lock
            key_hi = ("lock", Lock_peak)
        else:
            key_hi = (j_lo + 1, 0)
        dest[key_hi] = dest.get(key_hi, 0.0) + prob * f

# Load the trade data ===================================================================

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


# Transition function for pre-lock value function ===========================================

def transition_pre(j, k, coin, risk):
    """All landings from pre-lock cell (j, k) for one coin. Returns {destination: prob}."""
    m = m_grid[j]              # the peak of this cell
    y = y_grid[k]              # the drawdown of this cell
    x = m-y               # current equity
    dest = {}
    for prob, R in coin:
        x_prime =  x + R*risk    # where this outcome lands
        add_to_grid_pre(m, x_prime, prob, dest)
    return dest