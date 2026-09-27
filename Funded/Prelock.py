from math import floor
import numpy as np
import pickle

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
    print(f"  landing x'={x_prime:>6}   peak m={m}   floor={floor_now}", end="   -> ")

    if x_prime <= floor_now:
        print("BREACH (at or below floor)")
        dest["breach"] = dest.get("breach", 0.0) + prob

    elif x_prime >= Lock_peak:
        print(f"LOCK (hand off to post-lock at x'={x_prime}, "
              f"cushion above $25,100 floor = {x_prime - 100})")
        key = ("lock", x_prime)                       # x' kept in the key
        dest[key] = dest.get(key, 0.0) + prob

    elif x_prime < m:                                 # SAME PEAK -> (m' = m, y' = m - x')
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





# --- routing test from state m=500, y=200 (equity x=300, floor=-500) ---
for x_prime in [130, 100, -480]:
    dest = {}
    add_to_grid_pre(500, x_prime, 1.0, dest)
    print("      dest:", dest, "  total:", sum(dest.values()))