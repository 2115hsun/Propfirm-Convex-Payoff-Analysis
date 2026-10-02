"""
Prelock.py — pre-lock DP for the Lucid Pro 25K funded account.

State (m, y) = (peak, drawdown), in dollars above the $25,000 start.
Breach when drawdown reaches DD_limit. Lock when the peak reaches Lock_peak;
the account then hands off to the post-lock model (Postlock.py).
"""
from math import floor
import numpy as np
import Postlock as post 
from Config import PLAN        


VERBOSE     = False    


DD_limit  = PLAN.dd_limit       # trailing drawdown that breaches ($)
Lock_peak = PLAN.lock_peak       # peak (above start) that locks the floor at $25,100
Dx        = PLAN.dx         # grid step ($)

m_grid = np.arange(0, Lock_peak, Dx)     # peaks:     0 ... 1050   (22 rows)
y_grid = np.arange(0, DD_limit, Dx)      # drawdowns: 0 ... 950    (20 columns)

Coins            = post.Coins
risk             = post.risk
R_ALL            = post.RETURN_ARRAYS
collapse_to_coin = post.collapse_to_coin

# ====================================================================================
# Transition: route one trade's landing to breach / lock / a pre-lock cell
# ====================================================================================

def add_to_grid_pre(m, x_prime, prob, dest):
    """Route ONE landing x_prime, from a state with peak m, adding prob to its destination(s)."""
    floor_now = m - DD_limit
    if VERBOSE:
        print(f"  landing x'={x_prime:>7}   peak m={m}   floor={floor_now}", end="   -> ")

    if x_prime <= floor_now:                              # BREACH
        if VERBOSE: print("BREACH")
        dest["breach"] = dest.get("breach", 0.0) + prob

    elif x_prime >= Lock_peak:                            # LOCK -> hand off at x'
        if VERBOSE: print(f"LOCK at x'={x_prime}")
        key = ("lock", x_prime)
        dest[key] = dest.get(key, 0.0) + prob

    elif x_prime < m:                                     # SAME PEAK -> (m, y' = m - x')
        if VERBOSE: print(f"SAME PEAK  y'={m - x_prime}")
        j     = int(round(m / Dx))                        # row of the current peak
        y_new = m - x_prime
        k_lo  = floor(y_new / Dx)                         # column at or below y_new
        f     = (y_new - y_grid[k_lo]) / Dx               # fraction toward the next column

        dest[(j, k_lo)] = dest.get((j, k_lo), 0.0) + prob * (1 - f)
        key_hi = "breach" if k_lo + 1 >= len(y_grid) else (j, k_lo + 1)   # past last column = breach
        dest[key_hi] = dest.get(key_hi, 0.0) + prob * f

    else:                                                 # NEW PEAK -> (m' = x', y' = 0)
        if VERBOSE: print(f"NEW PEAK   m'={x_prime}")
        j_lo = floor(x_prime / Dx)                        # row at or below x'
        f    = (x_prime - m_grid[j_lo]) / Dx              # fraction toward the next row

        dest[(j_lo, 0)] = dest.get((j_lo, 0), 0.0) + prob * (1 - f)
        key_hi = ("lock", Lock_peak) if j_lo + 1 >= len(m_grid) else (j_lo + 1, 0)   # past last row = lock
        dest[key_hi] = dest.get(key_hi, 0.0) + prob * f


def transition_pre(j, k, coin, risk):
    """All landings from pre-lock cell (j, k) for one coin. Returns {destination: prob}."""
    m = m_grid[j]                  # peak of this cell
    y = y_grid[k]                  # drawdown of this cell
    x = m - y                      # current equity
    dest = {}
    for prob, R in coin:
        add_to_grid_pre(m, x + R * risk, prob, dest)
    return dest


# ====================================================================================
# Handoff: post-lock value at any equity (interpolated between post-lock cells)
# ====================================================================================

def v_post(x, V_post, at_target=1.0):
    """Post-lock value at equity x (dollars above the $25,000 start)."""
    if x >= post.Target:
        return at_target                                      # at or past $27,100: payout
    if x <= post.Floor:
        return 0.0                                        # at or below $25,100: dead

    lo   = post.cell_index(x, post.Floor, post.Dx)
    x_lo = post.Floor + (lo + 1) * post.Dx
    f    = (x - x_lo) / post.Dx

    v_lo = 0.0 if lo < 0 else V_post[lo]
    v_hi = at_target if lo + 1 >= len(post.grid) else V_post[lo + 1]
    return (1 - f) * v_lo + f * v_hi


# ====================================================================================
# Solver and evaluation
# ====================================================================================
                       # 20 drawdown columns per peak row
n_states = len(m_grid) * len(y_grid)
ny = len(y_grid)

def cell_number(j, k):
    return j * ny + k

def build_matrices(coins, risk, V_post, lock_one=False, at_target=1.0):
    """
    Build one transition matrix P and one lock-reward vector r per RR.

    P[s, s2] is the probability that a trade from cell s ends in cell s2.
    r[s] is the value collected right away from cell s by trades that lock.
    A breach is worth 0, so it adds nothing to either.
    With these, the backup for every cell at once is r + P @ V.
    """
    Ps, rs = [], []
    for rho in coins:
        P = np.zeros((n_states, n_states))
        r = np.zeros(n_states)
        for j in range(len(m_grid)):
            for k in range(len(y_grid)):
                s = cell_number(j, k)
                for d, prob in transition_pre(j, k, coins[rho], risk).items():
                    if d == "breach":
                        continue
                    if d[0] == "lock":
                        r[s] += prob * (1.0 if lock_one else v_post(d[1], V_post, at_target=at_target))
                    else:
                        P[s, cell_number(d[0], d[1])] += prob
        Ps.append(P)
        rs.append(r)
    return Ps, rs

def solve_pre_fast(coins, V_post, risk=risk, lock_one=False, tol=1e-10, max_sweeps=100_000):
    """
    Value iteration with a max over RR, using matrices.
    Each sweep computes Q = r + P @ V for every RR and every cell in one step each,
    then keeps the best RR per cell. Returns (V, policy) as 22 x 20 tables.
    """
    Ps, rs = build_matrices(coins, V_post=V_post, risk=risk, lock_one=lock_one)
    rhos = list(coins)
    V = np.zeros(n_states)
    for sweep in range(max_sweeps):
        Q = np.stack([r + P @ V for P, r in zip(Ps, rs)])    # one row per RR, one column per cell
        newV = Q.max(axis=0)                                  # best value in each cell
        delta = np.abs(newV - V).max()
        V = newV
        if delta < tol:
            break
    print(f"converged after {sweep + 1} sweeps")
    best = Q.argmax(axis=0)                                   # which RR won in each cell
    policy = np.array([rhos[i] for i in best], dtype=object).reshape(len(m_grid), ny)
    return V.reshape(len(m_grid), ny), policy


def evaluate_policy_pre_fast(policy, coins, V_post, risk=risk, lock_one=False):
    """
    Exact value of following a fixed policy table, with no sweeps.
    With the RR fixed in every cell, the values satisfy V = r_pi + P_pi @ V,
    which is a linear system: (I - P_pi) V = r_pi.
    """
    Ps, rs = build_matrices(coins, V_post=V_post, risk=risk, lock_one=lock_one)
    rhos = list(coins)
    choice = [rhos.index(rho) for rho in policy.flatten()]           # RR index per cell
    P_pi = np.stack([Ps[a][s] for s, a in enumerate(choice)])        # each cell's row, for its RR
    r_pi = np.array([rs[a][s] for s, a in enumerate(choice)])
    V = np.linalg.solve(np.eye(n_states) - P_pi, r_pi)
    return V.reshape(len(m_grid), ny)

def expected_trades_pre(policy, coins, T_post, risk=risk):
    """
    Expected trades from each pre-lock cell until payout or breach, following `policy`.
    A lock at x' contributes the post-lock trades still ahead, T_post(x').
    """
    Ps, rs = build_matrices(coins, risk=risk, V_post=T_post, at_target=0.0)
    rhos = list(coins)
    choice = [rhos.index(rho) for rho in policy.flatten()]
    P_pi = np.stack([Ps[a][s] for s, a in enumerate(choice)])
    r_pi = np.array([rs[a][s] for s, a in enumerate(choice)])
    T = np.linalg.solve(np.eye(n_states) - P_pi, np.ones(n_states) + r_pi)    # each trade costs 1, plus lock time
    return T.reshape(len(m_grid), ny)


def fixed_policy(rho):
    """A policy table that uses the same RR in every state."""
    return np.full((len(m_grid), len(y_grid)), rho, dtype=object)


def show_row(label, V, V_ref=None):
    """One aligned result line: label, value at the start, and (optionally) the gap to V_ref."""
    line = f"  {label:<30} V(0,0) = {V[0, 0]:.4f}"
    if V_ref is not None:
        line += f"   gap to optimal: {(V_ref[0, 0] - V[0, 0]) * 100:5.2f} points"
    print(line)




