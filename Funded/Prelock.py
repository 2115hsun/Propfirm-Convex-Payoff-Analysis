"""
Prelock.py — pre-lock DP for the Lucid Pro 25K funded account.

State (m, y) = (peak, drawdown), in dollars above the $25,000 start.
Breach when drawdown reaches DD_limit. Lock when the peak reaches Lock_peak;
the account then hands off to the post-lock model (Postlock.py).
"""
from math import floor
import numpy as np
import Postlock as post          # post-lock model: its solved values, coins, and helpers
import time 


# ---- switches ----------------------------------------------------------------------
VERBOSE     = False    # print every landing inside add_to_grid_pre (small tests only)
RUN_CHECKS  = False    # re-run the all-cells transition check (after changing transition code)
SHOW_TABLES = False    # print the policy table, the start ranking, and the gap map
RUN_MAIN = True 

# ---- rules and grid ----------------------------------------------------------------
DD_limit  = 1000       # trailing drawdown that breaches ($)
Lock_peak = 1100       # peak (above start) that locks the floor at $25,100
Dx        = 50         # grid step ($)

m_grid = np.arange(0, Lock_peak, Dx)     # peaks:     0 ... 1050   (22 rows)
y_grid = np.arange(0, DD_limit, Dx)      # drawdowns: 0 ... 950    (20 columns)

# ---- data, shared with post-lock so both stages use the same trades ----------------
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

def v_post(x, V_post):
    """Post-lock value at equity x (dollars above the $25,000 start)."""
    if x >= post.Target:
        return 1.0                                        # at or past $27,100: payout
    if x <= post.Floor:
        return 0.0                                        # at or below $25,100: dead

    lo   = post.cell_index(x, post.Floor, post.Dx)
    x_lo = post.Floor + (lo + 1) * post.Dx
    f    = (x - x_lo) / post.Dx

    v_lo = 0.0 if lo < 0 else V_post[lo]
    v_hi = 1.0 if lo + 1 >= len(post.grid) else V_post[lo + 1]
    return (1 - f) * v_lo + f * v_hi


# ====================================================================================
# Solver and evaluation
# ====================================================================================

def build_transitions(coins, risk):
    """Compute every (cell, RR) transition once. Returns {(j, k, rho): dest}."""
    trans = {}
    for j in range(len(m_grid)):
        for k in range(len(y_grid)):
            for rho in coins:
                trans[(j, k, rho)] = transition_pre(j, k, coins[rho], risk)
    return trans

def backup_cached(dest, V, lock_one=False):
    """Expected value of one stored transition, using the estimates in V."""
    expected = 0.0
    for d, prob in dest.items():
        if d == "breach":
            v = 0.0
        elif d[0] == "lock":
            v = 1.0 if lock_one else v_post(d[1])
        else:
            v = V[d]
        expected += prob * v
    return expected

n_states = len(m_grid) * len(y_grid)    # 440 pre-lock cells
ny = len(y_grid)                         # 20 drawdown columns per peak row


def cell_number(j, k):
    return j * ny + k

def build_matrices(coins, risk, V_post, lock_one=False):
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
                        r[s] += prob * (1.0 if lock_one else v_post(d[1], V_post))
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

def solve_pre(coins, risk=risk, lock_one=False, tol=1e-10, max_sweeps=10000):
    """Value iteration with a max over RR. Returns (V, policy)."""
    V = np.zeros((len(m_grid), len(y_grid)))
    policy = np.empty(V.shape, dtype=object)
    trans = build_transitions(coins, risk)

    for sweep in range(max_sweeps):
        delta = 0.0
        newV = V.copy()
        for j in range(len(m_grid)):
            for k in range(len(y_grid)):
                q_by_rr = [(bellman_backup_pre(j, k, coins[rho], V, risk, lock_one), rho) for rho in coins]
                best_q, best_rho = max(q_by_rr, key=lambda t: t[0])
                newV[j, k]   = best_q
                policy[j, k] = best_rho
                delta = max(delta, abs(newV[j, k] - V[j, k]))
        V = newV
        if sweep % 25 == 0:
            print(f"  sweep {sweep}: delta = {delta:.2e}")
        if delta < tol:
            print(f"converged after {sweep + 1} sweeps")
            break
    return V, policy


def evaluate_policy_pre(policy, coins, risk=risk, lock_one=False, tol=1e-10, max_sweeps=10000):
    """Value of FOLLOWING a given policy table (one RR per state), scored on `coins`."""
    V = np.zeros((len(m_grid), len(y_grid)))
    trans = build_transitions(coins, risk)
    
    for sweep in range(max_sweeps):
        delta = 0.0
        newV = V.copy()
        for j in range(len(m_grid)):
            for k in range(len(y_grid)):
                newV[j, k] = bellman_backup_pre(j, k, coins[policy[j, k]], V, risk, lock_one)
                delta = max(delta, abs(newV[j, k] - V[j, k]))
        V = newV
        if delta < tol:
            break
    return V


def fixed_policy(rho):
    """A policy table that uses the same RR in every state."""
    return np.full((len(m_grid), len(y_grid)), rho, dtype=object)


def show_row(label, V, V_ref=None):
    """One aligned result line: label, value at the start, and (optionally) the gap to V_ref."""
    line = f"  {label:<30} V(0,0) = {V[0, 0]:.4f}"
    if V_ref is not None:
        line += f"   gap to optimal: {(V_ref[0, 0] - V[0, 0]) * 100:5.2f} points"
    print(line)


def bellman_backup_pre(j, k, coin, V, risk=risk, lock_one=False):
    """Expected value of one trade from cell (j, k).
    lock_one=True values every lock at 1 (P(lock)) instead of V_post (P(payout))."""
    expected = 0.0
    for d, prob in transition_pre(j, k, coin, risk).items():     # which risk?
        if d == "breach":
            v = 0.0
        elif d[0] == "lock":
            v = 1 if lock_one else v_post(d[1])
        else:
            v = V[d]
        expected += prob * v
    return expected

# ====================================================================================
# Run
# ====================================================================================


print(f"\npre-lock grid: {len(m_grid)} peak rows x {len(y_grid)} drawdown columns "
      f"= {len(m_grid) * len(y_grid)} states")

if RUN_CHECKS:
    for j in range(len(m_grid)):
        for k in range(len(y_grid)):
            for rho in Coins:
                d = transition_pre(j, k, Coins[rho], risk)
                assert abs(sum(d.values()) - 1.0) < 1e-9, f"mass != 1 at ({j},{k}), RR {rho}"
                assert min(d.values()) >= 0, f"negative prob at ({j},{k}), RR {rho}: {d}"
    print("transition_pre check passed: all cells, all RRs")

# ---- 1. solve on all 884 trades -----------------------------------------------------
V_post_full, _ = post.solve_post(Coins, risk)       # post-lock at the same risk, all 884 trades

if RUN_MAIN:
    # ---- import time ----------------------------------------------------------------------
    t0 = time.time()                                    # NEW: start the clock
    V_pre, policy_pre = solve_pre_fast(Coins, V_post_full)
    print(f"solve_pre_fast took {time.time() - t0:.1f}s")    # NEW: stop the clock and print
    print(f"\nP(first payout | fresh funded account) = {V_pre[0, 0]:.4f}   best RR at start: {policy_pre[0, 0]}")

    if SHOW_TABLES:
        print("\nOptimal RR by state   (rows: peak m, columns: drawdown y)")
        print("  m\\y  " + " ".join(f"{int(y):>4}" for y in y_grid[::2]))
        for j in range(len(m_grid) - 1, -1, -1):
            print(f"{int(m_grid[j]):>5}  " + " ".join(f"{policy_pre[j, k]:>4}" for k in range(0, len(y_grid), 2)))

        qs = sorted([(bellman_backup_pre(0, 0, Coins[rho], V_pre), rho) for rho in Coins], reverse=True)
        print("\nAt the start (0,0), all RRs ranked:")
        for v, rho in qs:
            print(f"  RR {rho:<4}  V = {v:.4f}")

        print("\nGap between best and 2nd-best RR, x1000  (rows: peak m, columns: drawdown y)")
        print("  m\\y  " + " ".join(f"{int(y):>4}" for y in y_grid[::2]))
        for j in range(len(m_grid) - 1, -1, -1):
            gaps = []
            for k in range(0, len(y_grid), 2):
                q = sorted([bellman_backup_pre(j, k, Coins[rho], V_pre) for rho in Coins], reverse=True)
                gaps.append(f"{(q[0] - q[1]) * 1000:>4.1f}")
            print(f"{int(m_grid[j]):>5}  " + " ".join(gaps))

    # ---- 2. optimal policy vs simple rules, all scored the same way ---------------------
    V_opt = evaluate_policy_pre_fast(policy_pre, Coins, V_post_full)

    print("\nOptimal policy vs simple rules (all 884 trades):")
    show_row("optimal policy", V_opt, V_opt)
    for rho in [1.0, 1.5, 3.0, 5.0]:
        show_row(f"always RR {rho}", evaluate_policy_pre_fast(fixed_policy(rho), Coins, V_post_full), V_opt)

    # ---- 3. are the edges stable over time? ----------------------------------------------
    half = len(R_ALL[1.0]) // 2
    print(f"\nEdge by RR, first half vs second half  ({half} trades each)")
    for rho in sorted(R_ALL):
        print(f"  RR {rho:<4}  first {R_ALL[rho][:half].mean():+.3f}R   second {R_ALL[rho][half:].mean():+.3f}R")

    # ---- 4. out-of-sample: build on the first half, score on the second -----------------
    Coins_A = {rho: collapse_to_coin(R_ALL[rho][:half]) for rho in R_ALL}     # earlier trades
    Coins_B = {rho: collapse_to_coin(R_ALL[rho][half:]) for rho in R_ALL}     # later trades

    V_post_A, _ = post.solve_post(Coins_A, risk)    # post-lock from the first half only
    V_post_B, _ = post.solve_post(Coins_B, risk)    # post-lock from the second half only

    V_A, policy_A = solve_pre_fast(Coins_A, V_post_A)                                         # built on A only

    print("\nOut-of-sample test (policy built on the FIRST half):")
    show_row("built policy, scored on A", V_A)                                 # in-sample
    show_row("built policy, scored on B", evaluate_policy_pre_fast(policy_A, Coins_B, V_post_B))   # out-of-sample
    for rho in [1.0, 5.0]:
        show_row(f"always RR {rho}, scored on B", evaluate_policy_pre_fast(fixed_policy(rho), Coins_B, V_post_B))

    # ---- 5. swap: build on the SECOND half, score on the first ---------------------------
    V_B, policy_B = solve_pre_fast(Coins_B, V_post_B)                                         # built on B only

    print("\nSwap test (policy built on the SECOND half):")
    show_row("built policy, scored on B", V_B)                                 # in-sample
    show_row("built policy, scored on A", evaluate_policy_pre_fast(policy_B, Coins_A, V_post_A))   # out-of-sample
    for rho in [1.0, 5.0]:
        show_row(f"always RR {rho}, scored on A", evaluate_policy_pre_fast(fixed_policy(rho), Coins_A, V_post_A))

    # ---- 6. fair-coin benchmark: P(lock) must equal (20/21)^22 --------------------------
    fair = {1.0: [(0.5, 1.0), (0.5, -1.0)]}
    V_fair, _ = solve_pre_fast(fair, V_post_full, risk=50, lock_one=True, tol=1e-9)
    exact = (20 / 21) ** 22
    print(f"\nfair-coin P(lock): solver {V_fair[0, 0]:.6f}   exact {exact:.6f}")

    # ---- 7. P(lock) under the payout-optimal policy --------------------------------------
    V_lock = evaluate_policy_pre_fast(policy_pre, Coins, V_post_full, lock_one=True)    # follow the SAME policy,
                                                                            # but count any lock as 1
    p_lock = V_lock[0, 0]
    print(f"\nP(lock)                  = {p_lock:.4f}")
    print(f"P(first payout)          = {V_pre[0, 0]:.4f}")
    print(f"P(payout | locked)       = {V_pre[0, 0] / p_lock:.4f}")
    print(f"(compare: post-lock at the lock entry = {V_post_full[post.cell_index(post.Entry, post.Floor, post.Dx)]:.4f})")

# By using evaluate_policy_pre_fast and solve_pre_fast, the program stopped redoing work whose answer never changes: where each trade outcome lands depends only on the trades and the risk, 
# not on the current value estiamtes, so working it out and reusing it (caching) removes the same routing from every one of 167 sweeps. Second, the arithmetic is hadned over in bulk as plain Python does
# each tiny multiply with a lot of per-step ooverhead, while numpy's P @ V does all 440 cells' multiply-and-add in one pass, so a sweep dropped from about 0.3 seconds to about a millisecond. 
