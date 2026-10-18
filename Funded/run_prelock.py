"""run_prelock.py — solve, comparisons, and out-of-sample tests for the pre-lock model."""
import time
import numpy as np
import Sweep 
import Postlock as post
from Config import PLAN
from Prelock import (m_grid, y_grid, Coins, risk, R_ALL, collapse_to_coin,   # grid and data
                     transition_pre, solve_pre_fast, evaluate_policy_pre_fast,  # model
                     fixed_policy, show_row, expected_trades_pre, q_values)    

print(f"{PLAN.name}, ${risk} per R, MAE {'on' if post.USE_MAE else 'off'}, {len(R_ALL[1.0])} trades")
# =====================================================================================
# switches 
# =====================================================================================
RUN_CHECKS  = False    # re-run the all-cells transition check
SHOW_TABLES = True    # print the optimal-RR table
RUN_MAIN    = True     # run the solve, comparisons, and out-of-sample tests

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

# =====================================================================================
# 1. solve on all 884 trades
# =====================================================================================
V_post_full, _ = post.solve_post(Coins, risk)       # post-lock at the same risk, all 884 trades

if RUN_MAIN:
    # ---- import time ----------------------------------------------------------------------
    t0 = time.time()                                   
    V_pre, policy_pre = solve_pre_fast(Coins, V_post_full)
    print(f"solve_pre_fast took {time.time() - t0:.1f}s")    
    print(f"\nP(first payout | fresh funded account) = {V_pre[0, 0]:.4f}   best RR at start: {policy_pre[0, 0]}")

    if SHOW_TABLES:
        print("\nOptimal RR by state   (rows: peak m, columns: drawdown y)")
        print("  m\\y  " + " ".join(f"{int(y):>4}" for y in y_grid[::2]))
        for j in range(len(m_grid) - 1, -1, -1):
            print(f"{int(m_grid[j]):>5}  " + " ".join(f"{policy_pre[j, k]:>4}" for k in range(0, len(y_grid), 2)))


    Q = q_values(Coins, V_post_full, V_pre)          # [RR, peak, drawdown]
    print("\nAt the start (0, 0), every RR ranked:")
    for a in np.argsort(-Q[:, 0, 0]):
        print(f"  RR {list(Coins)[a]:<5} V = {Q[a, 0, 0]:.4f}")

    top2 = -np.sort(-Q, axis=0)[:2]
    gap = (top2[0] - top2[1]) * 1000
    print("\nGap between best and second-best RR, x1000")
    for j in range(len(m_grid) - 1, -1, -1):
        print(f"{int(m_grid[j]):>5}  " + " ".join(f"{gap[j, k]:>5.1f}" for k in range(0, len(y_grid), 2)))

    # =====================================================================================
    # 2. optimal policy vs simple rules, all scored the same way 
    # =====================================================================================
    V_opt = evaluate_policy_pre_fast(policy_pre, Coins, V_post_full)

    print("\nOptimal policy vs simple rules (all 884 trades):")
    show_row("optimal policy", V_opt, V_opt)
    for rho in [1.0, 1.5, 3.0, 5.0]:
        show_row(f"always RR {rho}", evaluate_policy_pre_fast(fixed_policy(rho), Coins, V_post_full), V_opt)

    # =====================================================================================
    # 3. are the edges stable over time? -
    # =====================================================================================
    half = len(R_ALL[1.0]) // 2
    print(f"\nEdge by RR, first half vs second half  ({half} trades each)")
    for rho in sorted(R_ALL):
        print(f"  RR {rho:<4}  first {R_ALL[rho][:half].mean():+.3f}R   second {R_ALL[rho][half:].mean():+.3f}R")

    # =====================================================================================
    # 4. out-of-sample: build on the first half, score on the second 
    # =====================================================================================
    mae = post.MAE if post.USE_MAE else None
    Coins_A = {rho: collapse_to_coin(R_ALL[rho][:half], None if mae is None else mae[rho][:half]) for rho in R_ALL}
    Coins_B = {rho: collapse_to_coin(R_ALL[rho][half:], None if mae is None else mae[rho][half:]) for rho in R_ALL}

    V_post_A, _ = post.solve_post(Coins_A, risk)    # post-lock from the first half only
    V_post_B, _ = post.solve_post(Coins_B, risk)    # post-lock from the second half only

    V_A, policy_A = solve_pre_fast(Coins_A, V_post_A)                                         

    print("\nOut-of-sample test (policy built on the FIRST half):")
    show_row("built policy, scored on A", V_A)                                 
    show_row("built policy, scored on B", evaluate_policy_pre_fast(policy_A, Coins_B, V_post_B))  
    for rho in [1.0, 5.0]:
        show_row(f"always RR {rho}, scored on B", evaluate_policy_pre_fast(fixed_policy(rho), Coins_B, V_post_B))

    # =====================================================================================
    # 5. swap: build on the SECOND half, score on the first 
    # =====================================================================================

    V_B, policy_B = solve_pre_fast(Coins_B, V_post_B)                                         

    print("\nSwap test (policy built on the SECOND half):")
    show_row("built policy, scored on B", V_B)                                 
    show_row("built policy, scored on A", evaluate_policy_pre_fast(policy_B, Coins_A, V_post_A))   
    for rho in [1.0, 5.0]:
        show_row(f"always RR {rho}, scored on A", evaluate_policy_pre_fast(fixed_policy(rho), Coins_A, V_post_A))

    # =====================================================================================
    # 6. fair-coin benchmark: P(lock) must equal (20/21)^22 
    # =====================================================================================
    fair = {1.0: [(0.5, 1.0), (0.5, -1.0)]}
    step = post.Dx                                      # one grid cell per trade
    V_fair, _ = solve_pre_fast(fair, None, risk=step, lock_one=True, tol=1e-9)
    d, a = round(post.PLAN.dd_limit / step), round(post.PLAN.lock_peak / step)
    print(f"\nfair-coin P(lock): solver {V_fair[0, 0]:.6f}   exact {(d / (d + 1)) ** a:.6f}")

    # =====================================================================================
    # 7. P(lock) under the payout-optimal policy 
    # =====================================================================================
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

    # =====================================================================================
    # time layer
    # =====================================================================================
    _, policy_post = post.solve_post(Coins, risk)
    T_post = post.expected_trades_post(Coins, policy_post, risk)
    T_pre  = expected_trades_pre(policy_pre, Coins, T_post)
    T_pre_only = expected_trades_pre(policy_pre, Coins, np.zeros_like(T_post))
    print(f"\nexpected trades, fresh funded account -> payout or death: {T_pre[0, 0]:.1f}")
    print(f"   of which pre-lock (until lock or breach):              {T_pre_only[0, 0]:.1f}")

# =====================================================================================
# Risk sweep
# =====================================================================================
print(Sweep.run_sweep(Sweep.default_risks()).round(3).to_string(index=False))