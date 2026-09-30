"""run_prelock.py — solve, comparisons, and out-of-sample tests for the pre-lock model."""
import time
import numpy as np
import Postlock as post
from Prelock import (m_grid, y_grid, Coins, risk, R_ALL, collapse_to_coin,   # grid and data
                     transition_pre, solve_pre_fast, evaluate_policy_pre_fast,  # model
                     fixed_policy, show_row, expected_trades_pre)                                     # helpers

# ---- switches ----
RUN_CHECKS  = False    # re-run the all-cells transition check
SHOW_TABLES = False    # print the optimal-RR table
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

    # ---- time layer -------------------------------------------------------------------
    _, policy_post = post.solve_post(Coins, risk)
    T_post = post.expected_trades_post(Coins, policy_post, risk)
    T_pre  = expected_trades_pre(policy_pre, Coins, T_post)
    T_pre_only = expected_trades_pre(policy_pre, Coins, np.zeros_like(T_post))
    print(f"\nexpected trades, fresh funded account -> payout or death: {T_pre[0, 0]:.1f}")
    print(f"   of which pre-lock (until lock or breach):              {T_pre_only[0, 0]:.1f}")


# ---- stage 11: risk sweep ---------------------------------------------------------------
import Lifecycle as life

RISKS = list(range(100, 1001, 50))       # $ per R = 2 to 10 MNQ
i_entry = post.cell_index(post.Entry, post.Floor, post.Dx)

eval_cost, eval_days = life.eval_stage(life.Eval_fee, life.Eval_pass_rate, life.Days_pass, life.Days_fail)

print(f"\neval side, per funded account: ${eval_cost:.0f} over {eval_days:.1f} days")

print(f"\n{'risk':>6} {'MNQ':>4} {'P(lock)':>8} {'P(pay1)':>8} {'P(again)':>9} "
      f"{'trades1':>8} {'trades/cyc':>11} {'payouts':>8} {'cash':>7} {'days':>7}"
      f" {'$/day':>7} {'fleet':>7} {'ROI':>7}")

for r in RISKS:
    V_post_r, pol_post_r = post.solve_post(Coins, r)                        # post-lock at this risk
    T_post_r = post.expected_trades_post(Coins, pol_post_r, r)

    V_pre_r, pol_pre_r = solve_pre_fast(Coins, V_post_r, risk=r)          # hands off to THIS post-lock
    p_lock = evaluate_policy_pre_fast(pol_pre_r, Coins, V_post_r, risk=r, lock_one=True)[0, 0]
    T_first = expected_trades_pre(pol_pre_r, Coins, T_post_r, risk=r)[0, 0]

    payouts, cash, fdays = life.funded_stage(q=V_pre_r[0,0], p=V_post_r[i_entry],
        trades_first=T_first, trades_cycle=T_post_r[i_entry],
        payout=life.Payout, split=life.Split, trades_per_day=life.Trades_per_day)

    net, cdays, per_day, fleet, roi = life.cycle_economics(eval_cost, eval_days, cash, fdays, life.N_slots)

    print(f"{r:>6} {r // 50:>4} {p_lock:>8.3f} {V_pre_r[0, 0]:>8.3f} {V_post_r[i_entry]:>9.3f} "
          f"{T_first:>8.1f} {T_post_r[i_entry]:>11.1f} {payouts:>8.2f} {cash:>7.0f} {fdays:>7.1f}"
          f" {per_day:>7.1f} {fleet:>7.0f} {roi:>7.0%}")