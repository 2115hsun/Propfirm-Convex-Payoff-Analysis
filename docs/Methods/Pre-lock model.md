# Pre-lock model

State $(m, y)$: peak and drawdown, 22 × 20 = 440 cells. A trade ends in a breach, a lock, a new peak, or a new drawdown under the same peak.

**Backward induction across stages**: [[Post-lock model]] is solved first. A trade that locks at $x'$ is worth $V_{post}(x')$.

Solved with matrices: one sweep is $r + P V$ (`solve_pre_fast`). A fixed policy is solved exactly from $(I - P_\pi)V = r_\pi$.

Result at $250: P(first payout) 0.3733, P(lock) 0.5221. See [[Out-of-sample test]].
