# Post-lock model

Floor frozen at $25,100, payout at $27,100. Value = probability of reaching the payout first:

$$V(x) = \max_{RR} \sum_{x'} P(x' \mid x, RR)\, V(x')$$

with $V = 1$ at the payout and $V = 0$ at the floor. Solved by value iteration in `solve_post`.

Result at $250: 0.6401 from the lock entry. Checks: [[Validation]]. Feeds [[Pre-lock model]] through the hand-off.
