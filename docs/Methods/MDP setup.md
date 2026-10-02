# MDP setup

The funded account as a Markov decision process.

- **States**: post-lock, the balance $x$. Pre-lock, the peak $m$ and drawdown $y$.
- **Actions**: the reward-to-risk target RR for the next trade.
- **Transitions**: from the trade distribution at each RR (`ledgers.pkl`), on a $50 grid with linear splitting between grid points.
- **Reward**: 1 at the payout, 0 at a breach, 0 otherwise.
- **Discount**: none ($\gamma = 1$).

Markov: the future depends only on the state. The trailing floor puts the peak in the pre-lock state.

Related: [[Post-lock model]], [[Pre-lock model]]
