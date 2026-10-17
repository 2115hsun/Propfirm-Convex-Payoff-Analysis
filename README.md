# Prop firm funded accounts as a stochastic control problem

This project asks how to trade a prop firm funded account to maximize the probability of a payout and, as a secondary question, whether the cycle of buying evaluations and collecting payouts has positive expected value. It models the funded account as a Markov decision process with absorbing barriers, solves it by backward induction on real trade data, and checks the results out of sample.

The trading strategy supplies the trade outcomes and could be replaced by another. The contribution is the method: encoding a firm's rules as states and boundaries, finding the best action in every state, measuring how long it takes, and pricing the full cycle.

## Prop firms

A proprietary trading firm ("prop firm") sells evaluations. A trader pays a fee, roughly \$50 to \$200, to trade a simulated futures account under fixed rules, such as a maximum loss. A trader who reaches the profit target without breaking the rules receives a funded account, trades it under a similar rule set, and keeps about 90% of each withdrawal.

The levels of a Lucid 25K Flex funded account, measured from the \$25,000 start:

| Level | Balance | Meaning |
|---|---|---|
| +2,100 | \$27,100 | Payout level used in the model: withdraw the \$1,000 cap and keep a full drawdown of cushion |
| +1,100 | \$26,100 | Peak that locks the floor |
| +100 | \$25,100 | Floor after the lock |
| 0 | \$25,000 | Start |
| −1,000 | \$24,000 | Starting floor, trailing \$1,000 below the highest end-of-day balance |

The rules that shape the problem, with the Lucid 25K Flex values:

| Rule | How it works | Lucid 25K Flex |
|---|---|---|
| Trailing drawdown | The account closes if the balance falls a fixed amount below its highest end-of-day balance. The floor rises with the peak and never falls. | \$1,000 |
| Lock | Once the peak is high enough, the floor stops trailing and freezes. | Locks at \$25,100 once the peak reaches \$26,100 |
| Intraday breach | Some firms close the account if the balance touches the floor during a trade, even if it recovers by the close. | To verify |
| Payout size | A share of profit, capped per withdrawal. | 50% of profit, up to \$1,000 |
| Profitable days | A minimum number of days above a profit threshold in each payout cycle. | 5 days of at least \$100 |
| Payout limit | A fixed number of payouts before the account moves to a live brokerage account. | 5 |
| Consistency | The best day cannot exceed a share of total profit. | 50% in the evaluation, none when funded |
| Daily loss limit | A cap on one day's loss, which either ends the day or closes the account. | None |

## The problem

Trading with prop firms is a repeating cycle. A trader pays for an evaluation, which may or may not pass. Each pass yields a funded account, which pays out until it breaches, and the cycle starts again. The evaluation behaves like an option: the fee is the most that can be lost, and the funded account is the payoff. Whether the cycle has positive expected value depends on what each funded account costs in fees and time, and on how many payouts it delivers before it breaches.

Within the funded stage, the account sits between two barriers. Below it is a floor that trails the highest balance, so every gain also raises the level that ends the account. Above it is the lock, where the floor freezes, and beyond that the payout. Before every trade, the trader chooses a profit target, measured as a reward-to-risk ratio (RR): an RR of 2 means the trade aims to win twice what it risks. A small target wins often but moves the balance slowly; a large one wins rarely but can clear the lock in one trade. With a small edge, the best choice depends on how much room the account has above the floor and how far it is from the next milestone.

Prop firms are usually discussed as a test of trading skill. This project treats the funded account as a first-passage problem with path-dependent barriers: the state is the balance and the peak, the action is the RR, and the objective is the probability of reaching a payout before the floor. That solution, combined with the evaluation's fee, pass rate and duration, prices the full cycle.

## Data

The data are 1-minute open, high, low, close and volume bars for NQ (Nasdaq-100 E-mini futures) from Databento, covering June 2021 to June 2026. Databento provides one row per listed instrument per minute, including individual contract months and calendar spreads, with prices stored as fixed-point integers.

The raw file was converted to a continuous series in three steps. Prices were rescaled to index points and timestamps parsed in UTC. At each minute, only the instrument with the highest traded volume was kept, so the series follows the most liquid contract and rolls to the next contract month when its volume overtakes the expiring one. Rows with a close outside 1,000 to 40,000 points, an open below 1,000, or a non-positive low were removed, which excludes spreads and erroneous prints. This reduced 2,595,575 rows to 1,767,373, with no duplicate timestamps. Timestamps are converted to New York time for analysis. The series is not back-adjusted, so prices step at each roll by the difference between contract months.

## Strategy

The trade outcomes come from `dual_kernel_v2`, an intraday rule inspired by trend following. It trades NQ long and short between 9:30 and 11:15 New York time.

The rule estimates the short-term trend with two Nadaraya-Watson kernel regressions, which are weighted averages of recent closing prices. The fast one weights the last few bars most heavily; the slow one weights a longer window more evenly. When the fast estimate crosses above the slow one, recent prices are rising faster than the longer trend, and the rule looks for a long entry; a cross in the other direction looks for a short.

Each trade has a fixed 25-point stop, which defines one unit of risk (1R). It exits at the stop, at the chosen target, or after 60 minutes, whichever comes first. Because the entries and stop are fixed, the same 884 trades (January to June 2026) can be scored at every target from 1R to 5R:

| RR | 1.0 | 1.25 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | 4.5 | 5.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| Win rate | 52.6% | 47.2% | 42.3% | 33.7% | 29.2% | 25.5% | 22.6% | 20.4% | 18.0% | 15.0% |
| Mean return (R/trade) | +0.055 | +0.065 | +0.062 | +0.027 | +0.054 | +0.074 | +0.092 | +0.125 | +0.133 | +0.106 |

The rule was selected from roughly 25 candidate designs tested on NQ, including mean-reversion, opening-range, VWAP, gap and volatility-breakout rules, and was the only one whose mean return exceeded a zero-edge benchmark at a usable trade frequency. The estimates are therefore subject to selection bias. Because the strategy has not been tested across market regimes, it is treated as a realistic input distribution, with no expectation that its returns persist.

Trades that reach neither the stop nor the target close at the time limit with an intermediate result. These time-outs keep the mean return positive at high targets, so the model uses each target's full distribution of outcomes rather than a simple win/loss approximation.

## Model

### Markov decision process

A Markov decision process (MDP) is a tuple $(S, A, P, R, \gamma)$: states, actions, a transition kernel $P(s' \mid s, a)$ giving the probability of moving to state $s'$ from $s$ under action $a$, rewards, and a discount factor.

A toy version shows the structure. A token sits on states $\{0, 1, 2, 3, 4\}$, where 0 (breach) and 4 (payout) are absorbing. In each interior state the player chooses a tight action (+1 with probability 0.7, −1 with 0.3) or a wide action (+2 with probability 0.4, −1 with 0.6). The funded account has the same shape at a larger scale:

| Part | Funded account |
|---|---|
| States | After the lock: balance $x$. Before the lock: peak $m$ and drawdown $y$ below it. |
| Actions | RR target for the next trade, 1.0 to 5.0 |
| Transitions | Empirical trade outcomes at each RR; a landing between grid points is split between its neighbors by distance |
| Reward | 1 at the payout, 0 at a breach, so a state's value is its probability of being paid |
| Discount | None ($\gamma = 1$) |

### 1) Markov property

The process is Markov if the next state depends only on the current state and action:

$$\Pr(s_{t+1} \mid s_t, \dots, s_0, a_t) = \Pr(s_{t+1} \mid s_t, a_t).$$

Balance alone fails this test before the lock. The floor is $m - D$, where $m = \max_{\tau \le t} x_\tau$ is the running peak and $D$ is the drawdown limit, so two accounts at the same balance can have different floors. Adding the peak restores the property: pre-lock states are $(m, y)$ with $y = m - x$. After the lock the floor is fixed and balance alone suffices. The two stages are solved separately, and the number of payouts taken is tracked in the lifecycle economics.

### 2) Bellman equation

The value of a state under a policy $\pi$ is its expected total discounted reward:

$$V^\pi(s) = \mathbb{E}\big[R_0 + \gamma R_1 + \gamma^2 R_2 + \cdots\big].$$

Splitting off the first reward, the remainder $\gamma(R_1 + \gamma R_2 + \cdots)$ is $\gamma$ times the value of whichever state $s'$ the first step lands in. Because $s'$ is random, it is averaged over the transition kernel:

$$V^\pi(s) = R(s, \pi(s)) + \gamma \sum_{s'} P\big(s' \mid s, \pi(s)\big)\, V^\pi(s').$$

Choosing the best action in each state gives the Bellman optimality equation. Here rewards arrive only at the barriers and $\gamma = 1$, so it reduces to

$$V(s) = \max_{RR} \sum_{s'} P(s' \mid s, RR)\, V(s'), \qquad V = 1 \text{ at the payout}, \quad V = 0 \text{ at a breach}.$$

The value of a state is the best available average of the values one trade away. It is solved by value iteration: start from zero, apply the equation to every state, and repeat until the values stop changing.

### 3) Post-lock stage

The two stages are solved in reverse order, because the value of reaching the lock depends on what happens after it. Once a Lucid 25K account locks, the floor is fixed at \$25,100 and the state is the balance $x$ alone. Its value $V(x)$ is the probability of reaching \$27,100 before \$25,100, given by the equation above with $V = 1$ at the payout and $V = 0$ at the floor. The balance is kept on a \$50 grid. At \$250 per trade (5 Micro Nasdaq contracts), an account that has just locked reaches the payout with probability 0.640.

### 4) Pre-lock stage

Before the lock, the floor trails the peak, so the state needs two numbers: the peak $m$ and the drawdown below it, $y$. A trade ends in one of four ways: a breach ($y$ reaches \$1,000), a lock (the peak reaches \$1,100), a new peak ($y$ resets to 0), or a new drawdown under the same peak. A trade that locks at balance $x'$ is worth the post-lock value $V_{\text{post}}(x')$, so a lock that overshoots with extra cushion is worth more than one at exactly \$26,100. At \$250 per trade, a fresh funded account reaches its first payout with probability 0.373.

### 5) From model to results

The two solves give the probability of a payout from any state. Three more steps turn that into the quantities the dashboard reports.

**Time.** For the chosen policy, the expected number of trades until payout or breach satisfies $T(s) = 1 + \sum_{s'} P(s' \mid s)\, T(s')$, solved exactly as a linear system. At \$250 per trade, a fresh account takes about 19 trades to reach its first payout or breach, and each later payout cycle about 18.

**Repeat payouts.** After a payout the account returns to the lock level, so each later payout succeeds with the same probability $p = 0.640$. With first-payout probability $q$ and a payout limit $K$, the expected number of payouts per funded account is

$$\text{payouts} = q\,\frac{1 - p^{K}}{1 - p}.$$

**The full cycle.** With evaluation pass rate $p_{\text{pass}}$, each funded account costs $1/p_{\text{pass}}$ evaluation fees plus any activation fee. Combining payouts, the profit split, evaluation cost and total time gives the expected net value per cycle, its length in trading days, and the return on evaluation fees.

**Risk per trade.** These steps are repeated at every position size from 2 to 20 contracts, with both stages re-solved each time. Smaller sizes raise the payout probability but lengthen every cycle; larger sizes do the reverse.

The dashboard reads these results from an exported file per plan. It shows the sweep by position size, the distribution of payouts per funded account, simulated returns from a batch of evaluations, the optimal RR for any account state, and the rules of the selected plan.

## Validation

### 1) Exact benchmarks

Two fair-coin cases have exact answers. A coin that wins or loses one \$50 grid cell with probability ½ turns post-lock into the gambler's ruin problem: from the lock entry, 20 steps from both the floor and the payout, the expected number of trades is $20 \times 20 = 400$. Before the lock, each new high starts a race between one step up and twenty steps down, which the coin wins with probability $20/21$; locking requires 22 such wins, so $P(\text{lock}) = (20/21)^{22} = 0.341850$. The solver reproduces both values exactly, which verifies the state routing, the barriers and the time calculation.

These cases move exactly one cell and count a lock as worth 1, so they do not test the interpolation between grid points or the hand-off to post-lock. Two further checks cover those: the fast matrix solver agrees with the original solver to about $10^{-16}$ in every state, and a 50K account at \$500 per trade, which has the same geometry in units of risk as a 25K at \$250, gives an identical post-lock payout probability (0.640).

### 2) Out-of-sample test

The optimal policy changes RR with the account's state. To check that this is not fitted noise, the policy is built on the first 442 trades and scored on the last 442, then the other way around. Post-lock is re-solved on each half, so nothing from the test half leaks into the policy. Values are the probability of a first payout:

| Built on | Scored on | Optimal policy | Always RR 1.0 | Always RR 5.0 |
|---|---|---|---|---|
| First half | Second half | 0.351 | 0.298 | 0.291 |
| Second half | First half | 0.390 | 0.301 | 0.335 |

Choosing RR by state raises the first-payout probability by about 5 percentage points over the best fixed rule in both directions, so the gain does not depend on one split. A larger sample is needed to confirm it.

## Results

All figures use the Lucid 25K Flex rules at \$250 per trade and closing prices unless stated.

**Payout probabilities.** A fresh funded account reaches its first payout with probability 0.373. It locks with probability 0.522, and a locked account is paid with probability 0.715, higher than the 0.640 of an account sitting exactly at the lock level because many locks overshoot with extra cushion. With Lucid's limit of five payouts, a funded account delivers about 0.92 payouts on average.

**When choosing RR by state helps.** The out-of-sample gain depends on how large each trade is relative to the drawdown. With four full losses of room (Lucid 25K at \$250, or 50K at \$500), the gain is about +5 points in both directions. With eight losses of room (50K at \$250), it is −4.3 points in one direction and +3.8 in the other. When risk is a large share of the drawdown, a single trade can decide whether the account locks, and the optimal policy exploits this; with more room, a fixed target does about as well.

**Intra-trade breaches.** Counting dips to the floor during a trade, measured up to each target's actual exit, lowers the 50K first-payout probability at \$500 from 0.385 to 0.354. The out-of-sample gain from choosing RR by state falls to about +3 to +4 points but stays positive in both directions.

**Thresholds in position size.** The number of full losses to breach is the drawdown divided by the risk per trade, rounded up, so performance drops each time risk crosses the drawdown divided by an integer. On a \$1,000 drawdown, \$450 per trade survives three losses and \$500 survives two.

**Probability against time.** Smaller risk raises every payout probability but lengthens every cycle; larger risk does the reverse. Net value per unit of time rises with size over the range examined, which makes this result sensitive to the rules not yet modeled. Results above about 70% of the drawdown depend on how the trade size aligns with the grid and the rule levels, and are treated as provisional.

**Independence of outcomes.** Against 5,000 random reorderings, the trade sequence shows no excess clustering of losses at the trade or day level. The largest single event, 13 consecutive losing trades on April 23 and 24, 2026, did not recur. This supports treating trades and days as independent draws, within the limits of a 161-day sample.

## Limitations

**Data.** The trade outcomes cover January to June 2026: 884 trades over 161 trading days, from a strategy selected after a search. The estimates are subject to selection bias, reflect one market regime, and carry no uncertainty ranges. Extending the sample to mid-2024 through 2026 with the strategy's parameters held fixed would allow a genuine out-of-sample test of the strategy itself.

**Rules not yet modeled.** Every plan encoded here requires a minimum number of profitable days per payout cycle, which penalizes fast, high-risk cycles most, so the best position size is likely lower than the current sweep suggests. Position limits, daily loss limits and post-payout floor changes are also absent.

**Costs and execution.** Commission and slippage are excluded. At about \$1 per contract per round trip, commission is roughly 0.02R per trade against a mean return near 0.05R. On the bar where a target is reached, 1-minute data cannot show whether the bar's low came first; the model assumes it did, which is conservative.

**Structure.** Each account takes one trade per day, and accounts are treated as independent. Risk per trade is fixed within a solve, the objective is the probability of a payout rather than value per unit of time, payouts are always taken at the same level, and an account that reaches its payout limit is treated as finished.

**Economics.** The evaluation pass rate and duration were measured on the Lucid 25K and reused for other plans. Lifecycle figures are steady-state averages that ignore the timing of fees and payouts.
