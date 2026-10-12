## Prop firm funded accounts as a stochastic control problem 

This project asks how to trade a prop firm funded account to get paid as often as possible, and whether buying evaluations and collecting payouts generates wealth overall. It models the funded account as a Markov decision process with absorbing barriers, solves it by backward induction on real trade data, and checks the results out of sample. 

The trading strategy is an input to the model. The contribution is the method: encoding a firm's rules as states and boundaries, finding the best action in every state, measuring how long it takes, and pricing the full cycle. 

### Introduction to Prop Firms 

A prop firm (not the investment entity) sells product called evaluation. A retail trader pay roughly $50 to $200 to trade a simulated futures account of $X value (i.e. Topstep Trader's $50,000 account) under fixed rules (i.e. max losses is $2,000, maximum profit per day cannot exceed 50% of total profit target, etc). If you reach the profit target without breaking the loss limit, you get a funded account and keep about 90% of what you withdraw. 
| Rule | Effect |
|---|---|
| Trailing drawdown | The account closes if the balance falls a fixed amount (e.g. \$1,000 on a 25K) below its highest end-of-day balance. The floor rises with the peak and never falls. |
| Lock | Once the peak is high enough, the floor stops trailing and freezes (e.g. at \$25,100). |
| Payouts | Withdrawals need a minimum profit and a number of profitable days, and are capped in size. |
| Payout limit | Some firms allow a fixed number of payouts per account before moving it to a live account. |

### The Problem

Trading prop firms is a repeating cycle. A trader pays a fee for an evaluation, which may or may not pass. Each pass buys a funded account, which pays out until it breaches, and then the cycle starts again. The evaluation behaves like an option contract: the fee is the most that can be lost, and the funded account is the payoff. Whether the cycle makes money depends on two things: what each funded account costs in evaluation fees and time, and how many payouts it delivers before it breaches.

A funded account is a race between two barriers. Below it sits a floor that trails the account's highest balance, so every gain also raises the level that ends the account. Above it sits the lock, where the floor freezes, and beyond that the payout. Before every trade, the trader picks a reward-to-risk target (RR) and a size. A small target wins often but slowly; a large one wins rarely but can clear the lock in one trade. With a small edge, the best choice depends on the account's room above the floor and its distance to the next milestone.

Prop firms are usually treated as a test of discretionary trading skill by retail trading enthusiasts. This project treats the funded account as a first-passage problem with path-dependent barriers: the state is the balance and peak, the action is the RR, and the objective is the probability of a payout before the floor. Its solution, combined with the evaluation's fee, pass rate, and duration, prices the full cycle. 

### Data

NQ (Nasdaq-100 E-mini) 1-minute bars, 2021 to 2026, about 1.77 million rows in Parquet, timestamps in America/New York. 

The contracts are pulled from Databento, and a custom rolling mechanism is designed for contract rollover. 

### Strategy 

The trade distribution comes from `dual_kernel_v2`, an intraday rule inspired by trend following. It trades NQ 1-minute bars, long and short, between 9:30 and 11:15 New York time.

The rule estimates the short-term trend with two Nadaraya-Watson kernel regressions, which are weighted averages of recent closing prices. The fast regression gives most of its weight to the last few bars; the slow one weights a longer window more evenly. When the fast estimate crosses above the slow one, recent prices are rising faster than the longer trend, and the rule looks for a long entry. A crossover in the other direction looks for a short. 

Each trade has a fixed 25-point stop, which defines one unit of risk (1R). It exits at the stop, at the chosen reward-to-risk target, or after 60 minutes, whichever comes first. Because the stop and the entries are fixed, the same 884 trades can be scored at every target from 1R to 5R:

| RR | 1.0 | 1.25 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | 4.5 | 5.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| Win rate | 52.6% | 47.2% | 42.3% | 33.7% | 29.2% | 25.5% | 22.6% | 20.4% | 18.0% | 15.0% |
| Mean return (R/trade) | +0.055 | +0.065 | +0.062 | +0.027 | +0.054 | +0.074 | +0.092 | +0.125 | +0.133 | +0.106 |

The rule was selected from roughly 25 candidate designs tested on NQ, including mean-reversion, opening-range, VWAP, gap and volatility-breakout rules. It was the only one whose mean return exceeded a zero-edge benchmark at a usable trade frequency. The estimates are therefore subject to selection bias. Because the strategy has not been tested across market regimes, it is treated here as a realistic input distribution, with no expectation that its returns persist.

Trades that reach neither the stop nor the target close at the time limit with an intermediate result. These time-outs keep the mean return positive at high targets, so the model uses each target's full empirical distribution of outcomes rather than a two-outcome win/loss approximation.
