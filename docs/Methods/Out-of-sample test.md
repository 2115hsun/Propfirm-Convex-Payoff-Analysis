# Out-of-sample test

Trades split in time: A = first 442, B = last 442. Build the policy on one half, score it on the other. Post-lock is re-solved on each half, so the test half never leaks in.

| Built on | Scored on | Built policy | Always RR 1.0 | Always RR 5.0 |
|---|---|---|---|---|
| A | B | 0.3507 | 0.2976 | 0.2908 |
| B | A | 0.3898 | 0.3013 | 0.3354 |

About 5 points better than the best simple rule in both directions. In-sample it looked like 7.6, so about 3 points of the in-sample gain was fitting.
