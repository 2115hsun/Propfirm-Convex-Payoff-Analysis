---
date: 2026-09-30
issue:
change: post-lock re-solved on each half in the out-of-sample test
result: gap stayed about 5 points both ways
status: done
---
# Clean out-of-sample split

## Question
Did the out-of-sample result depend on post-lock values learned from all 884 trades?

## Change
`run_prelock.py`: `V_post_A` and `V_post_B` solved on each half and used in the matching calls.

## Baseline before → after
| Check | Before | After |
|---|---|---|
| Built on A, scored on B | 0.3510 | 0.3507 |
| Built on B, scored on A | 0.3836 | 0.3898 |

## Conclusion
Keep. The leak barely mattered; the policy's advantage is real within 2026. See [[Out-of-sample test]].
