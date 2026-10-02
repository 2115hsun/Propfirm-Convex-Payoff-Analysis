# Baselines

Rerun after every change. A refactor must reproduce all of these; a modeling change should move them, and the new values go in an experiment note.

At $250 per R, 25-point stop, 884 trades (2026), no commission, no MAE.

| Check | Expected | Where |
|---|---|---|
| Post-lock P(payout) at $26,100 | 0.6401 | `run_postlock.py` |
| Post-lock at $150 | 0.7003 | `run_postlock.py` |
| Fair coin, trades from lock entry | 400.00 | `run_postlock.py` |
| Pre-lock P(first payout) | 0.3733 | `run_prelock.py` |
| P(lock) / P(payout given lock) | 0.5221 / 0.7150 | `run_prelock.py` |
| Fair coin P(lock) | 0.341850 = (20/21)^22 | `run_prelock.py` |
| Built on A, scored on B | 0.3507 vs always RR 1.0 0.2976 | `run_prelock.py` |
| Built on B, scored on A | 0.3898 vs always RR 1.0 0.3013 | `run_prelock.py` |
| Trades, fresh account to payout or breach | 18.8 (10.3 pre-lock) | `run_prelock.py` |
| Eval side ($60, 35%, 4 and 2 days) | $171 over 7.7 days | `Lifecycle.py` |
| $250 row: payouts / cash / funded days | 1.04 / $934 / 37.3 | risk sweep |
