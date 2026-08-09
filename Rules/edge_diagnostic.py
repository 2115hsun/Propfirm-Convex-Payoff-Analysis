"""
edge_diagnostic.py — is it "no edge" or "bad config / bug"? Run this on your
REAL data. It brackets your strategy's measured edge between two controls:

    LookaheadCheat : entries that PEEK at the future -> UPPER bound on the edge
                     the pipeline can possibly extract from this data. If this
                     is near zero, the data has no directional structure at this
                     horizon at all (nothing to capture) — or the pipeline is
                     broken. It should be strongly positive.
    RandomEntry    : random entries/directions -> the ZERO-edge baseline. Your
                     strategy must beat this to have any edge.

Read the output like this:
    cheat >> 0  AND  strat ~ random   -> data HAS structure but YOUR entry isn't
                                          capturing it (config/logic problem, NOT
                                          "no edge exists"). Tune trigger/stop/gates.
    cheat >> 0  AND  strat > random    -> you have real (if small) edge -> go OOS.
    cheat ~ 0                          -> no extractable structure at this horizon
                                          -> change timeframe, not parameters.
    strat < random                     -> entry is anti-predictive (e.g. buying
                                          into mean-reversion) -> flip the thesis.

Usage:
    from edge_diagnostic import bracket_edge
    from strategies.kernel_trend import get_strategy
    bracket_edge(get_strategy(trigger="cross"), bars, k_stop_atr=5,
                 max_hold=120, rr=1.5)
"""
from __future__ import annotations
import numpy as np
from Backtest import backtest_entries, ledger_for_rr, measured_winrate


class RandomEntry:
    name = "random"; wants_bars = True
    def __init__(self, n_target, seed=0):
        self.n = int(n_target); self.rng = np.random.default_rng(seed)
    def entries(self, bars):
        n = len(bars)
        k = min(self.n, max(n - 30, 1))
        idx = np.sort(self.rng.choice(np.arange(25, n - 5), k, replace=False))
        return idx, self.rng.choice([1, -1], len(idx))


class LookaheadCheat:
    """Peeks H bars ahead — NOT tradeable, only an upper bound on extractable edge."""
    name = "cheat"; wants_bars = True
    def __init__(self, H=20, every=5):
        self.H = int(H); self.every = int(every)
    def entries(self, bars):
        c = bars["close"].to_numpy(float); n = c.size
        idx = np.arange(25, n - self.H - 5, self.every)
        return idx, np.where(c[idx + self.H] > c[idx], 1, -1)


def _edge(E, rr):
    R, _ = ledger_for_rr(E, rr); return R.mean(), R.std()


def bracket_edge(strategy, bars, k_stop_atr=5, max_hold=120, rr=1.5,
                 cheat_H=20, seed=0, n_boot=400):
    """Print strategy edge vs random and lookahead controls, with a bootstrap
    CI on the strategy edge so you can see if it's distinguishable from zero."""
    Es = backtest_entries(strategy, bars, k_stop_atr=k_stop_atr, max_hold=max_hold)
    ns = len(Es.day_ids)
    es, ss = _edge(Es, rr)
    Ec = backtest_entries(LookaheadCheat(cheat_H), bars, k_stop_atr=k_stop_atr, max_hold=max_hold)
    Er = backtest_entries(RandomEntry(ns, seed), bars, k_stop_atr=k_stop_atr, max_hold=max_hold)
    ec, _ = _edge(Ec, rr); er, _ = _edge(Er, rr)

    # bootstrap CI on strategy edge (are we even distinguishable from zero?)
    R, _ = ledger_for_rr(Es, rr)
    rng = np.random.default_rng(seed)
    boots = np.array([R[rng.integers(0, R.size, R.size)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    name = getattr(strategy, "name", "strat")
    print(f"  {'strategy ('+name+')':<26} edge={es:+.3f}  WR={measured_winrate(Es,rr):.1%}  "
          f"n={ns}  95%CI=[{lo:+.3f}, {hi:+.3f}]")
    print(f"  {'lookahead cheat (upper)':<26} edge={ec:+.3f}   <- extractable edge ceiling")
    print(f"  {'random entry (zero base)':<26} edge={er:+.3f}")
    verdict = ("EDGE vs random, CI excludes 0" if lo > 0 and es > er + 0.02 else
               "NOT distinguishable from random/zero" if lo <= 0 <= hi else
               "ANTI-predictive (below random)" if es < er - 0.02 else "marginal")
    print(f"  -> {verdict}")
    return dict(edge=es, ci=(lo, hi), cheat=ec, random=er, n=ns)


def sweep_configs(get_strategy, bars, k_stop_list=(3, 5, 8), triggers=("regime", "cross"),
                  rr=1.5, max_hold=120):
    """Grid the two knobs that mattered most in the audit: trigger x stop width."""
    print(f"{'trigger':>8} {'k_stop':>7} {'edge':>8} {'WR':>6} {'n':>6}")
    for tr in triggers:
        for km in k_stop_list:
            E = backtest_entries(get_strategy(trigger=tr, slope_min=0, er_min=0),
                                 bars, k_stop_atr=km, max_hold=max_hold)
            e, _ = _edge(E, rr)
            print(f"{tr:>8} {km:>7} {e:>+8.3f} {measured_winrate(E,rr):>6.1%} {len(E.day_ids):>6}")