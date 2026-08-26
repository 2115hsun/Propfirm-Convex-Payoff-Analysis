"""edge_diagnostic.py — points-based edge bracket vs random + lookahead controls."""
from __future__ import annotations
import numpy as np
from Backtest import backtest_entries, ledger_for_rr, measured_winrate   # points version now


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
    name = "cheat"; wants_bars = True
    def __init__(self, H=20, every=5):
        self.H = int(H); self.every = int(every)
    def entries(self, bars):
        c = bars["close"].to_numpy(float); n = c.size
        idx = np.arange(25, n - self.H - 5, self.every)
        return idx, np.where(c[idx + self.H] > c[idx], 1, -1)


def _edge(E, rr):
    R, _ = ledger_for_rr(E, rr); return R.mean()


def bracket_edge(strategy, bars, stop_points=40.0, max_hold=40, rr=1.5,
                 cheat_H=20, seed=0, n_boot=400):
    """Strategy edge vs random/lookahead controls at ONE stop, with a bootstrap
    CI on the strategy edge (distinguishable from zero?)."""
    Es = backtest_entries(strategy, bars, stop_points=stop_points, max_hold=max_hold)
    ns = len(Es.day_ids)
    es = _edge(Es, rr)
    Ec = backtest_entries(LookaheadCheat(cheat_H), bars, stop_points=stop_points, max_hold=max_hold)
    Er = backtest_entries(RandomEntry(ns, seed), bars, stop_points=stop_points, max_hold=max_hold)
    ec, er = _edge(Ec, rr), _edge(Er, rr)

    R, _ = ledger_for_rr(Es, rr)
    rng = np.random.default_rng(seed)
    boots = np.array([R[rng.integers(0, R.size, R.size)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    verdict = ("EDGE vs random (CI excludes 0)" if lo > 0 and es > er + 0.02 else
               "NOT distinguishable from zero"   if lo <= 0 <= hi else
               "ANTI-predictive (below random)"  if es < er - 0.02 else "marginal")
    print(f"  stop={stop_points:>4g}pt  edge={es:+.3f}  WR={measured_winrate(Es,rr):.1%}  "
          f"n={ns}  CI=[{lo:+.3f},{hi:+.3f}]  cheat={ec:+.3f} rand={er:+.3f}  -> {verdict}")
    return dict(stop=stop_points, edge=es, ci=(lo, hi), cheat=ec, random=er, n=ns, verdict=verdict)


def bracket_edge_grid(strategy, bars, stop_grid, rr=1.5, max_hold=40, seed=0):
    """Run bracket_edge across a stop grid — the pre-sweep 'is there any stop
    where this beats random' check."""
    return [bracket_edge(strategy, bars, stop_points=sp, rr=rr, max_hold=max_hold, seed=seed)
            for sp in stop_grid]