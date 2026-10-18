"""
Walkforward.py: rolling walk-forward validation of the points-based 3-D
risk-geometry sweep, with the multi-account rotation measured per fold.

  * The param + geometry search runs on each rolling train block, is locked,
    then measured on the next test block. Pooling the test blocks gives
    hundreds of out-of-sample trades instead of one short tail.
  * Train windows are fixed width, not expanding. A fixed-points stop isn't
    stable across 2021-2026 (NQ went from 13k to 25k), so each fold picks the
    stop width for its own regime.
  * The winner is the combo with the highest p25 over its own 3-D pass grid,
    so a broad plateau beats a single lucky cell.
  * Each locked winner is measured on the test block twice: single-account
    (oos_check) and with the N-slot rotation (multi_account_replay).
  * PBO proxy: each fold ranks the train winner among the Stage-B survivors
    on the test block. PBO is the fraction of folds where the winner lands
    in the worse half (Lopez de Prado's definition, per fold).

This module only orchestrates Backtest, Analysis and MultiEval. The caller
passes `strategy_factory(**params)` returning a ready-to-backtest strategy
(already time-window wrapped), so nothing here is tied to one strategy.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import itertools
import numpy as np
import pandas as pd

from Backtest import backtest_entries, ledger_for_rr, measured_winrate
from Analysis import risk_geometry_3d, oos_check
from MultiEval import multi_account_replay


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------
@dataclass
class Fold:
    idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp     # exclusive-ish upper bound for the .loc slice (minus embargo)
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(bars, train_months=6, test_months=1, step_months=1, embargo_days=1):
    """Rolling (train, test) calendar folds over the span of `bars`. Train is
    a fixed-width window ending `embargo_days` before test_start; test is the
    next `test_months`; folds step forward by `step_months`."""
    tz = bars.index.tz
    start = bars.index.min().normalize()
    end = bars.index.max().normalize()
    folds = []
    cur = pd.Timestamp(start.year, start.month, 1, tz=tz)
    i = 0
    while True:
        tr_start = cur
        tr_end_full = tr_start + pd.DateOffset(months=train_months)
        te_start = tr_end_full
        te_end = te_start + pd.DateOffset(months=test_months)
        if te_start > end:
            break
        tr_end = tr_end_full - pd.Timedelta(days=embargo_days)   # purge the boundary
        folds.append(Fold(i, tr_start, tr_end, te_start, min(te_end, end + pd.Timedelta(days=1))))
        i += 1
        cur = cur + pd.DateOffset(months=step_months)
    return folds


def _slice(bars, a, b):
    """Inclusive day-based slice a..b on a full-24h intraday frame (keeps each
    calendar day's pre-09:30 warmup bars intact for reset_daily kernels)."""
    return bars.loc[a:b]


# ---------------------------------------------------------------------------
# Per-fold selection (Stage A prefilter -> Stage B 3-D geometry -> floor pick)
# ---------------------------------------------------------------------------
def _stage_a(strategy_factory, bars_tr, param_grid, stage_a_stop, max_hold,
             min_trades, min_trades_per_day, rr_check, top_n):
    """Fast edge_per_day prefilter over indicator params at ONE stop width.
    Returns the top_n param dicts (highest edge_per_day) that clear the floors."""
    keys = list(param_grid.keys())
    rows = []
    for combo in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        s = strategy_factory(**params)
        E = backtest_entries(s, bars_tr, stop_points=stage_a_stop, max_hold=max_hold)
        n = len(E.day_ids)
        tpd = n / max(E.n_days, 1)
        if n < min_trades or tpd < min_trades_per_day:
            continue
        edges = [ledger_for_rr(E, rr)[0].mean() for rr in rr_check]
        score = float(np.mean(edges))
        rows.append((params, score * tpd))          # rank key = edge_per_day
    rows.sort(key=lambda t: t[1], reverse=True)
    return [p for p, _ in rows[:top_n]]


def _combo_floor(g, max_days_to_pass):
    """p25 of a combo's finite 3-D pass cells (within the days ceiling if
    any), so the whole surface has to be high, not one cell."""
    pg = g["pass_grid"]
    dg = g.get("days_grid")
    m = np.isfinite(pg)
    if max_days_to_pass is not None and dg is not None:
        within = m & np.isfinite(dg) & (dg <= max_days_to_pass)
        if within.any():
            m = within
    vals = pg[m]
    return float(np.percentile(vals, 25)) if vals.size else float("nan")


@dataclass
class FoldResult:
    fold: Fold
    n_train_trades: int
    selected_params: dict = field(default_factory=dict)
    stop_points: float = float("nan")
    rr: float = float("nan")
    risk_pct: float = float("nan")
    train_pass_pct: float = float("nan")     # locked cell, measured on train
    train_floor: float = float("nan")        # p25 of the winning combo's grid (the selection score)
    test_pass_pct: float = float("nan")      # single-account Monte Carlo on test
    test_days_to_pass: float = float("nan")
    test_wr: float = float("nan")
    test_trades: int = 0
    multi_pass_rate: float = float("nan")    # N-slot rotation on test (passed/(passed+blown))
    multi_passed: int = 0
    multi_blown: int = 0
    multi_timeout: int = 0
    multi_entries_used: int = 0
    winner_test_rank_pct: float = float("nan")  # for PBO: winner's rank among survivors on test (0=best, 1=worst)
    note: str = ""


def _select_on_train(firm, strategy_factory, bars_tr, param_grid,
                     stop_grid, risk_pcts, rr_grid, max_hold, n_paths, max_trades,
                     min_trades, min_trades_per_day, max_days_to_pass,
                     stage_a_stop, rr_check, top_n, selection, seed):
    """Stage A -> Stage B -> floor/peak pick. Returns (winner dict, survivors list)
    where each survivor is dict(params, g, best) so the caller can re-score them
    on the test block for the PBO statistic."""
    survivors_params = _stage_a(strategy_factory, bars_tr, param_grid, stage_a_stop,
                                max_hold, min_trades, min_trades_per_day, rr_check, top_n)
    survivors = []
    for params in survivors_params:
        s = strategy_factory(**params)
        try:
            g = risk_geometry_3d(firm, s, bars_tr, stop_grid, risk_pcts, rr_grid,
                                 max_hold=max_hold, n_paths=n_paths, max_trades=max_trades,
                                 min_trades=min_trades, seed=seed,
                                 max_days_to_pass=max_days_to_pass)
        except ValueError:
            continue   # no cell cleared min_trades at any stop width this fold
        score = _combo_floor(g, max_days_to_pass) if selection == "floor" else g["best"]["pass_pct"]
        survivors.append(dict(params=params, g=g, best=g["best"], score=score))
    if not survivors:
        return None, []
    survivors.sort(key=lambda d: (np.nan_to_num(d["score"], nan=-1)), reverse=True)
    return survivors[0], survivors


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
@dataclass
class WalkForwardResult:
    folds: list                      # list[FoldResult]
    n_accounts: int
    def _resolved(self):
        return [f for f in self.folds if np.isfinite(f.test_pass_pct)]
    @property
    def pooled_test_pass_pct(self):
        r = self._resolved()
        if not r:
            return float("nan")
        w = np.array([f.test_trades for f in r], float)      # trade-weighted
        v = np.array([f.test_pass_pct for f in r], float)
        return float(np.average(v, weights=w)) if w.sum() else float(np.mean(v))
    @property
    def pooled_multi_pass_rate(self):
        P = sum(f.multi_passed for f in self.folds)
        B = sum(f.multi_blown for f in self.folds)
        return 100.0 * P / (P + B) if (P + B) else float("nan")
    @property
    def pbo(self):
        """Fraction of folds where the train winner landed in the worse OOS
        half of that fold's survivors (rank_pct > 0.5). ~0 good, ~0.5 random."""
        r = [f.winner_test_rank_pct for f in self.folds if np.isfinite(f.winner_test_rank_pct)]
        return float(np.mean([x > 0.5 for x in r])) if r else float("nan")
    def table(self):
        rows = []
        for f in self.folds:
            rows.append(dict(
                fold=f.fold.idx,
                train=f"{f.fold.train_start.date()}..{f.fold.train_end.date()}",
                test=f"{f.fold.test_start.date()}..{f.fold.test_end.date()}",
                **{k: f.selected_params.get(k) for k in f.selected_params},
                stop=f.stop_points, rr=f.rr, risk=f.risk_pct,
                train_pass=f.train_pass_pct, train_floor=f.train_floor,
                test_pass=f.test_pass_pct, test_days=f.test_days_to_pass,
                test_n=f.test_trades, multi_pass=f.multi_pass_rate,
                passed=f.multi_passed, blown=f.multi_blown, timeout=f.multi_timeout,
                rank_pct=f.winner_test_rank_pct, note=f.note))
        return pd.DataFrame(rows)


def walk_forward(firm, bars, strategy_factory, param_grid,
                 stop_grid, risk_pcts, rr_grid,
                 train_months=6, test_months=1, step_months=1, embargo_days=1,
                 max_hold=120, n_paths=2000, max_trades=300,
                 min_trades=150, min_trades_per_day=0.20, max_days_to_pass=10,
                 stage_a_stop_points=None, stage_a_rr_check=(1.0, 1.5, 2.0, 3.0),
                 stage_a_top_n=10, selection="floor",
                 multi_n_accounts=5, multi_max_days=25, multi_restart=True,
                 seed=0, verbose=True):
    """Run the whole walk-forward. `param_grid` is a dict of indicator-param
    lists, e.g. {"h":[5,8,10,12], "x_0":[5,8,10,12], "slope_min":[0.3,0.4,0.5]}.
    `strategy_factory(**params)` must return a ready-to-backtest strategy."""
    if stage_a_stop_points is None:
        stage_a_stop_points = list(stop_grid)[len(list(stop_grid)) // 2]
    folds = make_folds(bars, train_months, test_months, step_months, embargo_days)
    out = []
    for fd in folds:
        bars_tr = _slice(bars, fd.train_start, fd.train_end)
        bars_te = _slice(bars, fd.test_start, fd.test_end)
        fr = FoldResult(fold=fd, n_train_trades=0)

        winner, survivors = _select_on_train(
            firm, strategy_factory, bars_tr, param_grid,
            stop_grid, risk_pcts, rr_grid, max_hold, n_paths, max_trades,
            min_trades, min_trades_per_day, max_days_to_pass,
            stage_a_stop_points, list(stage_a_rr_check), stage_a_top_n, selection, seed)

        if winner is None:
            fr.note = "no combo cleared train floors"
            out.append(fr)
            if verbose:
                print(f"[fold {fd.idx}] {fr.note} (train {fd.train_start.date()}..{fd.train_end.date()})")
            continue

        b = winner["best"]
        fr.selected_params = winner["params"]
        fr.stop_points, fr.rr, fr.risk_pct = b["stop_points"], b["rr"], b["risk_pct"]
        fr.train_pass_pct = b["pass_pct"]
        fr.train_floor = winner["score"]
        fr.n_train_trades = b["trades"]

        # test block: single-account Monte Carlo
        s_win = strategy_factory(**winner["params"])
        oc = oos_check(firm, s_win, bars_te, stop_points=fr.stop_points, rr=fr.rr,
                       risk_pct=fr.risk_pct, max_hold=max_hold, n_paths=n_paths,
                       max_trades=max_trades, seed=seed)
        fr.test_trades = oc["trades"]
        if oc["warning"] is None:
            fr.test_pass_pct, fr.test_days_to_pass, fr.test_wr = oc["pass_pct"], oc["days_to_pass"], oc["wr"]
        else:
            fr.note = oc["warning"]

        # test block: N-slot rotation (needs detail=True)
        E_te = backtest_entries(s_win, bars_te, stop_points=fr.stop_points,
                                max_hold=max_hold, detail=True)
        if E_te.entry_time is not None and len(E_te.day_ids) > 0:
            res = multi_account_replay(E_te, firm, risk_pct=fr.risk_pct, rr=fr.rr,
                                       n_accounts=multi_n_accounts, max_days=multi_max_days,
                                       restart_on_fail=multi_restart)
            fr.multi_pass_rate = res.pass_rate * 100
            fr.multi_passed, fr.multi_blown = res.n_passed, res.n_blown
            fr.multi_timeout, fr.multi_entries_used = res.n_timeout, res.entries_used

        # PBO: rank the winner among survivors, scored on the test block
        if len(survivors) > 1:
            test_scores = []
            for sv in survivors:
                s_i = strategy_factory(**sv["params"])
                oc_i = oos_check(firm, s_i, bars_te, stop_points=sv["best"]["stop_points"],
                                 rr=sv["best"]["rr"], risk_pct=sv["best"]["risk_pct"],
                                 max_hold=max_hold, n_paths=n_paths, max_trades=max_trades, seed=seed)
                test_scores.append(oc_i["pass_pct"] if oc_i["warning"] is None else np.nan)
            test_scores = np.array(test_scores, float)
            wscore = test_scores[0]                                  # survivors[0] is the winner
            valid = np.isfinite(test_scores)
            if np.isfinite(wscore) and valid.sum() > 1:
                # rank_pct = fraction of valid survivors that beat the winner OOS
                better = np.sum(test_scores[valid] > wscore)
                fr.winner_test_rank_pct = float(better) / float(valid.sum() - 1)

        out.append(fr)
        if verbose:
            print(f"[fold {fd.idx}] {fd.test_start.date()}..{fd.test_end.date()}  "
                  f"params={winner['params']}  {fr.stop_points:g}pt/{fr.rr:g}R@{fr.risk_pct:.2f}%  "
                  f"train_floor={fr.train_floor:.0f} train_pass={fr.train_pass_pct:.0f}  "
                  f"-> TEST single={fr.test_pass_pct:.0f}% (n={fr.test_trades}) "
                  f"multi={fr.multi_pass_rate:.0f}% ({fr.multi_passed}P/{fr.multi_blown}B)")

    return WalkForwardResult(folds=out, n_accounts=multi_n_accounts)