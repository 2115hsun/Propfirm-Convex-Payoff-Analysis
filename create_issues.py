"""
create_issues.py — create the project's GitHub labels, task issues, and one tracking issue.

Requirements: the GitHub CLI (https://cli.github.com), logged in with `gh auth login`,
and this script run from inside your repo folder:   python create_issues.py
Run it once. Running it again creates duplicate issues.
"""
import re
import subprocess

LABELS = {
    "research":  ("C5DEF5", "Rules, data checks, no code"),
    "data":      ("FBCA04", "Ledger and export"),
    "model":     ("0E8A16", "Solvers, transitions, economics"),
    "stats":     ("5319E7", "Uncertainty, out-of-sample, robustness"),
    "dashboard": ("1D76DB", "Web dashboard"),
    "housekeeping": ("BFD4F2", "Cleanup, docs, tests"),
    "backlog":   ("EDEDED", "Not scheduled yet"),
}

# (title, labels, body).  Bodies end with an acceptance check.
NEXT = [
 ("1. Verify Lucid 25K and 50K Flex rules", ["research"], """
- [ ] Lock: peak that freezes the floor, and where it freezes
- [ ] Payout conditions: profit needed, 50% rule, cap, buffer
- [ ] Five profitable days per cycle: minimum $ per day
- [ ] Does an intraday touch of the floor liquidate, or only the close?
- [ ] Maximum contracts (25K and 50K)
- [ ] Payout count cap or move to a live account
- [ ] Limit on number of funded accounts
- [ ] Opposite positions across accounts allowed?
- [ ] Payout processing time and fees

**Check:** each answer recorded in `docs/Rules/` with source and date."""),
 ("2. Export v2: MAE, trading day, entry time, direction, exit time", ["data"], """
- [ ] MAE per trade
- [ ] Trading day, entry time, direction per trade
- [ ] Exit time per trade at each RR
- [ ] Print first-signal-of-day vs later-signal edge

**Check:** R arrays identical to `ledgers.pkl`; all baseline numbers unchanged."""),
 ("3. Commission and MAE in both stages", ["model"], """
- [ ] Commission per round turn, converted to R, subtracted in `collapse_to_coin`
- [ ] MAE breach check in both transitions, behind a switch
- [ ] Rerun risk sweep and out-of-sample tests

**Check:** commission 0 and MAE off reproduce the baseline; with them on, no probability goes up."""),
 ("4. Clustering test and bootstrap ranges", ["stats"], """
- [ ] Compare longest losing streaks (by trade and by day) with shuffled data
- [ ] Bootstrap the 884 trades, re-solve, report ranges for P(first payout), payouts, ROI
- [ ] Put out-of-sample and bootstrap values in the export

**Check:** decision recorded: is the independence assumption acceptable?"""),
 ("5. Day-level model: several trades per day, end-of-day trailing", ["model"], """
- [ ] One step per trading day, using the day's trades and intraday low
- [ ] Floor updates on the close only
- [ ] Compare 1 vs 2 vs all trades per account per day, and signal allocation across accounts

**Check:** with one trade per day, matches the per-trade model."""),
 ("6. Config refactor, then the 50K account", ["model"], """
- [ ] Move rule constants into one config object
- [ ] Sparse matrices or a $100 grid for the larger state space
- [ ] Run the full pipeline for 50K

**Check:** the 25K config reproduces the baseline exactly."""),
 ("7. Five profitable days rule (Flex)", ["model"], """
- [ ] Add profitable days this cycle (0 to 5) to the post-lock state
- [ ] Payout only at the target with 5 profitable days

**Check:** requirement set to 0 days reproduces the baseline."""),
 ("8. Withdrawal decision", ["model"], """
- [ ] Sweep the push-to level ($27,100, $27,600, $28,100, ...)
- [ ] Compare payouts, days, ROI

**Check:** the $27,100 row reproduces current results."""),
 ("9. Dashboard upgrades", ["dashboard"], """
- [ ] Per-account risk or contract column in My accounts
- [ ] Interpolate between grid cells
- [ ] Profitable-days counter; payout ready only when the rule is met
- [ ] Provenance and validation panel
- [ ] Grey out risks above the contract cap and fleets above the account limit
- [ ] Show out-of-sample and bootstrap ranges

**Check:** My accounts matches the Python model in three hand-picked states."""),
]

BACKLOG = [
 ("Objective: money per unit of time (Dinkelbach)", ["model", "backlog"]),
 ("Risk chosen per state", ["model", "backlog"]),
 ("Stop-size sweep with out-of-sample check", ["model", "backlog"]),
 ("Model the eval stage with the pre-lock solver", ["model", "backlog"]),
 ("Walk-forward over rolling windows", ["stats", "backlog"]),
 ("Earlier history, if the edge holds", ["data", "backlog"]),
 ("Compare real trade logs with the backtest", ["data", "backlog"]),
 ("Cash-flow view: fees early, payouts later, lowest bankroll", ["model", "backlog"]),
 ("Rotation replay with real exit times and the opposite-position rule", ["model", "backlog"]),
 ("Reference dictionary solver with an equivalence test", ["housekeeping", "backlog"]),
 ("Cleanup: SHOW_TABLES block, docstrings, leftover comments", ["housekeeping"]),
]


def gh(*args):
    out = subprocess.run(["gh", *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"gh {' '.join(args[:2])} failed:\n{out.stderr}")
    return out.stdout.strip()


def new_issue(title, labels, body):
    url = gh("issue", "create", "--title", title, "--body", body.strip(), "--label", ",".join(labels))
    number = int(re.search(r"/issues/(\d+)", url).group(1))
    print(f"  #{number}  {title}")
    return number


print("labels...")
for name, (color, desc) in LABELS.items():
    gh("label", "create", name, "--color", color, "--description", desc, "--force")

print("next-up issues...")
next_numbers = [(new_issue(t, l, b), t) for t, l, b in NEXT]

print("backlog issues...")
backlog_numbers = [(new_issue(t, l, "Not scheduled yet."), t) for t, l in BACKLOG]

tracking = "Progress for the funded account model. Boxes tick themselves when an issue is closed.\n\n## Next up\n"
tracking += "\n".join(f"- [ ] #{n}" for n, _ in next_numbers)
tracking += "\n\n## Backlog\n" + "\n".join(f"- [ ] #{n}" for n, _ in backlog_numbers)
print("tracking issue...")
t = new_issue("Roadmap (tracking)", ["research"], tracking)
gh("issue", "pin", str(t))
print(f"\nDone. Pinned tracking issue: #{t}")
