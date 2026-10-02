"""Config.py — prop-firm plan rules. Dollar levels are measured from the start balance.
Rules checked October 2026; confirm in each firm's dashboard before relying on them."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    name: str
    start: float                 # starting balance
    dd_limit: float              # used: trailing drawdown before the lock
    lock_peak: float             # used: peak (above start) that freezes the floor
    locked_floor: float          # used: where the floor freezes (above start)
    payout_target: float         # used: balance (above start) at which you withdraw
    payout_pct: float            # share of profit that can be withdrawn
    payout_cap: float            # largest single payout
    split: float                 # used: your share of each payout
    eval_fee: float              # used: cost per evaluation attempt
    activation_fee: float        # used: one-time fee when an eval passes
    max_payouts: int | None      # used: payouts before the account leaves the model (None = no limit)
    min_day_profit: float        # not yet modeled: profit needed for a qualifying day
    days_required: int           # not yet modeled: qualifying days per payout cycle
    consistency: float | None    # not yet modeled: best day / cycle profit cap (None = no rule)
    dx: float = 50.0             # grid step; keep about 20 steps across the drawdown

    @property
    def payout(self):
        """Amount withdrawn at payout_target: a share of profit, up to the cap."""
        return min(self.payout_cap, self.payout_pct * self.payout_target)


PLANS = {
    "lucid25flex": Plan("Lucid 25K Flex", start=25_000, dd_limit=1_000, lock_peak=1_100,
        locked_floor=100, payout_target=2_100, payout_pct=0.5, payout_cap=1_000, split=0.90,
        eval_fee=60, activation_fee=0, max_payouts=5, min_day_profit=100, days_required=5,
        consistency=None),
    "lucid50flex": Plan("Lucid 50K Flex", start=50_000, dd_limit=2_000, lock_peak=2_100,
        locked_floor=100, payout_target=4_100, payout_pct=0.5, payout_cap=2_000, split=0.90,
        eval_fee=140, activation_fee=0, max_payouts=5, min_day_profit=150, days_required=5,
        consistency=None, dx=100),
    "apex25eod": Plan("Apex 25K EOD", start=25_000, dd_limit=1_000, lock_peak=1_100,
        locked_floor=100, payout_target=2_100, payout_pct=0.5, payout_cap=1_000, split=1.00,   # VERIFY cap, split
        eval_fee=199, activation_fee=99, max_payouts=6, min_day_profit=100, days_required=5,     # VERIFY eval_fee
        consistency=0.50),
    "topstep50": Plan("Topstep 50K", start=50_000, dd_limit=2_000, lock_peak=2_000,
        locked_floor=0, payout_target=4_000, payout_pct=0.5, payout_cap=2_000, split=0.90,
        eval_fee=99, activation_fee=0, max_payouts=None, min_day_profit=150, days_required=5,
        consistency=None, dx=100),
    "topstep150": Plan("Topstep 150K", start=150_000, dd_limit=4_500, lock_peak=4_500,
        locked_floor=0, payout_target=10_000, payout_pct=0.5, payout_cap=5_000, split=0.90,
        eval_fee=199, activation_fee=149, max_payouts=None, min_day_profit=150, days_required=5,
        consistency=None, dx=225),
    "tradeify50flex": Plan("Tradeify 50K Select Flex", start=50_000, dd_limit=2_000, lock_peak=2_100,
        locked_floor=100, payout_target=4_100, payout_pct=0.5, payout_cap=3_000, split=0.90,
        eval_fee=159, activation_fee=0, max_payouts=5, min_day_profit=150, days_required=5,
        consistency=None, dx=100),
}

PLAN = PLANS[os.environ.get("PLAN", "lucid25flex")]