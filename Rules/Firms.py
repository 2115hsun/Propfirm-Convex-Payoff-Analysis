"""
Firms.py: eval rule configs per prop firm ($50K accounts, plus Lucid 25K).

Derived from the firms' rule .md files. Where those were unclear about
trailing vs static drawdown or hard vs soft daily-loss behaviour, the value is
a best guess marked `# VERIFY`. These settings move pass rates a lot, so
correct them once confirmed.

Where the firms differ:
  1. drawdown TYPE  : static floor vs EOD-trailing vs intraday-trailing
  2. trail_lock     : the equity level at which a trailing DD stops moving up
                      (e.g. 50000 = floor locks at breakeven once you're up)
  3. daily loss     : none / soft (ends the day) / hard (fails the account)
  4. consistency    : best single day must be <= pct of total profit
  5. min_days       : minimum number of distinct trading days to pass
"""

from dataclasses import dataclass
from enum import Enum


class DrawdownType(str, Enum):
    STATIC = "static"                 # fixed floor, never moves
    EOD_TRAIL = "eod_trail"           # trails highest end-of-day balance
    INTRADAY_TRAIL = "intraday_trail" # trails highest intraday equity peak (harshest)


class DLLBehavior(str, Enum):
    NONE = "none"
    SOFT_LOCK = "soft"                # hitting DLL just ends the trading day
    HARD_FAIL = "hard"                # hitting DLL fails the account


@dataclass(frozen=True)
class FirmConfig:
    key: str
    name: str
    start_balance: float = 50_000.0
    profit_target: float = 3_000.0            # reach start + target => $53,000
    max_drawdown: float = 2_000.0             # distance from peak/floor => $48,000 initially
    dd_type: DrawdownType = DrawdownType.STATIC
    trail_lock: float | None = None           # equity level trailing stops at (None = never locks)
    daily_loss_limit: float | None = None
    dll_behavior: DLLBehavior = DLLBehavior.NONE
    consistency_pct: float | None = None      # best day <= pct * total profit (None = no rule)
    min_days: int = 1

    @property
    def target_equity(self) -> float:
        return self.start_balance + self.profit_target

    @property
    def initial_floor(self) -> float:
        return self.start_balance - self.max_drawdown


# ----------------------------------------------------------------------------
# The 5 firms, $50K accounts. Some offer multiple eval paths -> separate entries.
# ----------------------------------------------------------------------------
FIRMS: dict[str, FirmConfig] = {

    # Apex, EOD trailing path. Trailing DD locks at the starting balance once
    # the account is up ~$2k. DLL is a hard rule in eval.
    "apex_eod": FirmConfig(
        key="apex_eod", name="Apex (EOD Trailing)",
        profit_target=3_000, max_drawdown=2_000,
        dd_type=DrawdownType.EOD_TRAIL, trail_lock=50_000,  # VERIFY lock point
        daily_loss_limit=1_000, dll_behavior=DLLBehavior.HARD_FAIL,  # VERIFY hard
        consistency_pct=None, min_days=1,
    ),

    # Apex, Intraday trailing path. No DLL. Harshest barrier.
    "apex_intraday": FirmConfig(
        key="apex_intraday", name="Apex (Intraday Trailing)",
        profit_target=3_000, max_drawdown=2_000,
        dd_type=DrawdownType.INTRADAY_TRAIL, trail_lock=50_000,  # VERIFY lock
        daily_loss_limit=None, dll_behavior=DLLBehavior.NONE,
        consistency_pct=None, min_days=1,
    ),

    # Lucid, EOD drawdown, DLL $1,200. Trailing-vs-static not explicit in docs.
    "lucid": FirmConfig(
        key="lucid", name="Lucid (LucidPro)",
        profit_target=3_000, max_drawdown=2_000,
        dd_type=DrawdownType.EOD_TRAIL, trail_lock=50_000,  # VERIFY (could be STATIC)
        daily_loss_limit=1_200, dll_behavior=DLLBehavior.HARD_FAIL,  # VERIFY hard
        consistency_pct=None, min_days=1,  # consistency is a payout rule, not an eval rule
    ),

    # Tradeify, Select eval. EOD DD, no daily loss limit in eval,
    # 40% consistency, min 3 trading days.
    "tradeify_select": FirmConfig(
        key="tradeify_select", name="Tradeify (Select)",
        profit_target=3_000, max_drawdown=2_000,
        dd_type=DrawdownType.EOD_TRAIL, trail_lock=50_000,  # VERIFY lock
        daily_loss_limit=None, dll_behavior=DLLBehavior.NONE,
        consistency_pct=0.40, min_days=3,
    ),

    # Tradeify, Growth eval. EOD trailing DD, soft DLL $1,250 (ends day, no fail),
    # no consistency in eval, can pass in 1 day.
    "tradeify_growth": FirmConfig(
        key="tradeify_growth", name="Tradeify (Growth)",
        profit_target=3_000, max_drawdown=2_000,
        dd_type=DrawdownType.EOD_TRAIL, trail_lock=50_000,  # VERIFY lock
        daily_loss_limit=1_250, dll_behavior=DLLBehavior.SOFT_LOCK,
        consistency_pct=None, min_days=1,
    ),

    # MyFundedFutures, Rapid. Intraday trailing DD, no DLL,
    # 50% consistency in eval, min 2 days.
    "mffu_rapid": FirmConfig(
        key="mffu_rapid", name="MyFundedFutures (Rapid)",
        profit_target=3_000, max_drawdown=2_000,
        dd_type=DrawdownType.INTRADAY_TRAIL, trail_lock=None,  # VERIFY (may lock at BE)
        daily_loss_limit=None, dll_behavior=DLLBehavior.NONE,
        consistency_pct=0.50, min_days=2,
    ),

    # Topstep, Trading Combine. Trailing MLL ($2,000), no DLL,
    # consistency = best day <= 50% of total profit, min 2 days.
    "topstep": FirmConfig(
        key="topstep", name="Topstep (Combine)",
        profit_target=3_000, max_drawdown=2_000,
        dd_type=DrawdownType.EOD_TRAIL, trail_lock=50_000,  # VERIFY (EOD vs intraday trail)
        daily_loss_limit=None, dll_behavior=DLLBehavior.NONE,
        consistency_pct=0.50, min_days=2,
    ),

    # Lucid 25K accounts (verified online, Aug 2026)
    "lucid_pro_25k": FirmConfig(
        key="lucid_pro_25k",
        name="Lucid Pro 25K",
        start_balance=25_000.0,
        profit_target=1_250.0,               # target = $26,250
        max_drawdown=1_000.0,                # floor starts at $24,000, EOD-trails
        dd_type=DrawdownType.EOD_TRAIL,
        trail_lock=25_000.0,                 # locks (static) once floor reaches start balance
        daily_loss_limit=None,               # no DLL on the 25K
        dll_behavior=DLLBehavior.NONE,
        consistency_pct=None,                # LucidPro eval: no consistency rule
        min_days=1,                          # 1-day pass allowed
    ),
    "lucid_flex_25k": FirmConfig(
        key="lucid_flex_25k",
        name="Lucid Flex 25K",
        start_balance=25_000.0,
        profit_target=1_250.0,
        max_drawdown=1_000.0,
        dd_type=DrawdownType.EOD_TRAIL,
        trail_lock=25_000.0,                 # EOD trail, converts to static at start+target
        daily_loss_limit=None,               # Flex: no DLL
        dll_behavior=DLLBehavior.NONE,
        consistency_pct=0.50,                # Flex eval: 50% consistency rule
        min_days=1,                          # Flex: min 1 trading day
    ),
}


def get_firm(key: str) -> FirmConfig:
    if key not in FIRMS:
        raise KeyError(f"Unknown firm '{key}'. Options: {list(FIRMS)}")
    return FIRMS[key]