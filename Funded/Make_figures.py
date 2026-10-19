"""make_figures.py: figures and the validation table for the README.

Run from the Funded folder:  python make_figures.py
Output goes to figures/ in the repo root:
    account_path.png     one simulated account: balance, floor, lock and payout
    policy_map.png       best RR in each pre-lock state
    value_map.png        P(first payout) from each pre-lock state
    postlock_value.png   P(payout) against balance after the lock
    risk_sweep.png       P(first payout) and trades to resolution at each risk level
    oos_result.png       out-of-sample P(first payout), state-based policy vs fixed RR
    validation_table.md  fair-coin checks and the out-of-sample table

PLAN and USE_MAE are read from the environment like the other scripts. Risk per trade is
the plan default (a quarter of the drawdown). The figures have no titles; the README
captions describe them.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

import Postlock as post
import Prelock as pre
import Sweep
from Config import PLAN

# ====================================================================================
# Setup: output folder, settings and plot style
# ====================================================================================
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)

RISK = post.risk
START = PLAN.start                   # balances in the model are relative to the start
I_ENTRY = post.cell_index(post.Entry, post.Floor, post.Dx)
SETTING = f"{PLAN.name}, ${RISK:g} per trade, MAE {'on' if post.USE_MAE else 'off'}"

PROVISIONAL = 0.7 * PLAN.dd_limit    # sweep above this risk is shaded (see README, Results)
NEAR_TIE = 0.001                     # policy map: grey out states where the top two RRs are this close

BLUE, ORANGE, GREEN, GREY = "tab:blue", "tab:orange", "tab:green", "0.55"
plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.9",
    "grid.linewidth": 0.8,
    "legend.frameon": False,
})


# ====================================================================================
# Helpers
# ====================================================================================
def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def half_coins(sl):
    """Coins from the trades in slice sl, with MAE if USE_MAE=1 (same as run_prelock)."""
    mae = post.MAE if post.USE_MAE else None
    return {rho: post.collapse_to_coin(R[sl], None if mae is None else mae[rho][sl])
            for rho, R in post.RETURN_ARRAYS.items()}


# ====================================================================================
# Solve once at the plan's default risk (figures 1-4 use this)
# ====================================================================================
coins = post.Coins
rrs = sorted(coins)
V_post, pol_post = post.solve_post(coins, RISK)
V_pre, pol_pre = pre.solve_pre_fast(coins, V_post, risk=RISK)

# imshow centres each cell on its grid point, so the edges are half a step outside
h = pre.Dx / 2
extent = [pre.y_grid[0] - h, pre.y_grid[-1] + h, pre.m_grid[0] - h, pre.m_grid[-1] + h]


# ====================================================================================
# 1. Pre-lock policy map
# ====================================================================================
# Where the best and second-best RR are nearly tied the colour means little, so those
# states are drawn in grey and only the states where the choice matters keep a colour.
Q = pre.q_values(coins, V_post, V_pre, risk=RISK)          # shape (RR, peak, drawdown)
best_two = -np.sort(-Q, axis=0)[:2]
tied = (best_two[0] - best_two[1]) < NEAR_TIE

rr_idx = np.vectorize(lambda r: rrs.index(float(r)))(pol_pre).astype(float)
cmap = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.3, 1.0, len(rrs))))
cmap.set_bad("0.88")
norm = BoundaryNorm(np.arange(len(rrs) + 1) - 0.5, len(rrs))

fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.grid(False)
im = ax.imshow(np.ma.masked_where(tied, rr_idx), origin="lower", aspect="auto",
               extent=extent, cmap=cmap, norm=norm)
cb = fig.colorbar(im, ax=ax, ticks=range(len(rrs)), fraction=0.046, pad=0.03)
cb.ax.set_yticklabels([f"{r:g}" for r in rrs])
cb.set_label("best RR")
ax.set_xlabel("drawdown below peak ($)")
ax.set_ylabel("peak above start ($)")
print(f"policy map: {tied.mean():.0%} of states are near-ties (grey)")
save(fig, "policy_map.png")


# ====================================================================================
# 2. Pre-lock value map
# ====================================================================================
fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.grid(False)
im = ax.imshow(V_pre, origin="lower", aspect="auto", extent=extent, cmap="Oranges", vmin=0, vmax=1)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("P(first payout)")
ax.set_xlabel("drawdown below peak ($)")
ax.set_ylabel("peak above start ($)")
save(fig, "value_map.png")


# ====================================================================================
# 3. Post-lock value against balance
# ====================================================================================
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.plot(START + post.grid, V_post, color=BLUE, lw=1.8)
ax.plot(START + post.Entry, V_post[I_ENTRY], "o", color=BLUE, ms=5)
ax.annotate(f"at the lock: {V_post[I_ENTRY]:.3f}", (START + post.Entry, V_post[I_ENTRY]),
            xytext=(8, -14), textcoords="offset points", fontsize=9)
ax.set_ylim(0, 1)
ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
ax.set_xlabel("balance")
ax.set_ylabel("P(payout before floor)")
save(fig, "postlock_value.png")


# ====================================================================================
# 4. One simulated account
# ====================================================================================
def simulate_account(seed, max_trades=200):
    """Trade one account with the solved policy, drawing each result from the ledger.
    Returns the balance and floor after every trade (relative to start), the trade on
    which it locked, and how it ended."""
    rng = np.random.default_rng(seed)
    x = peak = 0.0
    floor = -PLAN.dd_limit
    locked_at, xs, floors = None, [0.0], [floor]
    for t in range(1, max_trades + 1):
        if locked_at is None:
            j = min(int(peak // pre.Dx), len(pre.m_grid) - 1)
            k = min(int((peak - x) // pre.Dx), len(pre.y_grid) - 1)
            rr = float(pol_pre[j, k])
        else:
            i = int(np.clip(post.cell_index(x, post.Floor, post.Dx), 0, len(post.grid) - 1))
            rr = float(pol_post[i])
        coin = coins[rr]
        _, R, *rest = coin[rng.choice(len(coin), p=[c[0] for c in coin])]
        mae = rest[0] if rest else 0.0
        if x - mae * RISK <= floor:          # dipped to the floor before the trade closed
            xs.append(floor)
            floors.append(floor)
            return np.array(xs), np.array(floors), locked_at, "breach"
        x += R * RISK
        if locked_at is None:
            peak = max(peak, x)
            floor = peak - PLAN.dd_limit
            if peak >= PLAN.lock_peak:
                locked_at, floor = t, PLAN.locked_floor
        xs.append(x)
        floors.append(floor)
        if x <= floor:
            return np.array(xs), np.array(floors), locked_at, "breach"
        if locked_at is not None and x >= PLAN.payout_target:
            return np.array(xs), np.array(floors), locked_at, "payout"
    return np.array(xs), np.array(floors), locked_at, "open"


# Use the first seed that shows a drawdown, a lock and a payout in under 40 trades.
# It is an example of the rules at work, not a typical path.
for seed in range(1000):
    xs, floors, locked_at, outcome = simulate_account(seed)
    dipped = (np.maximum.accumulate(xs) - xs).max() > 0.3 * PLAN.dd_limit
    if outcome == "payout" and locked_at and 8 <= len(xs) <= 40 and dipped:
        break
print(f"account path: seed {seed}, locked on trade {locked_at}, paid on trade {len(xs) - 1}")

t = np.arange(len(xs))
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(t, START + xs, color=BLUE, lw=1.6, marker="o", ms=3, label="balance")
ax.step(t, START + floors, where="post", color=ORANGE, lw=1.6, label="floor")
ax.axhline(START + PLAN.lock_peak, color=GREY, lw=0.9, ls="--")
ax.axhline(START + PLAN.payout_target, color=GREY, lw=0.9, ls="--")
ax.text(0.2, START + PLAN.lock_peak, "lock", va="bottom", color="0.35", fontsize=9)
ax.text(0.2, START + PLAN.payout_target, "payout", va="bottom", color="0.35", fontsize=9)
ax.annotate("floor locks", (locked_at, START + PLAN.locked_floor), xytext=(6, -16),
            textcoords="offset points", fontsize=9)
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
ax.set_xlabel("trade")
ax.set_ylabel("balance")
ax.legend(loc="lower right")
save(fig, "account_path.png")


# ====================================================================================
# 5. Risk sweep (probability and time get separate panels)
# ====================================================================================
risks = Sweep.default_risks()
sweep = Sweep.run_sweep(risks)

fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.2))
left.plot(sweep.risk, sweep.p_pay1, color=BLUE, lw=1.6, marker="o", ms=3)
left.set_ylim(0, max(0.6, sweep.p_pay1.max() * 1.1))
left.set_ylabel("P(first payout)")
right.plot(sweep.risk, sweep.trades_first, color=BLUE, lw=1.6, marker="o", ms=3)
right.set_yscale("log")
right.set_ylabel("trades to first payout or breach")
for ax in (left, right):
    ax.set_xlabel("risk per trade ($)")
    ax.axvspan(PROVISIONAL, risks[-1], color="0.95", zorder=0)
    ax.text((PROVISIONAL + risks[-1]) / 2, 0.95, "provisional", transform=ax.get_xaxis_transform(),
            ha="center", va="top", color="0.45", fontsize=9)
fig.tight_layout()
save(fig, "risk_sweep.png")


# ====================================================================================
# 6. Validation table
# ====================================================================================
# Fair coin, one grid step up or down: post-lock is the gambler's ruin (a*b expected
# trades), and pre-lock locks with probability (d/(d+1))^k.
fair = {1.0: [(0.5, 1.0), (0.5, -1.0)]}
T_fair = post.expected_trades_post(fair, {i: 1.0 for i in range(len(post.grid))}, post.Dx)
a = round((post.Entry - post.Floor) / post.Dx)
b = round((post.Target - post.Entry) / post.Dx)
V_fair, _ = pre.solve_pre_fast(fair, None, risk=post.Dx, lock_one=True, tol=1e-12)
d = round(pre.DD_limit / post.Dx)
k = round(pre.Lock_peak / post.Dx)

rows = [
    f"Settings: {SETTING}, {len(post.RETURN_ARRAYS[1.0])} trades.",
    "",
    "| Check | Solver | Exact |",
    "|---|---|---|",
    f"| Post-lock fair coin: expected trades from the lock entry | {T_fair[I_ENTRY]:.2f} | {a * b} |",
    f"| Pre-lock fair coin: P(lock) | {V_fair[0, 0]:.6f} | {(d / (d + 1)) ** k:.6f} |",
    "",
    "| Built on | Scored on | Optimal policy | Always RR 1.0 | Always RR 5.0 |",
    "|---|---|---|---|---|",
]

# Build on one half of the trades, score on the other. Post-lock is solved separately on
# each half so the scoring half never informs the policy.
n_half = len(post.RETURN_ARRAYS[1.0]) // 2
halves = {"first half": half_coins(slice(None, n_half)),
          "second half": half_coins(slice(n_half, None))}
v_post_half = {name: post.solve_post(c, RISK)[0] for name, c in halves.items()}

oos = {}
for built, scored in [("first half", "second half"), ("second half", "first half")]:
    _, policy = pre.solve_pre_fast(halves[built], v_post_half[built], risk=RISK)

    def score(pol):
        return pre.evaluate_policy_pre_fast(pol, halves[scored], v_post_half[scored], risk=RISK)[0, 0]

    vals = (score(policy), score(pre.fixed_policy(1.0)), score(pre.fixed_policy(5.0)))
    oos[f"built on {built}\nscored on {scored}"] = vals
    rows.append(f"| {built.capitalize()} | {scored.capitalize()} | "
                f"{vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} |")

path = os.path.join(OUT, "validation_table.md")
with open(path, "w") as f:
    f.write("\n".join(rows) + "\n")
print("wrote", path)

# ====================================================================================
# 7. Out-of-sample chart
# ====================================================================================
fig, ax = plt.subplots(figsize=(6.4, 2.6))
names = [("state-based policy", BLUE), ("always RR 1.0", ORANGE), ("always RR 5.0", GREEN)]
for row, (label, vals) in enumerate(oos.items()):
    ax.plot([min(vals), max(vals)], [row, row], color="0.85", lw=1.2, zorder=1)
    for (name, colour), v in zip(names, vals):
        ax.plot(v, row, "o", color=colour, ms=7, zorder=2, label=name if row == 0 else None)
    ax.annotate(f"{vals[0]:.3f}", (vals[0], row), xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=9)
ax.set_yticks(range(len(oos)), list(oos))
ax.set_ylim(len(oos) - 0.4, -0.6)
ax.grid(axis="y", visible=False)
ax.set_xlabel("P(first payout) on the held-out half")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9)
save(fig, "oos_result.png")