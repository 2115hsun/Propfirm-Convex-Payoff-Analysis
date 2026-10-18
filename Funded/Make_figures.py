"""make_figures.py: the README figures and the validation table.

Run from the Funded folder:  python make_figures.py
Writes to figures/ at the repo root (next to the Funded folder):
    policy_map.png      best RR target in each pre-lock state
    value_map.png       P(first payout) from each pre-lock state
    postlock_value.png  P(payout) against balance after the lock
    risk_sweep.png      P(first payout) and expected trades at each risk level
    validation_table.md fair-coin checks and the out-of-sample table

The plan and the intra-trade breach check come from PLAN and USE_MAE, as in every other
script. Figures use the plan's default risk per trade (a quarter of the drawdown).
"""
import os

import matplotlib
matplotlib.use("Agg")                      # write files only; no window opens
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

import Postlock as post
import Prelock as pre
import Sweep
from Config import PLAN

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)

RISK = post.risk
START = PLAN.start                         # added to every balance so axes show account dollars
PROVISIONAL = 0.7 * PLAN.dd_limit          # sweep results above this risk are shaded (see README)
I_ENTRY = post.cell_index(post.Entry, post.Floor, post.Dx)
SETTING = f"{PLAN.name}, ${RISK:g} per trade, MAE {'on' if post.USE_MAE else 'off'}"

# Colors. RR is ordered, so the policy map uses one blue ramp from light (small target)
# to dark (large target). The value map uses orange so the two heatmaps are not confused.
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LINE = "#2a78d6"
BLUE = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ORANGE = LinearSegmentedColormap.from_list("orange", ["#fdeee6", "#f4a37f", "#eb6834", "#a8401a", "#5e220b"])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def half_coins(sl):
    """Coins built from the trades in slice `sl`, with MAE when USE_MAE=1 (as in run_prelock)."""
    mae = post.MAE if post.USE_MAE else None
    return {rho: post.collapse_to_coin(R[sl], None if mae is None else mae[rho][sl])
            for rho, R in post.RETURN_ARRAYS.items()}


# ====================================================================================
# Solve once at the headline risk; figures 1-3 all use these
# ====================================================================================
coins = post.Coins
rrs = sorted(coins)
V_post, pol_post = post.solve_post(coins, RISK)
V_pre, pol_pre = pre.solve_pre_fast(coins, V_post, risk=RISK)

# imshow draws each value as a block centred on its grid point, so the image edges sit
# half a step beyond the first and last points
half_step = pre.Dx / 2
extent = [pre.y_grid[0] - half_step, pre.y_grid[-1] + half_step,
          pre.m_grid[0] - half_step, pre.m_grid[-1] + half_step]

# ====================================================================================
# 1. pre-lock policy map 
# ====================================================================================
fig, ax = plt.subplots(figsize=(6.4, 4.6))
rr_index = np.vectorize(lambda r: rrs.index(float(r)))(pol_pre)        # RR -> position in rrs
cmap = ListedColormap(BLUE[:len(rrs)])
norm = BoundaryNorm(np.arange(len(rrs) + 1) - 0.5, len(rrs))         # one color band per RR
im = ax.imshow(rr_index, origin="lower", aspect="auto", extent=extent, cmap=cmap, norm=norm)
cb = fig.colorbar(im, ax=ax, ticks=range(len(rrs)), fraction=0.046, pad=0.03)
cb.ax.set_yticklabels([f"{r:g}" for r in rrs])
cb.set_label("best RR target")
cb.outline.set_visible(False)
ax.set_xlabel("drawdown below peak ($)")
ax.set_ylabel("peak above start ($)")
ax.set_title(f"Pre-lock policy at ${RISK:g} per trade")
save(fig, "policy_map.png")

# ====================================================================================
# 2. pre-lock value map 
# ====================================================================================
fig, ax = plt.subplots(figsize=(6.4, 4.6))
im = ax.imshow(V_pre, origin="lower", aspect="auto", extent=extent, cmap=ORANGE, vmin=0, vmax=1)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("P(first payout)")
cb.outline.set_visible(False)
ax.set_xlabel("drawdown below peak ($)")
ax.set_ylabel("peak above start ($)")
ax.set_title(f"Pre-lock value at ${RISK:g} per trade")
ax.annotate(f"start: {V_pre[0, 0]:.3f}", xy=(0, 0),
            xytext=(0.25 * PLAN.dd_limit, 0.12 * PLAN.lock_peak), color=INK, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
save(fig, "value_map.png")

# ====================================================================================
# 3. post-lock value against balance 
# ====================================================================================
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(START + post.grid, V_post, color=LINE, lw=2)
ax.axvline(START + post.Entry, color=MUTED, lw=1, ls=(0, (4, 3)))
ax.plot(START + post.Entry, V_post[I_ENTRY], "o", ms=7, color=LINE, mec=SURF, mew=2)
ax.annotate(f"lock entry: {V_post[I_ENTRY]:.3f}", xy=(START + post.Entry, V_post[I_ENTRY]),
            xytext=(12, -18), textcoords="offset points", color=INK, fontsize=9)
ax.set_ylim(0, 1)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_xlabel("balance ($)")
ax.set_ylabel("P(payout before floor)")
ax.set_title(f"Post-lock value at ${RISK:g} per trade")
save(fig, "postlock_value.png")

# ====================================================================================
# 4. Risk sweep: probability and time in two panels, since they have different units
# ====================================================================================
risks = Sweep.default_risks()
sweep = Sweep.run_sweep(risks)

fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.6))
left.plot(sweep.risk, sweep.p_pay1, color=LINE, lw=2, marker="o", ms=4)
left.set_ylim(0, max(0.6, sweep.p_pay1.max() * 1.1))
left.set_ylabel("P(first payout)")
left.set_title("Payout probability by position size")

right.plot(sweep.risk, sweep.trades_first, color=LINE, lw=2, marker="o", ms=4)
right.set_yscale("log")                    # trades span a wide range; a log scale keeps both ends readable
right.set_ylabel("trades to first payout or breach")
right.set_title("Time by position size")

for ax in (left, right):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_xlabel("risk per trade ($)")
    ax.axvspan(PROVISIONAL, risks[-1], color="#f0efec", zorder=0)
    # x in data units, y as a fraction of the panel height, so the label sits near the
    # top on both the linear and the log axis
    ax.text((PROVISIONAL + risks[-1]) / 2, 0.94, "provisional", transform=ax.get_xaxis_transform(),
            ha="center", color=MUTED, fontsize=9)
save(fig, "risk_sweep.png")

# ====================================================================================
# 5. Validation table
# ====================================================================================
# Fair coin: one grid step up or down with probability 1/2, so both stages have exact answers.
fair = {1.0: [(0.5, 1.0), (0.5, -1.0)]}

# Post-lock is the gambler's ruin: a steps to the floor and b to the target give a*b trades.
T_fair = post.expected_trades_post(fair, {i: 1.0 for i in range(len(post.grid))}, post.Dx)
a = round((post.Entry - post.Floor) / post.Dx)
b = round((post.Target - post.Entry) / post.Dx)

# Pre-lock: each new peak is a race of one step up against d steps down, won with
# probability d/(d+1); locking takes k wins in a row. Counting a lock as 1 gives P(lock).
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

# Out of sample: build the policy on one half of the trades and score it on the other.
# Post-lock is solved on each half separately, so nothing from the scoring half leaks in.
n_half = len(post.RETURN_ARRAYS[1.0]) // 2
halves = {"First half": half_coins(slice(None, n_half)),
          "Second half": half_coins(slice(n_half, None))}
v_post_half = {name: post.solve_post(c, RISK)[0] for name, c in halves.items()}

for built, scored in [("First half", "Second half"), ("Second half", "First half")]:
    _, policy = pre.solve_pre_fast(halves[built], v_post_half[built], risk=RISK)

    def score(pol):
        return pre.evaluate_policy_pre_fast(pol, halves[scored], v_post_half[scored], risk=RISK)[0, 0]

    rows.append(f"| {built} | {scored} | {score(policy):.3f} | "
                f"{score(pre.fixed_policy(1.0)):.3f} | {score(pre.fixed_policy(5.0)):.3f} |")

path = os.path.join(OUT, "validation_table.md")
with open(path, "w") as f:
    f.write("\n".join(rows) + "\n")
print("wrote", path)