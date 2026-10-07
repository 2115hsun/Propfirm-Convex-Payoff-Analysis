import pickle
import numpy as np 

N_SHUFFLES = 5000 # numbers of random reorderings  
RRS = [1.0, 1.5, 2.0, 2.5, 3.0] # which RR to target 
rng = np.random.default_rng(0) # random number generator for reproducibility


with open("ledgers.pkl", "rb") as f:
    R_ALL = pickle.load(f)

R = np.asarray(R_ALL[1.0], float)

def longest_losing_streak(R):
    best = run = 0
    for r in R:
        run = run + 1 if r < 0 else 0
        best = max(best, run)
    return best

def n_runs(R):
    """Number of win/loss runs"""
    s = np.sign(R)
    s=s[s != 0]  # remove zeros
    return 1 + int(np.sum(s[1:] != s[:-1]))  # count sign changes

def lag1(R):
    """Correlation between the each trade and the next"""
    return float(np.corrcoef(R[:-1], R[1:])[0, 1])

def worst_window(R, k=10):
    """Worst total over any 10 trades in a row: the stretch most likely to breach an account."""
    c = np.concatenate([[0.0], np.cumsum(R)])
    return float((c[k:] - c[:-k]).min())

STATS = {
    "longest losing streak":  (longest_losing_streak, "higher"),
    "number of runs":         (n_runs, "lower"),
    "lag-1 autocorrelation":  (lag1, "higher"),
    "worst 10-trade sum (R)": (worst_window, "lower"),
}

for rho in RRS:
    R = np.asarray(R_ALL[rho], float)
    print(f"\nRR {rho}: {len(R)} trades, {np.mean(R < 0):.0%} losers")
    shuffles = [rng.permutation(R) for _ in range(N_SHUFFLES)]
    for name, (fn, direction) in STATS.items():
        real = fn(R)
        sims = np.array([fn(x) for x in shuffles])
        p = np.mean(sims >= real) if direction == "higher" else np.mean(sims <= real)
        print(f"  {name:<24} real {real:8.3f}   shuffled median {np.median(sims):8.3f}   p = {p:.3f}")