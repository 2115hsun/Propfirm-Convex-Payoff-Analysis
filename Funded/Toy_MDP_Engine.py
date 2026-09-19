States = [0, 1, 2, 3, 4]
Interior = [1, 2, 3]
Absorbing = [0, 4]

Win_state = 4
Lose_state = 0 

#A: action spaces
Actions = ["Tight", "Wide"]

Gamma = 1.0 

Boundary_value = {Lose_state: 0, Win_state: 1}

def reward (state, action):
    return 0.0

Moves = {
    "Tight": [(0.7, +1), (0.3, -1)],
    "Wide": [(0.4, +2), (0.6, -1)]
}

def _clamp(landing):
    if landing >= Win_state:
        return Win_state     # reaching OR exceeding target => win
    if landing <= Lose_state:
        return Lose_state    # reaching OR passing floor => loss
    return landing
 
 
def transitions(state, action):
    merged = {}
    for prob, step in Moves[action]:
        s2 = _clamp(state + step)
        merged[s2] = merged.get(s2, 0.0) + prob
    return [(p, s2) for s2, p in merged.items()]
 

if __name__ == "__main__":
    print("Transition kernel P(s' | s, a):\n")
    for s in Interior:
        for a in Actions:
            branches = transitions(s, a)
            total = sum(p for p, _ in branches)
            print(f"  state {s}, {a:5s}: {branches}   sum(p)={total:.2f}")
            assert abs(total - 1.0) < 1e-9, f"probs != 1 at ({s},{a})"
            for _, s2 in branches:
                assert s2 in States, f"illegal landing {s2} at ({s},{a})"
    print("\nAll kernels valid: probabilities sum to 1, all landings are real states.")
    # the specific fix: state 3 Wide should now clamp 5 -> 4
    print("\nState-3 Wide (was landing on illegal '5'):", transitions(3, "Wide"))
 

