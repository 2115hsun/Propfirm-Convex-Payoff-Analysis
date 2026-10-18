import Toy_MDP_Engine as mdp 
import numpy as np 
import pickle 

def bellman_backup_expectation(state, action, V):
    r = mdp.reward(state, action)
    expected_next = 0.0
    for prob, next_state in mdp.transitions(state, action):
        expected_next += prob * V[next_state]
    return r + mdp.Gamma * expected_next

def evaluate_policy(policy, tol=1e-6, max_sweeps=10000, verbose=False):
    V = {s: 0.0 for s in mdp.Interior}  # initialize value function for interior states
    V.update(mdp.Boundary_value)  # set boundary values for absorbing states

    for sweep in range (max_sweeps):
        delta = 0.0
        newV = dict(V)

        for s in mdp.Interior:
            a = policy[s]
            newV[s] = bellman_backup_expectation(s, a, V)
            delta = max(delta, abs(newV[s] - V[s]))

        V=newV
        if verbose and sweep < 8:
            print(f" sweep {sweep+1}: "+
                  ", ".join(f"{s}:{V[s]:.6f}" for s in mdp.Interior))

        if delta< tol:
            if verbose:
                print (f"Converged after {sweep+1} sweeps with delta={delta:.2e}")

            break
    return V

# ====================================================================
# Value Iteration (Searching for the optimal policy)
# ====================================================================
def value_iteration(tol=1e-6, max_sweeps=10000, verbose=False):
    V = {s: 0.0 for s in mdp.Interior}  # initialize value function for interior states
    V.update(mdp.Boundary_value)  # set boundary values for absorbing states
    policy = {s: None for s in mdp.Interior}  # initialize policy

    for sweep in range(max_sweeps):
        delta = 0.0
        newV = dict(V)
        for s in mdp.Interior:
              q_by_action = [(bellman_backup_expectation(s, a, V), a)   
                for a in mdp.Actions]

              best_q, best_a = max(q_by_action)       
              newV[s]        = best_q       
              policy[s]      = best_a 

              delta = max(delta, abs(newV[s] - V[s]))
        V = newV
        if verbose and sweep < 8:
            print(f" sweep {sweep+1}: "+
                  " ".join(f"{s}:{V[s]:.6f}" for s in mdp.Interior))
        if delta < tol:
            if verbose:
                print(f"Converged after {sweep+1} sweeps with delta={delta:.2e}")
            break

    return V, policy

if __name__ == "__main__":
    tight_everywhere = {s: "Tight" for s in mdp.Interior}
 
    print("Evaluating: Tight at every interior state\n")
    V = evaluate_policy(tight_everywhere, verbose=True)
 
    print("\nFinal values:")
    for s in mdp.States:
        print(f"  V({s}) = {V[s]:.4f}")
 
    print("\nHand check (Part II):  V(1)=0.592  V(2)=0.845  V(3)=0.954")

