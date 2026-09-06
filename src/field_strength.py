"""
How much of the edge survives when opponents are not naive?

The headline comparison pits one value drafter against eleven consensus
followers. Real leagues contain other people who also draft on value. This
sweeps the number of rival value drafters from 0 to 8 and reports what happens
to our championship probability.
"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/user/fantasy")
import jax, numpy as np, pandas as pd
from src.draft_sim import load_board, run_draft
from src.season_sim import simulate, championship, round_robin

S = int(sys.argv[1]) if len(sys.argv) > 1 else 700
ME = 5
RIVAL_POOL = [0, 1, 2, 3, 7, 8, 9, 10]


def main():
    _, arr = load_board(); sched = round_robin(); rows = []
    for k in [0, 1, 2, 4, 6, 8]:
        rivals = RIVAL_POOL[:k]
        ros, _ = run_draft(arr, my_slot=ME, policy="MLV_VONA", S=S, seed=77,
                           smart_slots=rivals)
        tot = simulate(ros, arr, jax.random.PRNGKey(1234))
        champ, wins, _ = championship(tot, sched)
        mine = float((champ == ME).mean())
        naive = [t for t in range(12) if t != ME and t not in rivals]
        nv = float(np.mean([float((np.array(champ) == t).mean()) for t in naive])) if naive else 0.0
        rows.append(dict(rival_value_drafters=k, me=mine, avg_naive_opponent=nv,
                         my_wins=float(wins[:, ME].mean())))
        print(f"rivals={k}: me={mine:.4f}  avg naive opponent={nv:.4f}  my wins={rows[-1]['my_wins']:.2f}",
              flush=True)
    pd.DataFrame(rows).to_csv("data/processed/field_strength.csv", index=False)
    print("WROTE data/processed/field_strength.csv")


if __name__ == "__main__":
    main()
