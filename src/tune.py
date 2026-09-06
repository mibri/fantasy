"""Optimise the bench-depth weight by championship probability."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/user/fantasy")
import jax, numpy as np, pandas as pd, itertools, time
from src.draft_sim import load_board, run_draft
from src.season_sim import simulate, championship, round_robin

S = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SLOTS = [0, 3, 6, 9]                     # sample of slots, averaged

def main():
    b, arr = load_board(); sched = round_robin(); rows = []
    grid = [(add, bb) for add in (False, True)
                      for bb in (0.0, 0.15, 0.30, 0.50, 0.80, 1.20)]
    for add, bb in grid:
        cs = []
        for slot in SLOTS:
            ros, _ = run_draft(arr, my_slot=slot, policy="MLV", S=S, seed=21 + slot,
                               params=dict(additive=add, bench_base=bb))
            tot = simulate(ros, arr, jax.random.PRNGKey(500 + slot))
            champ, wins, _ = championship(tot, sched)
            cs.append(float((champ == slot).mean()))
        rows.append(dict(additive=add, bench_base=bb, champ=float(np.mean(cs)),
                         se=float(np.std(cs) / np.sqrt(len(cs)))))
        print(f"additive={add} bench_base={bb:.2f} -> champ={rows[-1]['champ']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv("data/processed/tuning.csv", index=False)
    print("WROTE data/processed/tuning.csv")

if __name__ == "__main__":
    main()
