"""Compare draft policies across all 12 slots by championship probability."""
import sys, time, json; sys.path.insert(0, "/home/user/fantasy")
import jax, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from src.draft_sim import load_board, run_draft, POLICIES
from src.season_sim import simulate, championship, round_robin

S = int(sys.argv[1]) if len(sys.argv) > 1 else 400
POLS = ["MLV", "MLV_VONA", "BPA", "ECR", "ZERO_RB", "HERO_RB", "ROBUST_RB"]

def main():
    b, arr = load_board()
    sched = round_robin()
    rows = []
    for pol in POLS:
        for slot in range(12):
            t0 = time.time()
            ros, _ = run_draft(arr, my_slot=slot, policy=pol, S=S, seed=100 + slot)
            tot = simulate(ros, arr, jax.random.PRNGKey(1000 + slot))
            champ, wins, pts = championship(tot, sched)
            rows.append(dict(policy=pol, slot=slot + 1,
                             champ=float((champ == slot).mean()),
                             wins=float(wins[:, slot].mean()),
                             pts=float(pts[:, slot].mean()),
                             playoff=float((np.argsort(-(np.array(wins) * 1e6 + np.array(pts)),
                                                       axis=1) == slot).argmax(axis=1).__lt__(6).mean()),
                             secs=round(time.time() - t0, 1)))
            print(f"{pol:10s} slot{slot+1:3d}  champ={rows[-1]['champ']:.3f} "
                  f"wins={rows[-1]['wins']:.2f}  ({rows[-1]['secs']}s)", flush=True)
    pd.DataFrame(rows).to_csv("data/processed/policy_results.csv", index=False)
    print("WROTE data/processed/policy_results.csv")

if __name__ == "__main__":
    main()
