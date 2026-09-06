"""What the model actually does at each draft slot, round by round."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/user/fantasy")
import numpy as np, pandas as pd, collections
from src.draft_sim import load_board, run_draft, POS_NAMES

S = int(sys.argv[1]) if len(sys.argv) > 1 else 600


def main():
    b, arr = load_board()
    rows, picks_rows = [], []
    for slot in range(12):
        ros, _ = run_draft(arr, my_slot=slot, policy="MLV", S=S, seed=7 + slot)
        mine = np.array(ros[:, slot, :])                       # (S, 15)
        for rnd in range(15):
            ids = mine[:, rnd]
            posn = b.pos.values[ids]
            cnt = collections.Counter(posn)
            top = cnt.most_common(1)[0]
            names = collections.Counter(b.player.values[ids]).most_common(3)
            rows.append(dict(slot=slot + 1, rnd=rnd + 1,
                             pos=top[0], pos_share=top[1] / S,
                             mix="/".join(f"{p}{round(100*c/S)}" for p, c in cnt.most_common(3)),
                             top_names="; ".join(f"{n} ({round(100*c/S)}%)" for n, c in names)))
    pd.DataFrame(rows).to_csv("data/processed/slot_plan.csv", index=False)
    print("wrote data/processed/slot_plan.csv")


if __name__ == "__main__":
    main()
