"""My edge on REAL seasons as the field gets sharper."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/user/fantasy")
import numpy as np, pandas as pd
from src.historical import board_for
from src.real_backtest import (draft, actuals, lineups, play_league, round_robin, N_TEAMS)

ME = 5
POOL = [0, 1, 2, 3, 7, 8, 9, 10, 11]
SCHED = round_robin()

def main():
    rows = []
    for k in [0, 2, 4, 6, 9]:
        rivals = tuple(POOL[:k])
        champ, wins, n = 0, 0.0, 0
        for yr in [2021, 2022, 2023, 2024, 2025]:
            b = board_for(yr); pts, played = actuals(b, yr)
            proj = {"pos": b.pos.values, "proj": b.proj_pts.values}
            for sd in range(8):
                r = draft(b, ME, "model_vona", seed=9100 + 13 * yr + sd, smart_slots=rivals)
                tw = np.array([lineups(r[i], proj, pts, played) for i in range(N_TEAMS)])
                ch, w, p = play_league(tw, SCHED)
                champ += int(ch == ME); wins += w[ME]; n += 1
        rows.append(dict(sharp_rivals=k, title=champ / n, wins=wins / n, n=n))
        print(f"rivals={k}: title={champ/n:.3f}  wins={wins/n:.2f}  (n={n})", flush=True)
    pd.DataFrame(rows).to_csv("data/processed/field_real.csv", index=False)

if __name__ == "__main__":
    main()
