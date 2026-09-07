"""
Phase 2: what real seasons say.

A. Policy comparison - draft on pre-season information only, score on the real
   season. No simulated variance anywhere in the outcome.
B. Skill vs luck - a league where all twelve seats draft the same way, so nobody
   has a process edge. How much does pre-season roster strength actually predict?
"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/user/fantasy")
import numpy as np, pandas as pd, time
from src.historical import board_for, SNAPSHOT
from src.real_backtest import (draft, actuals, lineups, play_league, round_robin,
                               replacement_levels, N_TEAMS, N_ROUNDS, BASE, N_FLEX, POS)

YEARS = [2021, 2022, 2023, 2024, 2025]
SEEDS_A = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SEEDS_B = int(sys.argv[2]) if len(sys.argv) > 2 else 60
SCHED = round_robin()


def strength(roster, proj, pos):
    """Pre-season projected optimal starting lineup for one team."""
    v = proj[roster]; g = pos[roster]
    by = {p: np.sort(v[g == p])[::-1] for p in POS}
    take = lambda p, k: list(by[p][:k]) + [0.0] * max(0, k - len(by[p]))
    base = sum(sum(take(p, BASE[p])) for p in POS)
    flex = sorted([x for p in ("RB", "WR", "TE") for x in by[p][BASE[p]:]], reverse=True)
    return base + sum(flex[:N_FLEX])


def main():
    rows_a, rows_b = [], []
    for yr in YEARS:
        b = board_for(yr)
        pts, played = actuals(b, yr)
        proj = {"pos": b.pos.values, "proj": b.proj_pts.values}
        pv, pg = b.proj_pts.values, b.pos.values
        t0 = time.time()

        # ---- A. policy comparison ----
        for pol in ["model_vona", "model", "adp"]:
            for slot in range(N_TEAMS):
                for sd in range(SEEDS_A):
                    r = draft(b, slot, pol, seed=1000 * yr + 17 * slot + sd)
                    tw = np.array([lineups(r[i], proj, pts, played) for i in range(N_TEAMS)])
                    ch, w, p = play_league(tw, SCHED)
                    rows_a.append(dict(year=yr, policy=pol, slot=slot, seed=sd,
                                       champ=int(ch == slot), wins=int(w[slot]),
                                       pts=float(p[slot]),
                                       playoff=int(np.argsort(-(w * 1e6 + p)).tolist().index(slot) < 6)))
        # ---- B. equal-skill league ----
        for sd in range(SEEDS_B):
            r = draft(b, 0, "adp", seed=77000 + 31 * yr + sd, all_same=True)
            tw = np.array([lineups(r[i], proj, pts, played) for i in range(N_TEAMS)])
            ch, w, p = play_league(tw, SCHED)
            st = np.array([strength(r[i], pv, pg) for i in range(N_TEAMS)])
            rows_b.append(dict(year=yr, seed=sd, champ=int(ch),
                               strongest=int(st.argmax()),
                               corr=float(np.corrcoef(st, w)[0, 1]),
                               spread=float(st.std()), mean=float(st.mean()),
                               champ_strength_rank=int((-st).argsort().tolist().index(int(ch)))))
        print(f"{yr} done in {time.time()-t0:.0f}s", flush=True)

    pd.DataFrame(rows_a).to_csv("data/processed/phase2_policy.csv", index=False)
    pd.DataFrame(rows_b).to_csv("data/processed/phase2_luck.csv", index=False)
    print("WROTE phase2 outputs")


if __name__ == "__main__":
    main()
