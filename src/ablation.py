"""
How much of the model's simulated edge survives when the SEASON is generated
from projections the model did not author?

Three worlds, same drafts, same opponents:
  full      - truth = our board (the original, self-referential setup)
  shrunk    - truth = market production x ratio shrunk by the MEASURED
              persistence slope (QB .398, RB .524, WR .511, TE .214), i.e. the
              ratio is real but weaker than our board assumes
  no_ratio  - truth = market production x positional-average ratio, i.e. the
              player-specific scoring edge does not exist at all
"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/user/fantasy")
import jax, jax.numpy as jnp, numpy as np, pandas as pd
from src.draft_sim import load_board, run_draft
from src.season_sim import simulate, championship, round_robin

SLOPE = {"QB": 0.398, "RB": 0.524, "WR": 0.511, "TE": 0.214}
S = int(sys.argv[1]) if len(sys.argv) > 1 else 700
SLOTS = [0, 2, 5, 8, 11]
POLICIES = ["MLV_VONA", "MLV", "ECR", "BPA"]


def truth_variants():
    b = pd.read_csv("data/processed/board_final.csv")
    b = b[b.proj_pts.notna()].reset_index(drop=True)
    out = {"full": b.proj_pts.values.astype(np.float32)}
    for name, f in [("shrunk", lambda r, m, sl: m + sl * (r - m)),
                    ("no_ratio", lambda r, m, sl: m)]:
        v = b.proj_pts.values.copy().astype(np.float64)
        for pos, sl in SLOPE.items():
            m_ = b.loc[b.pos == pos, "ratio"].mean()
            sel = (b.pos == pos) & b.ratio.notna() & b.market_std_pts.notna()
            newr = f(b.loc[sel, "ratio"].values, m_, sl)
            v[sel.values] = b.loc[sel, "market_std_pts"].values * newr
        out[name] = v.astype(np.float32)
    return b, out


def main():
    b, tv = truth_variants()
    _, arr = load_board()
    sched = round_robin()
    rows = []
    for world, tp in tv.items():
        tpj = jnp.array(tp)
        for pol in POLICIES:
            cs = []
            for slot in SLOTS:
                ros, _ = run_draft(arr, my_slot=slot, policy=pol, S=S, seed=31 + slot)
                tot = simulate(ros, arr, jax.random.PRNGKey(900 + slot), truth_proj=tpj)
                champ, _, _ = championship(tot, sched)
                cs.append(float((champ == slot).mean()))
            rows.append(dict(world=world, policy=pol, champ=float(np.mean(cs))))
            print(f"{world:9s} {pol:9s} champ={rows[-1]['champ']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv("data/processed/ablation.csv", index=False)
    print("WROTE data/processed/ablation.csv")


if __name__ == "__main__":
    main()
