"""
The question that actually matters for a league of experienced drafters:

Everyone else is sharp and drafts on value - but on ORDINARY rankings, because
almost nobody re-derives player values under a league's custom scoring. Is
having done that worth anything?

Three fields, all eleven opponents sophisticated:
  standard  - they draft on value computed from standard-PPR projections
  custom    - they draft on value computed under this league's actual scoring
  naive     - they just follow consensus rank (the flattering baseline)
"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/user/fantasy")
import jax, jax.numpy as jnp, numpy as np, pandas as pd
from src.draft_sim import load_board, run_draft
from src.season_sim import simulate, championship, round_robin
from src.vorp import add_vorp

S = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SLOTS = [1, 4, 7, 10]


def standard_board():
    sb = pd.read_csv("data/processed/board_final.csv")
    sb = sb[sb.proj_pts.notna()].reset_index(drop=True)
    std = sb.market_std_pts.copy()
    kd = sb.pos.isin(["K", "DST"])
    std[kd] = sb.loc[kd, "proj_pts"] / 1.1
    sb["proj_pts"] = std.fillna(sb.proj_pts)
    sb, _, rep = add_vorp(sb)
    for pos in ["K", "DST"]:
        v = np.sort(sb.loc[sb.pos == pos, "proj_pts"].values)[::-1]
        sb.loc[sb.pos == pos, "vorp"] = sb.loc[sb.pos == pos, "proj_pts"] - (
            v[12] if len(v) > 12 else v[-1])
    return dict(
        proj=jnp.array(sb.proj_pts.values, dtype=jnp.float32),
        vorp=jnp.array(sb.vorp.values, dtype=jnp.float32),
        repl=jnp.array([rep.get(p, float(sb[sb.pos == p].proj_pts.median()))
                        for p in ["QB", "RB", "WR", "TE", "K", "DST"]], dtype=jnp.float32))


def main():
    b, arr = load_board(); sched = round_robin(); opp = standard_board(); rows = []
    fields = [("opponents sharp, STANDARD values", "standard"),
              ("opponents sharp, YOUR scoring too", "custom"),
              ("opponents just follow rankings", "naive")]
    for label, kind in fields:
        cs, ws = [], []
        for slot in SLOTS:
            smart = [] if kind == "naive" else [t for t in range(12) if t != slot]
            ob = opp if kind == "standard" else None
            ros, _ = run_draft(arr, my_slot=slot, policy="MLV_VONA", S=S, seed=300 + slot,
                               smart_slots=smart, opp_board=ob)
            tot = simulate(ros, arr, jax.random.PRNGKey(4000 + slot))
            champ, w, _ = championship(tot, sched)
            cs.append(float((champ == slot).mean())); ws.append(float(w[:, slot].mean()))
        m = float(np.mean(cs)); se = float(np.sqrt(m * (1 - m) / (S * len(SLOTS))))
        rows.append(dict(field=label, champ=m, se=se, wins=float(np.mean(ws))))
        print(f"{label:34s} title={m:.4f} ±{se:.4f}   wins={rows[-1]['wins']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv("data/processed/custom_edge.csv", index=False)
    a, c = rows[0]["champ"], rows[1]["champ"]
    sd = np.sqrt(rows[0]["se"] ** 2 + rows[1]["se"] ** 2)
    print(f"\ncustom-scoring edge vs equally sharp opponents: "
          f"{a:.4f} vs {c:.4f}  (+{100*(a/c-1):.0f}% relative, {(a-c)/sd:.1f} SE)")
    print(f"baseline if all twelve teams were equal: 0.0833")


if __name__ == "__main__":
    main()
