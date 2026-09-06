"""Export the draft board to compact JSON for the live draft assistant."""
import pandas as pd, numpy as np, json, sys
sys.path.insert(0, "/home/user/fantasy")
from src.tiers import all_tiers

b = pd.read_csv("data/processed/board_final.csv")
t = all_tiers()[["player", "pos", "tier"]]
b = b.merge(t, on=["player", "pos"], how="left")
b["tier"] = b.tier.fillna(9).astype(int)
b = b.sort_values("vorp", ascending=False).reset_index(drop=True)
b = b.head(300)

recs = []
for i, r in b.iterrows():
    recs.append(dict(
        i=int(i), n=r.player, p=r.pos, tm=(r.team if isinstance(r.team, str) else ""),
        ecr=round(float(r.ecr), 1), pts=round(float(r.proj_pts)),
        vorp=round(float(r.vorp)), tier=int(r.tier),
        sd=round(float(r.proj_sd)),
        ratio=(round(float(r.ratio), 3) if pd.notna(r.ratio) else None),
        bye=(int(r.bye) if pd.notna(r.bye) else 0),
    ))
repl = {p: round(float(b[b.pos == p].replacement.iloc[0])) for p in b.pos.unique()}
out = dict(players=recs, replacement=repl,
           starters=dict(QB=1, RB=2, WR=2, TE=1, FLEX=2, K=1, DST=1),
           n_teams=12, roster=15, generated="2026-09-06", source_asof="2026-09-04")
json.dump(out, open("data/processed/board.json", "w"))
print("exported", len(recs), "players; replacement:", repl)
