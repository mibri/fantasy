"""Join the 2026 FantasyPros redraft consensus board to nflverse player IDs & history."""
import pandas as pd, numpy as np, re, sys
RAW, OUT = "data/raw", "data/processed"

def norm(s):
    s = str(s).lower()
    s = re.sub(r"[.'`,]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

ecr = pd.read_csv(f"{RAW}/fpecr.csv", low_memory=False)
ecr = ecr[ecr.page_type == "redraft-overall"].copy()
ecr["pos"] = ecr["pos"].astype(str).str.replace(r"\d+", "", regex=True)
ecr["merge_name"] = ecr.player.map(norm)

ids = pd.read_csv(f"{RAW}/dp_ids.csv", low_memory=False)
ids["merge_name"] = ids.name.map(norm)

# 1) join on FantasyPros id where available, 2) fall back to name+position
ecr["fp_id"] = pd.to_numeric(ecr["id"], errors="coerce")
ids["fantasypros_id"] = pd.to_numeric(ids["fantasypros_id"], errors="coerce")
m = ecr.merge(ids[["fantasypros_id", "gsis_id", "sleeper_id", "birthdate", "draft_year", "position"]],
              left_on="fp_id", right_on="fantasypros_id", how="left")
need = m.gsis_id.isna()
fb = m.loc[need, ["merge_name", "pos"]].merge(
    ids[["merge_name", "position", "gsis_id", "sleeper_id", "birthdate", "draft_year"]]
      .drop_duplicates("merge_name"),
    on="merge_name", how="left")
for c in ["gsis_id", "sleeper_id", "birthdate", "draft_year"]:
    m.loc[need, c] = fb[c].values

m["age"] = (pd.Timestamp("2026-09-06") - pd.to_datetime(m.birthdate, errors="coerce")).dt.days / 365.25
skill = m[m.pos.isin(["QB", "RB", "WR", "TE"])].copy()
print(f"board rows={len(m)}  skill={len(skill)}  gsis matched={skill.gsis_id.notna().sum()} "
      f"({skill.gsis_id.notna().mean():.1%})")
print("unmatched skill players (likely rookies):")
print(skill[skill.gsis_id.isna()].nsmallest(12, "ecr")[["player", "pos", "team", "ecr"]].to_string(index=False))
m.to_csv(f"{OUT}/board_2026.csv", index=False)
