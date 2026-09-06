"""Assemble the full draftable board: skill players + K + DST, with projections."""
import pandas as pd, numpy as np, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/fantasy")
from config.league import DEF_POINTS_ALLOWED, DEFENSE
from src.projections import build, add_season_projection
from src.kdst import kicker_seasons, dst_seasons
from src.vorp import add_vorp
from sklearn.isotonic import IsotonicRegression
OUT = "data/processed"


def dst_full():
    d = dst_seasons()
    pa = pd.read_csv(f"{OUT}/points_allowed.csv")
    d = d.merge(pa, on=["season", "week", "team"], how="left")
    d["pa"] = d.pa.fillna(22.6)
    tier = np.zeros(len(d))
    for lo, hi, p in DEF_POINTS_ALLOWED:
        tier += np.where((d.pa >= lo) & (d.pa <= hi), p, 0.0)
    d["pts_total"] = d.pts + tier + DEFENSE["points_per_point_allowed"] * d.pa
    return d.groupby(["team", "season"], as_index=False).pts_total.sum()


def rank_curve(vals_by_season, n_keep):
    """Average season points at each positional finish rank."""
    arr = []
    for s, v in vals_by_season:
        v = np.sort(np.asarray(v))[::-1][:n_keep]
        arr.append(np.pad(v, (0, max(0, n_keep - len(v))), constant_values=v[-1] if len(v) else 0))
    return np.mean(np.vstack(arr), axis=0)


def main():
    skill = add_season_projection(build())
    skill = skill[["player", "pos", "team", "ecr", "sd", "bye", "age", "gsis_id",
                   "proj_ppg", "exp_games", "proj_pts", "proj_sd", "ratio",
                   "ratio_raw", "raw_games", "market_std_pts"]]

    # --- kickers: map ECR positional rank -> historical points at that rank ---
    k = kicker_seasons(); k = k[k.g >= 8]
    kc = rank_curve([(s, g.pts.values) for s, g in k.groupby("season")], 32)
    # --- DST ---
    d = dst_full()
    dc = rank_curve([(s, g.pts_total.values) for s, g in d.groupby("season")], 32)

    board = pd.read_csv(f"{OUT}/board_2026.csv", low_memory=False)
    extra = []
    for pos, curve, sd in [("K", kc, 24.0), ("DST", dc, 30.0)]:
        b = board[board.pos == pos].copy().sort_values("ecr")
        b["pos_rank"] = np.arange(len(b))
        idx = np.clip(b.pos_rank.values, 0, len(curve) - 1)
        b["proj_pts"] = curve[idx]
        b["proj_sd"] = sd
        b["proj_ppg"] = b.proj_pts / 16.0
        b["exp_games"] = 16.0
        for c in ["ratio", "ratio_raw", "raw_games", "market_std_pts"]:
            b[c] = np.nan
        extra.append(b[["player", "pos", "team", "ecr", "sd", "bye", "age", "gsis_id",
                        "proj_ppg", "exp_games", "proj_pts", "proj_sd", "ratio",
                        "ratio_raw", "raw_games", "market_std_pts"]])

    full = pd.concat([skill] + extra, ignore_index=True)
    full = full[full.proj_pts.notna()].copy()
    full, counts, repl = add_vorp(full)
    # K/DST replacement = 12th best at position
    for pos in ["K", "DST"]:
        v = np.sort(full.loc[full.pos == pos, "proj_pts"].values)[::-1]
        r = v[12] if len(v) > 12 else v[-1]
        full.loc[full.pos == pos, "replacement"] = r
        full.loc[full.pos == pos, "vorp"] = full.loc[full.pos == pos, "proj_pts"] - r
    full = full.sort_values("vorp", ascending=False).reset_index(drop=True)
    full["vorp_rank"] = np.arange(1, len(full) + 1)
    full.to_csv(f"{OUT}/board_final.csv", index=False)
    print("startable counts:", counts)
    print("replacement:", {k2: round(v2, 1) for k2, v2 in repl.items()},
          "K:", round(full[full.pos=="K"].replacement.iloc[0],1),
          "DST:", round(full[full.pos=="DST"].replacement.iloc[0],1))
    print("board size:", len(full), full.pos.value_counts().to_dict())
    return full


if __name__ == "__main__":
    main()
