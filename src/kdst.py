"""Score kickers and team defenses under the league's rules."""
import pandas as pd, numpy as np, sys
sys.path.insert(0, "/home/user/fantasy")
from config.league import KICKING, DEFENSE, DEF_POINTS_ALLOWED, DEF_YARDS_ALLOWED
RAW, OUT = "data/raw", "data/processed"


def kicker_seasons(seasons=(2023, 2024, 2025)):
    df = pd.concat([pd.read_csv(f"{RAW}/sw_{s}.csv", low_memory=False) for s in seasons])
    df = df[(df.season_type == "REG") & (df.position == "K")]
    n = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0)
    pts = (3*n("fg_made_0_19") + 3*n("fg_made_20_29") + 3*n("fg_made_30_39") +
           4*n("fg_made_40_49") + 5*n("fg_made_50_59") + 6*n("fg_made_60_") +
           1*n("pat_made") - 1*n("fg_missed") - 1*n("pat_missed"))
    df = df.assign(pts=pts)
    return df.groupby(["player_id", "player_display_name", "season"], as_index=False).agg(
        pts=("pts", "sum"), g=("week", "count"))


def dst_seasons(seasons=(2023, 2024, 2025)):
    """Aggregate individual defensive stats to team level, add PBP drive events."""
    out = []
    for s in seasons:
        df = pd.read_csv(f"{RAW}/sw_{s}.csv", low_memory=False)
        df = df[df.season_type == "REG"]
        n = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0) if c in df else 0.0
        df = df.assign(
            _sack=n("def_sacks"), _int=n("def_interceptions"), _ff=n("def_fumbles_forced"),
            _fr=n("fumble_recovery_opp"), _td=n("def_tds"), _sty=n("def_safeties"),
            _tfl=n("def_tackles_for_loss"), _qbh=n("def_qb_hits"), _pd=n("def_pass_defended"),
            _blk=n("def_punt_blocks") + n("def_pat_blocks") + n("def_fg_blocks"),
            _sttd=n("special_teams_tds"))
        t = df.groupby(["team", "season", "week"], as_index=False)[
            ["_sack", "_int", "_ff", "_fr", "_td", "_sty", "_tfl", "_qbh", "_pd", "_blk", "_sttd"]].sum()
        t["pts_core"] = (DEFENSE["def_sacks"]*t._sack + DEFENSE["def_interceptions"]*t._int +
                         DEFENSE["def_fumbles_forced"]*t._ff + DEFENSE["fumble_recovery_opp"]*t._fr +
                         6*t._td + DEFENSE["def_safeties"]*t._sty +
                         DEFENSE["def_tackles_for_loss"]*t._tfl + DEFENSE["def_qb_hits"]*t._qbh +
                         DEFENSE["def_pass_defended"]*t._pd + DEFENSE["blocked_kick"]*t._blk +
                         6*t._sttd)
        out.append(t)
    d = pd.concat(out)

    pbp = pd.read_csv(f"{OUT}/team_defense_pbp.csv")
    d = d.merge(pbp, on=["team", "season", "week"], how="left").fillna(0)
    d["pts_drive"] = (DEFENSE["three_and_out"]*d.three_and_out +
                      DEFENSE["fourth_down_stop"]*d.fourth_down_stop)

    ya = d.yards_allowed.values
    ytier = np.zeros(len(d))
    for lo, hi, p in DEF_YARDS_ALLOWED:
        ytier += np.where((ya >= lo) & (ya <= hi), p, 0.0)
    d["pts_yards"] = ytier
    d["pts"] = d.pts_core + d.pts_drive + d.pts_yards   # points-allowed term added below
    return d
