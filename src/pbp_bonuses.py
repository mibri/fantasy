"""
Extract from play-by-play the scoring events that weekly stat tables do not carry:
  * 40+ / 50+ yard TD bonuses (passing, rushing, receiving)
  * pick-sixes thrown
  * team-defense drive events (3-and-outs, 4th down stops) and points/yards allowed
"""
import pandas as pd, numpy as np, os

RAW = "/home/user/fantasy/data/raw"
OUT = "/home/user/fantasy/data/processed"

PBP_COLS = [
    "season", "week", "season_type", "play_type", "yards_gained", "touchdown",
    "pass_touchdown", "rush_touchdown", "passer_player_id", "receiver_player_id",
    "rusher_player_id", "interception", "return_touchdown", "posteam", "defteam",
    "fixed_drive", "fixed_drive_result", "drive_play_count", "sack", "safety",
    "third_down_converted", "third_down_failed", "fourth_down_converted",
    "fourth_down_failed", "punt_blocked", "field_goal_result", "game_id",
]


def long_td_bonuses(season: int) -> pd.DataFrame:
    """Per player-week counts of 40+/50+ yard TDs, and pick-sixes thrown."""
    df = pd.read_csv(f"{RAW}/pbp_{season}.csv", usecols=PBP_COLS, low_memory=False)
    df = df[df.season_type == "REG"]
    y = df.yards_gained.fillna(0)
    recs = []

    def add(mask, id_col, prefix):
        sub = df[mask & df[id_col].notna()]
        if sub.empty:
            return
        yy = sub.yards_gained.fillna(0)
        g = pd.DataFrame({
            "player_id": sub[id_col], "season": sub.season, "week": sub.week,
            f"{prefix}_td_40": (yy >= 40).astype(int),
            f"{prefix}_td_50": (yy >= 50).astype(int),
        }).groupby(["player_id", "season", "week"], as_index=False).sum()
        recs.append(g)

    ptd = df.pass_touchdown.fillna(0) == 1
    rtd = df.rush_touchdown.fillna(0) == 1
    add(ptd, "passer_player_id", "pass")
    add(ptd, "receiver_player_id", "rec")
    add(rtd, "rusher_player_id", "rush")

    # pick-six thrown: interception on the play AND the return was a TD
    p6 = df[(df.interception.fillna(0) == 1) &
            (df.return_touchdown.fillna(0) == 1) & df.passer_player_id.notna()]
    if not p6.empty:
        recs.append(pd.DataFrame({
            "player_id": p6.passer_player_id, "season": p6.season, "week": p6.week,
            "pick_six_thrown": 1,
        }).groupby(["player_id", "season", "week"], as_index=False).sum())

    out = recs[0]
    for r in recs[1:]:
        out = out.merge(r, on=["player_id", "season", "week"], how="outer")
    return out.fillna(0)


def team_defense(season: int) -> pd.DataFrame:
    """Per team-week: 3-and-outs forced, 4th down stops, points & yards allowed."""
    df = pd.read_csv(f"{RAW}/pbp_{season}.csv", usecols=PBP_COLS, low_memory=False)
    df = df[df.season_type == "REG"]

    # drive-level: a 3-and-out is a punt/turnover-on-downs drive with <=3 plays
    dr = (df.dropna(subset=["fixed_drive", "posteam", "defteam"])
            .groupby(["game_id", "season", "week", "posteam", "defteam", "fixed_drive"],
                     as_index=False)
            .agg(plays=("drive_play_count", "max"), result=("fixed_drive_result", "first")))
    dr["three_and_out"] = ((dr.plays <= 3) &
                           dr.result.isin(["Punt", "Turnover on downs"])).astype(int)
    tao = (dr.groupby(["season", "week", "defteam"], as_index=False)["three_and_out"].sum())

    fds = (df.assign(stop=df.fourth_down_failed.fillna(0))
             .groupby(["season", "week", "defteam"], as_index=False)["stop"].sum()
             .rename(columns={"stop": "fourth_down_stop"}))

    # yards allowed = offensive yards gained by posteam against this defteam
    ya = (df[df.play_type.isin(["pass", "run"])]
            .groupby(["season", "week", "defteam"], as_index=False)["yards_gained"].sum()
            .rename(columns={"yards_gained": "yards_allowed"}))

    out = tao.merge(fds, on=["season", "week", "defteam"], how="outer") \
             .merge(ya, on=["season", "week", "defteam"], how="outer").fillna(0)
    return out.rename(columns={"defteam": "team"})


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    seasons = [2021, 2022, 2023, 2024, 2025]
    pd.concat([long_td_bonuses(s) for s in seasons]).to_csv(
        f"{OUT}/long_td_bonuses.csv", index=False)
    pd.concat([team_defense(s) for s in seasons]).to_csv(
        f"{OUT}/team_defense_pbp.csv", index=False)
    print("wrote long_td_bonuses.csv and team_defense_pbp.csv")
