"""Apply the league's custom scoring to nflverse weekly stat lines."""
import pandas as pd, numpy as np, sys
sys.path.insert(0, "/home/user/fantasy")
from config.league import SCORING, KICKING, DEF_POINTS_ALLOWED, DEF_YARDS_ALLOWED

RAW, OUT = "/home/user/fantasy/data/raw", "/home/user/fantasy/data/processed"
SKILL = ("QB", "RB", "WR", "TE", "FB")


def load_weekly(seasons):
    df = pd.concat([pd.read_csv(f"{RAW}/sw_{s}.csv", low_memory=False) for s in seasons])
    return df[df.season_type == "REG"].copy()


def score_weekly(df, long_td=None, stack_long_td=True):
    """Return df with a `pts` column: league points for that player-week.

    stack_long_td: if True a 55-yd TD earns BOTH the 40+ and 50+ bonus
    (separate stat triggers, the usual platform behaviour). If False the
    50+ tier replaces the 40+ tier.
    """
    d = df.copy()
    num = lambda c: pd.to_numeric(d[c], errors="coerce").fillna(0) if c in d else 0.0

    if long_td is not None:
        d = d.merge(long_td, on=["player_id", "season", "week"], how="left")
        for c in ["pass_td_40", "pass_td_50", "rec_td_40", "rec_td_50",
                  "rush_td_40", "rush_td_50", "pick_six_thrown"]:
            d[c] = pd.to_numeric(d.get(c), errors="coerce").fillna(0)
    else:
        for c in ["pass_td_40", "pass_td_50", "rec_td_40", "rec_td_50",
                  "rush_td_40", "rush_td_50", "pick_six_thrown"]:
            d[c] = 0.0

    att, comp = num("attempts"), num("completions")
    incomp = (att - comp).clip(lower=0)
    ry, recy, py = num("rushing_yards"), num("receiving_yards"), num("passing_yards")
    carries = num("carries")

    p = 0.0
    # passing
    p += SCORING["passing_yards"] * py
    p += SCORING["passing_tds"] * num("passing_tds")
    p += SCORING["passing_interceptions"] * num("passing_interceptions")
    p += SCORING["pick_six_thrown"] * d["pick_six_thrown"]
    p += SCORING["completions"] * comp
    p += SCORING["incompletions"] * incomp
    p += SCORING["passing_40"] * num("passing_40")
    p += SCORING["passing_2pt_conversions"] * num("passing_2pt_conversions")
    # rushing
    p += SCORING["rushing_yards"] * ry
    p += SCORING["rushing_tds"] * num("rushing_tds")
    p += SCORING["rushing_40"] * num("rushing_40")
    p += SCORING["rushing_2pt_conversions"] * num("rushing_2pt_conversions")
    # receiving
    p += SCORING["receptions"] * num("receptions")
    p += SCORING["receiving_yards"] * recy
    p += SCORING["receiving_tds"] * num("receiving_tds")
    p += SCORING["receiving_40"] * num("receiving_40")
    p += SCORING["receiving_2pt_conversions"] * num("receiving_2pt_conversions")
    # long TD bonuses
    if stack_long_td:
        p += SCORING["pass_td_40"] * d.pass_td_40 + SCORING["pass_td_50"] * d.pass_td_50
        p += SCORING["rush_td_40"] * d.rush_td_40 + SCORING["rush_td_50"] * d.rush_td_50
        p += SCORING["rec_td_40"] * d.rec_td_40 + SCORING["rec_td_50"] * d.rec_td_50
    else:
        p += SCORING["pass_td_40"] * (d.pass_td_40 - d.pass_td_50) + SCORING["pass_td_50"] * d.pass_td_50
        p += SCORING["rush_td_40"] * (d.rush_td_40 - d.rush_td_50) + SCORING["rush_td_50"] * d.rush_td_50
        p += SCORING["rec_td_40"] * (d.rec_td_40 - d.rec_td_50) + SCORING["rec_td_50"] * d.rec_td_50
    # misc
    p += SCORING["fumbles_lost_total"] * num("fumbles_lost_total")
    p += SCORING["fumble_recovery_tds"] * num("fumble_recovery_tds")
    p += 6.0 * num("special_teams_tds")

    # weekly game bonuses (mutually exclusive tiers)
    p += np.where(ry >= 200, SCORING["bonus_rush_200"],
         np.where(ry >= 100, SCORING["bonus_rush_100"], 0.0))
    p += np.where(recy >= 200, SCORING["bonus_rec_200"],
         np.where(recy >= 100, SCORING["bonus_rec_100"], 0.0))
    p += np.where(py >= 400, SCORING["bonus_pass_400"],
         np.where(py >= 300, SCORING["bonus_pass_300"], 0.0))
    p += np.where(carries >= 20, SCORING["bonus_carries_20"], 0.0)

    # first-down bonus: RB/WR/TE only (per league settings, not QB passing 1Ds)
    fd = num("rushing_first_downs") + num("receiving_first_downs")
    is_skill_fd = d.position.isin(["RB", "WR", "TE", "FB"]).values
    p += SCORING["first_down_skill"] * fd * is_skill_fd

    d["pts"] = p
    return d


def score_kicker_weekly(df):
    d = df.copy()
    num = lambda c: pd.to_numeric(d[c], errors="coerce").fillna(0) if c in d else 0.0
    p = sum(v * num(k) for k, v in KICKING.items() if k not in ("fg_missed", "pat_missed"))
    p += KICKING["fg_missed"] * num("fg_missed") + KICKING["pat_missed"] * num("pat_missed")
    d["pts"] = p
    return d


def tier_points(x, tiers):
    out = np.zeros(len(x))
    for lo, hi, pts in tiers:
        out += np.where((x >= lo) & (x <= hi), pts, 0.0)
    return out
