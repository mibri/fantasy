"""
League configuration — mibri 2026 fantasy football league.

Every number here was transcribed from the league's published settings page.
Anything the settings page did not state explicitly is marked ASSUMPTION.
"""

LEAGUE = dict(
    n_teams=12,
    platform="sleeper",
    draft_type="snake",          # confirmed by the completed draft
    # CORRECTED after the draft: the league uses NO KICKER. Nine starters, not
    # ten, and fourteen rounds, not fifteen. Skill-position replacement levels
    # are unaffected -- they are solved from QB/RB/WR/TE and the FLEX slots only,
    # so the board ordering that drove the draft is unchanged.
    starters=dict(QB=1, RB=2, WR=2, TE=1, FLEX=2, DST=1),
    flex_eligible=("RB", "WR", "TE"),
    bench=5,
    ir=1,
    uses_kicker=False,
)
LEAGUE["n_starters"] = sum(LEAGUE["starters"].values())          # 10
LEAGUE["roster_size"] = LEAGUE["n_starters"] + LEAGUE["bench"]   # 15
LEAGUE["n_drafted"] = LEAGUE["roster_size"] * LEAGUE["n_teams"]  # 180

# ---------------------------------------------------------------- scoring ---
# Yardage/event coefficients applied to season or weekly stat columns.
SCORING = {
    # ---- passing ----
    "passing_yards":            0.04,   # 25 yds = 1 pt
    "passing_tds":              6.0,    # NON-STANDARD (typical = 4)
    "passing_2pt_conversions":  2.0,
    "passing_interceptions":   -3.0,    # NON-STANDARD (typical = -2)
    "pick_six_thrown":         -1.0,    # additional, on top of the INT
    "completions":              0.27,   # NON-STANDARD, large
    "incompletions":           -0.50,   # NON-STANDARD, large
    "passing_40":               0.50,   # 40+ yard completion bonus
    "pass_td_40":               0.50,   # 40+ yard passing TD bonus
    "pass_td_50":               1.50,   # 50+ yard passing TD bonus

    # ---- rushing ----
    "rushing_yards":            0.10,
    "rushing_tds":              6.0,
    "rushing_2pt_conversions":  2.0,
    "rushing_40":               0.50,
    "rush_td_40":               0.50,
    "rush_td_50":               1.50,

    # ---- receiving ----
    "receptions":               1.00,   # full PPR
    "receiving_yards":          0.10,
    "receiving_tds":            6.0,
    "receiving_2pt_conversions": 2.0,
    "receiving_40":             0.50,
    "rec_td_40":                0.50,
    "rec_td_50":                1.50,

    # ---- misc ----
    "fumbles_lost_total":      -2.0,
    "fumble_recovery_tds":      6.0,

    # ---- weekly game bonuses (computed per game, then summed) ----
    "bonus_rush_100":           3.0,    # 100-199 rushing yards in a game
    "bonus_rush_200":           5.0,    # 200+  (replaces the 100-199 award)
    "bonus_rec_100":            3.0,
    "bonus_rec_200":            5.0,
    "bonus_pass_300":           2.0,    # 300-399 passing yards
    "bonus_pass_400":           3.0,
    "bonus_carries_20":         2.0,    # 20+ carries in a game

    # ---- first down bonuses (RB/WR/TE only, per league settings) ----
    "first_down_skill":         0.25,
}

# ---- kicking ----
KICKING = {
    "fg_made_0_19": 3.0, "fg_made_20_29": 3.0, "fg_made_30_39": 3.0,
    "fg_made_40_49": 4.0, "fg_made_50_59": 5.0, "fg_made_60_": 6.0,
    "pat_made": 1.0, "fg_missed": -1.0, "pat_missed": -1.0,
}

# ---- team defense / special teams ----
DEFENSE = {
    "def_tds": 6.0, "special_teams_tds": 6.0,
    "def_sacks": 1.0, "def_interceptions": 2.0,
    "fumble_recovery_opp": 1.0, "def_fumbles_forced": 1.0,
    "def_safeties": 4.0, "def_tackles_for_loss": 0.50,
    "def_qb_hits": 0.10, "def_pass_defended": 0.25,
    "blocked_kick": 3.0, "def_2pt_returns": 4.0,
    "three_and_out": 0.25, "fourth_down_stop": 0.50,
    "points_per_point_allowed": -0.10,
}
# Points-allowed tiers: (lower_inclusive, upper_inclusive, points). 0-13 is 0.
DEF_POINTS_ALLOWED = [(14, 20, -1.0), (21, 27, -2.0), (28, 34, -3.0), (35, 999, -4.0)]
# Yards-allowed tiers
DEF_YARDS_ALLOWED = [(350, 399, -1.0), (400, 449, -2.0), (450, 499, -3.0),
                     (500, 549, -4.0), (550, 9999, -5.0)]

# The break-even completion rate implied by the completion/incompletion terms:
#   0.27*p - 0.50*(1-p) = 0  ->  p = 0.50/0.77
BREAKEVEN_COMP_PCT = 0.50 / 0.77   # 0.6494
