"""
2026 projections in the league's custom scoring.

Architecture
------------
Two independent information sources are combined:

  (a) MARKET  - the FantasyPros redraft consensus (ECR). Hundreds of analysts
      watching film and camp reports. Best available estimate of a player's
      2026 *role and volume*. Expressed in standard-PPR terms.

  (b) HISTORY - each player's own realised per-game production, re-scored under
      THIS league's rules. Captures the scoring-translation edge (completion
      bonus, first-down bonus, big-game bonuses) the market does not price.

For each position we fit an isotonic (monotone-decreasing) curve
    positional ECR rank -> expected league points per game,
then shrink each player's own history toward that curve with an
empirical-Bayes weight that grows with their sample size.
"""
import pandas as pd, numpy as np, sys
from sklearn.isotonic import IsotonicRegression

OUT = "data/processed"
SEASON_W = {2025: 0.60, 2024: 0.28, 2023: 0.12}   # recency decay
MIN_GAMES_FULL_TRUST = 24                          # ~1.5 seasons -> mostly own history


def player_history(sc):
    """Recency-weighted league points per game, and weighted games, per player."""
    h = sc[sc.season.isin(SEASON_W)].copy()
    h["w"] = h.season.map(SEASON_W)
    # only count games the player actually appeared in (any usage)
    used = (h.attempts.fillna(0) + h.carries.fillna(0) + h.targets.fillna(0)) > 0
    h = h[used]
    g = h.groupby("player_id").apply(
        lambda d: pd.Series({
            "hist_ppg": np.average(d.pts, weights=d.w),
            "wgames": d.w.sum(),
            "raw_games": len(d),
            "ppg_sd": d.pts.std(ddof=1) if len(d) > 3 else np.nan,
        }), include_groups=False)
    return g.reset_index()


def fit_market_curve(df, pos):
    """Isotonic fit: positional ECR rank -> league PPG, using players with history."""
    d = df[(df.pos == pos) & df.hist_ppg.notna() & (df.raw_games >= 6)].copy()
    d = d.sort_values("pos_rank")
    if len(d) < 8:
        return None
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(d.pos_rank.values, d.hist_ppg.values)
    return iso


# Empirical within-player aging factors, normalised so a typical-age player = 1.0.
# Measured from 2021-2025 year-over-year PPG ratios; shrunk to 70% of raw effect
# because the raw ratios also contain regression-to-the-mean, not just aging.
AGE_KNOTS = {
    "RB": [(23, 1.066), (25, 1.026), (27, 0.962), (29, 0.781), (32, 0.702)],
    "WR": [(23, 1.140), (25, 0.949), (27, 0.970), (29, 0.820), (32, 0.739)],
    "TE": [(23, 1.170), (25, 0.875), (27, 1.032), (29, 0.845), (32, 0.993)],
    "QB": [(23, 0.990), (25, 1.041), (27, 1.027), (29, 0.988), (32, 0.865)],
}
AGE_SHRINK = 0.70


def age_factor(pos, age):
    """Smooth, shrunk multiplicative aging adjustment for the history component."""
    if pos not in AGE_KNOTS or not np.isfinite(age):
        return 1.0
    xs = [k[0] for k in AGE_KNOTS[pos]]; ys = [k[1] for k in AGE_KNOTS[pos]]
    raw = float(np.interp(age, xs, ys))
    return 1.0 + AGE_SHRINK * (raw - 1.0)


def build(blend_floor=0.15):
    sc = pd.read_parquet(f"{OUT}/weekly_scored.parquet")
    board = pd.read_csv(f"{OUT}/board_2026.csv", low_memory=False)
    board = board[board.pos.isin(["QB", "RB", "WR", "TE"])].copy()
    board["pos_rank"] = board.groupby("pos").ecr.rank(method="first")

    hist = player_history(sc)
    df = board.merge(hist, left_on="gsis_id", right_on="player_id", how="left")

    # age-adjust the history component (the market curve already prices age)
    df["age_factor"] = [age_factor(p, a) for p, a in zip(df.pos, df.age)]
    df["hist_ppg_raw"] = df["hist_ppg"]
    df["hist_ppg"] = df["hist_ppg"] * df["age_factor"]

    df["market_ppg"] = np.nan
    for pos in ["QB", "RB", "WR", "TE"]:
        iso = fit_market_curve(df, pos)
        mask = df.pos == pos
        if iso is not None:
            df.loc[mask, "market_ppg"] = iso.predict(df.loc[mask, "pos_rank"].values)

    # empirical-Bayes shrinkage weight on own history
    w = (df.wgames.fillna(0) / MIN_GAMES_FULL_TRUST).clip(0, 1) * (1 - blend_floor)
    df["w_hist"] = w
    df["proj_ppg"] = np.where(df.hist_ppg.notna(),
                              w * df.hist_ppg + (1 - w) * df.market_ppg,
                              df.market_ppg)
    return df


# ---- empirically measured year-over-year predictive residual SD (points/game) ----
# Estimated from 2021-2025 nflverse data, players with >=8 games in the prior season.
RESID_SD_PPG = {"QB": 5.50, "RB": 4.55, "WR": 3.73, "TE": 2.66}
# Expected games played for starter-calibre players (of 17), measured 2022-2025.
EXP_GAMES = {"QB": 15.8, "RB": 15.6, "WR": 15.4, "TE": 14.9}


def add_season_projection(df):
    """PPG -> season points, with expected games and predictive uncertainty."""
    d = df.copy()
    # expected games: blend positional base with the player's own availability record
    base = d.pos.map(EXP_GAMES)
    own = (d.raw_games / 3.0).clip(upper=17)          # games/season over 3 yrs of history
    trust = (d.raw_games.fillna(0) / 30.0).clip(0, 0.6)
    d["exp_games"] = np.where(d.raw_games.notna(), trust * own + (1 - trust) * base, base)
    d["proj_pts"] = d.proj_ppg * d.exp_games
    # season-level SD: predictive PPG error scaled by games, plus games-played variance
    sd_ppg = d.pos.map(RESID_SD_PPG)
    d["proj_sd"] = np.sqrt((sd_ppg * d.exp_games) ** 2 + (d.proj_ppg * 2.6) ** 2)
    return d
