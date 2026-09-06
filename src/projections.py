"""
2026 projections in the league's custom scoring.

Decomposition
-------------
A player's projected league points factor cleanly into two parts:

    league_points  =  standard_PPR_production  x  scoring_translation_ratio

We estimate each from the source that knows it best:

  VOLUME / ROLE  <- the market (FantasyPros redraft consensus, 2026-09-04).
      Hundreds of analysts with camp reports, depth charts and injury news.
      We do not try to out-project them on who will get the touches; an
      earlier version that projected volume from a player's own history
      systematically underrated second-year players whose roles were
      changing (Egbuka, Burden, Golden) and overrated declining veterans.

  TRANSLATION RATIO  <- the player's own history, re-scored under THIS league.
      This is where the edge is. Measured over 2023-2025 the ratio has
      spread sd=0.087 at QB (Burrow 1.33 vs Richardson 0.91) because of the
      +0.27/completion and -0.50/incompletion terms, but only 0.025 at WR
      and 0.014 at TE. The market prices standard scoring; it does not
      price this. Shrunk toward the positional mean by sample size.
"""
import pandas as pd, numpy as np
from sklearn.isotonic import IsotonicRegression

OUT = "data/processed"
HIST_SEASONS = (2023, 2024, 2025)
RATIO_SHRINK_GAMES = 20          # empirical-Bayes prior strength, in games
SEASON_W = {2025: 0.60, 2024: 0.28, 2023: 0.12}

# empirically measured year-over-year predictive residual SD (points/game)
RESID_SD_PPG = {"QB": 5.50, "RB": 4.55, "WR": 3.73, "TE": 2.66}
EXP_GAMES = {"QB": 15.8, "RB": 15.6, "WR": 15.4, "TE": 14.9}

# Measured year-over-year regression slopes (see docs/methodology.md). A preseason
# rank is a *forecast* of a finish, so expected production is the finish-rank curve
# shrunk toward the positional mean by exactly this slope.
PERSIST_SLOPE = {"QB": 0.581, "RB": 0.710, "WR": 0.797, "TE": 0.795}


def finish_curve(sc, pos, n=60, seasons=(2022, 2023, 2024, 2025)):
    """Mean standard-PPR season points by end-of-season positional finish rank."""
    used = (sc.attempts.fillna(0) + sc.carries.fillna(0) + sc.targets.fillna(0)) > 0
    s = sc[used].groupby(["player_id", "position", "season"], as_index=False).agg(
        std=("fantasy_points_ppr", "sum"))
    rows = []
    for yr in seasons:
        v = np.sort(s[(s.position == pos) & (s.season == yr)]["std"].values)[::-1][:n]
        if len(v) == 0:
            continue
        rows.append(np.pad(v, (0, max(0, n - len(v))), constant_values=v[-1]))
    return np.vstack(rows).mean(axis=0)


def expected_by_rank(sc, pos, ranks):
    """E[standard-PPR season points | preseason positional rank]."""
    curve = finish_curve(sc, pos)
    m = curve[:24].mean()
    slope = PERSIST_SLOPE[pos]
    idx = np.clip(np.round(np.asarray(ranks)).astype(int) - 1, 0, len(curve) - 1)
    return m + slope * (curve[idx] - m)


def player_history(sc):
    """Per player: standard-PPR ppg, league ppg, games, and translation ratio."""
    h = sc[sc.season.isin(HIST_SEASONS)].copy()
    used = (h.attempts.fillna(0) + h.carries.fillna(0) + h.targets.fillna(0)) > 0
    h = h[used]
    h["w"] = h.season.map(SEASON_W)
    g = h.groupby("player_id").apply(lambda d: pd.Series({
        "lg_pts": d.pts.sum(),
        "std_pts": d.fantasy_points_ppr.sum(),
        "raw_games": len(d),
        "std_ppg_w": np.average(d.fantasy_points_ppr, weights=d.w),
        "wgames": d.w.sum(),
    }), include_groups=False).reset_index()
    g["ratio_raw"] = g.lg_pts / g.std_pts.replace(0, np.nan)
    return g


def build():
    sc = pd.read_parquet(f"{OUT}/weekly_scored.parquet")
    board = pd.read_csv(f"{OUT}/board_2026.csv", low_memory=False)
    board = board[board.pos.isin(["QB", "RB", "WR", "TE"])].copy()
    board["pos_rank"] = board.groupby("pos").ecr.rank(method="first")

    hist = player_history(sc)
    df = board.merge(hist, left_on="gsis_id", right_on="player_id", how="left")

    # ---- 1. market volume: preseason positional rank -> expected season points ----
    # Fitting historical ppg against current rank (an earlier approach) depressed
    # the top of the curve, because highly ranked young players with short or weak
    # histories sat in the fit set and dragged it down - Drake Maye, the market's
    # QB3, came out BELOW QB12 replacement. The finish-rank curve, shrunk by the
    # measured persistence slope, is the right estimator.
    df["market_std_pts"] = np.nan
    for pos in ["QB", "RB", "WR", "TE"]:
        m = (df.pos == pos)
        df.loc[m, "market_std_pts"] = expected_by_rank(sc, pos, df.loc[m, "pos_rank"].values)

    # ---- 2. player-specific scoring translation ratio, shrunk to positional mean ----
    pos_mean = df.groupby("pos").apply(
        lambda d: np.average(d.ratio_raw.dropna()) if d.ratio_raw.notna().any() else 1.1,
        include_groups=False)
    n = df.raw_games.fillna(0)
    prior = df.pos.map(pos_mean)
    df["ratio"] = ((n * df.ratio_raw.fillna(0) + RATIO_SHRINK_GAMES * prior)
                   / (n + RATIO_SHRINK_GAMES))
    df["ratio_prior"] = prior

    # ---- 3. combine ----
    # The finish-rank curve is already a SEASON total including games missed, so
    # availability must not be multiplied in again.
    df["proj_pts"] = df.market_std_pts * df.ratio
    df["proj_ppg"] = df.proj_pts / df.pos.map(EXP_GAMES)
    return df


def add_season_projection(df):
    d = df.copy()
    d["exp_games"] = d.pos.map(EXP_GAMES)
    sd_ppg = d.pos.map(RESID_SD_PPG)
    d["proj_sd"] = np.sqrt((sd_ppg * d.exp_games) ** 2 + (d.proj_ppg * 2.6) ** 2)
    return d
