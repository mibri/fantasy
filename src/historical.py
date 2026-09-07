"""
Point-in-time boards: what the model would have produced before a given season,
using ONLY information available at that moment.

Every ingredient is recomputed per season - the finish-rank curve, the
persistence slopes, and each player's scoring-translation ratio - so nothing
leaks backwards from seasons that had not happened yet.
"""
import pandas as pd, numpy as np, re, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/fantasy")
from config.league import LEAGUE

# preseason consensus snapshots, taken days before Week 1 (2025 is the latest available)
SNAPSHOT = {2021: "2021-09-09", 2022: "2022-09-09", 2023: "2023-09-08",
            2024: "2024-09-06", 2025: "2025-08-08"}
RATIO_PRIOR_GAMES = 20
POSITIONS = ("QB", "RB", "WR", "TE")


def norm(s):
    s = str(s).lower(); s = re.sub(r"[.'`,]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _crosswalk():
    ids = pd.read_csv("data/raw/dp_ids.csv", low_memory=False)
    ids["mn"] = ids.name.map(norm)
    return ids.dropna(subset=["gsis_id"]).drop_duplicates("mn").set_index("mn").gsis_id


def finish_curve(tot, pos, before, n=60, lookback=4):
    """Standard-PPR season points by finish rank, from the seasons before `before`."""
    yrs = sorted([y for y in tot.season.unique() if y < before])[-lookback:]
    rows = []
    for y in yrs:
        v = np.sort(tot[(tot.position == pos) & (tot.season == y)]["sp"].values)[::-1][:n]
        if len(v):
            rows.append(np.pad(v, (0, max(0, n - len(v))), constant_values=v[-1]))
    return np.vstack(rows).mean(axis=0) if rows else None


def persistence_slope(tot, pos, before):
    """Year-over-year regression slope of season points, using only prior seasons."""
    t = tot[(tot.position == pos) & (tot.season < before) & (tot.g >= 8)]
    nxt = t.copy(); nxt["season"] = nxt.season - 1
    j = t.merge(nxt[["player_id", "season", "sp"]], on=["player_id", "season"],
                suffixes=("", "_n"))
    if len(j) < 25:
        return 0.6
    return float(np.clip(np.polyfit(j["sp"], j["sp_n"], 1)[0], 0.2, 1.0))


def ratio_prior(tot, before):
    """Scoring-translation ratio per player from strictly prior seasons, shrunk."""
    p = tot[tot.season < before].groupby("player_id").agg(
        lg=("lg", "sum"), sp=("sp", "sum"), g=("g", "sum")).reset_index()
    p["ratio_raw"] = p.lg / p.sp.replace(0, np.nan)
    return p


def board_for(year, tot=None, ecr=None, xw=None):
    """The draftable board as it would have looked just before `year` kicked off."""
    if tot is None:
        tot = pd.read_parquet("data/processed/season_totals.parquet")
    if ecr is None:
        ecr = pd.read_parquet("data/processed/ecr_history.parquet")
    if xw is None:
        xw = _crosswalk()

    day = SNAPSHOT[year]
    e = ecr[ecr.scrape_date.astype(str).str[:10] == day].copy()
    e["pos"] = e["pos"].astype(str).str.replace(r"\d+", "", regex=True)
    e = e[e.pos.isin(POSITIONS)].copy()
    e["gsis_id"] = e.player.map(norm).map(xw)
    e = e.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")
    e["pos_rank"] = e.groupby("pos").ecr.rank(method="first")

    rp = ratio_prior(tot, year)
    e = e.merge(rp, left_on="gsis_id", right_on="player_id", how="left")

    e["proj_std"] = np.nan
    for pos in POSITIONS:
        m = e.pos == pos
        curve = finish_curve(tot, pos, year)
        if curve is None:
            continue
        base = curve[:24].mean()
        slope = persistence_slope(tot, pos, year)
        idx = np.clip(np.round(e.loc[m, "pos_rank"].values).astype(int) - 1, 0, len(curve) - 1)
        e.loc[m, "proj_std"] = base + slope * (curve[idx] - base)

    # ratio: sample-size shrinkage, then capped by measured persistence of the ratio
    pos_mean = e.groupby("pos").ratio_raw.transform("mean")
    n = e.g.fillna(0)
    w = (n / (n + RATIO_PRIOR_GAMES)).fillna(0.0)
    rslope = ratio_persistence(tot, year)
    e["ratio"] = pos_mean + e.pos.map(rslope).fillna(0.4) * w * (
        e.ratio_raw.fillna(pos_mean) - pos_mean)
    e["proj_pts"] = e.proj_std * e.ratio
    return e.dropna(subset=["proj_pts"]).reset_index(drop=True)


def ratio_persistence(tot, before):
    """How much of a player's scoring ratio carries forward, from prior seasons only."""
    out = {}
    t = tot[tot.season < before].copy()
    t["ratio"] = t.lg / t["sp"].replace(0, np.nan)
    t = t[(t.g >= 8) & t.ratio.notna()]
    nxt = t.copy(); nxt["season"] = nxt.season - 1
    j = t.merge(nxt[["player_id", "season", "ratio"]], on=["player_id", "season"],
                suffixes=("", "_n"))
    for pos in POSITIONS:
        x = j[j.position == pos]
        out[pos] = float(np.clip(np.polyfit(x.ratio, x.ratio_n, 1)[0], 0.0, 1.0)) \
            if len(x) >= 25 else 0.4
    return out
