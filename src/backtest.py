"""
A real out-of-sample backtest, now that historical preseason ECR is available.

For each season 2021-2025 we take the FantasyPros consensus snapshot from the
days before Week 1, use ONLY data available before that season to compute each
player's scoring-translation ratio, and ask a single question:

    after controlling for what the market already knew, does the ratio
    predict who outscored whom in this league's scoring?

That is the incremental-validity test for the model's one genuine claim.
"""
import pandas as pd, numpy as np, re, sys, warnings; warnings.filterwarnings("ignore")

SNAP = {2021: "2021-09-09", 2022: "2022-09-09", 2023: "2023-09-08",
        2024: "2024-09-06", 2025: "2025-08-08"}


def norm(s):
    s = str(s).lower(); s = re.sub(r"[.'`,]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    return re.sub(r"\s+", " ", s).strip()


def build():
    ecr = pd.read_parquet("data/processed/ecr_history.parquet")
    ecr["scrape_date"] = pd.to_datetime(ecr.scrape_date).dt.strftime("%Y-%m-%d")
    ids = pd.read_csv("data/raw/dp_ids.csv", low_memory=False)
    ids["mn"] = ids.name.map(norm)
    xw = ids.dropna(subset=["gsis_id"]).drop_duplicates("mn").set_index("mn").gsis_id
    tot = pd.read_parquet("data/processed/season_totals.parquet")

    rows = []
    for yr, day in SNAP.items():
        e = ecr[ecr.scrape_date == day].copy()
        if e.empty:
            continue
        e["pos"] = e["pos"].astype(str).str.replace(r"\d+", "", regex=True)
        e = e[e.pos.isin(["QB", "RB", "WR", "TE"])]
        e["gsis_id"] = e.player.map(norm).map(xw)
        e = e.dropna(subset=["gsis_id"])
        e["season"] = yr

        # realised outcome that season
        act = tot[tot.season == yr][["player_id", "lg", "g"]]
        e = e.merge(act, left_on="gsis_id", right_on="player_id", how="left")
        e["lg"] = e.lg.fillna(0.0)          # undrafted/never played = 0 points

        # ratio from STRICTLY PRIOR seasons only
        prior = tot[tot.season < yr].groupby("player_id").agg(
            lg_p=("lg", "sum"), sp_p=("sp", "sum"), g_p=("g", "sum")).reset_index()
        prior["ratio_prior"] = prior.lg_p / prior.sp_p.replace(0, np.nan)
        e = e.merge(prior, left_on="gsis_id", right_on="player_id", how="left")
        rows.append(e)
    return pd.concat(rows, ignore_index=True)


def main():
    df = build()
    df["pos_rank"] = df.groupby(["season", "pos"]).ecr.rank(method="first")
    print("=== backtest sample ===")
    print(df.groupby("season").agg(players=("player", "size"),
                                   with_prior=("ratio_prior", "count")).to_string())

    # shrink the prior ratio the same way the live model does
    out = []
    for pos in ["QB", "RB", "WR", "TE"]:
        d = df[(df.pos == pos) & df.ratio_prior.notna() & (df.g_p >= 16)].copy()
        if len(d) < 60:
            continue
        # control for the market: expected points given preseason positional rank,
        # fitted WITHIN season so it carries no look-ahead across seasons
        d["mkt"] = d.groupby("season").apply(
            lambda g: pd.Series(np.interp(g.pos_rank,
                                          np.sort(g.pos_rank),
                                          g.sort_values("pos_rank").lg.rolling(9, center=True,
                                                                               min_periods=3).mean()),
                                index=g.index), include_groups=False).reset_index(level=0, drop=True)
        d = d.dropna(subset=["mkt"])
        r_m = np.corrcoef(d.mkt, d.lg)[0, 1]
        resid = d.lg - d.mkt
        r_ratio = np.corrcoef(d.ratio_prior, resid)[0, 1]
        # does ratio beat market alone?
        pred2 = d.mkt * (d.ratio_prior / d.ratio_prior.mean())
        r_2 = np.corrcoef(pred2, d.lg)[0, 1]
        out.append((pos, len(d), r_m, r_ratio, r_2))
    print()
    print("=== INCREMENTAL VALIDITY OF THE SCORING RATIO (out of sample) ===")
    print(f"{'pos':4s} {'n':>5s} {'market alone':>13s} {'ratio vs resid':>15s} {'market x ratio':>15s}")
    for pos, n, a, b, c in out:
        print(f"{pos:4s} {n:5d} {a:13.3f} {b:15.3f} {c:15.3f}")
    df.to_parquet("data/processed/backtest.parquet")


if __name__ == "__main__":
    main()
