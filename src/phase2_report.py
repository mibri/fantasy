"""Report Phase 2 with year-clustered uncertainty.

Draft seeds within a season share that season's actual results, so the naive
binomial standard error is far too small. The effective sample is the five
seasons: compute each season's mean, then the spread across seasons.
"""
import pandas as pd, numpy as np, sys; sys.path.insert(0, "/home/user/fantasy")


def clustered(df, value, by="year"):
    per = df.groupby(by)[value].mean()
    return float(per.mean()), float(per.std(ddof=1) / np.sqrt(len(per))), per


def main():
    a = pd.read_csv("data/processed/phase2_policy.csv")
    b = pd.read_csv("data/processed/phase2_luck.csv")

    print("=== A. POLICY COMPARISON ON REAL SEASONS ===")
    print(f"   {len(a)} drafted teams | {a.year.nunique()} seasons | baseline title rate = {1/12:.3f}\n")
    print(f"{'policy':12s} {'title':>7s} {'±clust':>8s} {'playoffs':>9s} {'wins':>6s}")
    for pol in ["model_vona", "model", "adp"]:
        d = a[a.policy == pol]
        m, se, _ = clustered(d, "champ")
        pm, _, _ = clustered(d, "playoff")
        wm, _, _ = clustered(d, "wins")
        print(f"{pol:12s} {m:7.4f} {se:8.4f} {pm:9.3f} {wm:6.2f}")

    print("\n   title rate by season:")
    piv = a.pivot_table(index="year", columns="policy", values="champ")
    print(piv.round(3).to_string())

    mv = a[a.policy == "model_vona"]; ad = a[a.policy == "adp"]
    d1, d2 = clustered(mv, "champ")[2], clustered(ad, "champ")[2]
    diff = d1 - d2
    print(f"\n   model_vona - adp, per season: {[round(x,3) for x in diff.tolist()]}")
    print(f"   mean {diff.mean():+.4f}  clustered SE {diff.std(ddof=1)/np.sqrt(len(diff)):.4f} "
          f"-> t = {diff.mean()/(diff.std(ddof=1)/np.sqrt(len(diff))):+.2f}  (4 df)")

    print("\n=== B. SKILL vs LUCK, REAL SEASONS, EQUAL-SKILL LEAGUE ===")
    m, se, per = clustered(b, "corr")
    print(f"   corr(preseason roster strength, actual wins) = {m:+.3f} ± {se:.3f}")
    print(f"     -> explains {100*m**2:.1f}% of win variance in a season")
    print(f"   by season: {[round(x,3) for x in per.tolist()]}")
    b["best_won"] = (b.champ == b.strongest).astype(int)
    m2, se2, per2 = clustered(b, "best_won")
    print(f"\n   strongest preseason roster wins the title = {100*m2:.1f}% ± {100*se2:.1f}%   (chance 8.3%)")
    print(f"   by season: {[round(100*x,1) for x in per2.tolist()]}")
    print(f"\n   champion's preseason strength rank (0 = strongest, 11 = weakest):")
    vc = b.champ_strength_rank.value_counts(normalize=True).sort_index()
    print("   " + "  ".join(f"{i}:{100*v:.0f}%" for i, v in vc.items()))
    print(f"   mean rank of champion = {b.champ_strength_rank.mean():.2f} (5.5 = pure chance)")
    print(f"\n   roster-strength spread within a league = {b.spread.mean():.0f} pts "
          f"on a mean of {b['mean'].mean():.0f} ({100*b.spread.mean()/b['mean'].mean():.1f}%)")


if __name__ == "__main__":
    main()
