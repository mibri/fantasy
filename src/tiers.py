"""Tier detection: find the natural cliffs in positional value.

A 'tier' is a group of players close enough in projected value that which one
you get barely matters. The decision that matters at the draft table is whether
a tier is about to run out - that is where waiting actually costs you points.
"""
import pandas as pd, numpy as np, sys
sys.path.insert(0, "/home/user/fantasy")


def tier_position(df, pos, max_players=45, n_tiers=7):
    """Jenks natural breaks (1-D k-means) on projected points.

    Preferred over a fixed gap threshold: a single huge gap (e.g. the TE1 cliff)
    otherwise swamps the threshold and collapses everyone else into one tier.
    """
    from sklearn.cluster import KMeans
    d = df[df.pos == pos].nlargest(max_players, "proj_pts").reset_index(drop=True)
    v = d.proj_pts.values.reshape(-1, 1)
    k = min(n_tiers, max(2, len(d) // 3))
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(v)
    # relabel clusters so tier 1 = highest scoring
    order = np.argsort(-km.cluster_centers_.ravel())
    remap = {c: i + 1 for i, c in enumerate(order)}
    d["tier"] = [remap[c] for c in km.labels_]
    return d.sort_values(["tier", "proj_pts"], ascending=[True, False]).reset_index(drop=True)


def all_tiers(path="data/processed/board_final.csv"):
    df = pd.read_csv(path)
    return pd.concat([tier_position(df, p) for p in ["QB", "RB", "WR", "TE", "K", "DST"]],
                     ignore_index=True)


if __name__ == "__main__":
    t = all_tiers()
    t.to_csv("data/processed/tiers.csv", index=False)
    for pos in ["RB", "WR", "TE", "QB"]:
        d = t[t.pos == pos]
        print(f"\n===== {pos} TIERS =====")
        for tier, g in d.groupby("tier"):
            if tier > 6: break
            names = ", ".join(g.player.head(7))
            extra = f" (+{len(g)-7} more)" if len(g) > 7 else ""
            print(f"  T{tier} [{g.proj_pts.min():.0f}-{g.proj_pts.max():.0f} pts, n={len(g)}]: {names}{extra}")
