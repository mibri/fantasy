"""Flex-aware value over replacement for the league's exact starting lineup."""
import pandas as pd, numpy as np, sys
sys.path.insert(0, "/home/user/fantasy")
from config.league import LEAGUE

def replacement_levels(df, value_col="proj_pts"):
    """Solve for how many of each position are startable once FLEX is allocated."""
    n = LEAGUE["n_teams"]; st = LEAGUE["starters"]
    base = {"QB": st["QB"] * n, "RB": st["RB"] * n, "WR": st["WR"] * n, "TE": st["TE"] * n}
    d = df.sort_values(value_col, ascending=False)

    # players already locked into a base starting slot
    locked = {p: set(d[d.pos == p].head(base[p]).index) for p in base}
    used = set().union(*locked.values())
    # remaining flex-eligible pool competes for the FLEX slots
    pool = d[d.pos.isin(LEAGUE["flex_eligible"]) & ~d.index.isin(used)]
    n_flex = st["FLEX"] * n
    flex_take = pool.head(n_flex)
    counts = {p: base[p] + int((flex_take.pos == p).sum()) for p in base}

    repl = {}
    for p, k in counts.items():
        pos_d = d[d.pos == p][value_col].values
        # replacement = the best player who does NOT start (next man up)
        repl[p] = float(pos_d[k]) if len(pos_d) > k else float(pos_d[-1])
    return counts, repl


def add_vorp(df, value_col="proj_pts"):
    counts, repl = replacement_levels(df, value_col)
    d = df.copy()
    d["replacement"] = d.pos.map(repl)
    d["vorp"] = d[value_col] - d["replacement"]
    return d, counts, repl
