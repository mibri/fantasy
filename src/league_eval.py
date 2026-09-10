"""Score every team in the actual 2026 draft against our board."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/fantasy"); sys.path.insert(0, "/home/user/fantasy/data/league")
import pandas as pd, numpy as np
from draft_2026 import ROSTERS, ME

BASE = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
N_FLEX = 2


def board():
    b = pd.read_csv("data/processed/board_final.csv").sort_values("vorp", ascending=False)
    b = b.reset_index(drop=True); b["rk"] = np.arange(1, len(b) + 1)
    b["key"] = b.player.str.lower().str.replace(".", "", regex=False).str.replace("'", "", regex=False)
    return b


def match(b, ini, last, tm, claimed=None):
    """A player belongs to exactly one roster, so already-claimed rows are excluded.
    Without this, a late-round 'B. Robinson' (Brian, WAS) silently resolved to
    Bijan Robinson, who went 1.3 to another team, and inflated that roster."""
    k = last.lower().replace(".", "").replace("'", "")
    c = b[b.key.str.contains(k, na=False, regex=False)]
    if claimed is not None:
        c = c[~c.rk.isin(claimed)]
    if len(c) > 1 and tm is not None:
        c2 = c[c.team == tm]
        if len(c2): c = c2
    if len(c) > 1:
        c2 = c[c.key.str.startswith(ini.lower())]
        if len(c2): c = c2
    return c.iloc[0] if len(c) else None


def lineup_points(rows):
    """Best 9: QB, RB, RB, WR, WR, TE, FLEX, FLEX, DEF."""
    by = {p: sorted([r.proj_pts for r in rows if r.pos == p], reverse=True) for p in
          ["QB", "RB", "WR", "TE", "DST"]}
    take = lambda p, n: by[p][:n] + [0.0] * max(0, n - len(by[p]))
    total = sum(sum(take(p, BASE[p])) for p in BASE)
    flex = sorted(by["RB"][2:] + by["WR"][2:] + by["TE"][1:], reverse=True)
    total += sum((flex + [0.0, 0.0])[:N_FLEX])
    total += (by["DST"][0] if by["DST"] else 0.0)
    return total


def main():
    b = board(); out = []; claimed = set()
    # resolve in draft order so earlier picks claim the ambiguous names first
    for team, players in ROSTERS.items():
        rows = []
        for p in players:
            r = match(b, *p, claimed=claimed)
            if r is not None:
                claimed.add(int(r.rk)); rows.append(r)
        out.append(dict(team=team, pts=lineup_points(rows),
                        vorp=sum(max(r.vorp, 0) for r in rows),
                        best=min(int(r.rk) for r in rows),
                        rows=rows))
    out.sort(key=lambda x: -x["pts"])
    print("=== LEAGUE POWER RANKINGS (projected starting lineup, your scoring) ===\n")
    print(f"{'#':>2}  {'team':<16}{'starters':>9}{'vs avg':>8}   top picks")
    avg = np.mean([o["pts"] for o in out])
    for i, o in enumerate(out, 1):
        top = ", ".join(r.player for r in sorted(o["rows"], key=lambda r: r.rk)[:3])
        mark = "  <-- YOU" if o["team"] == ME else ""
        print(f"{i:>2}. {o['team']:<16}{o['pts']:>9.0f}{o['pts']-avg:>+8.0f}   {top[:46]}{mark}")
    print(f"\n    league average: {avg:.0f}")
    return out


if __name__ == "__main__":
    main()


# ---- season simulation for the REAL league (9 starters, no kicker) ----
SD_A = {"QB": 7.71, "RB": 2.90, "WR": 2.53, "TE": 1.50, "DST": 6.00}
SD_B = {"QB": 0.171, "RB": 0.391, "WR": 0.432, "TE": 0.513, "DST": 0.30}
REG, PLAYOFF = list(range(14)), [14, 15, 16]


def simulate_league(out, S=4000, seed=0):
    rng = np.random.default_rng(seed)
    T = len(out)
    names = [o["team"] for o in out]
    # per-team player arrays
    ppg, sd_s, pos = [], [], []
    for o in out:
        r = o["rows"]
        ppg.append([x.proj_pts / 17 for x in r])
        sd_s.append([x.proj_sd / 17 for x in r])
        pos.append([x.pos for x in r])
    W = 17
    totals = np.zeros((S, T, W))
    for t in range(T):
        n = len(ppg[t])
        mu = np.array(ppg[t])[None, :]                        # (1,n)
        sd = np.array(sd_s[t])[None, :]
        true = np.clip(mu + rng.normal(0, 1, (S, n)) * sd, 0.5, None)
        a = np.array([SD_A.get(p, 4.0) for p in pos[t]])[None, :]
        bcoef = np.array([SD_B.get(p, 0.35) for p in pos[t]])[None, :]
        sw = a + bcoef * true
        shape = np.clip((true / np.maximum(sw, 1e-3)) ** 2, 0.4, 200)
        scale = true / shape
        wk = rng.gamma(shape[:, :, None], scale[:, :, None], size=(S, n, W))
        wk *= (rng.random((S, n, W)) < 0.92)
        # optimal weekly lineup: QB, RB, RB, WR, WR, TE, FLEX, FLEX, DEF
        P = np.array(pos[t])
        for slot_pos, k in [("QB", 1), ("RB", 2), ("WR", 2), ("TE", 1), ("DST", 1)]:
            idx = np.where(P == slot_pos)[0]
            if len(idx) == 0: continue
            v = np.sort(wk[:, idx, :], axis=1)[:, ::-1, :]
            totals[:, t, :] += v[:, :k, :].sum(axis=1)
        # flex from leftovers
        fl = []
        for p, k in [("RB", 2), ("WR", 2), ("TE", 1)]:
            idx = np.where(P == p)[0]
            if len(idx) > k:
                v = np.sort(wk[:, idx, :], axis=1)[:, ::-1, :]
                fl.append(v[:, k:, :])
        if fl:
            pool = np.sort(np.concatenate(fl, axis=1), axis=1)[:, ::-1, :]
            totals[:, t, :] += pool[:, :N_FLEX, :].sum(axis=1)
    # schedule + playoffs
    teams = list(range(T)); sched = []
    for _ in range(14):
        opp = [0] * T
        for i in range(T // 2):
            x, y = teams[i], teams[T - 1 - i]; opp[x] = y; opp[y] = x
        sched.append(opp); teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    sched = np.array(sched)
    wins = np.zeros((S, T), int)
    for wi in range(14):
        for t in range(T):
            wins[:, t] += totals[:, t, wi] > totals[:, sched[wi, t], wi]
    pts = totals[:, :, REG].sum(axis=2)
    seed_ = np.argsort(-(wins * 1e6 + pts), axis=1)
    g = lambda a, b, w: np.where(totals[np.arange(S), a, w] >= totals[np.arange(S), b, w], a, b)
    s_ = [seed_[:, i] for i in range(6)]
    w1 = g(s_[2], s_[5], 14); w2 = g(s_[3], s_[4], 14)
    w3 = g(s_[0], w2, 15);    w4 = g(s_[1], w1, 15)
    champ = g(w3, w4, 16)
    return names, wins, champ, seed_


def report():
    out = main()
    names, wins, champ, seed_ = simulate_league(out)
    S = len(champ)
    print("\n=== SIMULATED SEASON, YOUR ACTUAL LEAGUE (4000 seasons) ===\n")
    print(f"{'team':<16}{'title':>8}{'playoffs':>10}{'wins':>7}")
    rows = []
    for i, nm in enumerate(names):
        title = (champ == i).mean()
        po = (seed_[:, :6] == i).any(axis=1).mean()
        rows.append((nm, title, po, wins[:, i].mean()))
    for nm, ti, po, wi in sorted(rows, key=lambda x: -x[1]):
        mark = "  <-- YOU" if nm == ME else ""
        print(f"{nm:<16}{ti:>8.3f}{po:>10.3f}{wi:>7.2f}{mark}")
    print(f"\n  (1-in-12 baseline = 0.083)")
