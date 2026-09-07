"""
Backtest against REAL seasons.

Draft using only pre-season information from year Y, then score the roster on
what actually happened in year Y. The season is no longer simulated, so the
week-to-week variance is the real NFL's rather than anything I chose.

Only the opponents' drafts remain simulated - we have no record of their draft
rooms - but outcomes are real.

Kickers and defenses are excluded: they are absent from player-level weekly
data, they are last-two-round picks, and they add near-identical noise to every
team. The draft is therefore 13 rounds with 8 starters
(QB, RB, RB, WR, WR, TE, FLEX, FLEX).
"""
import pandas as pd, numpy as np, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/fantasy")
from src.historical import board_for, SNAPSHOT

POS = ["QB", "RB", "WR", "TE"]
BASE = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
N_FLEX, N_TEAMS, N_ROUNDS = 2, 12, 13
CAP = {"QB": 2, "RB": 6, "WR": 7, "TE": 3}
MIN_NEED = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
REG_WEEKS, PLAYOFF_WEEKS = list(range(1, 15)), [15, 16, 17]


def snake_order():
    o = []
    for r in range(N_ROUNDS):
        o.extend(range(N_TEAMS) if r % 2 == 0 else range(N_TEAMS - 1, -1, -1))
    return o


def replacement_levels(proj, pos):
    """Flex-aware replacement, solved on this year's board."""
    rep = {}
    starters = {p: BASE[p] * N_TEAMS for p in POS}
    idx = np.argsort(-proj)
    taken = set()
    for p in POS:
        c = [i for i in idx if pos[i] == p][: starters[p]]
        taken.update(c)
    pool = [i for i in idx if pos[i] in ("RB", "WR", "TE") and i not in taken]
    for i in pool[: N_FLEX * N_TEAMS]:
        starters[pos[i]] += 1
    for p in POS:
        v = np.sort(proj[pos == p])[::-1]
        k = starters[p]
        rep[p] = float(v[k]) if len(v) > k else float(v[-1])
    return rep, starters


def draft(board, my_slot, policy, seed, noise=1.0, all_same=False, smart_slots=()):
    """One 12-team snake draft. Returns (N_TEAMS, N_ROUNDS) board indices.

    Only `my_slot` uses `policy`; the other eleven follow noisy consensus, unless
    all_same=True, which makes every seat draft identically - the setup for
    asking how much of an outcome is luck when nobody has a process edge.
    """
    rng = np.random.default_rng(seed)
    proj = board.proj_pts.values.astype(float)
    pos = board.pos.values
    ecr = board.ecr.values.astype(float)
    sd = np.nan_to_num(board.sd.values.astype(float), nan=8.0).clip(1, 40)
    rep, _ = replacement_levels(proj, pos)
    perceived = ecr + rng.normal(0, sd * noise)
    pidx = {g: i for i, g in enumerate(POS)}
    pc = np.array([pidx[g] for g in pos])            # position as an index
    rep_arr = np.array([rep[p] for p in POS])
    rep_of = rep_arr[pc]

    P = len(board)
    avail = np.ones(P, bool)
    rosters = np.full((N_TEAMS, N_ROUNDS), -1, int)
    counts = [{p: 0 for p in POS} for _ in range(N_TEAMS)]
    vals = [{p: [] for p in POS} for _ in range(N_TEAMS)]

    for n, t in enumerate(snake_order()):
        rnd = n // N_TEAMS
        legal = avail.copy()
        for p in POS:
            if counts[t][p] >= CAP[p]:
                legal &= (pos != p)
        need = {p: max(0, MIN_NEED[p] - counts[t][p]) for p in POS}
        if sum(need.values()) >= (N_ROUNDS - rnd):
            m = np.zeros(P, bool)
            for p in POS:
                if need[p] > 0:
                    m |= (pos == p)
            legal &= m
        if not legal.any():
            legal = avail.copy()

        uses_policy = all_same or (t == my_slot) or (t in smart_slots)
        if policy == "adp" or not uses_policy:
            score = -perceived
        else:
            # marginal value to the optimal starting lineup, floored at replacement
            th = {}
            flex_pool = sorted(
                [v for p in ("RB", "WR", "TE") for v in vals[t][p][BASE[p]:]], reverse=True)
            f = [flex_pool[i] if i < len(flex_pool) else 0.0 for i in range(N_FLEX)]
            flex_rep = max(rep["RB"], rep["WR"], rep["TE"])
            fmin = max(min(f) if f else 0.0, flex_rep)
            for p in POS:
                own = vals[t][p]
                base_slot = min(own[:BASE[p]]) if len(own) >= BASE[p] else 0.0
                base_slot = max(base_slot, rep[p])
                th[p] = base_slot if p == "QB" else min(base_slot, fmin)
            th_arr = np.array([th[p] for p in POS])
            depth_arr = np.array([max(0, counts[t][p] - MIN_NEED[p]) for p in POS])
            gain = np.maximum(proj - th_arr[pc], 0.0)
            bench = 0.25 * (0.45 ** depth_arr[pc]) * np.maximum(proj - rep_of, 0.0)
            score = gain + bench
            if policy == "model_vona":
                gap = 2 * (N_TEAMS - 1 - t) + 1 if rnd % 2 == 0 else 2 * t + 1
                soon = np.argsort(np.where(avail, perceived, 1e9))[:gap]
                nxt = {}
                for p in POS:
                    k = int((pos[soon] == p).sum())
                    rest = np.sort(proj[(pos == p) & avail])[::-1]
                    nxt[p] = float(rest[k]) if len(rest) > k else rep[p]
                nxt_arr = np.array([max(0.0, nxt[p] - rep[p]) for p in POS])
                score = score - 0.5 * nxt_arr[pc]

        score = np.where(legal, score, -np.inf)
        pick = int(np.argmax(score))
        avail[pick] = False
        rosters[t, rnd] = pick
        counts[t][pos[pick]] += 1
        vals[t][pos[pick]] = sorted(vals[t][pos[pick]] + [proj[pick]], reverse=True)
    return rosters


def actuals(board, year, weekly=None):
    """Real weekly points and availability for every player on the board."""
    if weekly is None:
        weekly = pd.read_parquet("data/processed/weekly_scored_all.parquet")
    w = weekly[(weekly.season == year)]
    use = (w.attempts.fillna(0) + w.carries.fillna(0) + w.targets.fillna(0)) > 0
    w = w[use]
    P, W = len(board), 18
    pts = np.zeros((P, W + 1)); played = np.zeros((P, W + 1), bool)
    pos_of = {g: i for i, g in enumerate(board.gsis_id.values)}
    for pid, wk, v in zip(w.player_id.values, w.week.values, w.pts.values):
        i = pos_of.get(pid)
        if i is not None and 1 <= wk <= W:
            pts[i, wk] += v; played[i, wk] = True
    return pts, played


def lineups(roster, proj, pts, played, hindsight=False):
    """Weekly points scored by one team, choosing starters WITHOUT hindsight.

    A manager knows who is inactive but not who will score, so starters are the
    highest PRE-SEASON PROJECTED players among those available. `hindsight=True`
    instead picks the actual best - an upper bound, not a fair evaluation.
    """
    pos = proj["pos"][roster]
    pr = proj["proj"][roster]
    p = pts[roster]                      # (R, W+1)
    av = played[roster]
    W = p.shape[1]
    key = p if hindsight else np.repeat(pr[:, None], W, axis=1)
    key = np.where(av, key, -np.inf)
    order = np.argsort(-key, axis=0)     # (R, W) priority per week

    total = np.zeros(W)
    for w in range(W):
        o = order[:, w]
        filled = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
        flex = 0; s = 0.0
        for i in o:
            if not av[i, w]:
                continue
            g = pos[i]
            if filled[g] < BASE[g]:
                filled[g] += 1; s += p[i, w]
            elif g in ("RB", "WR", "TE") and flex < N_FLEX:
                flex += 1; s += p[i, w]
        total[w] = s
    return total


def round_robin(n=N_TEAMS, weeks=len(REG_WEEKS)):
    teams = list(range(n)); sched = []
    for _ in range(weeks):
        opp = [0] * n
        for i in range(n // 2):
            a, b = teams[i], teams[n - 1 - i]
            opp[a] = b; opp[b] = a
        sched.append(opp)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return np.array(sched)


def play_league(team_weeks, sched):
    """team_weeks (T, W+1) real points -> (champion, wins, points)."""
    reg = team_weeks[:, REG_WEEKS]
    wins = np.zeros(N_TEAMS, int)
    for wi in range(len(REG_WEEKS)):
        for t in range(N_TEAMS):
            if reg[t, wi] > reg[sched[wi, t], wi]:
                wins[t] += 1
    pts = reg.sum(axis=1)
    seed = np.argsort(-(wins * 1e6 + pts))
    g = lambda a, b, wk: a if team_weeks[a, wk] >= team_weeks[b, wk] else b
    w1 = g(seed[2], seed[5], 15); w2 = g(seed[3], seed[4], 15)
    w3 = g(seed[0], w2, 16);      w4 = g(seed[1], w1, 16)
    return g(w3, w4, 17), wins, pts
