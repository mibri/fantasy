"""
Vectorised Monte-Carlo draft + season simulator (JAX).

Simulates S independent 12-team snake drafts in parallel. Opponents draft from a
noisy perception of consensus rank (calibrated to FantasyPros expert dispersion)
under realistic roster rules. Our team follows a parameterised policy.
"""
import jax, jax.numpy as jnp, numpy as np, pandas as pd

QB, RB, WR, TE, K, DST = range(6)
POS_NAMES = ["QB", "RB", "WR", "TE", "K", "DST"]
CAP = jnp.array([2, 6, 7, 3, 1, 1])
MIN_NEED = jnp.array([1, 2, 2, 1, 1, 1])
N_TEAMS, N_ROUNDS = 12, 15
N_PICKS = N_TEAMS * N_ROUNDS
KDST_MIN_ROUND = 12
MAXPOS = 8
# Bench-depth weight, TUNED by championship probability rather than assumed
# (data/processed/tuning.csv, 4 slots x 700 sims per point, SE ~0.007).
# Once the season simulation models a WAIVER WIRE, this parameter stops
# mattering: the whole grid 0.0-1.2 scores 0.185-0.192, every point within ~1 SE.
# That is the honest result. Any value in 0-0.5 is equivalent; 0.25 is used.
#
# Two earlier sweeps pointed elsewhere, and both were artefacts:
#   * a very LARGE weight looked optimal while the empty-slot threshold bug
#     over-scaled the starter term (only a big depth term could balance it);
#   * a small weight looked clearly optimal once that was fixed, but only
#     because a manager with an injured starter still scored ZERO from that
#     slot, which massively overstates the worth of rostering backups.
# With free agency modelled, bench depth is close to free - as it is in a real
# league - and the parameter washes out.
BENCH_BASE, BENCH_DECAY = 0.25, 0.45
MLV_ADDITIVE = True


def snake_order(n_teams=N_TEAMS, n_rounds=N_ROUNDS):
    o = []
    for r in range(n_rounds):
        o.extend(range(n_teams) if r % 2 == 0 else range(n_teams - 1, -1, -1))
    return jnp.array(o)


def load_board(path="data/processed/board_final.csv"):
    b = pd.read_csv(path)
    b = b[b.proj_pts.notna()].reset_index(drop=True)
    pos = b.pos.map({p: i for i, p in enumerate(POS_NAMES)}).values
    repl = b.groupby("pos").replacement.first()
    return b, dict(
        pos=jnp.array(pos),
        ecr=jnp.array(b.ecr.values, dtype=jnp.float32),
        ecr_sd=jnp.array(np.nan_to_num(b.sd.values, nan=8.0).clip(1.0, 40.0), dtype=jnp.float32),
        proj=jnp.array(b.proj_pts.values, dtype=jnp.float32),
        vorp=jnp.array(b.vorp.values, dtype=jnp.float32),
        proj_sd=jnp.array(b.proj_sd.values, dtype=jnp.float32),
        repl=jnp.array([repl.get(p, 0.0) for p in POS_NAMES], dtype=jnp.float32),
    )


def allowed_mask(avail, pos_count, pos, rnd):
    cap_ok = pos_count[:, pos] < CAP[pos][None, :]
    early_block = ((pos == K) | (pos == DST))[None, :] & (rnd < KDST_MIN_ROUND)
    need = jnp.maximum(0, MIN_NEED[None, :] - pos_count)
    must_fill = need.sum(axis=1) >= (N_ROUNDS - rnd)
    ok = avail & cap_ok & (~early_block)
    return jnp.where(must_fill[:, None], ok & (need[:, pos] > 0), ok)


def lineup_and_thresholds(rv, repl=None):
    """rv: (S,6,MAXPOS) sorted desc. Returns (lineup_total, threshold_by_pos).

    `repl` clamps every threshold to at least replacement level. Without it an
    EMPTY slot has threshold 0, so a candidate's marginal value equals his full
    projected points - which systematically flatters quarterbacks, who score the
    most raw points. An empty slot can always be filled later from the waiver
    wire, so the true alternative is a replacement-level player, never nothing.
    """
    qb, rb, wr, te, k, dst = (rv[:, i, :] for i in range(6))
    flex = jnp.sort(jnp.concatenate([rb[:, 2:], wr[:, 2:], te[:, 1:]], axis=1), axis=1)[:, ::-1]
    f1, f2 = flex[:, 0], flex[:, 1]
    total = (qb[:, 0] + rb[:, 0] + rb[:, 1] + wr[:, 0] + wr[:, 1] + te[:, 0]
             + f1 + f2 + k[:, 0] + dst[:, 0])
    fmin = jnp.minimum(f1, f2)
    if repl is None:
        th = jnp.stack([qb[:, 0],
                        jnp.minimum(jnp.minimum(rb[:, 0], rb[:, 1]), fmin),
                        jnp.minimum(jnp.minimum(wr[:, 0], wr[:, 1]), fmin),
                        jnp.minimum(te[:, 0], fmin),
                        k[:, 0], dst[:, 0]], axis=1)
        return total, th

    # Each slot is floored by what could replace THAT slot for free. A base slot
    # is floored at its own position's replacement, but a FLEX slot is floored at
    # the best replacement-level flex-eligible player - max(RB, WR, TE) - because
    # any of the three can fill it. Flooring a tight end's flex option at the TE
    # replacement (158) rather than the WR one (197) made tight ends look far
    # better as flex fillers than they are, and the simulator over-drafted them.
    flex_repl = jnp.max(repl[jnp.array([RB, WR, TE])])
    ff = jnp.maximum(fmin, flex_repl)
    th = jnp.stack([
        jnp.maximum(qb[:, 0], repl[QB]),
        jnp.minimum(jnp.maximum(jnp.minimum(rb[:, 0], rb[:, 1]), repl[RB]), ff),
        jnp.minimum(jnp.maximum(jnp.minimum(wr[:, 0], wr[:, 1]), repl[WR]), ff),
        jnp.minimum(jnp.maximum(te[:, 0], repl[TE]), ff),
        jnp.maximum(k[:, 0], repl[K]),
        jnp.maximum(dst[:, 0], repl[DST])], axis=1)
    return total, th


def kth_available_vorp(avail, pos_order, pos_vorp, k):
    a = avail[:, pos_order]
    cum = jnp.cumsum(a.astype(jnp.int32), axis=1)
    return ((a & (cum == k[:, None])) * pos_vorp[None, :]).sum(axis=1)


POLICIES = ("MLV", "MLV_VONA", "BPA", "ECR", "ZERO_RB", "HERO_RB", "ROBUST_RB")


def policy_scores(name, c, params=None):
    params = params or {}
    pos, vorp, rnd = c["pos"], c["vorp"], c["rnd"]
    S, P = c["avail"].shape
    proj = c["proj"]
    base_vorp = jnp.broadcast_to(vorp[None, :], (S, P))

    # --- marginal value to the optimal starting lineup, plus bench option value ---
    th_p = c["th"][:, pos]                              # (S,P) threshold for each player
    start_gain = proj[None, :] - th_p
    depth = jnp.maximum(c["pos_count"][:, pos] - MIN_NEED[pos][None, :], 0)
    bb = params.get("bench_base", BENCH_BASE)
    bd = params.get("bench_decay", BENCH_DECAY)
    bench_w = bb * (bd ** depth)
    # not floored at zero: below replacement the ordering still carries
    # information, and flooring makes every late candidate tie at exactly 0
    bench_gain = bench_w * (proj[None, :] - c["repl"][pos][None, :])
    # `additive` stacks starter gain and depth value; otherwise a player is
    # counted as either a starter upgrade or bench depth, never both.
    mlv = jnp.where(params.get("additive", MLV_ADDITIVE),
                    jnp.maximum(start_gain, 0.0) + bench_gain,
                    jnp.where(start_gain > 0, start_gain, bench_gain))

    if name == "MLV":
        return mlv
    if name == "MLV_VONA":                              # subtract opportunity cost
        return mlv - 0.5 * c["next_turn_vorp_by_pos"][:, pos]
    if name == "BPA":
        return base_vorp
    if name == "ECR":
        return jnp.broadcast_to(-c["ecr"][None, :], (S, P))
    if name == "ZERO_RB":
        return mlv + jnp.where((pos == RB) & (rnd < 4), -1e4, 0.0)[None, :]
    if name == "HERO_RB":
        got = c["pos_count"][:, RB] >= 1
        return mlv + jnp.where((pos == RB)[None, :] & got[:, None] & (rnd < 4), -1e4, 0.0)
    if name == "ROBUST_RB":
        return mlv + jnp.where((pos == RB) & (rnd < 2), 60.0, 0.0)[None, :]
    raise ValueError(name)


def run_draft(d, my_slot, policy, S=1000, seed=0, ecr_noise=1.0, params=None,
              smart_slots=None, opp_board=None):
    """smart_slots: other seats that ALSO draft on value rather than consensus.
    The default (None) makes every opponent a consensus-follower, which is the
    optimistic case; a real league usually contains a few value drafters.

    opp_board: an alternative projection/VORP set those smart opponents draft
    from. Passing a STANDARD-scoring board here models the realistic case - a
    sharp opponent doing value-based drafting off ordinary rankings, rather than
    one who has re-derived every value under this league's custom rules."""
    pos, ecr, vorp, proj = d["pos"], d["ecr"], d["vorp"], d["proj"]
    P = pos.shape[0]
    order = snake_order()
    noise = jax.random.normal(jax.random.PRNGKey(seed), (S, P)) * (d["ecr_sd"][None, :] * ecr_noise)
    perceived = ecr[None, :] + noise

    pos_order, pos_vorp, pos_len = [], [], []
    for g in range(6):
        idx = np.where(np.array(pos) == g)[0]
        idx = idx[np.argsort(-np.array(vorp)[idx])]
        pos_order.append(jnp.array(idx)); pos_vorp.append(vorp[jnp.array(idx)]); pos_len.append(len(idx))

    ar = jnp.arange(S)

    def body(carry, n):
        avail, pos_count, rv = carry
        t = order[n]; rnd = n // N_TEAMS
        pc_t = pos_count[:, t, :]
        rv_t = rv[:, t]                                   # (S,6,MAXPOS)
        legal = allowed_mask(avail, pc_t, pos, rnd)
        _, th = lineup_and_thresholds(rv_t, d["repl"])

        # picks until THIS team's next turn (works for any seat, not just ours)
        gap = jnp.where(rnd % 2 == 0, 2 * (N_TEAMS - 1 - t) + 1, 2 * t + 1)
        rank = jnp.argsort(jnp.argsort(perceived + jnp.where(avail, 0.0, 1e6), axis=1), axis=1)
        soon = (rank < gap) & avail
        nxt = [kth_available_vorp(avail, pos_order[g], pos_vorp[g],
                                  jnp.clip((soon & (pos == g)[None, :]).sum(axis=1) + 1, 1, pos_len[g]))
               for g in range(6)]
        ctx = dict(pos=pos, vorp=vorp, proj=proj, ecr=ecr, rnd=rnd, avail=avail,
                   pos_count=pc_t, th=th, repl=d["repl"],
                   next_turn_vorp_by_pos=jnp.stack(nxt, axis=1))

        smart = (t == my_slot)
        if smart_slots:
            for extra in smart_slots:
                smart = smart | (t == extra)
        my_score = policy_scores(policy, ctx, params)
        if opp_board is not None:
            octx = dict(ctx, proj=opp_board["proj"], vorp=opp_board["vorp"],
                        repl=opp_board["repl"])
            _, oth = lineup_and_thresholds(rv_t, opp_board["repl"])
            octx["th"] = oth
            opp_smart = policy_scores(policy, octx, params)
            my_score = jnp.where(t == my_slot, my_score, opp_smart)
        score = jnp.where(smart, my_score, -perceived)
        pick = jnp.argmax(jnp.where(legal, score, -jnp.inf), axis=1)

        avail = avail.at[ar, pick].set(False)
        g = pos[pick]
        pos_count = pos_count.at[ar, t, g].add(1)
        row = rv[:, t][ar, g]                             # (S,MAXPOS)
        newrow = jnp.sort(jnp.concatenate([row, proj[pick][:, None]], axis=1), axis=1)[:, ::-1][:, :MAXPOS]
        rv = rv.at[ar, t, g].set(newrow)
        return (avail, pos_count, rv), pick

    init = (jnp.ones((S, P), bool), jnp.zeros((S, N_TEAMS, 6), jnp.int32),
            jnp.zeros((S, N_TEAMS, 6, MAXPOS), jnp.float32))
    (_, _, rv_final), picks = jax.lax.scan(body, init, jnp.arange(N_PICKS))
    picks = picks.T
    rosters = jnp.zeros((S, N_TEAMS, N_ROUNDS), jnp.int32)
    for n in range(N_PICKS):
        rosters = rosters.at[:, int(order[n]), n // N_TEAMS].set(picks[:, n])
    return rosters, rv_final
