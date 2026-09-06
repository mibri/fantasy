"""
Simulate a full fantasy season from drafted rosters and score policies on
CHAMPIONSHIP PROBABILITY.

Three nested sources of randomness, all measured from data rather than assumed:
  1. projection error   - each player's true per-game rate is drawn around our
                          projection using the empirically measured year-over-year
                          predictive residual (QB 5.50, RB 4.55, WR 3.73, TE 2.66 ppg)
  2. week-to-week noise - gamma-distributed around that rate, matching the observed
                          mean/SD relationship and right skew (skew ~0.85 at RB/WR/TE)
  3. availability       - each week played with prob = expected games / 17
Then: optimal weekly lineups, a 14-week head-to-head schedule, and a 6-team
playoff bracket in weeks 15-17.
"""
import jax, jax.numpy as jnp, numpy as np

N_TEAMS, N_ROUNDS, N_WEEKS, REG_WEEKS = 12, 15, 17, 14
QB, RB, WR, TE, K, DST = range(6)
# weekly SD = a + b*mean, fitted from 2021-2025 (see docs/methodology.md)
SD_A = jnp.array([7.71, 2.90, 2.53, 1.50, 4.00, 6.00])
SD_B = jnp.array([0.171, 0.391, 0.432, 0.513, 0.30, 0.30])


def round_robin(n=N_TEAMS, weeks=REG_WEEKS):
    """Circle-method schedule: (weeks, n) array giving each team's opponent."""
    teams = list(range(n)); sched = []
    for w in range(weeks):
        opp = [0] * n
        for i in range(n // 2):
            a, b = teams[i], teams[n - 1 - i]
            opp[a] = b; opp[b] = a
        sched.append(opp)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return jnp.array(sched)


def simulate(rosters, board_arr, key, waiver=True):
    """rosters (S,T,R) player indices -> (weekly team totals (S,T,W))."""
    S = rosters.shape[0]
    pos = board_arr["pos"][rosters]                       # (S,T,R)
    ppg = board_arr["proj"][rosters] / 16.0               # per-game rate
    sd_season = board_arr["proj_sd"][rosters] / 16.0      # projection error, per game
    avail_p = jnp.clip(board_arr["proj"][rosters] * 0 + 0.92, 0.5, 1.0)

    k1, k2, k3 = jax.random.split(key, 3)
    # (1) projection error: true rate per player per sim
    true_ppg = jnp.clip(ppg + jax.random.normal(k1, ppg.shape) * sd_season, 0.5, None)

    # (2) weekly gamma noise
    sd_w = SD_A[pos] + SD_B[pos] * true_ppg
    shape = jnp.clip((true_ppg / jnp.maximum(sd_w, 1e-3)) ** 2, 0.4, 200.0)
    scale = true_ppg / shape
    g = jax.random.gamma(k2, shape[..., None], shape=true_ppg.shape + (N_WEEKS,))
    weekly = g * scale[..., None]

    # (3) availability
    played = jax.random.uniform(k3, weekly.shape) < avail_p[..., None]
    weekly = weekly * played

    # optimal weekly lineup
    scores = jnp.moveaxis(weekly, -1, 0)                  # (W,S,T,R)
    posb = jnp.broadcast_to(pos[None], scores.shape)

    # Weekly points a freely available waiver-wire player would give at each slot.
    # Without this a manager whose starter is hurt scores ZERO from that slot, which
    # massively overstates the value of rostering backups. In a real league you pick
    # up a replacement-level player for nothing.
    wv = (board_arr["repl"] / 17.0) if waiver else jnp.zeros(6)
    flex_wv = jnp.max(wv[jnp.array([RB, WR, TE])])

    def best_lineup(sc, ps):
        # sc,ps: (...,R) -> per-position sorted desc
        per = []
        for gpos in range(6):
            masked = jnp.where(ps == gpos, sc, -jnp.inf)
            per.append(jnp.sort(masked, axis=-1)[..., ::-1])
        qb, rb, wr, te, kk, ds = per
        f = lambda x, floor: jnp.maximum(jnp.where(jnp.isneginf(x), 0.0, x), floor)
        flex = jnp.concatenate([rb[..., 2:4], wr[..., 2:4], te[..., 1:3]], axis=-1)
        flex = jnp.sort(flex, axis=-1)[..., ::-1]
        return (f(qb[..., 0], wv[QB]) + f(rb[..., 0], wv[RB]) + f(rb[..., 1], wv[RB])
                + f(wr[..., 0], wv[WR]) + f(wr[..., 1], wv[WR]) + f(te[..., 0], wv[TE])
                + f(flex[..., 0], flex_wv) + f(flex[..., 1], flex_wv)
                + f(kk[..., 0], wv[K]) + f(ds[..., 0], wv[DST]))

    totals = best_lineup(scores, posb)                    # (W,S,T)
    return jnp.moveaxis(totals, 0, -1)                    # (S,T,W)


def championship(totals, sched):
    """totals (S,T,W) -> (champ_onehot (S,T), wins (S,T), points (S,T))."""
    S, T, W = totals.shape
    reg = totals[:, :, :REG_WEEKS]
    opp = sched                                            # (REG_WEEKS,T)
    opp_scores = jnp.take_along_axis(
        jnp.moveaxis(reg, 1, 2), jnp.broadcast_to(opp[None], (S, REG_WEEKS, T)), axis=2)
    my = jnp.moveaxis(reg, 1, 2)                           # (S,W,T)
    wins = (my > opp_scores).sum(axis=1)                   # (S,T)
    pts = reg.sum(axis=2)
    seed_key = wins * 1e6 + pts                            # wins, points as tiebreak
    order = jnp.argsort(-seed_key, axis=1)                 # (S,T) team ids by seed
    s = [order[:, i] for i in range(6)]

    def game(a, b, week):
        sa = jnp.take_along_axis(totals[:, :, week], a[:, None], 1)[:, 0]
        sb = jnp.take_along_axis(totals[:, :, week], b[:, None], 1)[:, 0]
        return jnp.where(sa >= sb, a, b)

    w1 = game(s[2], s[5], 14); w2 = game(s[3], s[4], 14)   # wildcard, week 15
    w3 = game(s[0], w2, 15);   w4 = game(s[1], w1, 15)     # semis, week 16
    champ = game(w3, w4, 16)                               # final, week 17
    return champ, wins, pts
