"""
How much of winning a fantasy league is skill and how much is variance?

Two decompositions, both measured rather than argued:

  1. SEASON LUCK.  Freeze one drafted league - twelve fixed rosters - and replay
     the season many times. If the best roster only wins the title occasionally,
     then even a decisively better team is mostly at the mercy of variance.

  2. DRAFT SKILL.  Let every team draft with the SAME policy, so nobody has a
     process edge, and see how much of the season still tracks roster quality.
"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/user/fantasy")
import jax, jax.numpy as jnp, numpy as np, pandas as pd
from src.draft_sim import load_board, run_draft, MAXPOS
from src.season_sim import simulate, championship, round_robin

QB, RB, WR, TE, K, DST = range(6)


def starter_strength(rosters, arr):
    """Projected optimal starting-lineup points for every team. (S,T)"""
    pos = np.array(arr["pos"])[np.array(rosters)]          # (S,T,R)
    proj = np.array(arr["proj"])[np.array(rosters)]
    S, T, R = proj.shape
    out = np.zeros((S, T))
    for g in range(6):
        pass
    for s in range(S):
        for t in range(T):
            by = {g: np.sort(proj[s, t][pos[s, t] == g])[::-1] for g in range(6)}
            take = lambda g, k: list(by[g][:k]) + [0.0] * max(0, k - len(by[g]))
            base = take(QB, 1) + take(RB, 2) + take(WR, 2) + take(TE, 1) + take(K, 1) + take(DST, 1)
            flex = sorted(list(by[RB][2:]) + list(by[WR][2:]) + list(by[TE][1:]), reverse=True)[:2]
            out[s, t] = sum(base) + sum(flex)
    return out


def main():
    b, arr = load_board(); sched = round_robin()
    everyone = [t for t in range(12) if t != 5]

    # --- equal-skill league: all twelve teams draft on value ---
    ros, _ = run_draft(arr, my_slot=5, policy="MLV_VONA", S=200, seed=404,
                       smart_slots=everyone)
    strength = starter_strength(ros, arr)                   # (200,12)

    # 1. season luck: freeze rosters, replay the season 40x per drafted league
    champ_counts = np.zeros((200, 12))
    wins_all = []
    for rep in range(40):
        tot = simulate(ros, arr, jax.random.PRNGKey(5000 + rep))
        champ, wins, _ = championship(tot, sched)
        champ_counts[np.arange(200), np.array(champ)] += 1
        wins_all.append(np.array(wins))
    p_champ = champ_counts / 40.0                           # (200,12)
    best = strength.argmax(axis=1)
    p_best = p_champ[np.arange(200), best]

    print("=== 1. SEASON VARIANCE (rosters fixed, season replayed 40x) ===")
    print(f"  strongest roster wins the title      : {100*p_best.mean():.1f}% of seasons")
    print(f"  best any single team ever manages    : {100*p_champ.max(axis=1).mean():.1f}%")
    print(f"  if it were pure chance               : 8.3%")
    print(f"  weakest roster still wins            : "
          f"{100*p_champ[np.arange(200), strength.argmin(axis=1)].mean():.1f}%")

    # 2. how much does roster quality explain?
    W = np.mean(wins_all, axis=0)                           # (200,12) mean wins
    rs = []
    for s in range(200):
        rs.append(np.corrcoef(strength[s], W[s])[0, 1])
    print()
    print("=== 2. DOES A BETTER ROSTER WIN MORE? (equal-skill league) ===")
    print(f"  corr(projected starters, regular-season wins) = {np.nanmean(rs):+.3f}")
    print(f"  spread in roster strength within a league     = {strength.std(axis=1).mean():.0f} pts "
          f"(mean {strength.mean():.0f})")
    # single-season correlation, not averaged over 40 replays
    tot1 = simulate(ros, arr, jax.random.PRNGKey(99))
    _, w1, _ = championship(tot1, sched); w1 = np.array(w1)
    r1 = np.nanmean([np.corrcoef(strength[s], w1[s])[0, 1] for s in range(200)])
    print(f"  same, but for ONE season only                 = {r1:+.3f}")
    print(f"  -> variance explained in one season: {100*r1**2:.0f}%")


if __name__ == "__main__":
    main()
