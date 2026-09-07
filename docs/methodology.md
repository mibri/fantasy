# Methodology

A quantitative draft model for a 12-team, custom-scoring Sleeper league.
Everything here is reproducible from `src/` against public data.

## 1. The league

| Setting | Value |
|---|---|
| Teams | 12 |
| Starters | QB, RB, RB, WR, WR, TE, FLEX, FLEX, K, DEF (10) |
| Bench / IR | 5 / 1 |
| Total drafted | 12 x 15 = **180 players** |
| Scoring | custom (see `config/league.py`) |

Two structural consequences, both of which drive the strategy:

1. **Two FLEX spots** mean 7 of 10 starters are RB/WR/TE. Across the league
   that is 84 skill starters, so replacement level sits very deep:
   **RB29 and WR43**, not RB24/WR24.
2. **Only 5 bench spots.** Two thirds of your roster starts every week. There
   is almost no room to stash lottery tickets; every pick must be a plausible
   starter.

## 2. Data

| Source | Use | As of |
|---|---|---|
| [nflverse](https://github.com/nflverse/nflverse-data) play-by-play, 2021-2025 | 40+/50+ yard TD bonuses, pick-sixes, drive events for DST | 2025 complete |
| nflverse weekly player stats, 2016-2025 | all other scoring inputs | 2025 complete |
| [DynastyProcess](https://github.com/dynastyprocess/data) mirror of FantasyPros ECR | 2026 redraft consensus + expert dispersion | **2026-09-04** |

Live fantasy APIs (Sleeper, FantasyPros direct) are blocked by this
environment's egress policy, so consensus data comes via the DynastyProcess
GitHub mirror. The `redraft-overall` slice (525 players) carries `ecr`, `sd`,
`best` and `worst` per player - consensus rank *and its dispersion*, which is
what the opponent model needs.

## 3. Scoring engine

`src/scoring.py` implements the league's rules exactly, including the parts
generic tools ignore: `+0.27` per completion, `-0.50` per incompletion, 6-point
passing TDs, `-3` interceptions plus `-1` pick-six, 40+/50+ yard TD bonuses,
100/200-yard and 300/400-yard game bonuses, the 20+ carry bonus, and
`+0.25` per first down for RB/WR/TE.

**Validation.** Computing 40+ yard TDs independently from the passer and
receiver sides of play-by-play yields exactly 464 each over 2021-2025, and
50+ yard TDs 281 each - they must match by construction, and they do.
Pick-sixes total 181 over five seasons (~36/yr), matching known NFL rates.

Two rules the settings page left ambiguous, with the choice made explicit:

- **Incompletions** = `attempts - completions`, so interceptions count as
  incompletions (the standard stat definition).
- **Long-TD bonuses stack**: a 55-yard TD earns both the 40+ and the 50+
  bonus, as separate stat triggers. `stack_long_td=False` flips this. Measured
  over 2021-2025, the choice is worth a mean of **0.18 points per
  player-season**, exceeds 2 points in 0.4% of player-seasons (7 of 1881), and
  never exceeds 4. It cannot change a draft decision.

## 4. Projections

The core decomposition:

```
league_points  =  standard-PPR production  x  scoring-translation ratio
```

Each factor is estimated from whichever source knows it best.

### Volume and role <- the market

The FantasyPros consensus reflects camp reports, depth charts and injury news.
A preseason rank is a *forecast of a finish*, so expected production is the
observed points-by-finish-rank curve (2022-2025) shrunk toward the positional
mean by the measured year-over-year persistence slope:

```
E[season points | preseason rank r] = m + slope x (finish_curve(r) - m)
```

> **Two model errors worth recording.**
>
> 1. The first version projected volume from each player's own recency-weighted
>    history. It systematically underrated second-year players whose roles were
>    changing (Egbuka, Burden, Golden fell 100+ spots below consensus) and
>    overrated declining veterans. An empirical aging curve helped but did not
>    fix the cause: *history is a poor estimator of next season's volume for
>    anyone whose role is changing.*
> 2. The second version fitted an isotonic curve of historical points-per-game
>    against *current* rank. This depressed the top of the curve, because highly
>    ranked young players with short or weak histories sat inside the fit set and
>    dragged it down. Drake Maye - the market's QB3 - came out **below QB12
>    replacement level**, which is plainly wrong. The finish-rank curve above
>    fixed it (Maye 291 -> 381).
>
> The cost of this choice is real and worth stating: player-specific volume
> information is now discarded entirely. Jared Goff's recent production is well
> above his consensus rank, and the model no longer credits him for it. The
> defence is that the market has already seen that history *and* the reasons it
> may not repeat, and the measured QB persistence slope of 0.581 says heavy
> regression is warranted.

### Scoring translation <- the player

For each player, `ratio = league_points / standard_PPR_points` over 2023-2025,
shrunk toward the positional mean with an empirical-Bayes prior of 20 games.
This is the part the market does not price, because the market prices standard
scoring. Measured spread:

| Position | mean ratio | sd | p10-p90 |
|---|---|---|---|
| **QB** | 1.164 | **0.087** | 1.064 - 1.259 |
| RB | 1.120 | 0.035 | 1.075 - 1.162 |
| WR | 1.086 | 0.025 | 1.056 - 1.123 |
| TE | 1.062 | 0.014 | 1.048 - 1.080 |

**The edge is concentrated at QB** and comes from the completion terms. A pass
attempt is worth `0.27p - 0.50(1-p) = 0.77p - 0.50`, so the break-even
completion rate is **64.9%**. Accurate pocket passers (Burrow 1.33, Goff 1.32,
Tua 1.32, Prescott 1.27) convert far better than inaccurate ones
(Richardson 0.91, Ward 1.01). Between two QBs the market prices equally, the
higher-ratio one is worth ~15-20% more here.

At RB the ratio rewards workhorses (Taylor, Henry, Barkley ~1.18) through the
20+ carry and 100-yard game bonuses.

### Uncertainty

Measured, not assumed, from year-over-year persistence (players with >= 8 games
in the prior season, 2021-2025):

| Position | corr(t, t+1) | slope | residual SD |
|---|---|---|---|
| QB | 0.536 | 0.581 | **5.50 ppg** |
| RB | 0.686 | 0.710 | 4.55 ppg |
| WR | 0.761 | 0.797 | 3.73 ppg |
| TE | 0.757 | 0.795 | 2.66 ppg |

QB is the **least** predictable position, which cuts against its inflated
scoring. All slopes are well below 1.0, confirming heavy regression to the mean.

## 5. Replacement level and VORP

Replacement is solved *with FLEX allocated*, not assumed: fill base slots
(12 QB, 24 RB, 24 WR, 12 TE), then let the best remaining RB/WR/TE compete for
the 24 FLEX spots. Equilibrium startable counts and replacement values:

| Position | startable | replacement (season pts) |
|---|---|---|
| QB | 12 | **309** |
| RB | 29 | 170 |
| WR | 43 | 172 |
| TE | 12 | 143 |
| K | 12 | 137 |
| DST | 12 | 114 |

QB replacement is extremely high because only 12 start. Even the top QB clears
it by ~124 points, which ranks him around 12th overall - **elite QB is not a
first-round pick here despite the inflated QB scoring.**

## 6. Draft and season simulation

`src/draft_sim.py` (JAX) runs S 12-team snake drafts in parallel as a
`lax.scan` over all 180 picks, with per-simulation availability masks.

- **Opponents** draft by `ECR + N(0, sd)` using each player's own expert
  dispersion, under roster caps, a K/DST embargo before round 12, and forced
  filling of unmet mandatory slots when picks run out.
- **Our team** follows a policy. The principled one is **Marginal Lineup
  Value**: a player is worth what he adds to the *optimal starting lineup*,
  i.e. `max(0, projection - threshold)` where the threshold is the weakest
  starter he could displace (correctly handling the FLEX chain), plus a
  decaying bench option value.

  Naive "best VORP available" drafts 6 RBs and 3 WRs, because raw VORP ignores
  that only 2 RB + 2 FLEX can start. MLV fixes this and yields realistic
  4.5 RB / 3.8 WR / 2.8 TE / 1.9 QB rosters.

  > **A third model error worth recording.** The threshold for an *empty* lineup
  > slot was initially 0, which made a candidate's marginal value equal his full
  > projected points. That systematically flattered quarterbacks, who score the
  > most raw points in this format: the simulator drafted Josh Allen in **round 2**
  > at nearly every slot, flatly contradicting the VORP table showing elite QB as
  > roughly the 15th most valuable asset. An empty slot can always be filled later
  > from the waiver wire, so the true alternative is a **replacement-level player,
  > never nothing**. Flooring every threshold at replacement level fixed it, and
  > QB moved back to round 3. The bug had also inflated the tuned bench weight,
  > because a large depth term was the only thing counterbalancing an over-scaled
  > starter term - so the weight had to be re-tuned after the fix.

  **The bench-depth weight was tuned, not assumed.** Sweeping it against
  championship probability (`data/processed/tuning.csv`, 4 slots x 700 sims per
  point, SE ~0.007) gives a flat optimum for weights **<= 0.5** (0.189-0.194)
  that falls away above it (0.185 at 0.80, 0.182 at 1.20). The model uses 0.25.

  This is worth recording because an earlier sweep pointed the *opposite* way,
  appearing to show that a very large depth weight was best. That was an
  artefact of the empty-slot threshold bug described above: with the starter
  term over-scaled, only a large depth term could counterbalance it. Fixing the
  bug reversed the tuning result. A tuned hyperparameter is only as trustworthy
  as the objective underneath it.

`src/season_sim.py` then plays each drafted league out: projection error, then
gamma-distributed weekly noise matching the measured mean-SD relationship and
right skew, then availability; optimal weekly lineups; a 14-week round-robin;
and a 6-team playoff bracket in weeks 15-17. Policies are scored on
**championship probability**.

### Interpreting the simulated win rates

Our policies beat ECR-following opponents by a wide margin, but that number is
**biased upward**: our own projections define truth inside the simulation, so
any policy optimising against them is flattered. The internally valid
comparison is **policy versus policy at the same draft slot**, where all
policies share the same projections and the same opponents.

## Backtest (added after the fact)

The original write-up said the projections could not be backtested because
historical preseason consensus was unavailable. That was wrong: DynastyProcess
publishes the full FantasyPros ECR archive as `db_fpecr.csv.gz` (100 MB), which
I had missed by only probing the uncompressed filename. It carries
`redraft-overall` snapshots from days before Week 1 for 2021-2024, plus an
August 2025 snapshot. `src/backtest.py` uses them.

### How good is the market?

Preseason consensus rank predicts realised season points in this league's
scoring at **r = 0.79 to 0.82** across QB/RB/WR/TE. That is the bar.

### Does the scoring ratio add anything out of sample?

This is the model's one genuine claim, so it is the thing to test. Using only
data available before each season:

| Position | market alone | ratio vs. residual | market x ratio |
|---|---|---|---|
| QB | 0.793 | **+0.053** | 0.788 |
| RB | 0.806 | -0.052 | 0.804 |
| WR | 0.815 | -0.045 | 0.814 |
| TE | 0.792 | +0.079 | 0.793 |

**Essentially nothing.** The ratio is a real, persistent player trait - prior
ratio predicts current ratio at r = +0.55 (QB), +0.46 (RB), +0.35 (WR),
+0.42 (TE) - but its effect on a season's points is swamped:

| Position | predictable ratio spread | worth | share of outcome variance |
|---|---|---|---|
| QB | 0.045 | **~12.5 pts/season** | 1.22% |
| RB | 0.025 | ~3.9 pts | 0.14% |
| WR | 0.012 | ~1.7 pts | 0.03% |
| TE | 0.009 | ~1.0 pts | 0.02% |

Against a season-points SD near 100, choosing between two similarly ranked
quarterbacks on their conversion rate is worth about **one good week**.

**This corrects an earlier overstatement.** The custom-scoring edge measured at
+21% relative title odds was computed in a simulation that *assumed* the ratio
was known and true. Out of sample it is close to undetectable. What survives is
the *structural* half of the scoring work - replacement levels, positional
value, flex-aware lineup construction - and that half needs no forecasting at
all. It is arithmetic on the league's own rules: QB replacement really is 303
points here, whatever anyone predicts.

### Is the consensus systematically biased anywhere?

Scanning preseason features against the residual (points beyond what the rank
implies):

| Signal | Strongest correlation |
|---|---|
| QB, prior-season points | +0.198 |
| QB, prior-season games | +0.147 |
| WR, prior-season games | +0.126 |
| WR rookies/2nd-year vs veterans | -7.3 vs +3.4 points |
| everything else (age, experience, expert disagreement, rank) | \|r\| < 0.10 |

The market is close to efficient. The largest bias found is r = 0.20 at
quarterback, about 4% of residual variance.

**Caveat on power.** Five seasons, ~2,500 player-seasons, and a crude
within-season rolling smoother as the market control. An effect of r = 0.10
would be hard to detect here. "No signal found" is not "no signal exists".

## Phase 1-2: backtest against REAL seasons

The simulated-season results above have a structural weakness: the variance is a
model output, so "the season is mostly luck" could be an artefact of how the
season was generated. This section removes the simulated season entirely.

**Design.** For each year 2021-2025, build the board from the preseason
FantasyPros snapshot taken days before Week 1 and projections computed from
*strictly prior* seasons (`src/historical.py` recomputes the finish-rank curve,
persistence slopes and every player's ratio per year). Run a 12-team draft. Then
score the resulting rosters on **what actually happened**: real weekly points,
real injuries, real breakouts. Only the opponents' drafts stay simulated.

Two design choices that matter:

* **Lineups are set without hindsight.** Starters are the highest *preseason
  projected* players among those who actually played that week - a manager knows
  who is inactive, not who will score. Using the realised-optimal lineup would
  inflate every team and specifically reward rostering boom/bust players nobody
  would have started.
* **Kickers and defenses are excluded** (absent from player-level weekly data,
  last-two-round picks, near-identical noise for every team). The backtest is a
  13-round draft with 8 starters.

### A. Does the model actually draft better? (real outcomes)

900 drafted teams, 5 real seasons, 12 seats, uncertainty clustered by season
because draft seeds within a year share that year's results.

| Policy | Title rate | Playoffs | Wins |
|---|---|---|---|
| marginal lineup value | **0.167** | 0.647 | 7.97 |
| MLV + opportunity cost | 0.153 | 0.673 | 8.04 |
| draft by consensus (ADP) | **0.083** | 0.430 | 6.74 |

Drafting by consensus lands **exactly** on the 1-in-12 baseline, which is a
reassuring sanity check. The model roughly doubles it, and gains **+1.55 wins
per season** - a far better-powered statistic than title rate.

The difference averages +0.070 with a clustered SE of 0.040, so **t = 1.75 on 4
degrees of freedom** - suggestive, not conclusive. Five seasons is five seasons.

An earlier version scored 0.177 (t = 2.03) with the bench term floored at zero.
That floor was removed because it made every late-round candidate tie at exactly
+0, so the live tool's advice from about round 9 onward was whatever happened to
sort first - it left rosters with two receivers. The backtest difference between
the two forms is well inside the clustered standard error, so sensible tool
behaviour was worth more than a number that could not be distinguished from
noise. Recorded here rather than quietly keeping the better figure.

**Mechanism.** The model builds rosters **+161 preseason points** stronger than
the field, every year (range +141 to +178), against a natural spread among ADP
drafters of only ~50 points. That is roughly 3.2 standard deviations. (This
metric is denominated in the model's own projections, so it explains *why*
rather than proving anything - the proof is the real title rate above.)

### B. Skill vs luck, measured on real seasons

A league where all twelve seats draft the same way, so nobody has a process edge:

| Measure | Real seasons | My simulation had |
|---|---|---|
| corr(preseason roster strength, actual wins) | **+0.133** | +0.152 |
| share of one season's win variance explained | **1.8%** | ~2% |
| strongest preseason roster wins the title | **9.7%** | 13.0% |
| champion's mean strength rank (5.5 = chance) | **5.06** | - |

**The simulated variance model was approximately right.** Real seasons behave
much as it predicted, so "mostly luck" was not an artefact of how I generated
the season. If anything the simulation was slightly *optimistic* about skill -
the real strongest roster wins 9.7% of the time, not 13%.

These two results are consistent, not contradictory: within a league of
similarly-drafted teams the spread is only ~2.5% of roster value and luck
dominates, but a better process opens a gap several times that spread.

### C. The same field-strength collapse, on real seasons

| Rival value drafters | My title rate | My wins |
|---|---|---|
| 0 | **27.5%** | 8.45 |
| 2 | 7.5% | 7.92 |
| 4 | 5.0% | 8.20 |
| 6 | 7.5% | 7.75 |
| 9 | 5.0% | 7.00 |

(n = 40 per cell, so individual cells are noisy; the contrast is not.) Against
even two other value drafters the measured edge is gone at this resolution. The
edge is taken from opponents who do not have it.

## An end-to-end test of the live tool (and three bugs it caught)

The published draft board had only ever been checked at pick 1 with a full
board. Driving it through a complete 180-pick draft in a real browser found
three faults that would have wrecked a live draft:

1. **No positional caps.** Nothing stopped it recommending a fifth kicker.
2. **No K/DEF embargo.** It recommended a defense in round 5 and a kicker in
   round 6, because both have low replacement levels, so *every* kicker scores
   well above replacement.
3. **A stale bench weight** of 0.70, left over from before the waiver-wire
   retune, against 0.25 in the validated model.

The resulting roster was 3 RB, 1 QB, 1 TE, **5 K and 5 DEF**, filling 7 of 10
lineup slots. After the fix: a legal roster, 10/10 slots filled at every draft
seat, and projected starters up from 1843 to ~2430.

The lesson is not subtle. The simulator and the tool were two independent
implementations of the same policy, and only the simulator was ever tested. Any
logic a user actually touches needs its own end-to-end test.
