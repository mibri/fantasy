# Fantasy Draft Model — 12-team custom-scoring league, 2026

A quantitative draft model built for one specific league, whose scoring is
unusual enough that off-the-shelf rankings are systematically wrong for it.

**Deliverables**
- **[Draft plan](https://claude.ai/code/artifact/64a7d8bb-6d1e-43bb-94f0-dfefd98628b6)** — the strategy, written for a first-time drafter
- **[Live draft board](https://claude.ai/code/artifact/36285c62-c9f5-4f37-b2b1-6d32def47087)** — use it during the draft; tap players off as they go
- `docs/methodology.md` — how it was built, including the errors found along the way
- `docs/strategy.md` — the same plan in markdown

## The headline

Your league pays `+0.27` per completion and `−0.50` per incompletion, so the
break-even accuracy is **64.9%**. Measured over 2023–2025, the same standard-PPR
production converts into this league's points at rates ranging from **0.91×**
(Anthony Richardson) to **1.33×** (Joe Burrow). The market prices standard
scoring and does not price this.

The model's clearest call: **Joe Burrow in round 3**, taken in 57–73% of
simulated drafts from slots 6–12.

## Results

Championship probability, 8,400 simulated seasons per strategy (SE ≈ 0.004);
baseline is 0.083 if all twelve teams were equal.

| Strategy | Titles |
|---|---|
| Marginal lineup value + opportunity cost | **19.1%** |
| Marginal lineup value | 18.8% |
| Best value available | 17.8% |
| Follow consensus rankings | 15.0% |
| Zero-RB | 14.1% |

## Layout

```
config/league.py    exact league rules, transcribed from the settings page
src/scoring.py      scoring engine (hand-validated to 0.0000 on a full stat line)
src/pbp_bonuses.py  long-TD bonuses and drive events from play-by-play
src/projections.py  market volume x player-specific scoring translation
src/vorp.py         flex-aware replacement levels
src/draft_sim.py    JAX Monte Carlo snake draft
src/season_sim.py   weekly lineups, schedule, playoffs, waiver wire
src/tune.py         hyperparameter sweep against championship probability
tests/              validation tests
```

## Reproducing

```bash
pip install numpy pandas scipy scikit-learn jax pyarrow
python src/pbp_bonuses.py     # needs nflverse CSVs in data/raw
python src/build_board.py
python src/experiment.py 700
python tests/test_scoring.py
```

Data: [nflverse](https://github.com/nflverse/nflverse-data) play-by-play and
weekly stats 2021–2025; FantasyPros redraft consensus via the
[DynastyProcess](https://github.com/dynastyprocess/data) mirror, 2026-09-04.
