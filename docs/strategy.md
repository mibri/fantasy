# Draft Plan

Written for someone who has never drafted before. Everything here comes out of
the model in this repo; `docs/methodology.md` explains how it was built.

## What a draft actually is

Twelve people take turns picking NFL players. Once someone is taken, nobody
else can have him. You end up with 15 players; each week you start 10 of them
(`QB, RB, RB, WR, WR, TE, FLEX, FLEX, K, DEF`) and their real-life statistics
turn into points. FLEX means "any running back, wide receiver, or tight end."

It's a **snake** draft: the order reverses each round. If you pick 6th, you pick
6th, then 19th, then 30th, then 43rd, and so on. So your picks come in pairs -
long gaps, then two picks close together.

**The one idea that matters.** A player's worth is not his point total. It is
how much better he is than the *worst player you could have started instead*.
That baseline is called **replacement level**, and in this league it is:

| Position | Replacement | Meaning |
|---|---|---|
| QB | **303 pts** | 12 teams start 1 QB, so QB13 is free. The bar is very high. |
| RB | 194 pts | 32 RBs start once FLEX spots are filled |
| WR | 197 pts | 40 WRs start - the deepest position |
| TE | 158 pts | only 12 start |
| K | 137 pts | any of them |
| DEF | 114 pts | any of them |

A quarterback who scores 340 is only worth **+37** over a free one. A running
back who scores 340 is worth **+146**. That single table is why you do not draft
a quarterback early, even though quarterbacks score the most raw points.

## Three things that make your league unusual

**1. Two FLEX spots and only five bench spots.** Seven of your ten starters are
RB/WR/TE. Across the league that's 84 skill starters, so useful players go much
deeper than normal - but you have almost no room to stash lottery tickets.
Nearly every pick has to be someone you would actually play.

**2. Quarterbacks get paid for completions.** `+0.27` per completion and
`-0.50` per incompletion means the break-even accuracy is **64.9%**. Accurate
passers gain a lot; inaccurate ones bleed. Measured over 2023-2025, the same
standard production converts into your league's points at wildly different
rates: **Joe Burrow 1.26x, Jared Goff 1.27x, Dak Prescott 1.23x** versus
**Anthony Richardson 0.91x**. Nobody else in your league is pricing this.

**3. Workhorse running backs get bonuses.** `+2` for 20 carries, `+3` for a
100-yard game, `+0.25` per first down. High-volume backs (Jonathan Taylor,
Derrick Henry, Saquon Barkley, all ~1.18x) gain more than committee backs.

## The value cliff

Value disappears fast. Best player available, by pick:

| After N picks | Best remaining | Value over replacement |
|---|---|---|
| 0 | Ja'Marr Chase | **+206** |
| 11 (end of round 1) | Chase Brown | +112 |
| 23 (end of round 2) | Omarion Hampton | +82 |
| 35 | Zay Flowers | +64 |
| 47 | Tyler Warren | +49 |
| 71 | TreVeyon Henderson | +31 |
| 119 (end of round 10) | Michael Wilson | **+0.3** |

Two conclusions. Your **first three picks are most of your season**. And from
round 11 on, everyone left is essentially replacement level - those picks are
insurance and lottery tickets, so that's where K and DEF belong.

## What the simulation actually drafts

700 simulated drafts per slot, against 11 opponents following consensus rankings.
The modal pick by round, which barely changes across draft slots:

| Round | Pick | Typical name |
|---|---|---|
| 1 | **RB** (81-97%) | Jonathan Taylor, Bijan Robinson, Ja'Marr Chase at slot 1 |
| 2 | **RB or Trey McBride** (RB 55-60 / TE 38-39) | James Cook III, Saquon Barkley, McBride |
| 3 | **QB** (69-77%) | **Joe Burrow** (57-73%) |
| 4 | **RB** (76-83%) | Derrick Henry, Breece Hall, Kyren Williams |
| 5 | **RB** (64-71%) | D'Andre Swift, Bucky Irving, Tyler Warren |
| 6 | RB / TE | Harold Fannin Jr., David Montgomery |
| 7-8 | **WR** (63-72%) | Chris Godwin Jr., DK Metcalf, Brian Thomas Jr. |
| 9-10 | QB2 / TE | Brock Purdy, Jared Goff, Dalton Kincaid |
| 14-15 | K, DEF | anyone |

Receivers come late on purpose. Wide receiver is the deepest position in this
format (40 of them start), so the 30th-best receiver is still nearly a starter,
while the running back board falls off a cliff after about 16 names.

## How the strategies compare

Championship probability, 8,400 simulated seasons per strategy. If all twelve
teams were equal, everyone would win 8.3% of the time.

| Strategy | Titles won |
|---|---|
| **Marginal lineup value + opportunity cost** | **19.1%** |
| Marginal lineup value | 18.8% |
| Best value available | 17.8% |
| Force RB early | 17.5% |
| One RB then receivers | 16.4% |
| Follow consensus rankings | 15.0% |
| Zero-RB (avoid RB early) | **14.1%** |

Two things to take from this. Drafting on value beats following consensus
rankings by **27% in relative terms** (7.1 standard errors - not noise). And
**Zero-RB is the worst strategy tested** - in a league with two FLEX spots,
avoiding running backs early is actively harmful.

## Rules for draft day

1. **Rounds 1-2: take the best running back or wide receiver.** Not a QB, not a
   tight end (one exception below). This is where the value is steepest.
2. **Never draft a kicker or defense before the last two rounds.** They are worth
   under 50 points of edge and the model shows the pick is always better spent
   elsewhere.
3. **Watch tiers, not ranks.** The difference between the 4th and 5th best RB is
   noise. The difference between the last player in a tier and the first player
   in the next one is real. If a tier is down to its last one or two players and
   your next pick is 20 slots away, that's when to reach.
4. **Two picks in a row (the turn) = take the scarce position first.** At the
   turn you effectively get both players, so take the one less likely to survive.
5. **Your bench is replaceable.** Once the simulation models a waiver wire, the
   weight you put on bench depth stops mattering - every setting lands within
   noise. A hurt starter costs a waiver pickup, not a zero. Plan bye weeks
   instead: week 11 holds 10 of the top 60 players' byes, weeks 6 and 13 nine
   each. Don't leave four starters idle in the same week.

## The tight end decision

**Trey McBride is the single biggest positional edge on the board.** He projects
280 points against a TE replacement level of 158 - **+122**, which ranks 10th
overall. The next tier of tight ends (Kittle, Bowers, Kelce) sits around 185-205.
If McBride is there in round 2, taking him is defensible; after him, tight end
becomes a position to fill late, because TE5 and TE10 are nearly identical.

## The quarterback decision - your biggest edge

The model's clearest actionable call. QB replacement is 303 points, so most
quarterbacks are worth almost nothing over a free one. But two things combine:
your league inflates QB scoring, and it rewards accuracy specifically.

| QB | Consensus rank | Ratio | Projected | Value over replacement |
|---|---|---|---|---|
| Josh Allen | 26 | 1.17 | 402 | **+99** |
| **Joe Burrow** | **45** | **1.26** | **394** | **+91** |
| Lamar Jackson | 32 | 1.19 | 390 | +87 |
| Drake Maye | 37 | 1.19 | 381 | +78 |
| Jayden Daniels | 57 | 1.16 | 347 | +44 |
| Dak Prescott | 78 | 1.23 | 341 | +38 |
| Brock Purdy | 96 | 1.21 | 333 | +30 |

**Joe Burrow is the pick, and the simulation is emphatic about it.** Across 700
simulated drafts he is the round-3 selection **57-73% of the time** at slots 6-12.
He is the second-most valuable quarterback in your scoring but goes about 19 picks
later than Josh Allen, because the market prices standard scoring, where his
completion rate earns nothing extra.

If you pick 1st, Josh Allen is instead the round-3 pick (38%). Same logic further
down: **Dak Prescott** and **Brock Purdy** are the round 8-9 versions.

One correction worth stating: an earlier version of this plan said to wait until
rounds 4-5. The simulation says round 3 - Burrow rarely survives to round 4.
