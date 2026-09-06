"""Validation tests for the league scoring engine."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/user/fantasy")
import pandas as pd, numpy as np
from config.league import BREAKEVEN_COMP_PCT


def test_breakeven_completion_rate():
    # 0.27p - 0.50(1-p) = 0  ->  p = 0.50/0.77
    assert abs(BREAKEVEN_COMP_PCT - 0.6494) < 1e-3


def test_long_td_symmetry():
    """Every 40+ yard passing TD is also a 40+ yard receiving TD, by construction."""
    b = pd.read_csv("data/processed/long_td_bonuses.csv")
    assert b.pass_td_40.sum() == b.rec_td_40.sum()
    assert b.pass_td_50.sum() == b.rec_td_50.sum()


def test_hand_computed_game():
    """Josh Allen, 2025 week 11, computed term by term from the raw stat line."""
    sc = pd.read_parquet("data/processed/weekly_scored.parquet")
    r = sc[(sc.player_display_name == "Josh Allen") & (sc.season == 2025)
           & (sc.week == 11)].iloc[0]
    expected = (0.04 * 317 + 6 * 3 - 3 * 2          # pass yds, TDs, INTs
                + 0.27 * 19 - 0.50 * 11             # completions / incompletions
                + 0.5 * 2                           # 40+ yard completions
                + 0.1 * 40 + 6 * 3                  # rushing
                + 2.0                               # 300-399 yard passing game
                + 2.5)                              # long-TD bonuses
    assert abs(float(r.pts) - expected) < 1e-6, (float(r.pts), expected)


def test_replacement_levels_ordered():
    """QB replacement must far exceed RB/WR because only 12 QBs start."""
    b = pd.read_csv("data/processed/board_final.csv")
    rep = b.groupby("pos").replacement.first()
    assert rep["QB"] > rep["RB"] + 80
    assert rep["TE"] < rep["WR"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"PASS {name}")
