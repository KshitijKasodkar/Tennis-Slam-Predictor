"""Central configuration: paths, splits and constants shared by every step."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"            # put TennisMyLife YYYY.csv (or Sackmann atp_matches_YYYY.csv) files here
PROC_DIR = ROOT / "data" / "processed"     # generated, git-ignored
SITE_DATA = ROOT / "docs" / "data"         # JSON consumed by the GitHub Pages site

FIRST_YEAR = 2008      # 2008-2009 are loaded only as warm-up history for rolling features
TRAIN_YEARS = (2010, 2021)
VAL_YEAR = 2022
TEST_YEARS = (2023, 2024)
SEED = 42

# Tour-level events only: Grand Slams, Masters 1000, ATP 250/500, Tour Finals, Olympics.
# Davis Cup ('D') is excluded: different format, home advantage, dead rubbers.
LEVELS = {"G", "M", "A", "250", "500", "F", "O"}   # TML codes ATP 250/500 as "250"/"500"

ROUND_ORDER = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7,
               "RR": 4, "BR": 6, "ER": 0}
SLAM_ROUNDS = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]

SLAM_NAMES = {
    "australian open": "Australian Open",
    "roland garros": "French Open",
    "french open": "French Open",
    "wimbledon": "Wimbledon",
    "us open": "US Open",
}

# Rolling windows (days) and shrinkage for small samples
FORM_WINDOW = 182          # "preceding 6 months"
SURFACE_WINDOW = 730       # "preceding 12-24 months" -> we use the full 24 months
SHRINK_K = 5               # pseudo-matches at 50% added to every win-rate (Bayesian shrinkage)
UNRANKED_RANK = 1500       # fixed domain value for players with no ATP ranking

FEATURES = [
    "log_rank_diff",        # log(rank A) - log(rank B)
    "rank_points_diff",     # points A - points B
    "win_pct_6m_diff",      # shrunk win rate, last 6 months
    "matches_6m_diff",      # match count, last 6 months (activity / fitness proxy)
    "surface_win_pct_diff", # shrunk win rate on this surface, last 24 months
    "h2h_win_diff",         # prior head-to-head wins A - B
    "age_diff",
    "height_diff",
    "best_of_5",            # context (orientation-free)
    "surface_clay",
    "surface_grass",
]
