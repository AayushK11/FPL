# ------------------ FPL API Endpoints ------------------ #
FPL_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_API_URL = "https://fantasy.premierleague.com/api/fixtures/"

# For compatibility with scraper.py
FPL_BOOTSTRAP = FPL_API_URL
FPL_FIXTURES = FIXTURES_API_URL

# ------------------ Fetched Data Attributes ------------------ #
PLAYERS_RAW_ATTR = "players_raw"
FIXTURES_RAW_ATTR = "fixtures_raw"

# ------------------ File Paths ------------------ #
PLAYERS_RAW_PATH = "data_fetched/players_raw.csv"
FIXTURES_RAW_PATH = "data_fetched/fixtures_raw.csv"
XGXA_CSV_PATH = "data_fetched/xg_xa.csv"

STARTERS_CSV_PATH = "output/best_starters.csv"
BENCH_CSV_PATH = "output/best_bench.csv"
ALLPLAYERS_CSV_PATH = "output/all_players.csv"
TRANSFER_SUGGESTION_CSV_PATH_TEAM1 = "output/transfer_suggestions_team1.csv"
TRANSFER_SUGGESTION_CSV_PATH_TEAM2 = "output/transfer_suggestions_team2.csv"

# ------------------ Scoring Weights for Expected Points ------------------ #
SCORE_WEIGHTS = {
    "total_points": 0.35,
    "form": 0.15,
    "expected_goals": 0.15,
    "expected_assists": 0.10,
    "points_per_game": 0.15,
    "expected_minutes_next": 0.10,
    "injury_penalty": 0.5,
    "minutes_threshold": 60,
    "chance_play_threshold": 80,
}

# ------------------ FPL Rules ------------------ #
FPL_RULES = {
    "budget": 100.0,
    "pos_limits": {"GKP": 1, "DEF": 3, "MID": 3, "FWD": 1},
    "pos_max": {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3},
    "team_limit": 3,
    "starting_xi": 11,
    "bench_size": 4,
    "bench_gkp_max": 1,
}

# ------------------ Points & Formation Parameters ------------------ #
POINTS_PARAM = {
    "goal_points": 4,
    "assist_points": 3,
    "appearance_points": 1,
    "bonus_base": 0.3,
    "minutes_scaler": 0.04,
    "fixture_weight": 1.0,
    "team_strength_weight": 0.8,
    "bookmaker_weight": 0.5,
}

FORMATION_RULES = {
    "GK": (1, 1),
    "DEF": (3, 5),
    "MID": (2, 5),
    "FWD": (1, 3),
}

MINUTES_FLOOR = 0.5
FIXTURE_EASY_BOOST = 1.1
FIXTURE_HARD_PENALTY = 0.85

# ------------------ User Team & Gameweek ------------------ #
TEAM1 = {
    "ENTRY_ID": 3816560,
    "TRANSFER_LIMIT": 2,
}
TEAM2 = {
    "ENTRY_ID": 10457709,
    "TRANSFER_LIMIT": 1,
}
GW = 2