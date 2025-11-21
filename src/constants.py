# ------------------ FPL API Endpoints ------------------ #
BASE_URL = "https://fantasy.premierleague.com/api/"
FPL_API_URL = BASE_URL + "bootstrap-static/"
FIXTURES_API_URL = BASE_URL + "fixtures/"
USER_TEAM_URL_TEMPLATE = BASE_URL + "entry/{entry_id}/event/{event_id}/picks/"

# ------------------ File Paths ------------------ #
PLAYERS_RAW_PATH = "data/players_raw.csv"
FIXTURES_RAW_PATH = "data/fixtures_raw.csv"
TEAMS_RAW_PATH = "data/teams_raw.csv"
XGXA_CSV_PATH = "data/xgxa.csv"
ALLPLAYERS_CSV_PATH = "output/all_players.csv"
BEST_ELEVEN_CSV_PATH = "output/best_eleven.csv"
TRANSFER_SUGGESTION_CSV_PATH = "output/{teamname}.csv"

# ------------------ Points & Multipliers ------------------ #
POINTS_PARAM = {
    "goal_points": 6,  # Points per goal scored
    "assist_points": 3,  # Points per assist
    "bonus_base": 0.5,  # Base bonus contribution
    "minutes_scaler": 0.05,  # Factor for minutes played
    "bookmaker_weight": 0.2,  # Weight for bookmaker odds
}

FIXTURE_EASY_BOOST = 1.2
FIXTURE_HARD_PENALTY = 0.8
MINUTES_FLOOR = 0.5  # Minimum minutes multiplier

# ------------------ Numeric Columns ------------------ #
NUMERIC_COLUMNS = [
    "total_points",
    "now_cost",
    "minutes",
    "form",
    "selected_by_percent",
]

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
REV_POSITION_MAP = {v: k for k, v in POSITION_MAP.items()}

# ------------------ Formation Rules ------------------ #
FORMATION_RULES = {
    "GK": (2, 2),
    "DEF": (3, 5),
    "MID": (3, 5),
    "FWD": (1, 3),
}

# ------------------ Output Columns ------------------ #
OUTPUT_COLS = [
    "web_name",
    "team_name",
    "position_code",
    "now_cost",
    "ep_next_3gw",
    "total_points",
    "form",
    "xG",
    "xA",
]

# ------------------ Misc ------------------ #
DEFAULT_BUDGET = 100.0  # Maximum team budget in million
SQUAD_SIZE = 11  # Size of optimized squad
TEAM_PLAYER_LIMIT = 3  # Max players per real-life team in squad

# ------------------ Gameweek & Teams ------------------ #
GW = 10

TEAMS = [
    {
        "NAME": "Paneer Tikka MoSalah",
        "MANAGER": "Aayush",
        "ENTRY_ID": 3816560,
        "TRANSFER_LIMIT": 2,
    },
    {
        "NAME": "Old Havertz Kai Hard",
        "MANAGER": "Aayush",
        "ENTRY_ID": 10457709,
        "TRANSFER_LIMIT": 3,
    },
    # {
    #     "NAME": "Lund Hope",
    #     "MANAGER": "Aayush",
    #     "ENTRY_ID": 4948948,
    #     "TRANSFER_LIMIT": 2,
    # },
]
