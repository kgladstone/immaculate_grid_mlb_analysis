import json
from pathlib import Path


"""
This module contains global constants and configuration settings for the immaculate grid MLB analysis.

Global Variables:
    MY_NAME (str): The name of the user running the script. This should be set based on the user.
    GRID_PLAYERS (dict): A dictionary containing player information, including phone numbers and colors.

File Paths:
    APPLE_TEXTS_DB_PATH (str): Path to the Apple Messages database.
    MESSAGES_CSV_PATH (str): Path for the CSV output file.
    PROMPTS_CSV_PATH (str): Path for the CSV prompts file.
    PDF_FILENAME (str): Path for the PDF output file.

Immutables:
    TEAM_LIST (list): A list of MLB team names.
    IMM_GRID_START_DATE (str): The start date of the immaculate grid.

Example config.json:
{
    "MY_NAME": "JohnDoe",
    "GRID_PLAYERS": {
        "JohnDoe": {
            "phone_number": "+1234567890",
            "color": "green"
        },
        "JaneDoe": {
            "phone_number": "+0987654321",
            "color": "red"
        },
    }
}
"""

# --------------------------------------------------------------------------------------------------
# File Paths

current_file = Path(__file__).resolve()  # Get the absolute path of the current script
root_dir = current_file.parent.parent.parent  # Move up to the root directory
csv_dir = root_dir / "csv/"  # Target the 'csv' folder in the root directory
bin_dir = root_dir / "bin/"

CONFIG_PATH = bin_dir / 'config.json'  # Path to the configuration file
TEAM_LIST_PATH = root_dir / "src" / "utils" / "team_list.json"
FRANCHID_MODERN_ALIGNMENT_PATH = root_dir / "src" / "utils" / "franchid_modern_alignment.json"
# Load configuration from a separate JSON file
with open(CONFIG_PATH, 'r') as config_file:
    config = json.load(config_file)

USER_HOME = Path(config.get("USER_HOME", str(Path.home())))
APPLE_TEXTS_DB_PATH = str(USER_HOME / "Library/Messages/chat.db")  # Path to the Apple Messages database
APPLE_IMAGES_PATH = str(USER_HOME / "Library/Messages/Attachments/")
MESSAGES_CSV_PATH = csv_dir / "results.csv"
PROMPTS_CSV_PATH = csv_dir / "prompts.csv"
PLAYER_HISTORY_CSV_PATH = csv_dir / "mlb_player_history.csv"
IMAGES_PATH = root_dir / "images/"  # Path to the images folder
IMAGES_METADATA_PATH = bin_dir / "images_metadata.json"  # Path to the images metadata file
IMAGES_METADATA_CSV_PATH = bin_dir / "images_metadata.csv"  # Path to the images metadata CSV file
IMAGES_METADATA_FUZZY_LOG_PATH = bin_dir / "images_metadata_fuzzy_log.csv"  # Path to the fuzzy matching log file
IMAGES_PARSER_PATH = bin_dir / "images_parser.json"  # Path to the images parser output file
PDF_FILENAME = bin_dir / "immaculate_grid_report.pdf"  # Path for the PDF output file
LOGO_DARK_PATH = bin_dir / "logo_dark.png"  # Path to the dark logo image
LOGO_LIGHT_PATH = bin_dir / "logo_light.png"  # Path to the light logo image

# --------------------------------------------------------------------------------------------------
# Global Variables

MY_NAME = config['MY_NAME']
GRID_PLAYERS = config['GRID_PLAYERS']
GRID_PLAYERS_RESTRICTED = {player: GRID_PLAYERS[player] for player in GRID_PLAYERS if GRID_PLAYERS[player]['restricted'] == "False"}

# --------------------------------------------------------------------------------------------------
# Immutables

def _load_team_list(path: Path) -> dict:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"TEAM_LIST JSON must be an object mapping names to codes: {path}")
    return data


TEAM_LIST = _load_team_list(TEAM_LIST_PATH)


def _load_franchid_alignment(path: Path) -> dict:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Franchise alignment JSON must be an object keyed by franchID: {path}")
    return data


FRANCHID_MODERN_ALIGNMENT = _load_franchid_alignment(FRANCHID_MODERN_ALIGNMENT_PATH)

# Normalize legacy/current franchise codes to a canonical representation.
# Example: Oakland appears as OAK historically and ATH in modern data.
FRANCHID_EQUIVALENTS = {
    "OAK": "ATH",
    "ATH": "ATH",
}


def canonicalize_franchid(value) -> str:
    code = str(value).strip().upper()
    return FRANCHID_EQUIVALENTS.get(code, code)

TEAM_ALIASES = {
    "Oakland Athletics": "Athletics",
}

CATEGORY_LIST = {
    # Overall
    "WAR" : "General Award",
    "All Star" : "General Award",
    "Hall of Fame": "General Award",
    "MVP" : "General Award",
    "Rookie of the Year": "General Award",
    "First Round Draft Pick" : "General Award",

    # Demographic
    "Born" : "Demographic",
    "Major" : "Demographic",
    "United States" : "Demographic",
    "Canada" : "Demographic",
    "Cuba" : "Demographic",
    "Venezuela" : "Demographic",
    "Puerto Rico" : "Demographic",
    "Mexico" : "Demographic",
    "Dominican Republic" : "Demographic",

    # One Team
    "Only One Team" : "One Team",

    # Fielding
    "Gold Glove" : "Position",
    "Field" : "Position", # specific outfield spots captured here
    "Outfield" : "Position",
    "Shortstop" : "Position",
    "Base" : "Position", # specific infield spots captured here
    "Catch" : "Position",
    "Designated Hitter": "Position",

    # Hitting
    "AVG": "Hitting",
    "HR" : "Hitting",
    "Hits" : "Hitting",
    "Silver Slugger" : "Hitting",
    "Batting" : "Hitting",
    "RBI" : "Hitting",
    "Run" : "Hitting",

    # Pitching
    "Cy Young": "Pitching",
    "Win" : "Pitching",
    "Pitch" : "Pitching",
    "Threw" : "Pitching",
    "ERA" : "Pitching",

    # Speed
    "SB" : "Speed"
}

IMM_GRID_START_DATE = "2023-04-02"  # Start date of the immaculate grid
