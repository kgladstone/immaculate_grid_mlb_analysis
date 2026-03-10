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
bin_dir = root_dir / "bin/"
json_dir = bin_dir / "json/"
csv_dir = bin_dir / "csv/"  # CSV files now live under bin/csv

CONFIG_PATH = json_dir / 'config.json'  # Path to the configuration file
TEAMS_CONFIG_PATH = json_dir / "teams.json"
# Load configuration from a separate JSON file
with open(CONFIG_PATH, 'r') as config_file:
    config = json.load(config_file)

USER_HOME = Path(config.get("USER_HOME", str(Path.home())))
APPLE_TEXTS_DB_PATH = str(USER_HOME / "Library/Messages/chat.db")  # Path to the Apple Messages database
APPLE_IMAGES_PATH = str(USER_HOME / "Library/Messages/Attachments/")
MESSAGES_CSV_PATH = csv_dir / "text_message_responses.csv"
PROMPTS_CSV_PATH = csv_dir / "prompts.csv"
IMAGES_PATH = root_dir / "images/"  # Path to the images folder
IMAGES_METADATA_PATH = json_dir / "images_metadata.json"  # Path to the images metadata file
IMAGES_METADATA_CSV_PATH = csv_dir / "images_metadata.csv"  # Path to the images metadata CSV file
IMAGES_METADATA_FUZZY_LOG_PATH = csv_dir / "images_metadata_fuzzy_log.csv"  # Path to the fuzzy matching log file
IMAGES_PARSER_PATH = json_dir / "images_parser.json"  # Path to the images parser output file
PDF_FILENAME = bin_dir / "immaculate_grid_report.pdf"  # Path for the PDF output file
LOGO_DARK_PATH = bin_dir / "logos" / "logo_dark.png"  # Path to the dark logo image
LOGO_LIGHT_PATH = bin_dir / "logos" / "logo_light.png"  # Path to the light logo image

# --------------------------------------------------------------------------------------------------
# Global Variables

MY_NAME = config['MY_NAME']
GRID_PLAYERS = config['GRID_PLAYERS']
GRID_PLAYERS_RESTRICTED = {player: GRID_PLAYERS[player] for player in GRID_PLAYERS if GRID_PLAYERS[player]['restricted'] == "False"}

# --------------------------------------------------------------------------------------------------
# Immutables

def _load_teams_config(path: Path) -> tuple[dict, dict, dict]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Teams config JSON must be an object: {path}")

    team_list = data.get("team_list", {})
    modern_alignment = data.get("franchid_modern_alignment", {})
    equivalents = data.get("franchid_equivalents", {})

    if not isinstance(team_list, dict):
        raise ValueError(f"`team_list` must be an object in: {path}")
    if not isinstance(modern_alignment, dict):
        raise ValueError(f"`franchid_modern_alignment` must be an object in: {path}")
    if not isinstance(equivalents, dict):
        raise ValueError(f"`franchid_equivalents` must be an object in: {path}")

    team_list = {str(k): str(v).upper() for k, v in team_list.items()}
    modern_alignment = {str(k).upper(): v for k, v in modern_alignment.items()}
    equivalents = {str(k).upper(): str(v).upper() for k, v in equivalents.items()}
    return team_list, modern_alignment, equivalents


TEAM_LIST, FRANCHID_MODERN_ALIGNMENT, FRANCHID_EQUIVALENTS = _load_teams_config(TEAMS_CONFIG_PATH)


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
