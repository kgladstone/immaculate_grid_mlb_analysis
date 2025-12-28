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
# Load configuration from a separate JSON file
with open(CONFIG_PATH, 'r') as config_file:
    config = json.load(config_file)

USER_HOME = Path(config.get("USER_HOME", str(Path.home())))
APPLE_TEXTS_DB_PATH = str(USER_HOME / "Library/Messages/chat.db")  # Path to the Apple Messages database
APPLE_IMAGES_PATH = str(USER_HOME / "Library/Messages/Attachments/")
MESSAGES_CSV_PATH = csv_dir / "results.csv"
PROMPTS_CSV_PATH = csv_dir / "prompts.csv"
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

TEAM_LIST = {
    "Cubs": "CHC",
    "Cardinals": "STL",
    "Brewers": "MIL",
    "Reds": "CIN",
    "Pirates": "PIT",
    "Nationals": "WSH",
    "Mets": "NYM",
    "Marlins": "MIA",
    "Phillies": "PHI",
    "Braves": "ATL",
    "Dodgers": "LAD",
    "Diamondbacks": "ARI",
    "Rockies": "COL",
    "Giants": "SFG",
    "Padres": "SDP",
    "Royals": "KCR",
    "White Sox": "CWS",
    "Twins": "MIN",
    "Guardians": "CLE",
    "Tigers": "DET",
    "Red Sox": "BOS",
    "Yankees": "NYY",
    "Blue Jays": "TOR",
    "Rays": "TBR",
    "Orioles": "BAL",
    "Angels": "LAA",
    "Athletics": "ATH",
    "Astros": "HOU",
    "Mariners": "SEA",
    "Rangers": "TEX"
}

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
