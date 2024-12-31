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
root_dir = current_file.parent.parent  # Move up to the root directory
csv_dir = root_dir / "csv/"  # Target the 'csv' folder in the root directory

CONFIG_PATH = root_dir / 'config.json'  # Path to the configuration file
APPLE_TEXTS_DB_PATH = '~/Library/Messages/chat.db'  # Path to the Apple Messages database
APPLE_IMAGES_PATH = "~/Library/Messages/Attachments/"
MESSAGES_CSV_PATH = csv_dir / "results.csv"
PROMPTS_CSV_PATH = csv_dir / "prompts.csv"
IMAGES_PATH = root_dir / "images/"  # Path to the images folder
IMAGES_METADATA_PATH = root_dir / "images_metadata.json"  # Path to the images metadata file
PDF_FILENAME = root_dir / "immaculate_grid_report.pdf"  # Path for the PDF output file
LOGO_DARK_PATH = root_dir / "logo_dark.png"  # Path to the dark logo image
LOGO_LIGHT_PATH = root_dir / "logo_light.png"  # Path to the light logo image

# --------------------------------------------------------------------------------------------------
# Global Variables

# Load configuration from a separate JSON file
with open(CONFIG_PATH, 'r') as config_file:
    config = json.load(config_file)

MY_NAME = config['MY_NAME']
GRID_PLAYERS = config['GRID_PLAYERS']

# --------------------------------------------------------------------------------------------------
# Immutables

TEAM_LIST = [
    "Cubs", "Cardinals", "Brewers", "Reds", "Pirates", "Nationals", "Mets", "Marlins", "Phillies", 
    "Braves", "Dodgers", "Diamondbacks", "Rockies", "Giants", "Padres", "Royals", "White Sox", 
    "Twins", "Guardians", "Tigers", "Red Sox", "Yankees", "Blue Jays", "Rays", "Orioles", "Angels", 
    "Athletics", "Astros", "Mariners", "Rangers"
]
IMM_GRID_START_DATE = "2023-04-02"  # Start date of the immaculate grid