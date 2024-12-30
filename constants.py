import json

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
# Global Variables

# Load configuration from a separate JSON file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

MY_NAME = config['MY_NAME']
GRID_PLAYERS = config['GRID_PLAYERS']

# --------------------------------------------------------------------------------------------------
# File Paths

APPLE_TEXTS_DB_PATH = '~/Library/Messages/chat.db'  # Path to the Apple Messages database
CSV_DIR = './csv/'  # Directory for CSV files
MESSAGES_CSV_PATH = './csv/results.csv'  # Path for the CSV output file
PROMPTS_CSV_PATH = './csv/prompts.csv'  # Path for the CSV prompts file
PDF_FILENAME = "./immaculate_grid_report.pdf"  # Path for the PDF output file

# --------------------------------------------------------------------------------------------------
# Immutables

TEAM_LIST = [
    "Cubs", "Cardinals", "Brewers", "Reds", "Pirates", "Nationals", "Mets", "Marlins", "Phillies", 
    "Braves", "Dodgers", "Diamondbacks", "Rockies", "Giants", "Padres", "Royals", "White Sox", 
    "Twins", "Guardians", "Tigers", "Red Sox", "Yankees", "Blue Jays", "Rays", "Orioles", "Angels", 
    "Athletics", "Astros", "Mariners", "Rangers"
]
IMM_GRID_START_DATE = "2023-04-02"  # Start date of the immaculate grid