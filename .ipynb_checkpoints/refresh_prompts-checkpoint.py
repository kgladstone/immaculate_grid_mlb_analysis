# Immaculate Grid Prompt Extractor
import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime
import pandas as pd
import ast

GRID_STORAGE_PATH = './csv/prompts.csv'
IMM_GRID_START_DATE = datetime(2023, 4, 2)

def get_today_grid_id():
    """
    Calculate the Immaculate Grid ID for today.
    """
    today = datetime.now()
    return (today - IMM_GRID_START_DATE).days

def _parse_raw_prompts_to_tuple(raw):
    """
    Parse a string in the format 'part1 + part2 + part3' into a tuple ('part1', 'part2 + part3').
    """
    parts = raw.split(" + ", 1)
    return tuple(part.strip() for part in parts) if len(parts) > 1 else (parts[0].strip(), '')

def fetch_grid_online(grid_id):
    """
    Fetch the Immaculate Grid data from the web by grid ID.
    """
    print("Attempting to pull Immaculate Grid #{}".format(str(grid_id)))
    url = f"https://www.immaculategrid.com/grid-{grid_id}"
    response = requests.get(url)
    response.raise_for_status()  # Raises HTTPError for bad responses
    soup = BeautifulSoup(response.text, 'html.parser')
    buttons = soup.find_all("button", attrs={'aria-label': True})
    labels = [str(grid_id)]
    labels.extend(_parse_raw_prompts_to_tuple(button['aria-label']) for button in buttons)
    return labels

def fetch_grids_online(index_list):
    """
    Fetch multiple Immaculate Grids and return as a DataFrame.
    """
    header = ['grid_id'] + [f"cell{i}" for i in range(1, 10)]
    grids_data = [fetch_grid_online(i) for i in index_list]
    return pd.DataFrame(grids_data, columns=header)

def refresh_prompts(GRID_STORAGE_PATH):
    # 1 - Read cached file and determine set of ids
    print("Reading cached prompts file...")
    data_previous = pd.read_csv(GRID_STORAGE_PATH)
    indices_previous = set(data_previous['grid_id'])

    # 2 - Determine set of ids between universe start and today
    today_index = get_today_grid_id()
    indices_universe = set(range(1, today_index + 1))
    
    # 3 - Determine remaining available ids to pull
    indices_remaining = list(indices_universe - indices_previous)

    if len(indices_remaining) == 0:
        print("No new data to pull!")
    
    # 4 - Execute the fetch of those ids
    data_incremental = fetch_grids_online(indices_remaining)
    
    # 5 - Merge into resultant dataframe
    data_combined = pd.concat([data_previous, data_incremental], ignore_index=True)

    # 6 - Save resultant dataframe into original location   
    print("Saving prompts file...")
    data_combined.to_csv(GRID_STORAGE_PATH, index=False)


if __name__ == "__main__":
    refresh_prompts(GRID_STORAGE_PATH)
