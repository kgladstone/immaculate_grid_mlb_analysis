import os
import pandas as pd
import requests
from bs4 import BeautifulSoup

from data.loader import Loader
from utils.utils import ImmaculateGridUtils

class PromptsLoader(Loader):
    def __init__(self, grid_storage_path):
        super().__init__(
            source=None,  # No primary source since grid IDs are dynamic
            cache_path=grid_storage_path,
            fetch_function=self._fetch_new_prompts,
            validate_function=self._validate_prompts
        )

    def _fetch_prompts_from_cache(self):
        """
        Reads the prompt data from a CSV file and processes it for further analysis.

        Args:
            filepath (str): Path to the input CSV file containing prompt data.

        Returns:
            pd.DataFrame: A cleaned DataFrame with the game ID and processed prompt values.
        """
        with open(os.path.expanduser(self.cache_path)) as f:
            prompts = pd.read_csv(f, header=None)  # Read raw CSV data
        # Assign column names to the DataFrame
        prompts.columns = ["game_id", "00", "01", "02", "10", "11", "12", "20", "21", "22"]
        prompts = prompts.iloc[1:]  # Remove the header row

        # Process each row to clean and reformat prompt values
        new_rows = []
        for _, row in prompts.iterrows():
            new_row = {}
            for col, val in row.items():
                for char in ["(", "'", ")"]:  # Remove unwanted characters
                    val = val.replace(char, "")
                new_row[col] = val.replace(", ", " + ")  # Replace separator for readability
            new_rows.append(new_row)

        # Create a new DataFrame with cleaned rows and convert 'game_id' to integer
        prompts = pd.DataFrame(new_rows)
        prompts['game_id'] = prompts['game_id'].astype(int)

        return prompts

    @staticmethod
    def _parse_raw_prompts_to_tuple(raw):
        """
        Parse a string in the format 'part1 + part2 + part3' into a tuple ('part1', 'part2 + part3').
        """
        parts = raw.split(" + ", 1)
        return tuple(part.strip() for part in parts) if len(parts) > 1 else (parts[0].strip(), '')

    @staticmethod
    def _fetch_grid_online(grid_id):
        """
        Fetch the Immaculate Grid data from the web by grid ID.
        """
        print(f"Attempting to pull Immaculate Grid #{grid_id}")
        url = f"https://www.immaculategrid.com/grid-{grid_id}"
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        buttons = soup.find_all("button", attrs={'aria-label': True})
        labels = [str(grid_id)]
        labels.extend(PromptsLoader._parse_raw_prompts_to_tuple(button['aria-label']) for button in buttons)
        return labels

    @staticmethod
    def _fetch_grids_online(index_list):
        """
        Fetch multiple Immaculate Grids and return as a DataFrame.
        """
        header = ['grid_id'] + [f"cell{i}" for i in range(1, 10)]
        grids_data = [PromptsLoader._fetch_grid_online(i) for i in index_list]
        return pd.DataFrame(grids_data, columns=header)

    def _fetch_new_prompts(self, _):
        """
        Fetch new prompts based on missing grid IDs.
        """
        # 1 - Read cached file and determine set of ids
        if os.path.exists(self.cache_path):
            print("Getting list of grid ids from cached file...")
            data_previous = pd.read_csv(self.cache_path)
            indices_previous = set(data_previous['grid_id'])
        else:
            data_previous = pd.DataFrame(columns=['grid_id'])
            indices_previous = set()

        # 2 - Determine set of ids between universe start and today
        today_index = ImmaculateGridUtils.get_today_grid_id()
        indices_universe = set(range(1, today_index + 1))

        # 3 - Determine remaining available ids to pull
        indices_remaining = list(indices_universe - indices_previous)

        if len(indices_remaining) == 0:
            print("Short Circuit Warning: Based on grid ID calculation, no need to pull new prompts from online.")
            return None

        print("Pulling new grids from online...")
        data_incremental = PromptsLoader._fetch_grids_online(indices_remaining)
        return pd.concat([data_previous, data_incremental], ignore_index=True)

    def _validate_prompts(self, data):
        """
        Validate the prompts data.
        """
        print(f"Validating {len(data)} prompts...")
        if 'grid_id' not in data.columns:
            raise ValueError("Invalid data: Missing 'grid_id' column.")