import sqlite3
import pandas as pd
import os
import datetime

from data.loader import Loader
from utils.grid_utils import ImmaculateGridUtils
from utils.constants import GRID_PLAYERS

class MessagesLoader(Loader):
    def __init__(self, db_path, cache_path):
        print("*"*20)
        print("Loading messages from Apple Messages database...")
        print("*"*20)
        super().__init__(
            source=db_path,
            cache_path=cache_path,
            fetch_function=self._fetch_messages,
            validate_function=self._validate_messages,
            dedupe_subset=["name", "grid_number"],
            dedupe_keep="first",
        )

    def _fetch_messages(self, db_path):
        # Extract data from SQL database
        query = '''
        SELECT
            message.rowid, 
            message.handle_id, 
            message.text, 
            message.date, 
            message.is_from_me, 
            handle.id as phone_number
        FROM 
            message 
        LEFT JOIN 
            handle 
        ON 
            message.handle_id = handle.rowid
        WHERE
            message.text LIKE '%Immaculate%'
        '''

        expanded_path = os.path.expanduser(db_path)
        if not os.path.exists(expanded_path):
            raise FileNotFoundError(f"Messages database not found at {expanded_path}")
        if not os.access(expanded_path, os.R_OK):
            raise PermissionError(
                f"Messages database not readable at {expanded_path}. "
                "On macOS grant Full Disk Access to the terminal/IDE process or provide a readable copy."
            )

        conn = sqlite3.connect(expanded_path)
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Extract name using phone number and "is_from_me" parameters
        print("Extracting usernames...")
        df['name'] = df.apply(lambda row: ImmaculateGridUtils._row_to_name(row['phone_number'], row['is_from_me']), axis=1)

        # Filter the DataFrame to include only valid messages using the _is_valid_message function
        print("Filter on valid messages...")
        df['valid'] = df.apply(lambda row: ImmaculateGridUtils._is_valid_message(row['name'], row['text']), axis=1)
        num_messages = len(df)
        df = df[df['valid'] == True]
        num_valid_messages = len(df)
        print("{} of {} messages were valid...".format(num_valid_messages, num_messages))
        self.diagnostics = {}
        def _date_as_ymd(val):
            sval = str(val)
            if len(sval) > 10 and sval.isdigit():
                sval = ImmaculateGridUtils._convert_timestamp(int(val))
            return sval

        df['date_str'] = df['date'].apply(_date_as_ymd)
        counts = df.groupby(['date_str', 'name']).size().reset_index(name='count')
        counts = counts.sort_values(by="date_str", ascending=False)
        cutoff_dt = pd.Timestamp.now().normalize() - pd.Timedelta(days=90)
        cutoff = cutoff_dt.strftime('%Y-%m-%d')
        counts_recent = counts[counts['date_str'] >= cutoff]

        # Add missing dates with zero counts
        all_days = pd.date_range(start=cutoff_dt, end=pd.Timestamp.now().normalize(), freq="D").strftime('%Y-%m-%d')
        idx = pd.MultiIndex.from_product([all_days, GRID_PLAYERS.keys()], names=["date_str", "name"])
        counts_recent = counts_recent.set_index(["date_str", "name"]).reindex(idx, fill_value=0).reset_index()

        print(f"Valid message counts by sender/date (last 90 days since {cutoff}):")
        for _, row in counts_recent.iterrows():
            print(f"  {row['date_str']}: {row['name']} -> {row['count']}")

        # Missing senders per day (last 90 days)
        expected_senders = set(GRID_PLAYERS.keys())
        missing_by_day = []
        all_days = pd.date_range(start=cutoff_dt, end=pd.Timestamp.now().normalize(), freq="D").strftime('%Y-%m-%d')
        counts_recent_indexed = counts_recent.set_index('date_str')
        for day in all_days:
            group = counts_recent_indexed.loc[[day]] if day in counts_recent_indexed.index else pd.DataFrame(columns=['name', 'count'])
            present = set(group['name']) if not group.empty else set()
            missing = expected_senders - present
            if missing:
                missing_by_day.append((day, sorted(missing)))
        self.diagnostics = {
            "counts_recent": counts_recent.to_dict(orient="records"),
            "cutoff": cutoff,
        }
        
        # Apply the _convert_timestamp function to clean the date field
        print("Formatting the date...")
        # Ensure you're modifying the DataFrame explicitly using .loc
        df.loc[:, 'date'] = df['date'].apply(ImmaculateGridUtils._convert_timestamp)

        # Sort the DataFrame by the cleaned date in ascending order
        print("Further cleaning...")
        df = df.sort_values(by="date", ascending=True)

        # Extract grid number from text
        df['grid_number'] = df['text'].apply(ImmaculateGridUtils._grid_number_from_text)

        # Extract correct from text
        df['correct'] = df['text'].apply(ImmaculateGridUtils._correct_from_text)

        # Extract score from text
        df['score'] = df['text'].apply(ImmaculateGridUtils._score_from_text)

        # Extract matrix from text
        df['matrix'] = df['text'].apply(ImmaculateGridUtils._matrix_from_text)

        # Keep only relevant columns
        print("Trimming columns...")
        columns_to_keep = ['grid_number', 'correct', 'score', 'date', 'matrix', 'name']
        df = df[columns_to_keep]

        # Drop duplicates
        df = df.drop_duplicates()
        
        print("Data extraction and transformation complete!")
        return df
    

    def _validate_messages(self, df):
        print(f"Validating {len(df)} messages...")

        pd.set_option('display.max_rows', None)  # Show all rows
        pd.set_option('display.max_columns', None)  # Show all columns

        if len(df) == 0:
            print("No messages to test...")
            return

        ################################# Examine range of dates in the messages #################################
        # Extract the minimum and maximum dates
        min_date = df['date'].min()
        max_date = df['date'].max()

        # Extract the minimum and maximum grid numbers
        min_grid = df[df['date'] == min_date]['grid_number'].max()
        max_grid = df[df['date'] == max_date]['grid_number'].max()

        print("Minimum date in dataset: {} (Grid Number: {})".format(min_date, min_grid))
        print("Maximum date in dataset: {} (Grid Number: {})".format(max_date, max_grid))
        
        ################################# Check for missing grid numbers #################################

        # Get today's grid number
        today_grid = ImmaculateGridUtils.get_today_grid_id()

        # Get the number of distinct names in the dataset
        num_names = df['name'].nunique()

        # Determine the valid grid range
        valid_grid_range = range(df['grid_number'].min(), today_grid + 1)

        # Count the number of entries for each grid number
        grid_counts = df['grid_number'].value_counts()

        # Identify grids with fewer than num_names entries within the valid range
        grid_counts_below_num_names = {
            grid: count
            for grid, count in grid_counts.items()
            if grid in valid_grid_range and count < num_names
        }

        # Add grids completely missing from the dataset
        all_grids_in_range = set(valid_grid_range)
        missing_grids = all_grids_in_range - set(grid_counts.index)
        for grid in missing_grids:
            grid_counts_below_num_names[grid] = 0

        # Convert to a DataFrame
        incomplete_grids_df = pd.DataFrame.from_dict(
            grid_counts_below_num_names, orient='index', columns=['message_count']
        ).reset_index().rename(columns={'index': 'grid_number'})

        # Add the maximum date for each grid number (use NaT for missing grids)
        min_date_per_grid = df.groupby('grid_number')['date'].min().reset_index()
        incomplete_grids_df = incomplete_grids_df.merge(min_date_per_grid, on='grid_number', how='left')

        # Sort by grid number
        incomplete_grids_df = incomplete_grids_df.sort_values(by='grid_number')

        # Replace NaT for completely missing grids
        incomplete_grids_df['date'] = incomplete_grids_df['date'].fillna('Missing')

        # Display results from the last 90 days
        print("\nValidation of grid text message results from the last 90 days:")
        ninety_days_ago = (pd.Timestamp.now() - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
        recent_incomplete_grids = incomplete_grids_df[
            (incomplete_grids_df['date'] != 'Missing') & 
            (incomplete_grids_df['date'] >= ninety_days_ago)
        ]
        print(recent_incomplete_grids)
        
        return
