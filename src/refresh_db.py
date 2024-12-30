"""
--------------------------------------------------------------------------------------
Script: Immaculate Grid Data Extraction and Processing
Description: This script extracts text message data from the Apple Messages database,
             processes Immaculate Grid results, and saves the data to a CSV file.
--------------------------------------------------------------------------------------
"""

# --------------------------------------------------------------------------------------
# Imports
import os
import pandas as pd
import sqlite3
import re
from pydantic import BaseModel
import json
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

from constants import MY_NAME, APPLE_TEXTS_DB_PATH, MESSAGES_CSV_PATH, PROMPTS_CSV_PATH, IMM_GRID_START_DATE, GRID_PLAYERS, csv_dir

# --------------------------------------------------------------------------------------
# Data Models
class ImmaculateGridResult(BaseModel):
    """
    A class to represent the results of an Immaculate Grid game using the Pydantic BaseModel.
    """

    grid_number: int
    correct: int
    score: int
    date: str
    matrix: list[list[bool]] = None
    name: str  # New field for player name (grid player; not MLB player)
    
    def to_dict(self):
        """
        Convert the ImmaculateGridResult instance to a dictionary format for easy CSV export.
        Returns:
            dict: A dictionary containing the result fields for CSV storage.
        """
        return {
            "grid_number": self.grid_number,
            "correct": self.correct,
            "score": self.score,
            "date": self.date,  # Date will now be in YYYY-MM-DD format
            "matrix": json.dumps(self.matrix),  # Convert matrix to JSON string for CSV
            "name": self.name,  # Include the name in the dictionary
        }

# --------------------------------------------------------------------------------------
class ImmaculateGridUtils:
    @staticmethod
    def df_to_immaculate_grid_objs(df):
        """
        Convert a DataFrame into a dictionary of ImmaculateGridResult objects grouped by name.
    
        Parameters:
            df (pd.DataFrame): The DataFrame containing the necessary columns to create ImmaculateGridResult objects.
    
        Returns:
            dict: A dictionary where the key is the player's name and the value is a list of ImmaculateGridResult objects.
        """
        results = {}
        for _, row in df.iterrows():
            try:
                # Extract values from the row
                grid_number = row['grid_number']
                correct = row['correct']
                score = row['score']
                date = row['date']
                matrix = json.loads(row['matrix']) if pd.notna(row['matrix']) else None
                name = row['name']
    
                # Add the result to the list for the specific player name
                result = ImmaculateGridResult(
                    grid_number=grid_number,
                    correct=correct,
                    score=score,
                    date=date,
                    matrix=matrix,
                    name=name
                )
    
                if name not in results:
                    results[name] = []
                results[name].append(result)
    
            except Exception as e:
                print(f"Error processing row: {row.to_dict()}\nField causing error: {e}")
                continue
    
        return results

    @staticmethod
    def _convert_timestamp(ts):
        """
        Convert Apple Messages timestamp to a human-readable date in YYYY-MM-DD format.
        
        Parameters:
            ts (int): The timestamp from Apple Messages database.
        
        Returns:
            str: The formatted date.
        """
        apple_timestamp_seconds = ts / 1e9
        unix_timestamp_seconds = apple_timestamp_seconds + 978307200
        return pd.to_datetime(unix_timestamp_seconds, unit='s').date().strftime('%Y-%m-%d')

    @staticmethod
    def _grid_number_from_text(text):
        try:
            match = re.search(r"Immaculate Grid (\d+) (\d)/\d", text)
            if not match:
                raise ValueError(f"No match found for text: '{text}'")
            else:
                parsed = match.groups()
                return int(parsed[0])
        except ValueError as e:
            print(e)
            print(text)
            return None

    @staticmethod
    def _fixed_date_from_grid_number(n):
        start_date = datetime(2023, 4, 2) # hardcoded start day of immaculate grid universe
        result_date = start_date + timedelta(days=n)
        return result_date.strftime('%Y-%m-%d')

    @staticmethod
    def _correct_from_text(text):
        parsed = re.search(r"Immaculate Grid (\d+) (\d)/\d", text).groups()
        return int(parsed[1])

    @staticmethod
    def _score_from_text(text):
        return int(re.search(r"Rarity: (\d{1,3})", text).groups()[0])

    @staticmethod
    def _matrix_from_text(text):
        """
        Extract matrix from the raw text of a message
        """
        matrix = []
        for text_row in text.split("\n"):
            current = []
            for char in text_row:
                if ord(char) == 11036:  # "⬜️":
                    current.append(False)
                elif ord(char) == 129001:  # "🟩":
                    current.append(True)
            if len(current) > 0:
                if len(current) != 3:
                    print(row.text)
                    assert len(current) == 3
                else:
                    matrix.append(current)
        assert len(matrix) == 3
        matrix = str(matrix).lower()
        return matrix

    @staticmethod
    def _row_to_name(phone_number, is_from_me):
        """
        Map phone numbers to known names or the user's own name.
        
        Parameters:
            phone_number (str): The phone number from the messages dataframe.
            is_from_me (bool): Flag indicating if the message is from the user.
        
        Returns:
            str: The name of the sender or recipient.
        """
        
        if is_from_me:
            return MY_NAME
        for name, details in GRID_PLAYERS.items():
            if details["phone_number"] == phone_number:
                return name
        return "Unknown"

    @staticmethod
    def _is_valid_message(name, text):
        """
        Validate messages based on specific content and exclusion criteria.
        
        Parameters:
            name (str): The name of the sender.
            text (str): The message text content.
        
        Returns:
            bool: True if the message is valid, False otherwise.
        """
        exclusion_keywords = [
            "Emphasized", "Laughed at", "Loved", 
            "Questioned", "Liked", "Disliked", "🏀"
        ]
    
        if text is not None:
            if name is not None and name != "Unknown":
                # Is not a reaction message
                if not any(keyword in text for keyword in exclusion_keywords):
                    # Has rarity
                    if "Rarity: " in text:                    
                        # Has the proper immaculate grid format
                        if "Immaculate Grid " in text:
                            return True
        return False

# --------------------------------------------------------------------------------------
# Message handling functions
def extract_messages(db_path):
    """
    Extract message contents from the Apple Messages database.
    
    Parameters:
        db_path (str): Path to the Apple Messages database file.
    
    Returns:
        pd.DataFrame: A dataframe containing extracted messages.
    """  

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

    try:      
        print("Querying message database...")
        conn = sqlite3.connect(os.path.expanduser(db_path))
        messages_df = pd.read_sql_query(query, conn)
        conn.close()
    except pd.errors.DatabaseError as e:
        print ("Warning: Messages query failed!")
        messages_df = pd.DataFrame()
    
    return messages_df

def process_messages(messages_df):
    # Extract name using phone number and "is_from_me" parameters
    print("Extracting usernames...")
    messages_df['name'] = messages_df.apply(lambda row: ImmaculateGridUtils._row_to_name(row['phone_number'], row['is_from_me']), axis=1)

    # Filter the DataFrame to include only valid messages using the _is_valid_message function
    print("Filter on valid messages...")
    messages_df['valid'] = messages_df.apply(lambda row: ImmaculateGridUtils._is_valid_message(row['name'], row['text']), axis=1)
    num_messages = len(messages_df)
    messages_df = messages_df[messages_df['valid'] == True]
    num_valid_messages = len(messages_df)
    print("{} of {} messages were valid...".format(num_valid_messages, num_messages))
    
    # Apply the _convert_timestamp function to clean the date field
    print("Formatting the date...")
    messages_df['date'] = messages_df['date'].apply(ImmaculateGridUtils._convert_timestamp)
    
    # Sort the DataFrame by the cleaned date in ascending order
    print("Further cleaning...")
    messages_df = messages_df.sort_values(by="date", ascending=True)

    # Extract grid number from text
    messages_df['grid_number'] = messages_df['text'].apply(ImmaculateGridUtils._grid_number_from_text)

    # Extract correct from text
    messages_df['correct'] = messages_df['text'].apply(ImmaculateGridUtils._correct_from_text)

    # Extract score from text
    messages_df['score'] = messages_df['text'].apply(ImmaculateGridUtils._score_from_text)

    # Extract matrix from text
    messages_df['matrix'] = messages_df['text'].apply(ImmaculateGridUtils._matrix_from_text)

    # Keep only relevant columns
    print("Trimming columns...")
    columns_to_keep = ['grid_number', 'correct', 'score', 'date', 'matrix', 'name']
    messages_df = messages_df[columns_to_keep]

    # Drop duplicates
    messages_df = messages_df.drop_duplicates()
    
    print("Data extraction and transformation complete!")
    return messages_df

def test_messages(messages_df):
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns


    if len(messages_df) == 0:
        print("No messages to test...")
        return

    ####################### Examine instances where grid number and date do not match ########################

    print("*" * 80)
    print("Printing instances of messages where date of message does not match date of grid result")
    print(messages_df[messages_df['date'] != messages_df['grid_number'].apply(ImmaculateGridUtils._fixed_date_from_grid_number)])
    print("Count of these instances: {}".format(
        len(messages_df[messages_df['date'] != messages_df['grid_number'].apply(ImmaculateGridUtils._fixed_date_from_grid_number)])
    ))


    ####################### Multiple texts from the same person for the same grid ########################
    print("*" * 80)
    print("Printing instances of multiple messages from same person for same grid")
    messages_by_person_by_grid = messages_df.groupby(['name','grid_number']).size().reset_index(name='message_count')
    print(messages_by_person_by_grid[messages_by_person_by_grid['message_count'] > 1])
    print("Count of these instances: {}".format(len(messages_by_person_by_grid[messages_by_person_by_grid['message_count'] > 1])))

    ################################# Examine range of dates in the messages #################################
    # Extract the minimum and maximum dates
    min_date = messages_df['date'].min()
    max_date = messages_df['date'].max()

    # Extract the minimum and maximum grid numbers
    min_grid = messages_df[messages_df['date'] == min_date]['grid_number'].max()
    max_grid = messages_df[messages_df['date'] == max_date]['grid_number'].max()

    print("Minimum date in dataset: {} (Grid Number: {})".format(min_date, min_grid))
    print("Maximum date in dataset: {} (Grid Number: {})".format(max_date, max_grid))
    
    # Generate a complete date range from min_date to max_date
    complete_date_range = pd.DataFrame(pd.date_range(start=min_date, end=max_date), columns=['date'])
    
    # Group messages by the cleaned date to count occurrences
    message_summary = messages_df.groupby('date').size().reset_index(name='message_count')
    
    # Ensure the cleaned_date column in message_summary is also of datetime type
    message_summary['date'] = pd.to_datetime(message_summary['date'])
    
    # Merge the complete date range with the message summary to ensure all dates are represented
    full_summary = complete_date_range.merge(message_summary, on='date', how='left').fillna(0)
    full_summary['message_count'] = full_summary['message_count'].astype(int)
    
    # Display the summary of the number of rows of data by cleaned date
    # print("Summary of message counts by date:")
    # print(full_summary)

    # Call out instances where there are zero messages on that date
    # print("\nDates with zero messages:")
    # zero = full_summary[full_summary['message_count'] == 0]
    # if not zero.empty:
    #     print(zero)
    # else:
    #     print("No dates with zero messages.")
    
    # Call out instances where there are zero or fewer than 5 messages on that date
    # print("\nDates with fewer than 5 messages (including zero):")
    # fewer_than_5 = full_summary[full_summary['message_count'] < 5]
    # if not fewer_than_5.empty:
    #     print(fewer_than_5)
    # else:
    #     print("No dates with fewer than 5 messages (including zero).")

    # # Call out instances where there are more than 5 messages on that date
    # print("\nDates with more than 5 messages:")
    # more_than_5 = full_summary[full_summary['message_count'] > 5]
    # if not more_than_5.empty:
    #     print(more_than_5)
    # else:
    #     print("No dates with more than 5 messages.")

    ###################################################################################################
    
    return

def refresh_results(apple_texts_file_path, output_results_file_path):
    
    # Check if the CSV file exists
    if os.path.exists(output_results_file_path):
        data_previous = pd.read_csv(output_results_file_path)
    else:
        data_previous = pd.DataFrame()  # Create an empty DataFrame if the file doesn't exist

    # Run tests on previous messages
    print("*" * 80)
    print("Running tests on cached messages...")
    test_messages(data_previous)
    
    # Extract formatted data from text messages
    data_latest_raw = extract_messages(apple_texts_file_path)

    if len(data_latest_raw) > 0:
        data_latest = process_messages(data_latest_raw)
        
        # Validate data from text messages
        print("*" * 80)
        print("Running tests on new messages...")
        test_messages(data_latest)

        # Count the unique rows in old DataFrame before combining
        initial_unique_previous = data_previous.drop_duplicates().shape[0]
        
        # Combine the data
        data_combined = pd.concat([data_previous, data_latest], ignore_index=True)
        
        # Drop duplicates to keep only unique rows
        data_combined_unique = data_combined.drop_duplicates()
    
        # Sort by 'date' first, then by 'name'
        data_sorted = data_combined_unique.sort_values(by=['date', 'name'])
        
        # Count unique rows in the combined DataFrame
        final_unique_count = data_combined_unique.shape[0]
        
        # Calculate the number of new unique rows created
        new_unique_rows_count = final_unique_count - initial_unique_previous
        
        # Print the result
        print(f"Number of new unique rows created: {new_unique_rows_count}")
    
        # Write result
        data_sorted.to_csv(output_results_file_path, index=False)

        return
        
    else:
        data_latest = pd.DataFrame()
        print("Warning: No new messages with 'Immaculate' found. Not changing the results file in {}".format(output_results_file_path))
        return

# --------------------------------------------------------------------------------------
# Prompt gathering functions

def get_today_grid_id():
    """
    Calculate the Immaculate Grid ID for today.
    """
    today = datetime.now()
    start_date = datetime.strptime(IMM_GRID_START_DATE, '%Y-%m-%d')
    return (today - start_date).days

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

def refresh_prompts(grid_storage_path):
    # 1 - Read cached file and determine set of ids
    print("Reading cached prompts file...")
    data_previous = pd.read_csv(grid_storage_path)
    indices_previous = set(data_previous['grid_id'])

    # 2 - Determine set of ids between universe start and today
    today_index = get_today_grid_id()
    indices_universe = set(range(1, today_index + 1))
    
    # 3 - Determine remaining available ids to pull
    indices_remaining = list(indices_universe - indices_previous)

    if len(indices_remaining) == 0:
        print("Warning: No new data to pull! Not changing the prompts file in {}".format(grid_storage_path))
        return

    else:
        print("Pulling new grids from online...")
        # 4 - Execute the fetch of those ids
        data_incremental = fetch_grids_online(indices_remaining)
        
        # 5 - Merge into resultant dataframe
        data_combined = pd.concat([data_previous, data_incremental], ignore_index=True)
    
        # 6 - Save resultant dataframe into original location   
        print("Saving prompts file...")
        data_combined.to_csv(grid_storage_path, index=False)

        return

# --------------------------------------------------------------------------------------
# Main Execution
if __name__ == "__main__":
    print("*" * 80)
    print("*" * 80)
    print("*" * 80)
    print("Running Immaculate Grid refresh process...")
    print("*" * 80)
    print("*" * 80)

    # Make directory at CSV_DIR
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    print("*" * 80)
    print("Handling results data...")
    refresh_results(APPLE_TEXTS_DB_PATH, MESSAGES_CSV_PATH)

    print("*" * 80)
    print("Handling prompts data...")
    refresh_prompts(PROMPTS_CSV_PATH)

    print("*" * 80)
    print("Complete!")
    print("*" * 80)

