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

# --------------------------------------------------------------------------------------
# Global Variables
MY_NAME = 'Keith'  # Change this variable based on the user running the script
APPLE_TEXTS_DB_PATH = '~/Library/Messages/chat.db'  # Path to the Apple Messages database
CSV_PATH = './results.csv'  # Path for the CSV output file

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

# --------------------------------------------------------------------------------------
# Helper Functions
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

def _fixed_date_from_grid_number(n):
    start_date = datetime(2023, 4, 2) # hardcoded start day of immaculate grid universe
    result_date = start_date + timedelta(days=n)
    return result_date.strftime('%Y-%m-%d')

def _correct_from_text(text):
    parsed = re.search(r"Immaculate Grid (\d+) (\d)/\d", text).groups()
    return int(parsed[1])

def _score_from_text(text):
    return int(re.search(r"Rarity: (\d{1,3})", text).groups()[0])

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
    
def _row_to_name(phone_number, is_from_me):
    """
    Map phone numbers to known names or the user's own name.
    
    Parameters:
        row (pd.Series): A row from the messages dataframe containing phone number information.
    
    Returns:
        str: The name of the sender or recipient.
    """
    if is_from_me:
        return MY_NAME
    elif phone_number == "+17736776982":
        return "Sam"
    elif phone_number == "+17736776717":
        return "Will"
    elif phone_number == "+17734281342":
        return "Rachel"
    elif phone_number == "+19087311244":
        return "Keith"
    elif phone_number == "+17329910081":
        return "Cliff"

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
        print(text)
        if name is not None:
            # Is not a reaction message
            if not any(keyword in text for keyword in exclusion_keywords):
                # Has rarity
                if "Rarity: " in text:                    
                    # Has the proper immaculate grid format
                    if "Immaculate Grid " in text:
                        return True
    return False

# --------------------------------------------------------------------------------------
# Core Functions
def extract_messages(db_path):
    """
    Extract message contents from the Apple Messages database.
    
    Parameters:
        db_path (str): Path to the Apple Messages database file.
    
    Returns:
        pd.DataFrame: A dataframe containing extracted messages.
    """  

    # Extract data from SQL database
    conn = sqlite3.connect(os.path.expanduser(db_path))
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
    '''

    print("Querying message database...")
    messages_df = pd.read_sql_query(query, conn)
    conn.close()

    return messages_df

def process_messages(messages_df):
    # Extract name using phone number and "is_from_me" parameters
    print("Extracting usernames...")
    messages_df['name'] = messages_df.apply(lambda row: _row_to_name(row['phone_number'], row['is_from_me']), axis=1)

    # Filter the DataFrame to include only valid messages using the _is_valid_message function
    print("Filter on valid messages...")
    messages_df['valid'] = messages_df.apply(lambda row: _is_valid_message(row['name'], row['text']), axis=1)
    num_messages = len(messages_df)
    messages_df = messages_df[messages_df['valid'] == True]
    num_valid_messages = len(messages_df)
    print("{} of {} messages were valid...".format(num_valid_messages, num_messages))
    
    # Apply the _convert_timestamp function to clean the date field
    print("Formatting the date...")
    messages_df['date'] = messages_df['date'].apply(_convert_timestamp)
    
    # Sort the DataFrame by the cleaned date in ascending order
    print("Further cleaning...")
    messages_df = messages_df.sort_values(by="date", ascending=True)

    # Extract grid number from text
    messages_df['grid_number'] = messages_df['text'].apply(_grid_number_from_text)

    # Extract correct from text
    messages_df['correct'] = messages_df['text'].apply(_correct_from_text)

    # Extract score from text
    messages_df['score'] = messages_df['text'].apply(_score_from_text)

    # Extract matrix from text
    messages_df['matrix'] = messages_df['text'].apply(_matrix_from_text)

    # Keep only relevant columns
    print("Trimming columns...")
    columns_to_keep = ['grid_number', 'correct', 'score', 'date', 'matrix', 'name']
    messages_df = messages_df[columns_to_keep]

    print("Data extraction and transformation complete!")
    return messages_df

def test_messages(messages_df, header):
    pd.set_option('display.max_rows', None)  # Show all rows

    print("\n*******\n{}...".format(header))

    if len(messages_df) == 0:
        print("No messages to test...")
        return

    # Extract the true date of the game, it may differ from date of message
    #messages_df['date'] = messages_df['grid_number'].apply(_fixed_date_from_grid_number)
    
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
    
    # # Display the summary of the number of rows of data by cleaned date
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
    print("\nDates with fewer than 5 messages (including zero):")
    fewer_than_5 = full_summary[full_summary['message_count'] < 5]
    if not fewer_than_5.empty:
        print(fewer_than_5)
    else:
        print("No dates with fewer than 5 messages (including zero).")

    # # Call out instances where there are more than 5 messages on that date
    # print("\nDates with more than 5 messages:")
    # more_than_5 = full_summary[full_summary['message_count'] > 5]
    # if not more_than_5.empty:
    #     print(more_than_5)
    # else:
    #     print("No dates with more than 5 messages.")
    return

# --------------------------------------------------------------------------------------
# Main Execution
if __name__ == "__main__":
    
    # Check if the CSV file exists
    if os.path.exists(CSV_PATH):
        data_previous = pd.read_csv(CSV_PATH)
    else:
        data_previous = pd.DataFrame()  # Create an empty DataFrame if the file doesn't exist
    
    # Extract formatted data from text messages
    data_latest_raw = extract_messages(APPLE_TEXTS_DB_PATH)
    data_latest = process_messages(data_latest_raw)

    # Validate data from text messages
    test_messages(data_latest, "Testing on new dataset")
  
    # Count the unique rows in old DataFrame before combining
    initial_unique_previous = data_previous.drop_duplicates().shape[0]
    
    # Combine the data
    data_combined = pd.concat([data_previous, data_latest], ignore_index=True)
    
    # Drop duplicates to keep only unique rows
    data_combined_unique = data_combined.drop_duplicates()

    # Sort by 'date' first, then by 'name'
    data_sorted = data_combined_unique.sort_values(by=['date', 'name'])

    # Run tests on full dataset
    #test_messages(data_sorted, "Testing on combined dataset")
    
    # Count unique rows in the combined DataFrame
    final_unique_count = data_combined_unique.shape[0]
    
    # Calculate the number of new unique rows created
    new_unique_rows_count = final_unique_count - initial_unique_previous
    
    # Print the result
    print(f"Number of new unique rows created: {new_unique_rows_count}")

    # Write result
    data_sorted.to_csv(CSV_PATH, index=False)