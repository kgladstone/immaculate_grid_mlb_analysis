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

    correct: int
    score: int
    date: str
    matrix: list[list[bool]] = None
    text: str
    name: str  # New field for player name (grid player; not MLB player)
    
    def to_dict(self):
        """
        Convert the ImmaculateGridResult instance to a dictionary format for easy CSV export.
        Returns:
            dict: A dictionary containing the result fields for CSV storage.
        """
        return {
            "correct": self.correct,
            "score": self.score,
            "date": self.date,  # Date will now be in YYYY-MM-DD format
            "matrix": json.dumps(self.matrix),  # Convert matrix to JSON string for CSV
            "text": self.text,
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
                correct = row['correct']
                score = row['score']
                date = row['date']
                matrix = json.loads(row['matrix']) if pd.notna(row['matrix']) else None
                text = row['text']
                name = row['name']
    
                # Add the result to the list for the specific player name
                result = ImmaculateGridResult(
                    correct=correct,
                    score=score,
                    date=date,
                    matrix=matrix,
                    text=text,
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
    def extract_grid_number_from_text(text):
        """
        Function to extract grid number from the text of ImmaculateGridResult
        """
        match = re.search(r"Immaculate Grid (\d+)", text)
        return int(match.group(1)) if match else None

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

def _row_to_name(row):
    """
    Map phone numbers to known names or the user's own name.
    
    Parameters:
        row (pd.Series): A row from the messages dataframe containing phone number information.
    
    Returns:
        str: The name of the sender or recipient.
    """
    if row.is_from_me:
        return MY_NAME
    elif row.phone_number == "+17736776982":
        return "Sam"
    elif row.phone_number == "+17736776717":
        return "Will"
    elif row.phone_number == "+17734281342":
        return "Rachel"
    elif row.phone_number == "+19087311244":
        return "Keith"
    elif row.phone_number == "+17329910081":
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
    
    if name and text and "Rarity:" in text:
        if not any(keyword in text for keyword in exclusion_keywords):
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
    messages_df = pd.read_sql_query(query, conn)
    conn.close()
    return messages_df

def test_messages_available(messages_df):
    for idx, row in messages_df[["text", "date", "phone_number", "is_from_me"]].iterrows():
        name = _row_to_name(row)
        if _is_valid_message(name, row.text):
            print("********************************************")
            print("Message from {} on {}: ".format(name, _convert_timestamp(row.date)))
            print(row.text)

def process_immaculate_grid_results(messages_df):
    """
    Process the messages dataframe to extract and organize Immaculate Grid results.
    
    Parameters:
        messages_df (pd.DataFrame): The dataframe containing message data.
    
    Returns:
        pd.DataFrame: A dataframe with the processed Immaculate Grid results.
    """
    texts = {}
    current_grid_number = 0
    for idx, row in messages_df[["text", "date", "phone_number", "is_from_me"]].iterrows():
        name = _row_to_name(row)
        if _is_valid_message(name, row.text):
            try:
                parsed = re.search(r"Immaculate Grid (\d+) (\d)/\d", row.text).groups()
            except Exception as e:
                continue
            grid_number = int(parsed[0])
            correct = int(parsed[1])
            score = int(re.search(r"Rarity: (\d{1,3})", row.text).groups()[0])
            date = _convert_timestamp(row.date)  # Date in YYYY-MM-DD format

            # Get specific correctness
            matrix = []
            for text_row in row.text.split("\n"):
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

            obj = ImmaculateGridResult(correct=correct, score=score, date=date, matrix=matrix, text=row.text, name=name)  # Include name here
            if name not in texts or grid_number not in texts[name] or (name in texts and grid_number in texts[name] and texts[name][grid_number].correct == correct):
                texts.setdefault(name, {}).setdefault(grid_number, obj)
            if grid_number >= current_grid_number:
                current_grid_number = grid_number

    all_results = []

    for name, results in texts.items():
        for grid_number, result in results.items():
            all_results.append(result.to_dict())

    # Convert to DataFrame and sort by date
    df = pd.DataFrame(all_results)
    df = df.sort_values(by="date")  # Sort by date
    
    return df

def write_results_to_csv(file_path, df):
    """
    Write the processed results to a CSV file, sorted by date.
    
    Parameters:
        file_path (str): The output path for the CSV file.
        df (pd.DataFrame): The dataframe to be saved.
    """
    df.to_csv(file_path, index=False)

def read_results_from_csv(file_path):
    """
    Read existing results from a CSV file.
    
    Parameters:
        file_path (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: A dataframe containing the data from the CSV file.
    """
    return pd.read_csv(file_path)

# --------------------------------------------------------------------------------------
# Main Execution
if __name__ == "__main__":
    
    # Check if the CSV file exists
    if os.path.exists(CSV_PATH):
        data_previous = read_results_from_csv(CSV_PATH)
    else:
        data_previous = pd.DataFrame()  # Create an empty DataFrame if the file doesn't exist

    # Validate data from text messages
    #test_messages_available(extract_messages(APPLE_TEXTS_DB_PATH))

    # Extract formatted data from text messages
    data_latest = process_immaculate_grid_results(extract_messages(APPLE_TEXTS_DB_PATH))
    
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
    
    write_results_to_csv(CSV_PATH, data_sorted)  # Write to CSV
