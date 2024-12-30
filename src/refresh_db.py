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
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from utils import ImmaculateGridUtils

from constants import APPLE_TEXTS_DB_PATH, MESSAGES_CSV_PATH, PROMPTS_CSV_PATH, IMM_GRID_START_DATE, csv_dir

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
    # Ensure you're modifying the DataFrame explicitly using .loc
    messages_df.loc[:, 'date'] = messages_df['date'].apply(ImmaculateGridUtils._convert_timestamp)

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

    # ####################### Examine instances where grid number and date do not match ########################

    # print("*" * 80)
    # print("Printing instances of messages where date of message does not match date of grid result")
    # print(messages_df[messages_df['date'] != messages_df['grid_number'].apply(ImmaculateGridUtils._fixed_date_from_grid_number)])
    # print("Count of these instances: {}".format(
    #     len(messages_df[messages_df['date'] != messages_df['grid_number'].apply(ImmaculateGridUtils._fixed_date_from_grid_number)])
    # ))

    ####################### Multiple texts from the same person for the same grid ########################
    print("*" * 80)
    print("Printing instances of multiple messages from same person for same grid")

    # Group by 'name' and 'grid_number', and count messages
    messages_by_person_by_grid = messages_df.groupby(['name', 'grid_number']).size().reset_index(name='message_count')

    # Filter for entries where the same person has more than one message for a grid
    filtered_df = messages_by_person_by_grid[messages_by_person_by_grid['message_count'] > 1]

    # Compute the maximum date for each grid_number
    min_date_per_grid = messages_df.groupby('grid_number')['date'].min().reset_index()

    # Merge the maximum date with the filtered DataFrame
    filtered_df = filtered_df.merge(min_date_per_grid, on='grid_number')

    # Sort the resulting DataFrame by 'grid_number' in ascending order
    sorted_filtered_df = filtered_df.sort_values(by='grid_number')

    # Print the final DataFrame
    print(sorted_filtered_df)

    # Print the count of these instances
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
    
    ################################# Check for missing grid numbers #################################

    # Get today's grid number
    today_grid = get_today_grid_id()

    # Get the number of distinct names in the dataset
    num_names = messages_df['name'].nunique()

    # Determine the valid grid range
    valid_grid_range = range(messages_df['grid_number'].min(), today_grid + 1)

    # Count the number of entries for each grid number
    grid_counts = messages_df['grid_number'].value_counts()

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
    min_date_per_grid = messages_df.groupby('grid_number')['date'].min().reset_index()
    incomplete_grids_df = incomplete_grids_df.merge(min_date_per_grid, on='grid_number', how='left')

    # Sort by grid number
    incomplete_grids_df = incomplete_grids_df.sort_values(by='grid_number')

    # Replace NaT for completely missing grids
    incomplete_grids_df['date'] = incomplete_grids_df['date'].fillna('Missing')

    # Display the final table
    print(incomplete_grids_df)

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

