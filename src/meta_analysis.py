# meta analysis

# This script is designed to perform a meta-analysis on the data collected from various sources.

# Load the text results
import pandas as pd
import json

from utils.constants import (
    MESSAGES_CSV_PATH,
    IMAGES_METADATA_PATH,
    IMAGES_PARSER_PATH,
)

# Load the data 
messages = pd.read_csv(MESSAGES_CSV_PATH)
images_metadata = json.load(open(IMAGES_METADATA_PATH, 'r'))
images_parser_metadata = json.load(open(IMAGES_PARSER_PATH, 'r'))


# print json nicely
# print(json.dumps(images_metadata, indent=4))

# The messages data has a column grid_number and name 
# analyze which messages have a corresponding entry in images_metadata (matching fields are grid_number and submitter)
# The images_metadata has a field called submitter which is the same as the grid_number in messages

def summarize_image_matches(messages, images_metadata):
    """
    Summarizes the matches between messages and images metadata.
    
    Args:
        messages (pd.DataFrame): DataFrame containing messages data.
        images_metadata (list): List of dictionaries containing image metadata.
        
    Returns:
        pd.DataFrame: Summary DataFrame with counts and percentages.
    """
    

    # Ensure types match
    messages['grid_number'] = messages['grid_number'].astype(int)
    messages['name'] = messages['name'].astype(str)

    # Build a set of (grid_number, submitter) from images_metadata
    image_keys = {
        (entry['grid_number'], entry['submitter']) 
        for entry in images_metadata
    }

    # Check for matches in messages
    messages['has_image'] = messages.apply(
        lambda row: (row['grid_number'], row['name']) in image_keys,
        axis=1
    )

    # Summary
    # Ensure date is datetime
    messages['date'] = pd.to_datetime(messages['date'])

    # Create a "year-month" string column
    messages['year_month'] = messages['date'].dt.to_period('M').astype(str)

    # Group by name and year_month, count how many have images vs not
    summary = messages.groupby(['name', 'year_month'])['has_image'].agg(
        total='count',
        with_image='sum'
    ).reset_index()

    # Calculate number without image and percentage
    summary['without_image'] = summary['total'] - summary['with_image']
    summary['pct_with_image'] = (summary['with_image'] / summary['total']) * 100

    # Show the summary
    print("Summary of messages with images:")
    for row in summary.itertuples(index=False):
        # Print each row in a formatted way
        # If the pct is then call it out with many asterisks
        if row.pct_with_image < 60:
            print(f"Name: {row.name}, Year-Month: {row.year_month}, Total: {row.total}, With Image: {row.with_image}, Without Image: {row.without_image}, Percentage with Image: {row.pct_with_image:.0f}% *****")
        else:
            # Print without asterisks
            print(f"Name: {row.name}, Year-Month: {row.year_month}, Total: {row.total}, With Image: {row.with_image}, Without Image: {row.without_image}, Percentage with Image: {row.pct_with_image:.0f}%")


# Find the message with the latest date that is not in images_metadata
def find_latest_message_without_image(messages, images_metadata):
    """
    Finds the latest message that does not have a corresponding image in images_metadata.
    
    Args:
        messages (pd.DataFrame): DataFrame containing messages data.
        images_metadata (list): List of dictionaries containing image metadata.
        
    Returns:
        pd.DataFrame: DataFrame with the latest message without an image.
    """
    
    # Ensure types match
    messages['grid_number'] = messages['grid_number'].astype(int)
    messages['name'] = messages['name'].astype(str)

    # Build a set of (grid_number, submitter) from images_metadata
    image_keys = {
        (entry['grid_number'], entry['submitter']) 
        for entry in images_metadata
    }

    # Check for matches in messages
    messages['has_image'] = messages.apply(
        lambda row: (row['grid_number'], row['name']) in image_keys,
        axis=1
    )

    # Find the latest message without an image
    latest_message_without_image = messages[~messages['has_image']].sort_values('date', ascending=False).head(1)

    return latest_message_without_image['name'].iloc[0], latest_message_without_image['date'].iloc[0]

def find_matching_image_parser_metadata(images_parser_metadata, name, date):
    """
    Finds matching images based on submitter and image_date.
    
    Args:
        images_parser_metadata (list): List of dictionaries containing image metadata.
        name (str): The name to match with submitter.
        date (str): The date to match with image_date.
        
    Returns:
        list: List of matching entries from images_parser_metadata.
    """
    
    # Convert date to string if it's a datetime object
    if isinstance(date, pd.Timestamp):
        date = date.strftime('%Y-%m-%d')

    # Find matching entries
    matches = [
        entry for entry in images_parser_metadata 
        if entry['submitter'] == name and entry['image_date'] == date
    ]

    return matches

def find_invalid_image_entries(images_parser_metadata, submitter):
    """
    Finds entries with invalid images based on the submitter.
    
    Args:
        images_parser_metadata (list): List of dictionaries containing image metadata.
        submitter (str): The submitter to match with.
        
    Returns:
        list: List of matching entries from images_parser_metadata.
    """
    
    # Find matching entries
    matches = [
        entry for entry in images_parser_metadata 
        if entry['submitter'] == submitter and "Warning: Invalid image" in entry['parser_message']
    ]

    # Sort by image_date
    matches.sort(key=lambda x: x['image_date'], reverse=False)

    return matches


def clean_invalid_immaculate_entries(json_path):
    """
    Reads a JSON file containing a list of dicts, removes entries where 
    'immaculate' and 'invalid image' are both in the 'parser_message' (case-insensitive),
    and overwrites the original file with the cleaned list.
    """
    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Safely filter out entries with problematic parser_message
    cleaned_data = []
    for entry in data:
        message = entry.get('parser_message')
        if isinstance(message, str):
            msg_lower = message.lower()
            if 'immaculate' in msg_lower and 'invalid image' in msg_lower:
                continue  # skip this entry
            elif 'warning: this grid already exists' in msg_lower:
                continue
        cleaned_data.append(entry)

    # Overwrite the original file
    with open(json_path, 'w') as f:
        json.dump(cleaned_data, f, indent=4)


# ----------------------------------

# Call the function to find the latest message without an image
latest_message_name, latest_message_date = find_latest_message_without_image(messages, images_metadata)
# Print the latest message without an image
print("Latest message without an image:")
print(f"Name: {latest_message_name}, Date: {latest_message_date}")

# # print(images_parser_metadata)
# # The images_parser_metadata is a JSON list of dicts with keys "submitter" and "image_date"
# # Return the list of entries that match submitter and image_date with latest_message_name and latest_message_date
# latest_message_parser_metadata = find_matching_image_parser_metadata(images_parser_metadata, latest_message_name, latest_message_date)
# print(json.dumps(latest_message_parser_metadata, indent=4))

# Get all entries from the image parser metadata for a given submitter where parser_message contains "Warning: Invalid image"
invalid_image_entries = find_invalid_image_entries(images_parser_metadata, latest_message_name)
print("Invalid image entries:")
print(json.dumps(invalid_image_entries, indent=4))


# Call the function to summarize image matches
summarize_image_matches(messages, images_metadata)

# clean_invalid_immaculate_entries(IMAGES_PARSER_PATH)