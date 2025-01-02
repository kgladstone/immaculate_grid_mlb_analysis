import pandas as pd
import os

from constants import TEAM_LIST
from utils import ImmaculateGridUtils

def to_percent(y, position):
    """Convert a decimal to a percentage string."""
    return f"{100 * y:.0f}%"

# Function to format each record with proper alignment
def format_record(rank, name, score, date, game_id, name_width=7, score_width=2, date_width=10, game_id_width=4):
    formatted_rank = f'{rank:<2}'
    formatted_name = f'{name:<{name_width}}'
    formatted_score = f'{str(score):<{score_width}}'
    formatted_date = f'{str(date):<{date_width}}'
    formatted_game_id = f'{str(game_id):<{game_id_width}}'
    
    return f'{formatted_rank} | {formatted_name} | {formatted_score} | {formatted_date} | {formatted_game_id}'

def make_color_map(grid_players):
    color_map = {}
    for player, details in grid_players.items():
        color_map[player] = details['color']
    return color_map

def preprocess_data_into_texts_structure(data):
    """Build up the "texts" nested structure"""
    num_rows = len(data)
    
    # Keeping only the highest score for each grid_number and name combination
    data = data.loc[data.groupby(['grid_number', 'name'])['score'].idxmax()]
    num_rows_post = len(data)
    
    print("Data was trimmed to {} from original size of {}, handling instances of multiple entries per person per grid".format(num_rows_post, num_rows))

    # Fixing date
    print("Fixing dates by using the grid number")
    data['date'] = data['grid_number'].apply(ImmaculateGridUtils._fixed_date_from_grid_number)
    data = data.drop_duplicates()
    
    return ImmaculateGridUtils.df_to_immaculate_grid_objs(data)

def pivot_texts_by_grid_id(texts): 
    """Reverse the texts data structure so that grid number points to player and their result"""
    # Initialize the reversed dictionary
    texts_by_grid_id = {}
    
    # Iterate over each name and the list of grid objects
    for name, grid_objects in texts.items():
        for grid_obj in grid_objects:
            # Extract the grid number from the text field of the object
            grid_number = grid_obj.grid_number
    
            if grid_number is not None:
                # Set up the reversed dictionary so that the grid number points to the player and their result
                texts_by_grid_id.setdefault(grid_number, {})[name] = grid_obj
    return texts_by_grid_id


def make_texts_melted(texts):
    """Convert texts into a pandas DataFrame"""
    
    # Initialize an empty list to store rows
    rows = []
    
    # Loop through the texts to gather data
    for person, grid_objects in texts.items():
        for grid_obj in grid_objects:
            # Extract the grid number from the text field of the object
            grid_number = grid_obj.grid_number
            
            if grid_number is not None:
                # Calculate average rarity
                total_score_of_correct_squares = grid_obj.score - (100 * (9 - grid_obj.correct))
                if grid_obj.correct == 0:
                    average_score_of_correct_squares = 100
                else:
                    average_score_of_correct_squares = total_score_of_correct_squares / grid_obj.correct
                
                # Produce dataset
                row = {
                    "grid_number": grid_number,  # Use a colon here
                    "name": grid_obj.name,
                    "correct": grid_obj.correct,
                    "score": grid_obj.score,
                    "average_score_of_correct": average_score_of_correct_squares,
                    "date": grid_obj.date,
                    "matrix": grid_obj.matrix
                }
                rows.append(row)  # Append the row to the list
    
    # Create the DataFrame from the list of rows
    texts_melted = pd.DataFrame(rows)
    
    # Ensure the 'date' column is in datetime format
    texts_melted['date'] = pd.to_datetime(texts_melted['date'])

    return texts_melted


#--------------------------------------------------------------------------------------------------
# Prompt Data Processing Functions
# -------------------------------------------------------------------------------------------------

def read_prompt_data(filepath):
    """
    Reads the prompt data from a CSV file and processes it for further analysis.

    Args:
        filepath (str): Path to the input CSV file containing prompt data.

    Returns:
        pd.DataFrame: A cleaned DataFrame with the game ID and processed prompt values.
    """
    with open(os.path.expanduser(filepath)) as f:
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


def category_is_team(category):
    """
    Determines if a given category is a team.

    Args:
        category (str): The category to check.

    Returns:
        bool: True if the category corresponds to a team, otherwise False.
    """
    for team in TEAM_LIST:
        if team in category:
            return True
    return False


def get_team_from_category(category):
    """
    Extracts the team name from a given category.

    Args:
        category (str): The category containing a team name.

    Returns:
        str: The extracted team name, or an empty string if no match is found.
    """
    for team in TEAM_LIST:
        if team in category:
            return team
    return ""


def get_categories_from_prompt(prompt):
    """
    Splits a prompt string into its two component categories.

    Args:
        prompt (str): The prompt string to split.

    Returns:
        tuple: A tuple containing the two categories (part_one, part_two).
    """
    parts = prompt.split(" + ")
    first, second = sorted(part.strip() for part in parts)
    return first, second


# -------------------------------------------------------------------------------------------------
# Category Data Structure Functions

def build_category_structure(texts, prompts):
    """
    Constructs a set of unique categories from the prompt data.

    Args:
        prompts (pd.DataFrame): DataFrame containing prompt data.

    Returns:
        set: A set of unique categories derived from the prompts.
    """

    categories = set()
    for person, games in texts.items():
        for game in games:
            prompt_rows = prompts[prompts['game_id'] == game.grid_number]
            if len(prompt_rows) != 1:  # Skip games with invalid or missing data
                continue
            prompt_row = prompt_rows.iloc[0][1:]  # Exclude the 'game_id' column
            for prompt in prompt_row:
                part_one, part_two = get_categories_from_prompt(prompt)
                categories.update([part_one, part_two])  # Add both parts to the category set
    return categories


def build_person_category_structure(texts, prompts, categories):
    """
    Builds a data structure mapping persons to category performance.

    Args:
        texts (dict): Dictionary of persons and their associated games.
        prompts (pd.DataFrame): DataFrame containing prompt data.
        categories (set): Set of unique categories.

    Returns:
        dict: A nested dictionary with performance metrics for each person and category.
    """
    # Initialize a structure where each person has a dictionary of categories with performance metrics
    person_to_category = {person: {cat: [0, 0] for cat in categories} for person in texts}

    for person, games in texts.items():
        for game in games:
            id = game.grid_number
            prompt_rows = prompts[prompts["game_id"] == id]
            if len(prompt_rows) != 1:
                continue
            prompt_row = prompt_rows.iloc[0][1:]  # Exclude the 'game_id' column

            # Update category performance based on the game's matrix
            matrix = game.matrix
            for i in range(3):
                for j in range(3):
                    part_one, part_two = get_categories_from_prompt(prompt_row[f"{i}{j}"])
                    if matrix[i][j]:  # Increment correct count if the matrix cell is correct
                        person_to_category[person][part_one][0] += 1
                        person_to_category[person][part_two][0] += 1
                    person_to_category[person][part_one][1] += 1  # Increment attempt count
                    person_to_category[person][part_two][1] += 1
    return person_to_category


def get_all_responses_from_game_id(texts, game_id):
    """
    Retrieves responses from all players for a given game_id

    Args:
        texts (dict): Dictionary of persons and their associated games.

    Returns:
        dict: A nested dictionary with prompts and responses for each game.
    """
    result = dict()
    for person, games in texts.items():
        for game in games:        
            if game.grid_number == game_id:
                result[person] = dict()
                result[person]['matrix'] = [item for sublist in game.matrix for item in sublist]
                result[person]['score'] = game.score
    return result
    

def build_game_prompt_response_structure(texts, prompts):
    """
    Builds a data structure mapping games to prompts and responses.

    Args:
        texts (dict): Dictionary of persons and their associated games.
        prompts (pd.DataFrame): DataFrame containing prompt data.

    Returns:
        dict: A nested dictionary with prompts and responses for each game.
    """

    result = dict()
    
    for game_id in range(min(prompts['game_id']), max(prompts['game_id'])):
        prompt = [item for item in prompts[prompts['game_id'] == game_id].iloc[0][1:]]
        response = get_all_responses_from_game_id(texts, game_id)
        result[game_id] = dict()
        result[game_id]['prompt'] = prompt
        result[game_id]['response'] = response

    return result

def build_intersection_structure(texts, prompts):
    """
    Build out a mapping of intersection to person-specific results
    """
    
    intersections = dict()
    game_prompt_response = build_game_prompt_response_structure(texts, prompts)

    for game_id, game_data in game_prompt_response.items():
        prompt = game_data['prompt']
        response = game_data['response']
        for i, prompt_i in enumerate(prompt):
            category_pair = get_categories_from_prompt(prompt_i)
            
            # Get everyone's performance
            for person in response:
                result = response[person]['matrix'][i]

                # If category pair not in 'intersections' yet, then initialize it
                if category_pair not in intersections:
                    # Add the category pair to our structure
                    intersections[category_pair] = dict()
                    
                # If person is not in the category pair yet, then initialize it
                if person not in intersections[category_pair]:
                    intersections[category_pair][person] = {
                        'attempts' : 0,
                        'successes' : 0,
                        'detail' : []
                    }

                # Collect total attempts per person
                intersections[category_pair][person]['attempts'] += 1

                # Collect total successes per person
                if result is True:
                    intersections[category_pair][person]['successes'] += 1

                # Compile list of raw detail per person
                intersections[category_pair][person]['detail'] += [{'game_id' : game_id, 'result' : result}]

    return intersections


def build_intersection_structure_for_person(texts, prompts, name):
    """
    Build the most common exact category intersections for a specific person.

    Args:
        texts (dict): Dictionary mapping persons to their games.
        prompts (pd.DataFrame): DataFrame containing prompt data.
        name (str): Name of the person whose intersections are being analyzed.

    Returns:
        dict: Dictionary where keys are category intersections, and values are their occurrence counts.
    """
    
    intersections = build_intersection_structure(texts, prompts)

    result = {}
    
    for key in intersections:
        if name in intersections[key]:
            result[key] = intersections[key][name]
            
    return result


def person_to_type_to_string(person_to_type):
    """
    Convert the person-to-type performance dictionary into a formatted string.

    Args:
        person_to_type (dict): Performance data for each person and category type.

    Returns:
        str: Formatted string summarizing performance metrics.
    """
    result = ""
    # Iterate through each person and their performance data
    for person in person_to_type:
        result += f"{person}\n"
        for tag, (correct, total) in person_to_type[person].items():
            acc = correct / total  # Calculate accuracy as a percentage
            line = f"{tag}: {round(100 * acc)}% ({total})"
            result += f"{line}\n"
        result += "\n"
    return result


def person_to_category_to_string(person_to_category, threshold=25):
    """
    Convert the person-to-category performance dictionary into a formatted string.

    Args:
        person_to_category (dict): Performance data for each person and category.
        threshold (int): Minimum attempts required for a category to be included.

    Returns:
        str: Formatted string summarizing performance metrics for each person and category.
    """
    result = ""
    # Iterate through each person's performance data
    for person, value in person_to_category.items():
        # Sort categories by accuracy in descending order
        rankings = sorted(
            [(cat, correct / total, total) for cat, (correct, total) in value.items()],
            key=lambda x: x[1],
            reverse=True
        )
        result += f"====={person}=====\n"
        count = 1
        # Include only categories that meet the attempt threshold
        for category, accuracy, total in rankings:
            if total > threshold:
                result += f"{count}. {category} ({round(accuracy, 2)}) ({total})\n"
                count += 1
        result += "\n\n"
    return result