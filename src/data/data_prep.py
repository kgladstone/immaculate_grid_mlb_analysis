import pandas as pd
import ast
import re
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.cluster import DBSCAN
import difflib

from utils.constants import TEAM_LIST, CATEGORY_LIST, TEAM_ALIASES
from utils.utils import ImmaculateGridUtils


def compare_flat_matrix_with_flat_image_responses(matrix_flat, responses_flat):
    for i, _ in enumerate(matrix_flat):
        if matrix_flat[i] and len(responses_flat[i]) == 0:
            parser_message = f"Warning: Expected a parsed value and found nothing"
            return False
        elif not matrix_flat[i] and len(responses_flat[i]) != 0:
            parser_message = f"Warning: Did not expect a parsed value and found something"
            return False
        else: # No issue
            continue
    return True


def to_percent(y, position):
    """Convert a decimal to a percentage string."""
    return f"{100 * y:.0f}%"

# Function to format each record with proper alignment
def format_record(rank, name, score, date, grid_id, name_width=7, score_width=2, date_width=10, grid_id_width=4):
    formatted_rank = f'{rank:<2}'
    formatted_name = f'{name:<{name_width}}'
    formatted_score = f'{str(score):<{score_width}}'
    formatted_date = f'{str(date):<{date_width}}'
    formatted_grid_id = f'{str(grid_id):<{grid_id_width}}'
    
    return f'{formatted_rank} | {formatted_name} | {formatted_score} | {formatted_date} | {formatted_grid_id}'

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

def get_team_name_without_city(full_name):
    full_name = normalize_team_aliases(full_name)
    # Iterate through the dictionary keys
    for team_name in TEAM_LIST:
        # Check if the team name appears at the end of the full name
        if full_name.endswith(team_name):
            return team_name
    
    # If no match is found
    return "Unknown Team"


def get_supercategory(category):
    for candidate in CATEGORY_LIST:
        if candidate in category:
            return CATEGORY_LIST[candidate]
    return category


def category_is_team(category):
    """
    Determines if a given category is a team.

    Args:
        category (str): The category to check.

    Returns:
        bool: True if the category corresponds to a team, otherwise False.
    """
    category = normalize_team_aliases(category)
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
    category = normalize_team_aliases(category)
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
    if isinstance(prompt, str):
        if " + " in prompt:
            parts = prompt.split(" + ")
            first, second = sorted(part.strip() for part in parts)
        elif isinstance(ast.literal_eval(prompt), tuple):
            first, second = ast.literal_eval(prompt)
        else:
            first = None
            second = None
    else:
        first, second = prompt
    return normalize_team_aliases(first), normalize_team_aliases(second)


def normalize_team_aliases(text):
    if not isinstance(text, str):
        return text
    normalized = text
    for alias, canonical in TEAM_ALIASES.items():
        normalized = normalized.replace(alias, canonical)
    return normalized


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
            prompt_rows = prompts[prompts['grid_id'] == game.grid_number]
            if len(prompt_rows) != 1:  # Skip games with invalid or missing data
                continue
            prompt_row = prompt_rows.iloc[0][1:]  # Exclude the 'grid_id' column
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

    prompts_clean = prompts.copy()

    prompts_clean.columns = ["grid_id", "00", "01", "02", "10", "11", "12", "20", "21", "22"]
    
    # Initialize a structure where each person has a dictionary of categories with performance metrics
    person_to_category = {person: {cat: [0, 0] for cat in categories} for person in texts}

    for person, games in texts.items():
        for game in games:
            id = game.grid_number
            prompt_rows = prompts_clean[prompts_clean["grid_id"] == id]
            if len(prompt_rows) != 1:
                continue
            prompt_row = prompt_rows.iloc[0][1:]  # Exclude the 'grid_id' column

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


def get_all_responses_from_grid_id(texts, grid_id):
    """
    Retrieves responses from all players for a given grid_id

    Args:
        texts (dict): Dictionary of persons and their associated games.

    Returns:
        dict: A nested dictionary with prompts and responses for each game.
    """
    result = dict()
    for person, games in texts.items():
        for game in games:        
            if game.grid_number == grid_id:
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
    
    for grid_id in range(min(prompts['grid_id']), max(prompts['grid_id'])):
        prompt = [item for item in prompts[prompts['grid_id'] == grid_id].iloc[0][1:]]
        response = get_all_responses_from_grid_id(texts, grid_id)
        result[grid_id] = dict()
        result[grid_id]['prompt'] = prompt
        result[grid_id]['response'] = response

    return result

def build_intersection_structure(texts, prompts):
    """
    Build out a mapping of intersection to person-specific results
    """
    
    intersections = dict()
    game_prompt_response = build_game_prompt_response_structure(texts, prompts)

    for grid_id, game_data in game_prompt_response.items():
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
                intersections[category_pair][person]['detail'] += [{'grid_id' : grid_id, 'result' : result}]

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


def get_image_metadata_entry(image_metadata, person, grid_number):
    """
    Quick search of metadata for a specific person and grid number.
    """

    # Filter the metadata DataFrame
    filtered = image_metadata[(image_metadata['grid_number'] == grid_number) & (image_metadata['submitter'] == person)]

    # Return the filtered DataFrame or None if empty
    if not filtered.empty:
        return filtered.iloc[0].to_dict()
    else:
        return None


def matrix_string_to_flat_list(matrix_string):
    # Replace JavaScript-style "true" with Python-style "True"
    python_style_string_performance = matrix_string.replace('true', 'True').replace('false', 'False')

    # Parse the string into a Python list
    nested_performance = ast.literal_eval(python_style_string_performance)

    return [item for sublist in nested_performance for item in sublist]


# Validate that performance data matches image data
def build_results_image_structure(texts, image_metadata):
    """
    Build the results structure for the image data
    """

    results = dict()

    for _, row in texts.iterrows():
        person = row['name']
        grid_number = int(row['grid_number'])

        # Replace JavaScript-style "true" with Python-style "True"
        performance = matrix_string_to_flat_list(row['matrix'])

        image_metadata_row = get_image_metadata_entry(image_metadata, person, grid_number)

        if person not in results.keys():
            results[person] = dict()

        if grid_number not in results[person].keys():
            results[person][grid_number] = dict()
            results[person][grid_number]['performance'] = performance
            results[person][grid_number]['image_metadata'] = image_metadata_row

    return results


def clean_image_parser_data(image_parser_data):
    """
    Cleans the parser_message column in the image_parser_data DataFrame 
    and preserves only grid_number, clean_parser_message, and submitter columns.

    Args:
        image_parser_data (pd.DataFrame): Input DataFrame with parser_message column.

    Returns:
        pd.DataFrame: Cleaned DataFrame with selected columns.
    """
    def _create_clean_parser_message(parser_message):
        if "Invalid image" in parser_message:
            return "Invalid image"
        elif "Failed to find logo" in parser_message:
            return "Failed to find logo"
        elif "grid already exists" in parser_message:
            return "Success"
        elif "Failed to divide grid cells" in parser_message:
            return "Failed to divide grid cells"
        elif "failed to extract grid number" in parser_message:
            return "Failed to extract grid number"
        elif "Success" in parser_message:
            return "Success"
        else:
            return parser_message
    
    # Create the clean_parser_message column
    image_parser_data['clean_parser_message'] = image_parser_data['parser_message'].apply(_create_clean_parser_message)
    
    # Return only the relevant columns
    return image_parser_data[['path', 'image_date', 'clean_parser_message', 'submitter']]


# Detailed grid cell view
def create_grid_cell_image_view(image_metadata):
    if len(image_metadata) == 0:
        return None
    df = pd.DataFrame()
    for _, row in image_metadata.iterrows():
        grid_number = row['grid_number']
        submitter = row['submitter']
        responses = row['responses']
        for position, response in responses.items():
            if response == '':
                continue
            result = {
                'grid_number': grid_number,
                'submitter': submitter,
                'position': position,
                'response': response
            }
            df = pd.concat(
                [df, pd.DataFrame([result])]
                ,ignore_index=True)
    df = df.sort_values(by='grid_number')
    return df


def create_disaggregated_results_df(
    image_metadata: pd.DataFrame,
    prompts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one tidy row per (submitter, grid_number, position) with:
    submitter | grid_number | image_filename | position | prompt | response | correct
    """

    # ------------------------------------------------------------------ #
    # 1. unify dtypes on join keys
    # ------------------------------------------------------------------ #
    def _coerce_keys(df: pd.DataFrame) -> pd.DataFrame:
        if "grid_number" in df.columns:
            df["grid_number"] = df["grid_number"].astype(str)
        if "position" in df.columns:
            df["position"] = df["position"].astype(str)
        return df

    image_metadata = _coerce_keys(image_metadata)
    prompts = _coerce_keys(prompts.rename(columns={"grid_id": "grid_number"}))

    # ------------------------------------------------------------------ #
    # 2. explode responses and melt to long
    # ------------------------------------------------------------------ #
    imd = image_metadata.copy()
    responses_expanded = (
        imd.pop("responses")                       # take the column out
        .map(lambda r: ast.literal_eval(r)      # parse if it's a str
                        if isinstance(r, str) else r)
        .apply(pd.Series)                       # one column per key
    )

    imd = pd.concat([imd, responses_expanded], axis=1)

    imd_long = imd.melt(
        id_vars=["grid_number", "date", "submitter", "image_filename"],
        var_name="position",
        value_name="response"
    )

    # ------------------------------------------------------------------ #
    # 3. tidy prompts (tuple kept intact)
    # ------------------------------------------------------------------ #
    prm_long = (
        prompts
        .melt(id_vars=["grid_number"], var_name="position", value_name="prompt")
    )
    prm_long["prompt"] = prm_long["prompt"].apply(ast.literal_eval)

    # ------------------------------------------------------------------ #
    # 4. merge image data with prompts
    # ------------------------------------------------------------------ #
    combined = imd_long.merge(
        prm_long,
        on=["grid_number", "position"],
        how="left"
    )

    # ------------------------------------------------------------------ #
    # 5. final ordering and types
    # ------------------------------------------------------------------ #
    final_cols = [
        "submitter",
        "date",
        "grid_number",
        "image_filename",
        "position",
        "prompt",
        "response",
    ]
    return (
        combined[final_cols]
        .sort_values(["grid_number", "submitter", "position"])
        .reset_index(drop=True)
    )
