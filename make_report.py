# Imports
import os
import pickle
import pandas as pd
import re
import json
from pydantic import BaseModel
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from refresh_db import ImmaculateGridUtils
import numpy as np
from copy import deepcopy 
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta
from io import StringIO
import math

#--------------------------------------------------------------------------------------------------
# Global Variables

INPUT_GRID_RESULTS_FILE_PATH = './csv/results.csv'
INPUT_PROMPT_DATA_PATH = './csv/prompts.csv'
COLOR_MAP = {"Sam": "red", "Keith": "blue", "Will": "purple", "Rachel": "green", "Cliff": "orange"}
PDF_FILENAME = "./immaculate_grid_report.pdf"
TEAM_LIST = ["Cubs", "Cardinals", "Brewers", "Reds", "Pirates", "Nationals", "Mets", "Marlins", "Phillies", "Braves", "Dodgers", "Diamondbacks", "Rockies", "Giants", "Padres", "Royals", "White Sox", "Twins", "Guardians", "Tigers", "Red Sox", "Yankees", "Blue Jays", "Rays", "Orioles", "Angels", "Athletics", "Astros", "Mariners", "Rangers"]

#--------------------------------------------------------------------------------------------------
# Data Prep functions

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


def make_reversed_dict(texts): 
    """Reverse the texts data structure so that grid number points to player and their result"""
    # Initialize the reversed dictionary
    reversed_dict = {}
    
    # Iterate over each name and the list of grid objects
    for name, grid_objects in texts.items():
        for grid_obj in grid_objects:
            # Extract the grid number from the text field of the object
            grid_number = grid_obj.grid_number
    
            if grid_number is not None:
                # Set up the reversed dictionary so that the grid number points to the player and their result
                reversed_dict.setdefault(grid_number, {})[name] = grid_obj
    return reversed_dict


def make_analysis_df(texts):
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
    analysis_df = pd.DataFrame(rows)
    
    # Ensure the 'date' column is in datetime format
    analysis_df['date'] = pd.to_datetime(analysis_df['date'])

    return analysis_df


# Function to calculate smoothed metrics (score, correct, average_score_of_correct) from analysis_df
def calculate_smoothed_metrics(texts, smoothness=28):
    """Generate a DataFrame of smoothed scores, correct values, and average scores over time."""
    metric_table = []

    # Group the data by 'name' to process each person individually
    analysis_df = make_analysis_df(texts)
    grouped = analysis_df.groupby('name')

    # Loop through each person
    for name, group in grouped:
        group = group.sort_values(by='date')  # Sort by date to ensure time-based smoothing
        scores = group['score'].tolist()  # Extract scores
        corrects = group['correct'].tolist()  # Extract correct values
        avg_scores = group['average_score_of_correct'].tolist()  # Extract average score of correct
        dates = group['date'].tolist()  # Extract dates

        # Apply smoothing with the specified window size
        for i in range(smoothness, len(scores)):
            # Extract windows of each metric
            score_window = scores[i - smoothness:i]
            correct_window = corrects[i - smoothness:i]
            avg_score_window = avg_scores[i - smoothness:i]
            
            # Calculate smoothed values for each metric
            valid_scores = [score for score in score_window if score is not None]
            valid_corrects = [correct for correct in correct_window if correct is not None]
            valid_avg_scores = [avg_score for avg_score in avg_score_window if avg_score is not None]

            smoothed_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
            smoothed_correct = sum(valid_corrects) / len(valid_corrects) if valid_corrects else None
            smoothed_avg_score = sum(valid_avg_scores) / len(valid_avg_scores) if valid_avg_scores else None
            smoothed_date = dates[i] if i < len(dates) else None

            # Only add rows where there are valid smoothed values
            if smoothed_score is not None and smoothed_correct is not None and smoothed_avg_score is not None:
                metric_table.append({
                    'name': name,
                    'grid_number': i,  # Could be i, or a corresponding column like group['grid_number']
                    'smoothed_score': smoothed_score,
                    'smoothed_correct': smoothed_correct,
                    'smoothed_avg_score': smoothed_avg_score,
                    'date': smoothed_date
                })

    # Create a DataFrame from the smoothed data
    return pd.DataFrame(metric_table, columns=["name", "grid_number", "smoothed_score", "smoothed_correct", "smoothed_avg_score", "date"]).dropna()


def calculate_win_rates(texts, criterion):
    """
    Calculate win rates based on a given criterion.

    Args:
        texts (dict): The games data.
        criterion (str): The criterion to calculate win rates ("overall", "correctness", "scores", "last_rate").

    Returns:
        dict: A dictionary of win rates for each person.
    """

    reversed_dict = make_reversed_dict(texts)
    
    wins = {person: 0 for person in texts}
    for game in reversed_dict.values():
        if criterion == "overall":
            best = max((game[person].correct * 1000) + (1000 - game[person].score) for person in game)
            for person in game:
                effective_score = (game[person].correct * 1000) + (1000 - game[person].score)
                if effective_score == best:
                    wins[person] += 1
        elif criterion == "correctness":
            best = max(game[person].correct for person in game)
            for person in game:
                if game[person].correct == best:
                    wins[person] += 1
        elif criterion == "scores":
            best = min(game[person].score for person in game)
            for person in game:
                if game[person].score == best:
                    wins[person] += 1
        elif criterion == "last_rate":
            best = min((game[person].correct * 1000) + (1000 - game[person].score) for person in game)
            for person in game:
                effective_score = (game[person].correct * 1000) + (1000 - game[person].score)
                if effective_score == best:
                    wins[person] += 1

    for person in wins:
        wins[person] /= len(reversed_dict.values())

    return wins


#--------------------------------------------------------------------------------------------------
# Formatting functions

def _to_percent(y, position):
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

#--------------------------------------------------------------------------------------------------
# Plotting functions

# Graph number of immaculates
def make_fig_1(texts, COLOR_MAP):
    counts = []
    for person in texts:
        data = [(1 if obj.correct == 9 else 0) for obj in texts[person]]
        counts.append(sum(data))
    plt.bar([person for person in texts], counts, color=[COLOR_MAP[person] for person in texts])
    plt.title("Number of Immaculates")
    plt.show()

    
# Graph distributions
def make_fig_2(texts, COLOR_MAP):
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))  # Create a 3x2 grid for 5 plots
    top_bar = 130
    
    # Flatten the axes array for easier indexing
    axs = axs.flatten()
    
    for i, person in enumerate(texts):
        distribution = [0 for _ in range(0, 10)]
        for row in texts[person]:
            distribution[row.correct] += 1
        
        # Plotting the distribution for each person
        axs[i].bar(range(0, 10), distribution, color=COLOR_MAP[person])
        axs[i].set_xticks(range(0, 10))
        axs[i].set_title(person)
        axs[i].set_ylim(0, 1.2*top_bar)
    
    # Hide the last subplot if it is not used
    if len(texts) < 6:
        axs[5].set_visible(False)
    
    fig.suptitle("Correctness Distribution")
    plt.subplots_adjust(hspace=0.5)
    plt.show()


# Graph average correct
def make_fig_3(texts, COLOR_MAP):

    analysis_df = make_analysis_df(texts)
    
    title = "Average Correct"
    analysis_summary = analysis_df.groupby('name')['correct'].mean().reset_index()
    
    plt.bar(
        analysis_summary.name, 
        analysis_summary.correct, 
        color=[COLOR_MAP[person] for person in analysis_summary.name])
    plt.title(title)
    plt.show()


# Graph average score
def make_fig_4(texts, COLOR_MAP):

    analysis_df = make_analysis_df(texts)
    
    title = "Average Score"
    analysis_summary = analysis_df.groupby('name')['score'].mean().reset_index()
    
    plt.bar(
        analysis_summary.name, 
        analysis_summary.score, 
        color=[COLOR_MAP[person] for person in analysis_summary.name])
    plt.title(title)
    plt.show()


# Graph average rarity of correct square
def make_fig_5(texts, COLOR_MAP):

    analysis_df = make_analysis_df(texts)
    
    title = "Average Rarity of Correct Square"
    analysis_summary = analysis_df.groupby('name')['average_score_of_correct'].mean().reset_index()
    
    plt.bar(
        analysis_summary.name, 
        analysis_summary.average_score_of_correct, 
        color=[COLOR_MAP[person] for person in analysis_summary.name])
    plt.title(title)
    plt.show()


# Function to plot smoothed metrics using the smoothed DataFrame
def plot_smoothed_metrics(texts, metric, title, ylabel, COLOR_MAP):
    """Plot the smoothed metrics (score, correct, or average score) over time."""
    smoothed_df = calculate_smoothed_metrics(texts, smoothness=28)
    
    plt.figure(figsize=(12, 6))

    # Plot smoothed metrics for each person
    for name in smoothed_df['name'].unique():
        person_data = smoothed_df[smoothed_df['name'] == name]
        
        # Plot line with proper date formatting for the selected metric
        plt.plot(person_data['date'], person_data[metric], label=name, color=COLOR_MAP.get(name, 'blue'))

    # Formatting the plot
    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)

    # Adjust x-axis date formatting and tick placement
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))  # Format to month and year
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))  # Limit number of x-ticks to avoid clutter

    plt.tight_layout()  # Adjust layout for better display
    plt.show()


def plot_win_rates(texts):
    """Plot win rates based on various criteria."""
    
    # Set a larger figure size to widen the graphs
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    criteria = ["overall", "correctness", "scores", "last_rate"]
    titles = ["Win Rates (Overall)", "Win Rates (Correctness Only)", "Win Rates (Scores Only)", "Last Rate (Overall)"]

    for ax, criterion, title in zip(axs.flat, criteria, titles):
        wins = calculate_win_rates(texts, criterion)
        ax.bar([person for person in wins], wins.values(), color=[COLOR_MAP[person] for person in wins])
        ax.set_title(title)
        ax.set_yticks([i / 5 for i in range(6)])
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(FuncFormatter(_to_percent))

    # Adjust the layout of the subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()


def plot_best_worst_scores(texts, fig_title='Best and Worst Scores (All Time)'):
    """
    Creates a summary page in the PDF with the best and worst scores.

    Parameters:
    - texts: Data used for creating score records.
    """

    # Prepare score records
    score_records = []
    for person, games in texts.items():
        for game in games:
            grid_id = game.grid_number
            score_records.append((person, game.score, game.date, grid_id))
    
    # Sort records by score
    sorted_records = sorted(score_records, key=lambda x: x[1])
    
    # Extract best and worst scores
    best_records = sorted_records[:25]
    worst_records = sorted_records[-25:][::-1]
    
    # Create a summary page with results
    plt.figure(figsize=(8.5, 11))
    plt.text(0.5, 0.97, fig_title, fontsize=25, ha='center', va='top', fontweight='bold')
        
    # Display best scores in a structured format with dynamic spacing
    plt.text(0, 0.85, 'Best Scores:', fontsize=16, ha='left', va='top', fontweight='bold')
    plt.text(0, 0.80, 'Rank | Name        | Score  | Date       | Game ID', fontsize=10, ha='left', va='top')
    
    for i, (name, score, date, game_id) in enumerate(best_records):
        record_text = format_record(i + 1, name, score, date, game_id)
        plt.text(0, 0.75 - i * 0.025, record_text, fontsize=10, ha='left', va='top', fontfamily='monospace')
    
    # Worst Scores Section
    plt.text(0.6, 0.85, 'Worst Scores:', fontsize=16, ha='left', va='top', fontweight='bold')
    plt.text(0.6, 0.80, 'Rank | Name        | Score  | Date       | Game ID', fontsize=10, ha='left', va='top')
    
    # Display worst scores in a structured format with dynamic spacing
    for i, (name, score, date, game_id) in enumerate(worst_records):
        record_text = format_record(i + 1, name, score, date, game_id)
        plt.text(0.6, 0.75 - i * 0.025, record_text, fontsize=10, ha='left', va='top', fontfamily='monospace')

    
    plt.axis('off')  # Hide axes for the results page
    plt.show()  # Close the figure
    

def plot_best_worst_scores_30(texts):

    # Get the current date
    current_date = datetime.now()
    
    # Calculate the date 30 days ago
    thirty_days_ago = str(current_date - timedelta(days=30))
    
    # Filter the texts dictionary
    filtered_texts = {
        person: [game for game in games if game.date >= thirty_days_ago]
        for person, games in texts.items()
    }

    fig_title = 'Best and Worst Scores (Last 30 Days)'
    plot_best_worst_scores(filtered_texts, fig_title)

    
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

def build_category_structure(prompts):
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

# -------------------------------------------------------------------------------------------------
# Category Analysis Functions

def get_category_clearing_threshold(categories, person_to_category, threshold=25):
    """
    Identify categories that exceed a minimum threshold of attempts.

    Args:
        categories (set): Set of all unique categories.
        person_to_category (dict): Dictionary mapping persons to their performance on categories.
        threshold (int): Minimum average attempt count required for a category to clear the threshold.

    Returns:
        list: Categories that have an average attempt count above the threshold.
    """
    # Initialize a dictionary to count attempts for each category
    categories_to_count = {category: [] for category in categories}
    
    # Populate the dictionary with total attempts per person for each category
    for _, value in person_to_category.items():
        for category, (_, total) in value.items():
            categories_to_count[category].append(total)
    
    # Filter categories based on whether their average attempts exceed the threshold
    categories_clearing_threshold = [
        cat for cat in categories_to_count 
        if sum(categories_to_count[cat]) / len(categories_to_count[cat]) > threshold
    ]
    return categories_clearing_threshold


def get_person_to_type(texts, prompts, person_to_category):
    """
    Analyze a person's performance by type of category pair (Team-Team, Team-Stat, or Stat-Stat).

    Args:
        texts (dict): Dictionary mapping persons to their games.
        prompts (pd.DataFrame): DataFrame containing prompt data.
        person_to_category (dict): Performance metrics for each person and category.

    Returns:
        dict: Nested dictionary with counts for each type of category pair (correct and total).
    """
    # Define category pair types and initialize the structure for storing results
    types = ["Team-Team", "Team-Stat", "Stat-Stat"]
    person_to_type = {person: {t: [0, 0] for t in types} for person in person_to_category}
    
    # Iterate through each person's games
    for person, games in texts.items():
        for game in games:
            id = game.grid_number
            prompt_rows = prompts[prompts["game_id"] == id]
            if len(prompt_rows) != 1:  # Skip invalid or missing prompts
                continue
            prompt_row = prompt_rows.iloc[0][1:]  # Exclude the 'game_id' column
            
            # Analyze the category pair for each cell in the game matrix
            matrix = game.matrix
            for i in range(3):
                for j in range(3):
                    part_one, part_two = get_categories_from_prompt(prompt_row[f"{i}{j}"])
                    tag = ""
                    if category_is_team(part_one) and category_is_team(part_two):
                        tag = "Team-Team"
                    elif category_is_team(part_one) != category_is_team(part_two):
                        tag = "Team-Stat"
                    else:
                        tag = "Stat-Stat"
                    if matrix[i][j]:  # Increment correct count if cell is correct
                        person_to_type[person][tag][0] += 1
                    person_to_type[person][tag][1] += 1  # Increment total count for attempts
    return person_to_type


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

def analyze_easiest_teams(categories, person_to_category):
    """
    Identify and rank the easiest teams based on average performance.

    Args:
        categories (set): Set of all unique categories.
        person_to_category (dict): Performance data for each person and category.

    Returns:
        str: A formatted string listing the easiest teams in descending order of average performance.
    """
    overall = []
    for category in categories:
        values = []
        counts = []
        for person in person_to_category:
            # Calculate accuracy for each person in the category
            values.append(person_to_category[person][category][0] / person_to_category[person][category][1])
            counts.append(person_to_category[person][category][1])
        if category_is_team(category):
            # Calculate average performance for the category
            overall.append((category, sum(values) / len(values)))

    # Sort teams by their average performance in descending order
    result = "Consensus Easiest Teams\n"
    overall = sorted(overall, key=lambda x: x[1], reverse=True)
    for i, (category, avg) in enumerate(overall):
        result += f"{i + 1}. {category} ({round(100 * avg)}%)\n"
    return result


def analyze_team_std_dev(categories, person_to_category):
    """
    Calculate standard deviation of team performance to identify variability.

    Args:
        categories (set): Set of all unique categories.
        person_to_category (dict): Performance data for each person and category.

    Returns:
        str: A formatted string listing teams ranked by performance variability.
    """
    overall = []
    for category in categories:
        values = []
        for person in person_to_category:
            # Calculate accuracy for each person in the category
            values.append(person_to_category[person][category][0] / person_to_category[person][category][1])
        if category_is_team(category):
            # Calculate standard deviation of performance for the category
            overall.append((category, np.std(values)))

    # Sort teams by their standard deviation in descending order
    result = "Biggest Team Standard Deviations\n"
    overall = sorted(overall, key=lambda x: x[1], reverse=True)
    for i, (category, std_dev) in enumerate(overall):
        result += f"{i + 1}. {category} ({round(100 * std_dev)}%)\n"
    return result


def analyze_person_prompt_performance(
    categories, person_to_category, categories_clearing_threshold=None, direction="Best", category_type="Team"
):
    """
    Analyze the best or worst performing person for teams or non-team categories.

    Args:
        categories (set): Set of all unique categories.
        person_to_category (dict): Performance data for each person and category.
        categories_clearing_threshold (list, optional): List of categories meeting the threshold. Required for non-team categories.
        direction (str): Whether to analyze "Best" or "Worst" performers. Default is "Best".
        category_type (str): Type of category to analyze ("Team" or "Category"). Default is "Team".

    Returns:
        str: A formatted string summarizing the analysis.
    """
    overall = []
    is_team = category_type == "Team"
    threshold_filter = (
        category_is_team if is_team 
        else lambda x: x in (categories_clearing_threshold or []) and not category_is_team(x)
    )
    comparator = max if direction == "Best" else min
    default_value = 0 if direction == "Best" else 101

    # Filter categories and find the best or worst performer for each
    for category in filter(threshold_filter, categories):
        extreme_acc = default_value
        for person in person_to_category:
            acc = person_to_category[person][category][0] / person_to_category[person][category][1]
            extreme_acc = comparator(extreme_acc, acc)

        # Identify all persons who achieved the extreme accuracy
        extreme_people = [
            person for person in person_to_category
            if abs(person_to_category[person][category][0] / person_to_category[person][category][1] - extreme_acc) < 0.0001
        ]
        overall.append((category, ", ".join(extreme_people)))

    # Format the results
    result_type = f"{direction} Person for Each {'Team' if is_team else 'Non-Team Category'}"
    result = f"{result_type}\n"
    spacing = 35  # Define a standard number of spaces for alignment
    for category, people in sorted(overall, key=lambda x: x[0]):
        result += f"{category.ljust(spacing)}{people}\n"
    return result


def analyze_hardest_teams(texts, prompts):
    """
    Identify the hardest team intersections for each person and overall consensus.

    Args:
        texts (dict): Dictionary mapping persons to their games.
        prompts (pd.DataFrame): DataFrame containing prompt data.

    Returns:
        str: A formatted string summarizing the hardest team intersections.
    """
    result = StringIO()
    hardest_teams = {person: {team: [0, 0] for team in TEAM_LIST} for person in texts}

    # Analyze each person's games for hardest team intersections
    for person, games in texts.items():
        for game in games:
            id = game.grid_number
            prompt_rows = prompts[prompts["game_id"] == id]
            if len(prompt_rows) != 1:
                continue
            prompt_row = prompt_rows.iloc[0][1:]

            matrix = game.matrix
            for i in range(3):
                for j in range(3):
                    part_one, part_two = get_categories_from_prompt(prompt_row[f"{i}{j}"])
                    if category_is_team(part_one) and category_is_team(part_two):
                        team_one = get_team_from_category(part_one)
                        team_two = get_team_from_category(part_two)
                        if matrix[i][j]:
                            hardest_teams[person][team_one][0] += 1
                            hardest_teams[person][team_two][0] += 1
                        hardest_teams[person][team_one][1] += 1
                        hardest_teams[person][team_two][1] += 1

    # Output results for each person
    for person in hardest_teams:
        result.write(f"====={person}=====\n")
        for i, (team, res) in enumerate(sorted(hardest_teams[person].items(), key=lambda x: x[1][0] / x[1][1], reverse=True)):
            result.write(f"{i + 1}. {team} ({round(100 * res[0] / res[1])}%)\n")
        result.write("\n\n")

    # Calculate consensus difficulty for all teams
    consensus_intersection_difficulty = {}
    for team in TEAM_LIST:
        right = sum(hardest_teams[person][team][0] for person in hardest_teams)
        total = sum(hardest_teams[person][team][1] for person in hardest_teams)
        consensus_intersection_difficulty[team] = right / total

    result.write("=====Consensus=====\n")
    for i, (team, pct) in enumerate(sorted(consensus_intersection_difficulty.items(), key=lambda x: x[1], reverse=True)):
        result.write(f"{i + 1}. {team} ({round(100 * pct)}%)\n")

    result_string = result.getvalue()
    result.close()
    return result_string


def analyze_hardest_team_stats(texts, prompts):
    """
    Identify the hardest team-to-stat intersections for each person and overall consensus.

    Args:
        texts (dict): Dictionary mapping persons to their games.
        prompts (pd.DataFrame): DataFrame containing prompt data.

    Returns:
        str: A formatted string summarizing the hardest team-to-stat intersections.
    """
    result = StringIO()
    hardest_team_stats = {person: {team: [0, 0] for team in TEAM_LIST} for person in texts}

    # Analyze each person's games for hardest team-to-stat intersections
    for person, games in texts.items():
        for game in games:
            id = game.grid_number
            prompt_rows = prompts[prompts["game_id"] == id]
            if len(prompt_rows) != 1:
                continue
            prompt_row = prompt_rows.iloc[0][1:]

            matrix = game.matrix
            for i in range(3):
                for j in range(3):
                    part_one, part_two = get_categories_from_prompt(prompt_row[f"{i}{j}"])
                    if category_is_team(part_one) and not category_is_team(part_two):
                        team_one = get_team_from_category(part_one)
                        if matrix[i][j]:
                            hardest_team_stats[person][team_one][0] += 1
                        hardest_team_stats[person][team_one][1] += 1
                    elif not category_is_team(part_one) and category_is_team(part_two):
                        team_two = get_team_from_category(part_two)
                        if matrix[i][j]:
                            hardest_team_stats[person][team_two][0] += 1
                        hardest_team_stats[person][team_two][1] += 1

    # Output results for each person
    for person in hardest_team_stats:
        result.write(f"====={person}=====\n")
        for i, (team, res) in enumerate(sorted(hardest_team_stats[person].items(), key=lambda x: x[1][0] / x[1][1], reverse=True)):
            result.write(f"{i + 1}. {team} ({round(100 * res[0] / res[1])}%)\n")
        result.write("\n\n")

    # Calculate consensus difficulty for all teams
    consensus_intersection_difficulty = {}
    for team in TEAM_LIST:
        right = sum(hardest_team_stats[person][team][0] for person in hardest_team_stats)
        total = sum(hardest_team_stats[person][team][1] for person in hardest_team_stats)
        consensus_intersection_difficulty[team] = right / total

    result.write("=====Consensus=====\n")
    for i, (team, pct) in enumerate(sorted(consensus_intersection_difficulty.items(), key=lambda x: x[1], reverse=True)):
        result.write(f"{i + 1}. {team} ({round(100 * pct)}%)\n")

    result_string = result.getvalue()
    result.close()
    return result_string


def analyze_most_successful_exact_intersections(texts, prompts, name):
    """
    Analyze and list the most common exact category intersections for a specific person.

    Args:
        texts (dict): Dictionary mapping persons to their games.
        prompts (pd.DataFrame): DataFrame containing prompt data.
        name (str): Name of the person whose intersections are being analyzed.

    Returns:
        str: A formatted string summarizing the most common exact intersections.
    """
    result = StringIO()

    # Compute intersections for the given person
    intersections = build_intersection_structure_for_person(texts, prompts, name)

    # Determine the maximum length of "{x} & {y}:" for alignment
    max_length = max(len(f"{x} & {y}:") for (x, y) in intersections.keys())

    # Output intersections with occurrence counts
    print(f"Most Successful Exact Intersections for {name}", file=result)
    for i, ((x, y), value) in enumerate(sorted(intersections.items(), key=lambda x: x[1]['successes'], reverse=True)):
        if value['successes'] >= 5:  # Only include intersections with at least 5 occurrences
            pct = 1.0 * value['successes'] / value['attempts']
            pair_text = f"{x} & {y}:"
            # Dynamically align the pair_text with the colon included
            print(
                f"{i + 1}. {pair_text:<{max_length}} {value['successes']} of {value['attempts']} ({_to_percent(pct, 2)})",
                file=result
            )

    # Get the full output as a string
    result_string = result.getvalue()
    result.close()  # Close the StringIO object
    return result_string


def analyze_empty_team_team_intersections(texts, prompts, name, categories):
    """
    Analyze and identify team-to-team intersections that are missing for a specific person.

    Args:
        texts (dict): Dictionary mapping persons to their games.
        prompts (pd.DataFrame): DataFrame containing prompt data.
        name (str): Name of the person whose intersections are being analyzed.
        categories (set): Set of all unique categories.

    Returns:
        str: A formatted string summarizing the missing team-to-team intersections.
    """
    result = StringIO()

    # Compute existing intersections for the given person
    intersections_for_person = build_intersection_structure_for_person(texts, prompts, name)

    # Map team names to their full category names and vice versa
    team_to_full_names = {}
    full_names_to_team = {}
    for team in TEAM_LIST:
        for category in categories:
            if team in category:
                team_to_full_names[team] = category
                full_names_to_team[category] = team

    missing = 0
    present = 0
    missing_maps = {}

    # Identify missing team-to-team intersections
    print(f"Empty Team-Team Intersections for {name}", file=result)
    
    for i, team in enumerate(sorted(TEAM_LIST)):
        for other in sorted(TEAM_LIST)[i + 1:]:
            team_1, team_2 = sorted((team_to_full_names[team], team_to_full_names[other]))
            key = (team_1, team_2)
            if key not in intersections_for_person:
                # This intersection is missing from the person's game
                print(f"Missing: {team_1} & {team_2}", file=result)
                missing += 1
                missing_maps[team_1] = missing_maps.get(team_1, 0) + 1
                missing_maps[team_2] = missing_maps.get(team_2, 0) + 1
            else:
                present += 1

    # Output total missing intersections for the given person
    print(f"\n\n\n\nTotal Missing for {name}", file=result)
    for i, (team, count) in enumerate(sorted(missing_maps.items(), key=lambda x: x[1], reverse=True)):
        if count > 0:
            print(f"{i + 1}. {team} ({count})", file=result)

    # Get the full output as a string
    result_string = result.getvalue()
    result.close()  # Close the StringIO object
    return result_string
    

#--------------------------------------------------------------------------------------------------
# Report production functions

def make_generic_text_page(func, args, page_title):

    output = func(*args)

    MAX_LINES_PER_NORMAL_PAGE = 45

    page_len_scalar = max([int(math.floor(output.count('\n') / MAX_LINES_PER_NORMAL_PAGE)),1])
    
    # Create a new figure and axes
    fig, ax = plt.subplots(figsize=(8, 11*page_len_scalar))
    
    # Hide the axes for a clean canvas
    ax.axis('off')

    plt.text(0.5, 1.1, page_title, fontsize=24, ha='center', va='top', fontweight='bold')

    # Add text to the plot
    plt.text(0.0, 1, output, fontsize=8, ha='left', va='top')
    
    # Display the plot
    plt.show()

def prepare_graph_functions(texts, prompts, COLOR_MAP):
    """
    This function prepares a list of graph_functions that will be executed later in PDF generation
    """
    # Prepare various structures using "texts" and "prompts"
    categories = build_category_structure(prompts)
    person_to_category = build_person_category_structure(texts, prompts, categories)
    game_prompt_response = build_game_prompt_response_structure(texts, prompts)
    intersections = build_intersection_structure(texts, prompts)
    categories_clearing_threshold = get_category_clearing_threshold(categories, person_to_category)
    person_to_type = get_person_to_type(texts, prompts, person_to_category)

  # List of graph-making functions with their respective arguments and titles
    graph_functions = [
        (make_fig_1, (texts, COLOR_MAP), "Number of Immaculates"),
        (make_fig_2, (texts, COLOR_MAP), "Correctness Distribution"),
        (make_fig_3, (texts, COLOR_MAP), "Average Correct"),
        (make_fig_4, (texts, COLOR_MAP), "Average Score"),
        (make_fig_5, (texts, COLOR_MAP), "Average Rarity of Correct Square"),
        (
            plot_smoothed_metrics, (
                texts, 
                'smoothed_score', 
                "Smoothed Scores Over Time", 
                "Smoothed Score", 
                COLOR_MAP
            ), "Smoothed Scores Over Time"
        ),
        (
            plot_smoothed_metrics, (
                texts, 
                'smoothed_correct', 
                "Smoothed Correct Over Time", 
                "Smoothed Correct", 
                COLOR_MAP
            ), 
            "Smoothed Correct Over Time"
        ),
        (
            plot_smoothed_metrics, (
                texts, 
                'smoothed_avg_score', 
                "Smoothed Avg Score of Correct Over Time", 
                "Smoothed Avg Score of Correct", 
                COLOR_MAP
            ), 
            "Smoothed Avg Score of Correct Over Time"
        ),
        (plot_win_rates, (texts, ), "Win Rates"),
        (plot_best_worst_scores, (texts, ), 'Best and Worst Scores (All Time)'),
        (plot_best_worst_scores_30, (texts, ), 'Best and Worst Scores (Last 30 Days)'),
        (make_generic_text_page, (person_to_type_to_string, (person_to_type, ), 'Type Performance Overview'), 'Type Performance Overview'),
        (make_generic_text_page, (person_to_category_to_string, (person_to_category, ), 'Category Performance Overview'), 'Category Performance Overview'),
        (make_generic_text_page, (analyze_easiest_teams, (categories, person_to_category, ), 'Easiest Teams Overview'), 'Easiest Teams Overview'),
        (make_generic_text_page, (analyze_team_std_dev, (categories, person_to_category, ), 'Easiest Teams Standard Deviation'), 'Easiest Teams Standard Deviation'),
        (make_generic_text_page, (analyze_person_prompt_performance, (categories, person_to_category, categories_clearing_threshold, "Best", "Team", ), 'Best Team Overview'), 'Best Team Overview'),
        (make_generic_text_page, (analyze_person_prompt_performance, (categories, person_to_category, categories_clearing_threshold, "Worst", "Team", ), 'Worst Team Overview'), 'Worst Team Overview'),
        (make_generic_text_page, (analyze_person_prompt_performance, (categories, person_to_category, categories_clearing_threshold, "Best", "Category", ), 'Best Category Overview'), 'Best Category Overview'),
        (make_generic_text_page, (analyze_person_prompt_performance, (categories, person_to_category, categories_clearing_threshold, "Worst", "Category", ), 'Worst Category Overview'), 'Worst Category Overview'),
        (make_generic_text_page, (analyze_hardest_teams, (texts, prompts, ), 'Hardest Teams Overview'), 'Hardest Teams Overview'),
        (make_generic_text_page, (analyze_hardest_team_stats, (texts, prompts, ), 'Hardest Teams Stats Overview'), 'Hardest Teams Stats Overview'),
        (make_generic_text_page, (analyze_most_successful_exact_intersections, (texts, prompts, "Keith"), 'Most Successful Intersections (Keith)'), 'Most Successful Intersections (Keith)'),
        (make_generic_text_page, (analyze_most_successful_exact_intersections, (texts, prompts, "Rachel"), 'Most Successful Intersections (Rachel)'), 'Most Successful Intersections (Rachel)'),
        (make_generic_text_page, (analyze_most_successful_exact_intersections, (texts, prompts, "Sam"), 'Most Successful Intersections (Sam)'), 'Most Successful Intersections (Sam)'),
        (make_generic_text_page, (analyze_most_successful_exact_intersections, (texts, prompts, "Will"), 'Most Successful Intersections (Will)'), 'Most Successful Intersections (Will)'),
        (make_generic_text_page, (analyze_most_successful_exact_intersections, (texts, prompts, "Cliff"), 'Most Successful Intersections (Cliff)'), 'Most Successful Intersections (Cliff)'),
        (make_generic_text_page, (analyze_empty_team_team_intersections, (texts, prompts, "Keith", categories), 'Never Shown Intersections (Keith)'), 'Never Shown Intersections (Keith)'),
        (make_generic_text_page, (analyze_empty_team_team_intersections, (texts, prompts, "Rachel", categories), 'Never Shown Intersections (Rachel)'), 'Never Shown Intersections (Rachel)'),
        (make_generic_text_page, (analyze_empty_team_team_intersections, (texts, prompts, "Sam", categories), 'Never Shown Intersections (Sam)'), 'Never Shown Intersections (Sam)'),
        (make_generic_text_page, (analyze_empty_team_team_intersections, (texts, prompts, "Will", categories), 'Never Shown Intersections (Will)'), 'Never Shown Intersections (Will)'),
        (make_generic_text_page, (analyze_empty_team_team_intersections, (texts, prompts, "Cliff", categories), 'Never Shown Intersections (Cliff)'), 'Never Shown Intersections (Cliff)'),
    ]

    return graph_functions
    

def generate_report(graph_functions, pdf_filename):
    """
    Creates a PDF booklet with a cover page, table of contents, various graphs, 
    and a summary table of best and worst scores based on the provided data.

    Parameters:
    - graph_functions: Data used for creating graphs.
    - pdf_filename: Name of the output PDF file.
    """

    def add_cover_page(pdf, today_date):
        """Helper function to create the cover page."""
        plt.figure(figsize=(8.5, 11))
        plt.text(0.5, 0.7, 'Immaculate Grid Analysis Results', fontsize=24, ha='center', va='center', fontweight='bold')
        plt.text(0.5, 0.6, f'Date of Analysis: {today_date}', fontsize=16, ha='center', va='center')
        plt.axis('off')
        pdf.savefig()
        plt.close()

    def add_toc_page(pdf, graph_functions):
        """Helper function to create the table of contents page."""
        plt.figure(figsize=(8.5, 11))
        plt.text(0.5, 0.9, 'Table of Contents', fontsize=20, ha='center', va='top', fontweight='bold')

        toc_item_y_position = 0.85
        for i, (_, _, title) in enumerate(graph_functions, start=1):
            plt.text(0.1, toc_item_y_position, f'{i}. {title}', fontsize=10, ha='left', va='top')
            toc_item_y_position -= 0.02

        plt.axis('off')
        pdf.savefig()
        plt.close()

    def add_graphs_to_pdf(pdf, graph_functions):
        """Helper function to generate graphs and add them to the PDF."""
        for func, args, _ in graph_functions:
            plt.figure()
            func(*args)
            pdf.savefig()
            plt.close()

    # Use a non-interactive backend to prevent plots from rendering to the screen
    plt.switch_backend('Agg')

    # Get today's date in a readable format
    today_date = datetime.now().strftime('%B %d, %Y')

    with PdfPages(pdf_filename) as pdf:
        # Add cover page
        add_cover_page(pdf, today_date)

        # Add Table of Contents page
        add_toc_page(pdf, graph_functions)

        # Add graphs
        add_graphs_to_pdf(pdf, graph_functions)


    print(f"PDF file '{pdf_filename}' has been created with a cover page, table of contents, and all graphs.")

    return


#--------------------------------------------------------------------------------------------------
# Main Execution
if __name__ == "__main__":

    # Read inputs
    raw_results = pd.read_csv(INPUT_GRID_RESULTS_FILE_PATH, index_col=False)
    texts = preprocess_data_into_texts_structure(raw_results)
    prompts = read_prompt_data(INPUT_PROMPT_DATA_PATH)

    # Prepare analysis
    graph_functions = prepare_graph_functions(texts, prompts, COLOR_MAP)

    # Generate report
    generate_report(graph_functions, PDF_FILENAME)