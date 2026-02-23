# analysis.py
import re
import os
import ast
import json
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional, Any, Dict, Tuple
from PIL import Image
from data.image_processor import ImageProcessor

from data.data_prep import (
    format_record,
    to_percent,
    make_texts_melted,
    pivot_texts_by_grid_id,
    category_is_team,
    get_team_from_category,
    get_categories_from_prompt,
    build_intersection_structure_for_person,
    build_results_image_structure,
    build_person_category_structure,
    clean_image_parser_data,
    create_grid_cell_image_view,
    get_team_name_without_city,
    get_supercategory,
)
from utils.constants import (
    TEAM_LIST,
    GRID_PLAYERS,
    GRID_PLAYERS_RESTRICTED,
    IMM_GRID_START_DATE,
    IMAGES_PATH,
    APPLE_TEXTS_DB_PATH,
    IMAGES_METADATA_PATH,
)

# Helpers for grid/date mapping and week buckets
_IMM_START_DT = datetime.strptime(IMM_GRID_START_DATE, "%Y-%m-%d")


def grid_to_date(grid_number: int) -> datetime:
    return _IMM_START_DT + timedelta(days=int(grid_number))


def week_start_from_grid(grid_number: int) -> datetime:
    grid_date = grid_to_date(grid_number)
    return grid_date - timedelta(days=grid_date.weekday())

# Function to calculate smoothed metrics (score, correct, average_score_of_correct) from texts_melted
def calculate_smoothed_metrics(texts, smoothness=28):
    """Generate a DataFrame of smoothed scores, correct values, and average scores over time."""
    metric_table = []

    # Group the data by 'name' to process each person individually
    texts_melted = make_texts_melted(texts)
    grouped = texts_melted.groupby('name')

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

    texts_by_grid_id = pivot_texts_by_grid_id(texts)
    
    wins = {person: 0 for person in texts}
    for game in texts_by_grid_id.values():
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
        wins[person] /= len(texts_by_grid_id.values())

    return wins
    
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

    prompts_clean = prompts.copy()
    prompts_clean.columns = ["grid_id", "00", "01", "02", "10", "11", "12", "20", "21", "22"]

    # Define category pair types and initialize the structure for storing results
    types = ["Team-Team", "Team-Stat", "Stat-Stat"]
    person_to_type = {person: {t: [0, 0] for t in types} for person in person_to_category}
    
    # Iterate through each person's games
    for person, games in texts.items():
        for game in games:
            id = game.grid_number
            prompt_rows = prompts_clean[prompts_clean["grid_id"] == id]
            if len(prompt_rows) != 1:  # Skip invalid or missing prompts
                continue
            prompt_row = prompt_rows.iloc[0][1:]  # Exclude the 'grid_id' column
            
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


def analyze_person_type_performance(texts, prompts, categories):
    """
    Convert the person-to-type performance dictionary into a crosstab DataFrame with a consensus row.

    Args:
        person_to_type (dict): Performance data for each person and category type.

    Returns:
        pd.DataFrame: A crosstab summarizing performance metrics with:
                      - Rows as persons (including a "Consensus" row).
                      - Columns as types.
                      - Values as "Accuracy% (Count)".
    """
    rows = []

    person_to_category = build_person_category_structure(texts, prompts, categories)
    person_to_type = get_person_to_type(texts, prompts, person_to_category)

    # Iterate through each person and their performance data
    for person, types in person_to_type.items():
        for tag, (correct, total) in types.items():
            acc = round(100 * (correct / total)) if total > 0 else 0  # Calculate accuracy
            rows.append({
                "Person": person,
                "Type": tag,
                "Correct": correct,
                "Total": total,
                "Value": f"{acc}% ({total})"
            })

    # Convert rows into a DataFrame
    df = pd.DataFrame(rows)

    # Calculate consensus accuracy for each type
    consensus_data = df.groupby("Type").apply(
        lambda group: (group["Correct"].sum(), group["Total"].sum())
    )

    # Prepare the consensus row
    consensus_row = {
        "Person": "Consensus",
    }
    for tag, (correct_sum, total_sum) in consensus_data.items():
        if total_sum > 0:
            consensus_acc = round(100 * correct_sum / total_sum)
            consensus_row[tag] = f"{consensus_acc}% ({total_sum})"
        else:
            consensus_row[tag] = "0% (0)"

    # Create a crosstab with persons as rows and types as columns
    crosstab_df = df.pivot(index="Person", columns="Type", values="Value").fillna("0% (0)")
    crosstab_df.reset_index(inplace=True)  # Convert the index into a regular column

    # Append the consensus row
    consensus_row_df = pd.DataFrame([consensus_row])
    crosstab_df = pd.concat([crosstab_df, consensus_row_df], ignore_index=True)

    return crosstab_df


def analyze_team_performance(texts, prompts, categories):
    """
    Analyze team performance by calculating consensus accuracy and standard deviation of accuracy.

    Args:
        categories (set): Set of all unique categories.
        person_to_category (dict): Performance data for each person and category.

    Returns:
        pd.DataFrame: A DataFrame with columns Team, Consensus Accuracy, and Std Deviation of Accuracy.
    """
    overall = []

    person_to_category = build_person_category_structure(texts, prompts, categories)

    for category in categories:
        values = []
        for person in person_to_category:
            if category in person_to_category[person]:
                correct, total = person_to_category[person][category]
                if total > 0:
                    values.append(correct / total)
        
        # Only process teams
        if category_is_team(category) and values:
            consensus_accuracy = np.mean(values) * 100  # Average accuracy as percentage
            std_deviation = np.std(values) * 100  # Standard deviation as percentage
            overall.append({
                "Team": category,
                "Consensus Accuracy": str(round(consensus_accuracy, 1)) + "%",
                "Std Deviation of Accuracy": str(round(std_deviation, 1)) + "%"
            })

    # Create and return the DataFrame
    df = pd.DataFrame(overall)
    df.sort_values(by="Consensus Accuracy", ascending=False, inplace=True)  # Sort by consensus accuracy
    return df


def analyze_person_prompt_performance(texts, prompts, categories, category_type="Team"
):
    """
    Analyze the best and worst performing persons for teams or non-team categories and return as a DataFrame.

    Args:
        categories (set): Set of all unique categories.
        person_to_category (dict): Performance data for each person and category.
        category_type (str): Type of category to analyze ("Team" or "Category"). Default is "Team".

    Returns:
        pd.DataFrame: A DataFrame summarizing the best and worst performers for each category.
    """
    overall = []
    is_team = category_type == "Team"

    person_to_category = build_person_category_structure(texts, prompts, categories)

    categories_clearing_threshold = get_category_clearing_threshold(categories, person_to_category)

    threshold_filter = (
        category_is_team if is_team 
        else lambda x: x in (categories_clearing_threshold or []) and not category_is_team(x)
    )

    # Filter categories and find the best and worst performers for each
    for category in filter(threshold_filter, categories):
        best_acc = 0
        worst_acc = 101
        best_people = []
        worst_people = []

        for person, data in person_to_category.items():
            if category not in data:
                continue
            correct, total = data[category]
            if total == 0:
                continue
            acc = correct / total

            # Update best performers
            if acc > best_acc:
                best_acc = acc
                best_people = [person]
            elif acc == best_acc:
                best_people.append(person)

            # Update worst performers
            if acc < worst_acc:
                worst_acc = acc
                worst_people = [person]
            elif acc == worst_acc:
                worst_people.append(person)

        if best_people or worst_people:
            overall.append({
                "Category": category,
                "Best Person": ", ".join(best_people),
                "Best Accuracy": f"{round(best_acc * 100, 1)}%",
                "Worst Person": ", ".join(worst_people),
                "Worst Accuracy": f"{round(worst_acc * 100, 1)}%",
            })

    # Create and return the DataFrame
    df = pd.DataFrame(overall)
    df.sort_values(by="Category", inplace=True)  # Sort by category
    return df


def analyze_hardest_intersections(texts, prompts, mode="team"):
    """
    Identify the hardest intersections aggregated at the team level for each person and overall consensus.

    Args:
        texts (dict): Dictionary mapping persons to their games.
        prompts (pd.DataFrame): DataFrame containing prompt data.
        mode (str): The mode of analysis, either "team" or "stat". Default is "team".

    Returns:
        pd.DataFrame: A DataFrame with rows for each team and columns for each person and consensus difficulty.
    """

    prompts_clean = prompts.copy()
    prompts_clean.columns = ["grid_id", "00", "01", "02", "10", "11", "12", "20", "21", "22"]

    # Initialize data structure for hardest intersections
    hardest_intersections = {person: {team: [0, 0] for team in TEAM_LIST} for person in texts}

    for person, games in texts.items():
        for game in games:
            grid_id = game.grid_number
            prompt_rows = prompts_clean[prompts_clean["grid_id"] == grid_id]
            if len(prompt_rows) != 1:
                continue
            prompt_row = prompt_rows.iloc[0][1:]

            matrix = game.matrix
            for i in range(3):
                for j in range(3):
                    part_one, part_two = get_categories_from_prompt(prompt_row[f"{i}{j}"])

                    if mode == "team":
                        # Handle team-to-team intersections
                        if category_is_team(part_one) and category_is_team(part_two):
                            team_one = get_team_from_category(part_one)
                            team_two = get_team_from_category(part_two)
                            if matrix[i][j]:
                                hardest_intersections[person][team_one][0] += 1
                                hardest_intersections[person][team_two][0] += 1
                            hardest_intersections[person][team_one][1] += 1
                            hardest_intersections[person][team_two][1] += 1

                    elif mode == "stat":
                        # Handle team-to-stat intersections
                        if category_is_team(part_one) and not category_is_team(part_two):
                            team = get_team_from_category(part_one)
                        elif not category_is_team(part_one) and category_is_team(part_two):
                            team = get_team_from_category(part_two)
                        else:
                            continue

                        if matrix[i][j]:
                            hardest_intersections[person][team][0] += 1
                        hardest_intersections[person][team][1] += 1

    # Build results into a DataFrame
    rows = []
    for team in TEAM_LIST:
        row = {"Team": team}
        correct_total = [0, 0]

        for person in texts:
            correct, total = hardest_intersections[person].get(team, [0, 0])
            accuracy = round(100 * correct / total, 1) if total > 0 else 0.0
            row[person] = f"{accuracy}%"
            correct_total[0] += correct
            correct_total[1] += total

        # Calculate consensus accuracy
        consensus_accuracy = round(100 * correct_total[0] / correct_total[1], 1) if correct_total[1] > 0 else 0.0
        row["Consensus"] = f"{consensus_accuracy}%"
        rows.append(row)

    # Create and return the DataFrame
    df = pd.DataFrame(rows)
    df.sort_values(by="Consensus", ascending=True, inplace=True)  # Sort by consensus difficulty
    return df


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
                f"{i + 1}. {pair_text:<{max_length}} {value['successes']} of {value['attempts']} ({to_percent(pct, 2)})",
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


def analyze_person_to_category(texts, prompts, categories, threshold=25):
    """
    Convert the person-to-category performance dictionary into a formatted string.

    Args:
        person_to_category (dict): Performance data for each person and category.
        threshold (int): Minimum attempts required for a category to be included.

    Returns:
        str: Formatted string summarizing performance metrics for each person and category.
    """
    # Create a dictionary to hold table data and consensus calculation data
    table_data = {}
    consensus_data = {}

    person_to_category = build_person_category_structure(texts, prompts, categories)

    # Process each person's performance data
    for person, category_data in person_to_category.items():
        for category, (correct, total) in category_data.items():
            # Only include categories that meet the attempt threshold
            if total > threshold:
                accuracy = (correct / total) * 100  # Calculate accuracy as percentage
                value = f"{round(accuracy, 1)}% ({total})"  # Format value
                if category not in table_data:
                    table_data[category] = {}
                    consensus_data[category] = {"weighted_sum": 0, "total_weight": 0}
                
                # Add accuracy data to table
                table_data[category][person] = value

                # Update consensus data
                consensus_data[category]["weighted_sum"] += accuracy * total
                consensus_data[category]["total_weight"] += total

    # Compute consensus for each category
    for category in consensus_data:
        total_weight = consensus_data[category]["total_weight"]
        weighted_sum = consensus_data[category]["weighted_sum"]
        consensus = weighted_sum / total_weight if total_weight > 0 else 0
        table_data[category]["Consensus"] = f"{round(consensus, 1)}%"

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(table_data).T  # Transpose to make categories the rows
    df.index.name = "Category"  # Name the index
    df.reset_index(inplace=True)  # Convert the index into a regular column
    df.fillna("N/A", inplace=True)  # Replace missing values with "N/A"

    # Sort the DataFrame by the Consensus column
    df["Consensus"] = df["Consensus"].str.rstrip('%').astype(float)  # Convert to numeric
    df.sort_values(by="Consensus", ascending=False, inplace=True)
    df['Consensus'] = df.apply(lambda x: str(x['Consensus']) + '%', axis=1)

    # Convert the DataFrame to a string with tabular formatting
    return df


def get_category_for_each_player_from_responses(image_metadata, prompts, grid_min=0, grid_max=99999):
    results = []

    grid_cell_image_view = create_grid_cell_image_view(image_metadata)

    for _, entry in grid_cell_image_view.iterrows():
        grid_number = int(entry['grid_number'])
        if grid_number >= grid_min and grid_number <= grid_max:
            submitter = entry['submitter']
            position = entry['position']
            player = entry['response']
            prompt_data = prompts[prompts['grid_id'] == grid_number][position]
            if prompt_data is not None and len(prompt_data) > 0:
                categories = get_categories_from_prompt(prompt_data.iloc[0])
                for category in categories:
                    results.append(
                        {
                            "submitter": submitter,
                            "player": player,
                            "category": category,
                            "grid_number": grid_number  # Keep the grid number in the results
                        }
                    )

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Aggregate by submitter, player, and category
    aggregated_results = (
        results_df.groupby(['submitter', 'player', 'category'])
        .agg(
            count=('grid_number', 'size'),  # Count occurrences
            list_of_grids=('grid_number', lambda x: list(x.unique()))  # Collect unique grid numbers
        )
        .reset_index()
    )

    return aggregated_results.sort_values(by='count', ascending=False)


def get_teams_for_each_player_from_responses(image_metadata, prompts, grid_min=0, grid_max=99999):

    df = get_category_for_each_player_from_responses(
        image_metadata, 
        prompts, 
        grid_min, 
        grid_max)
    
    filtered_df = df[df.apply(lambda row: category_is_team(row['category']), axis=1)]

    filtered_df.rename(columns={'category':'team'}, inplace=True)

    return filtered_df


def get_supercategory_for_each_player_from_responses(image_metadata, prompts, grid_min=0, grid_max=99999):
    df = get_category_for_each_player_from_responses(
        image_metadata, 
        prompts, 
        grid_min, 
        grid_max)
    
    filtered_df = df[df.apply(lambda row: not category_is_team(row['category']), axis=1)]

    filtered_df['supercategory'] = filtered_df.apply(lambda row: get_supercategory(row['category']), axis=1)

    # Aggregate by submitter, player, and category
    aggregated_results = (
        filtered_df.groupby(['submitter', 'player', 'supercategory'])
        .agg(
            count=('count', 'sum'),  # Count occurrences
            list_of_grids=('list_of_grids', lambda x: list(set(item for sublist in x for item in sublist)))  # Collect unique grid numbers
        )
        .reset_index()
    )

    return aggregated_results.sort_values(by='count', ascending=False)


def get_players_by_position_from_responses(image_metadata, prompts, grid_min=0, grid_max=99999):
    df = get_category_for_each_player_from_responses(
        image_metadata, 
        prompts, 
        grid_min, 
        grid_max)
    
    filtered_df = df[df.apply(lambda row: not category_is_team(row['category']), axis=1)]

    filtered_df = filtered_df[filtered_df.apply(lambda row: (get_supercategory(row['category']) == "Position") and (row['category'] != "Gold Glove"), axis=1)]

    filtered_df.rename(columns={'category':'position'}, inplace=True)

    filtered_df['position'] = filtered_df.apply(lambda row: row['position'].replace('Played', '').replace('min. 1 game', '').strip(), axis=1)

    return filtered_df

def get_favorite_player_by_attribute(image_metadata, prompts, attribute='team', grid_min=0, grid_max=99999):
    # Call the function to get aggregated results
    if attribute == 'team':
        aggregated_results = get_teams_for_each_player_from_responses(image_metadata, prompts, grid_min, grid_max)
    elif attribute == 'supercategory':
        aggregated_results = get_supercategory_for_each_player_from_responses(image_metadata, prompts, grid_min, grid_max)
    elif attribute == 'position': 
        aggregated_results = get_players_by_position_from_responses(image_metadata, prompts, grid_min, grid_max)
    else:
        return pd.DataFrame()

    # Calculate total counts by submitter and attribute
    total_counts = (
        aggregated_results.groupby(['submitter', attribute])['count']
        .sum()
        .reset_index()
        .rename(columns={'count': 'total_count'})
    )

    # Merge total counts back into the aggregated results
    aggregated_results = pd.merge(aggregated_results, total_counts, on=['submitter', attribute])

    # Determine the favorite player by attribute and submitter
    favorite_players = (
        aggregated_results
        .sort_values([attribute, 'submitter', 'count'], ascending=[True, True, False])
        .drop_duplicates([attribute, 'submitter'], keep='first')  # Keep the highest count for each attribute/submitter combo
    )

    # Add a column combining player name and counts (X of Y) for display
    favorite_players['player_with_count'] = (
        favorite_players['player'] + 
        "\n(" + 
        favorite_players['count'].astype(str) + 
        " of " + 
        favorite_players['total_count'].astype(str) + 
        ")"
    )

    # Pivot to create the desired cross-tab
    crosstab = favorite_players.pivot(index=attribute, columns='submitter', values='player_with_count')

    # Reset index to make 'attribute' a column
    crosstab = crosstab.reset_index()

    return crosstab


def get_players_used_for_most_teams(image_metadata, prompts, cutoff=25, grid_min=0, grid_max=99999):
    # Call the function to get aggregated results
    aggregated_results = get_teams_for_each_player_from_responses(image_metadata, prompts, grid_min, grid_max)

    # Group by player and team to get distinct team count per player
    player_team_data = (
        aggregated_results[['player', 'team']]
        .drop_duplicates()  # Ensure each player-team pair is unique
        .groupby('player')
        .agg({
            'team': ['count', lambda x: ', '.join(sorted(TEAM_LIST[get_team_name_without_city(team)] for team in x.unique()))]
        })
        .reset_index()
    )
    
    # Rename columns for readability
    player_team_data.columns = ['Player', 'TeamCount', 'Teams']

    # Sort by the number of distinct teams in descending order
    player_team_data = player_team_data.sort_values(by='TeamCount', ascending=False)

    # Apply the cutoff
    player_team_data = player_team_data.head(cutoff)

    # # Generate the formatted text output
    # output = "\n".join(
    #     f"{row['Player']} {row['TeamCount']} {row['Teams']}"
    #     for _, row in player_team_data.iterrows()
    # )

    return player_team_data
               

def grid_numbers_with_matrix_image_nonmatches(texts, image_metadata, person):
    """
    Identify grid numbers where there is a mismatch between the performance matrix 
    and image metadata responses for a specific person.

    Args:
        texts (dict): Parsed text data.
        image_metadata (dict): Image metadata for grids.
        person (str): The person whose grids are being analyzed.

    Returns:
        list: List of grid numbers with mismatches.
    """
    image_data_structure = build_results_image_structure(texts, image_metadata)
    grid_numbers = []

    for grid_number, results in image_data_structure.get(person, {}).items():
        image_metadata_day = results.get('image_metadata')

        # Skip grids with missing image metadata
        if image_metadata_day is None:
            continue

        # Calculate the identification rate using the helper function
        performance_matrix = results.get('performance', [])
        if not performance_matrix:
            continue

        id_rate = get_image_cell_id_rate(performance_matrix, image_metadata_day)

        # If identification rate is less than 100%, there's a mismatch
        if id_rate < 1.0:
            grid_numbers.append(grid_number)

    return grid_numbers


def get_image_cell_id_rate(matrix, image_metadata_day):
    """
    Get the image cell ID rate for a given person-grid-day
    """
    image_metadata_responses_list = list(image_metadata_day['responses'].values())
    correct_identification = 0

    # Count the number of "true" responses with a nonempty string in image cell position
    for i, value in enumerate(matrix):
        if value:
            if image_metadata_responses_list[i] != '':
                correct_identification += 1

    # Get number of True values in matrix
    potential_cells_to_identify_in_image = sum(matrix)

    return correct_identification / potential_cells_to_identify_in_image if potential_cells_to_identify_in_image > 0 else 0


def analyze_image_data_coverage(texts, image_metadata, image_parser_data):
    """
    Analyze the coverage of image metadata for text results.

    Text results are the performance data from the text messages
    Image results are the performance data from the image metadata
    Avg. Accuaracy of Results is the average accuracy of the text results that have image metadata in the right slots
    """
    image_data_structure = build_results_image_structure(texts, image_metadata)

    image_parser_data_clean = clean_image_parser_data(image_parser_data)

    results = []

    for person, data in image_data_structure.items():
        count_of_text_results = 0
        count_of_image_results = 0
        image_cell_id_rates = []

        for _, row in data.items():
            count_of_text_results += 1
            if row['image_metadata'] is not None:
                count_of_image_results += 1
                image_cell_id_rates.append(get_image_cell_id_rate(row['performance'], row['image_metadata']))

        average_image_cell_id_rate_unweighted = sum(image_cell_id_rates) / len(image_cell_id_rates) if len(image_cell_id_rates) > 0 else 0

        results.append({
            'Person': person,
            'Text Results': count_of_text_results,
            'Parsed Image Results': count_of_image_results,
            'Avg. Accuracy of Results': average_image_cell_id_rate_unweighted,
        })

    # Convert results to a DataFrame for better formatting
    df = pd.DataFrame(results).sort_values(by='Avg. Accuracy of Results', ascending=False)

    # Summarize image_parser_data_clean by person and parser_message_clean
    parser_data_aggregated = image_parser_data_clean.groupby(['submitter', 'clean_parser_message']).size().reset_index(name='count')

    # Sort by person and count in descending order
    parser_data_aggregated = parser_data_aggregated.sort_values(by=['submitter', 'count'], ascending=[True, False])

    # Drop rows where clean_parser_message is "Invalid image"
    #parser_data_aggregated = parser_data_aggregated[parser_data_aggregated['clean_parser_message'] != "Invalid image"]

    # Output the DataFrame to a StringIO buffer for a human-readable format
    output = StringIO()
    print("Image Data Coverage", file=output)
    df.to_string(buf=output, index=False)

    # Append the parser_data_aggregated to the buffer output
    print("\n", file=output)

    # iterate through names and print each name's parser_data_aggregated
    for name in parser_data_aggregated['submitter'].unique():
        print(f"\nParser results for {name}", file=output)
        person_specifc_aggregated = parser_data_aggregated[parser_data_aggregated['submitter'] == name]
        person_specifc_aggregated.to_string(buf=output, index=False)
        print("\n", file=output)

    return output.getvalue()


def analyze_top_players_by_month(image_metadata, cutoff):
    """
    Analyze consensus top players by month and return a formatted string with consistent spacing.

    Args:
        image_metadata (pd.DataFrame): DataFrame containing image metadata.
        cutoff (int): The number of top players to analyze for each month.

    Returns:
        str: A formatted string summarizing consensus top players for each month.
    """
    # Map grid numbers to months
    grid_month_mapping = [
        (entry['grid_number'], entry['date'].strftime('%Y-%m')) 
        for _, entry in image_metadata.iterrows()
    ]
    grid_month_dict = {}
    for grid_number, month in grid_month_mapping:
        if month not in grid_month_dict:
            grid_month_dict[month] = []
        grid_month_dict[month].append(grid_number)
    
    result = StringIO()

    # For each month, get the list of grids and compute consensus top players
    for month in dict(sorted(grid_month_dict.items())):
        grid_numbers = grid_month_dict[month]

        # Aggregate player frequencies across all submitters for the given grids
        player_frequency = {}
        filtered_metadata = image_metadata[image_metadata['grid_number'].isin(grid_numbers)]
        for _, row in filtered_metadata.iterrows():
            responses = row['responses'].values()
            for response in responses:
                if response != '':
                    player_frequency[response] = player_frequency.get(response, 0) + 1

        # Create a sorted list of top players for the month
        sorted_players = sorted(player_frequency.items(), key=lambda x: x[1], reverse=True)[:cutoff]

        # Add month header
        result.write(f"Month: {month}\n")
        result.write(f"{'Rank':<6}{'Player':<20}{'Frequency':<10}\n")  # Header with spacing
        result.write("-" * 40 + "\n")  # Separator

        # Add rows with consistent spacing
        for rank, (player, frequency) in enumerate(sorted_players, start=1):
            result.write(f"{rank:<6}{player:<20}{frequency:<10}\n")

        result.write("\n")

    return result.getvalue()


def analyze_top_players_by_submitter(image_metadata, cutoff, grid_number_list=None):
    """
    Analyze the top players by frequency for each submitter, outputting a DataFrame.

    Args:
        image_metadata (pd.DataFrame): DataFrame containing image metadata.
        cutoff (int): The number of top players to analyze.
        grid_number_list (list, optional): List of grid numbers to filter the data.

    Returns:
        pd.DataFrame: A DataFrame with columns for Rank and each submitter, where values are
                      the player at rank with the count in parentheses.
    """
    # Filter image metadata based on grid numbers if provided
    if grid_number_list is not None:
        image_metadata_preprocessed = image_metadata[image_metadata['grid_number'].isin(grid_number_list)]
    else:
        image_metadata_preprocessed = image_metadata

    # Initialize player frequency dictionary by submitter
    submitter_to_player_frequency = {}

    # Populate player frequency data for each submitter
    for _, row in image_metadata_preprocessed.iterrows():
        submitter_name = row['submitter']
        responses = row['responses'].values()
        if submitter_name not in submitter_to_player_frequency:
            submitter_to_player_frequency[submitter_name] = {}
        for response in responses:
            if response != '':
                submitter_to_player_frequency[submitter_name][response] = (
                    submitter_to_player_frequency[submitter_name].get(response, 0) + 1
                )

    # Create DataFrame for the top players
    data = []
    for submitter, player_freq in submitter_to_player_frequency.items():
        sorted_players = sorted(player_freq.items(), key=lambda x: x[1], reverse=True)[:cutoff]
        for rank, (player, freq) in enumerate(sorted_players, start=1):
            if len(data) < rank:
                data.append({"Rank": rank})
            data[rank - 1][submitter] = f"{player} ({freq})"

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    # Ensure all columns are present even if some submitters have fewer players
    for submitter in submitter_to_player_frequency:
        if submitter not in df.columns:
            df[submitter] = None

    # Sort columns so "Rank" is first
    df = df[["Rank"] + [col for col in df.columns if col != "Rank"]]

    return df


def analyze_submitter_specific_players(image_metadata, top_n=50, specificity_weight=0.9, usage_weight=0.01):
    """
    Analyze submitter-specific players and return a DataFrame with rank and players,
    including specificity scores, usage counts, and a composite score for ranking.

    Args:
        image_metadata (pd.DataFrame): DataFrame containing image metadata.
        top_n (int): Number of top players to analyze per submitter. Default is 50.
        specificity_weight (float): Weight for specificity in the composite score. Default is 0.9.
        usage_weight (float): Weight for usage count in the composite score. Default is 0.3.

    Returns:
        pd.DataFrame: A DataFrame with columns for Rank and each submitter, where values
                      are the player with specificity score, usage count, and composite score.
    """
    # Initialize player frequency dictionaries
    submitter_to_player_frequency = {}
    total_player_frequency = {}

    # Populate player frequency data for each submitter
    for _, row in image_metadata.iterrows():
        submitter_name = row['submitter']
        responses = row['responses'].values()
        if submitter_name not in submitter_to_player_frequency:
            submitter_to_player_frequency[submitter_name] = {}
        for response in responses:
            if response != '':
                submitter_to_player_frequency[submitter_name][response] = (
                    submitter_to_player_frequency[submitter_name].get(response, 0) + 1
                )
                total_player_frequency[response] = total_player_frequency.get(response, 0) + 1

    # Prepare the rank data
    rank_data = []
    for submitter, player_freq in submitter_to_player_frequency.items():
        # Calculate composite scores and sort players
        sorted_players = sorted(
            player_freq.items(),
            key=lambda x: (
                specificity_weight * (x[1] / total_player_frequency[x[0]]) + 
                usage_weight * x[1]
            ),
            reverse=True
        )[:top_n]

        for rank, (player, freq) in enumerate(sorted_players, start=1):
            total_freq = total_player_frequency[player]
            specificity_score = round((freq / total_freq) * 100, 2)  # Specificity as percentage
            composite_score = round(
                specificity_weight * (freq / total_freq) + usage_weight * freq,
                2
            )  # Composite score
            if len(rank_data) < rank:
                rank_data.append({"Rank": rank})
            rank_data[rank - 1][submitter] = (
                f"{player}\n(Spec: {specificity_score}%, Freq: {freq}, Score: {composite_score})"
            )

    # Convert the rank data to a DataFrame
    df = pd.DataFrame(rank_data)

    # Ensure all submitters are represented as columns
    for submitter in submitter_to_player_frequency.keys():
        if submitter not in df.columns:
            df[submitter] = None

    # Sort columns so "Rank" is first
    df = df[["Rank"] + [col for col in df.columns if col != "Rank"]]

    return df



def analyze_grid_cell_with_shared_guesses(image_metadata, prompts, player_list):
    """
    Analyze the grid cells and apply string analysis to each grouped response.

    Args:
        image_metadata: Input metadata for the grid cells.

    Returns:
        DataFrame: A DataFrame with analyzed responses and submitters for each grid cell.
    """

    # Step 0: Subset image_metadata on relevant submitters
    image_metadata_relevant = image_metadata[image_metadata['submitter'].isin(player_list)]

    # Step 1: Create the grid cell image view
    grid_view = create_grid_cell_image_view(image_metadata_relevant)

    if grid_view is None:
        return None

    # Step 2: Group by 'grid_number' and 'position', aggregate lists of responses and submitters
    grouped = grid_view.groupby(['grid_number', 'position']).agg({
        "response": list,
        "submitter": list  # Collect submitters for each cell
    }).reset_index()

    # Step 3: Normalize and map responses to submitters
    def normalize_response(response):
        """
        Normalize a response string by removing special characters, extra spaces, and converting to lowercase.
        """
        return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', response.strip().lower()))

    def build_analyzed_response(responses, submitters):
        """
        Build the analyzed response with normalized responses, counts, original responses, and submitters.
        """
        # Normalize responses and track their original values and submitters
        response_mapping = defaultdict(list)
        for resp, submitter in zip(responses, submitters):
            norm_resp = normalize_response(resp)
            response_mapping[norm_resp].append((resp, submitter))

        # Build the analyzed response
        analyzed = defaultdict(lambda: {"count": 0, "original_responses": [], "submitters": []})
        for norm_resp, details in response_mapping.items():
            # Count occurrences and collect original responses and submitters
            analyzed[norm_resp]["count"] = len(details)
            analyzed[norm_resp]["original_responses"] = list({resp for resp, _ in details})  # Unique original responses
            analyzed[norm_resp]["submitters"] = list({submitter for _, submitter in details})  # Unique submitters

        return dict(analyzed)

    grouped["analyzed_response"] = grouped.apply(
        lambda row: build_analyzed_response(row["response"], row["submitter"]), axis=1
    )

    # Step 4: Filter rows where at least one player is repeated
    def has_repeated_players(analyzed_response, n_repeats):
        return any(details["count"] > n_repeats for details in analyzed_response.values())

    filtered_rows = grouped[grouped["analyzed_response"].apply(
        lambda x: has_repeated_players(x, 2)
    )]

    # Step 5: Extract prompt by grid_number and position
    def get_prompt(prompts, grid_number, position):
        """
        Get the prompt value for a specific grid_number and position.

        Args:
            prompts (DataFrame): The DataFrame containing prompt data.
            grid_number (int): The grid number to filter on.
            position (str): The position in the grid.

        Returns:
            The prompt value for the given grid_number and position, or None if not found.
        """

        # Filter the DataFrame for the given grid number
        prompt_for_grid = prompts[prompts['grid_id'] == grid_number]

        # Ensure the position exists as a column
        if position in prompt_for_grid.columns:
            value = prompt_for_grid[position].iloc[0] if not prompt_for_grid.empty else None
        else:
            value = None  # Handle cases where the position column doesn't exist

        return value

    # Apply the function to add the 'prompt' column
    filtered_rows["prompt"] = filtered_rows.apply(
        lambda row: get_prompt(prompts, row["grid_number"], row["position"]), axis=1
    )

    def generate_analysis_dataframe(filtered_rows, player_list):
        """
        Generates a DataFrame summarizing the analysis of the filtered rows.
        
        Args:
            filtered_rows (DataFrame): Input DataFrame containing rows to be analyzed.
        
        Returns:
            DataFrame: A DataFrame with the columns:
                - grid_number
                - position
                - prompt
                - player_in_question
                - verdict (ban or save)
                - saved_by_person (if verdict == save)
        """
        result_data = []

        image_processor = ImageProcessor(APPLE_TEXTS_DB_PATH, IMAGES_METADATA_PATH, IMAGES_PATH)

        # Hardcoded value for ban_enactment_grid_number, when bans were enacted
        ban_enactment_grid_number = 574

        for _, row in filtered_rows.iterrows():
            grid_number = row['grid_number']
            position = row['position']
            prompt = row['prompt']

            for key, value in row['analyzed_response'].items():
                player = key.title()  # Capitalize the player's name
                count = value['count']
                
                if count < 3:
                    continue  # Skip players with less than 3 mentions
                
                if count >= 4 and grid_number >= ban_enactment_grid_number:
                    # Banned player
                    result_data.append({
                        "grid_number": grid_number,
                        "position": position,
                        "prompt": prompt,
                        "player": player,
                        "verdict": "ban",
                        "saved_by": None,
                        "reason": None
                    })
                elif count >= 4 and grid_number < ban_enactment_grid_number:
                    # Banned player before bans were enacted
                    result_data.append({
                        "grid_number": grid_number,
                        "position": position,
                        "prompt": prompt,
                        "player": player,
                        "verdict": "---",
                        "saved_by": None,
                        "reason": None
                    })
                elif count == 3:
                    # Saved player
                    excluded_submitters = [x for x in player_list if x not in value['submitters']]
                    remaining_submitter = excluded_submitters[0]
                    reason = image_processor.get_save_reason(remaining_submitter, grid_number, position)
                    result_data.append({
                        "grid_number": grid_number,
                        "position": position,
                        "prompt": prompt,
                        "player": player,
                        "verdict": "save",
                        "saved_by": ", ".join(excluded_submitters),  # Join the savers into a single string
                        "reason": reason
                    })

        # Create a DataFrame from the result data
        result_df = pd.DataFrame(result_data, columns=[
            "grid_number", "prompt", "player", "verdict", "saved_by", "reason"
        ])

        result_df = result_df.sort_values(by=['verdict', 'grid_number'])
        result_df.rename(columns={'grid_number':'grid'}, inplace=True)

        return result_df
    
    result_df = generate_analysis_dataframe(filtered_rows, player_list)

    return result_df


# - Longest immaculate day streaks 
def analyze_immaculate_streaks(texts):
    # Assume make_texts_melted(texts) returns a DataFrame with at least columns: 'name', 'date', 'correct'
    df = make_texts_melted(texts)
    df['date'] = pd.to_datetime(df['date'])
    
    # Only keep necessary columns and sort by name and date
    df = df[['name', 'date', 'correct']].sort_values(['name', 'date'])
    
    # Flag immaculate entries
    df['is_immaculate'] = df['correct'] == 9
    
    streaks = []
    
    # Process by each player
    for name, group in df.groupby('name'):
        # Only consider immaculate rows for streak calculation
        immaculate_group = group[group['is_immaculate']].reset_index(drop=True)
        if immaculate_group.empty:
            continue
        
        # Calculate the gap in days between consecutive immaculate entries
        immaculate_group['gap'] = immaculate_group['date'].diff().dt.days.fillna(0).astype(int)
        # A new streak starts if the gap is not exactly 1 day
        immaculate_group['new_streak'] = immaculate_group['gap'] != 1
        # Cumulatively sum to get a unique streak id for each streak
        immaculate_group['streak_id'] = immaculate_group['new_streak'].cumsum()
        
        # Process each streak
        for _, streak_group in immaculate_group.groupby('streak_id'):
            if len(streak_group) < 2:
                continue  # Skip 1-day streaks
            streaks.append({
                'name': name,
                'start_of_streak': streak_group['date'].iloc[0].strftime('%Y-%m-%d'),
                'end_of_streak': streak_group['date'].iloc[-1].strftime('%Y-%m-%d'),
                'length': len(streak_group)
            })
    
    # Compile the results into a DataFrame and sort by streak length (descending)
    streaks_df = pd.DataFrame(streaks)
    if not streaks_df.empty:
        streaks_df = streaks_df.sort_values(by=['length','start_of_streak'], ascending=False).head(25).reset_index(drop=True)
    
    return streaks_df

def analyze_splits(image_metadata, prompts, submitters):
    """
    Given a DataFrame of image metadata, a DataFrame of prompts, and a list of submitters,
    returns a table (as a DataFrame) with the grid number, response position, prompt text, 
    answer_1, answer_2, submitters who gave answer_1, and submitters who gave answer_2.
    
    A split is defined as an even split of responses among the provided submitters 
    (e.g., if there are 4 submitters, exactly 2 provided one answer and 2 provided the other).
    
    Assumes:
        - image_metadata has columns: grid_number, submitter, date, responses, image_filename.
          The 'responses' column contains a dict mapping positions (e.g., "top_left") to a string.
        - prompts has a column 'grid_id' (matching grid_number) and columns for each response position
          (e.g., "top_left", "bottom_right") containing prompt text.
        - submitters is a list of names.
    
    Returns:
        pd.DataFrame: with columns:
            - grid_number
            - position
            - prompt
            - answer_1
            - answer_2
            - submitters_answer_1 (list)
            - submitters_answer_2 (list)
    """
    results = []
    submitter_set = set(submitters)
    n_submitters = len(submitter_set)
    
    # For an even split, assume n_submitters is even.
    half_count = n_submitters // 2
    
    # Define the positions to check.
    positions = [
        "top_left", "top_center", "top_right",
        "middle_left", "middle_center", "middle_right",
        "bottom_left", "bottom_center", "bottom_right"
    ]
    
    # Group the image metadata by grid_number.
    for grid_number, group in image_metadata.groupby('grid_number'):
        # Filter rows to only those with a submitter in our list.
        group_filtered = group[group['submitter'].isin(submitter_set)]
        
        # Only consider if exactly all the desired submitters are present.
        if set(group_filtered['submitter']) != submitter_set:
            continue
        
        # For each response position:
        for pos in positions:
            responses = {}
            # Build a dictionary: response -> list of submitters who gave that response.
            for _, row in group_filtered.iterrows():
                # Assume 'responses' is already a dict. If not, try to load it.
                resp_val = row['responses']
                if not isinstance(resp_val, dict):
                    try:
                        resp_val = json.loads(resp_val)
                    except Exception:
                        resp_val = {}
                answer = resp_val.get(pos, '').strip()
                # If any answer is blank, skip this position.
                if answer == '':
                    responses = {}
                    break
                responses.setdefault(answer, []).append(row['submitter'])
            
            # We require exactly 2 distinct answers and an even split.
            if len(responses) == 2:
                counts = [len(v) for v in responses.values()]
                if all(c == half_count for c in counts):
                    # Sort answers (arbitrarily, so answer_1 and answer_2 are consistent)
                    sorted_answers = sorted(responses.items(), key=lambda x: x[0])
                    answer_1, submitters_answer_1 = sorted_answers[0]
                    answer_2, submitters_answer_2 = sorted_answers[1]
                    
                    # Look up the prompt text for this grid and position in the prompts DataFrame.
                    prompt_text = None
                    prompt_row = prompts[prompts['grid_id'] == grid_number]
                    if not prompt_row.empty and pos in prompt_row.columns:
                        prompt_text = prompt_row.iloc[0][pos].replace(" + ","\n")
                    
                    results.append({
                        "grid_number": grid_number,
                        "position": pos.replace("_", " ").title(),
                        "prompt": prompt_text,
                        "answers": answer_1 + "\n" + answer_2,
                        "submitters_1": ",\n".join(submitters_answer_1),
                        "submitters_2": ",\n".join(submitters_answer_2)
                    })
                    
    return pd.DataFrame(results)

def analyze_everyone_missed(texts, prompts, submitters):
    """
    Given texts (raw data that will be melted into a DataFrame), a DataFrame of prompts, and a list of submitters,
    returns a list of dictionaries for each grid number and response position where every one of the provided submitters 
    "missed" that position (i.e. the corresponding entry in the 'matrix' is False).
    
    The melted texts DataFrame (produced by make_texts_melted(texts)) should have these columns:
        - grid_number
        - name           (the submitter's name)
        - correct, score, average_score_of_correct, date, matrix
    where 'matrix' is a 3x3 list-of-lists of booleans.
    
    The prompts DataFrame should have a column 'grid_id' (which corresponds to grid_number) and additional columns 
    named after the positions (e.g., "top_left", "bottom_right", etc.) containing the prompt text for that grid.
    
    Args:
        texts: Raw texts data.
        prompts (pd.DataFrame): DataFrame of prompts.
        submitters (list): List of submitter names to check.
    
    Returns:
        List[dict]: Each dict has keys:
            - "grid_number": the grid number where all specified submitters missed a particular position.
            - "position": the response position (e.g., "top_left") that was missed by all.
            - "prompt": the prompt text for that grid and position (if available).
    """
    results = []
    submitter_set = set(submitters)
    
    # Mapping from response position to indices in the 3x3 matrix.
    position_to_indices = {
        "top_left": (0, 0),
        "top_center": (0, 1),
        "top_right": (0, 2),
        "middle_left": (1, 0),
        "middle_center": (1, 1),
        "middle_right": (1, 2),
        "bottom_left": (2, 0),
        "bottom_center": (2, 1),
        "bottom_right": (2, 2)
    }
    
    # Convert texts into a melted DataFrame.
    df = make_texts_melted(texts)
    
    # Group by grid_number.
    for grid_number, group in df.groupby('grid_number'):
        # Filter rows to only those with a submitter in our list (using column "name").
        group_filtered = group[group['name'].isin(submitter_set)]
        
        # Only proceed if all desired submitters are present in this grid.
        if set(group_filtered['name']) >= submitter_set:
            for pos, (i, j) in position_to_indices.items():
                all_missed = True
                for _, row in group_filtered.iterrows():
                    matrix_val = row['matrix']
                    # Check if the value at (i, j) is False (i.e. missed).
                    if matrix_val[i][j] != False:
                        all_missed = False
                        break
                if all_missed:
                    # Look up the corresponding prompt for this grid and position.
                    prompt_text = None
                    prompt_row = prompts[prompts['grid_id'] == grid_number]
                    if not prompt_row.empty and pos in prompt_row.columns:
                        prompt_text = prompt_row.iloc[0][pos]
                    
                    results.append({
                        "grid_number": grid_number,
                        "position": pos.replace("_", " ").title(),
                        "prompt": prompt_text
                    })
                    
    return pd.DataFrame(results)


def analyze_all_used_on_same_day(image_metadata, prompts, submitters):
    """
    Finds grids (days) where the same player was used by all specified submitters (though not necessarily in the same position).
    
    Returns a DataFrame with 4 columns:
      1. grid_number
      2. player_used (the common player)
      3. submitter_positions: a newline-separated string with "submitter: pos1, pos2" for each submitter.
      4. position_prompts: a newline-separated string with "position: prompt" for each combined position.
    
    Args:
        image_metadata (pd.DataFrame): DataFrame with columns:
            - grid_number
            - submitter
            - date
            - responses: a dict (or JSON string) mapping positions (e.g., "top_left") to a string (player name)
            - image_filename
        prompts (pd.DataFrame): DataFrame with a column 'grid_id' (matching grid_number) and columns for each position
            containing the prompt text.
        submitters (list): List of submitter names to check.
    
    Returns:
        pd.DataFrame: A table with columns: grid_number, player_used, submitter_positions, position_prompts.
    """
    results = []
    submitter_set = set(submitters)
    
    # Group by grid_number.
    for grid_number, group in image_metadata.groupby('grid_number'):
        # Filter rows to only those with a submitter in our list.
        group_filtered = group[group['submitter'].isin(submitter_set)]
        # Only proceed if exactly all the desired submitters are present.
        if set(group_filtered['submitter']) != submitter_set:
            continue
        
        # Build a mapping for each submitter: {submitter: {player: [position, ...]}}
        submitter_map = {}
        for _, row in group_filtered.iterrows():
            sub = row['submitter']
            responses = row['responses']
            # For this submitter, record positions for each non-empty player.
            for pos, player in responses.items():
                player = player.strip()
                if player:
                    submitter_map.setdefault(sub, {}).setdefault(player, []).append(pos)
        
        # Find common players across all submitters.
        common_players = None
        for sub in submitters:
            players = set(submitter_map.get(sub, {}).keys())
            if common_players is None:
                common_players = players
            else:
                common_players = common_players.intersection(players)
        if not common_players:
            continue
        
        # For each common player, gather the output.
        for player in common_players:
            # Build a string for each submitter: "submitter: pos1, pos2"
            sub_pos_lines = []
            combined_positions = set()
            for sub in submitters:
                pos_list = submitter_map[sub].get(player, [])
                if pos_list:
                    combined_positions.update(pos_list)
                    sub_pos_lines.append(f"{sub}: {', '.join(sorted([pos.replace("_", " ").title() for pos in pos_list]))}")
            submitter_positions_str = "\n".join(sub_pos_lines)
            
            # For each combined position, look up the prompt.
            prompt_dict = {}
            prompt_row = prompts[prompts['grid_id'] == grid_number]
            if not prompt_row.empty:
                for pos in sorted(combined_positions):
                    if pos in prompt_row.columns:
                        prompt_dict[pos] = str(prompt_row.iloc[0][pos])
            # Format prompt_dict as "position: prompt" lines.
            position_prompts_str = "\n".join([f"{pos.replace("_", " ").title()}: {prompt_dict[pos]}" for pos in sorted(prompt_dict)])

            same = True if len(prompt_dict.keys()) == 1 else False
            
            results.append({
                "grid": grid_number,
                "player": player.replace(" ", "\n"),
                "positions": submitter_positions_str,
                "prompts": position_prompts_str,
                "same": same
            })
    
    return pd.DataFrame(results)


def analyze_illegal_uses(image_metadata, prompts, submitters):
    """
    Analyzes image_metadata for illegal uses of banned players.
    
    First, it calls analyze_grid_cell_with_shared_guesses (assumed to exist) to obtain a ban_df,
    which has a column 'verdict' (only rows with verdict=='ban' are used) and columns 'grid' and 'player',
    indicating that the player is banned starting at that grid.
    
    Then, for each banned player, this function scans image_metadata for any grid (i.e. day)
    with grid_number greater than the ban grid in which the banned player appears in the responses.
    
    Only rows where the submitter is in the provided submitters list are considered.
    
    The output is grouped by grid and banned player and returned as a DataFrame with 4 columns:
      1. grid_number
      2. player_used
      3. submitter_positions – a newline-separated string with entries like "submitter: pos1, pos2"
      4. position_prompts – a newline-separated string with entries like "position: prompt"
    
    Args:
        image_metadata (pd.DataFrame): DataFrame with columns including:
            - grid_number
            - submitter
            - responses (a dict or JSON string mapping positions to player names)
            - image_filename (and possibly other columns)
        prompts (pd.DataFrame): DataFrame with a column 'grid_id' (matching grid_number) and columns for positions.
        submitters (list): List of submitter names to consider.
    
    Returns:
        pd.DataFrame: A table with columns as described.
    """
    # Get banned players and grids using your existing function.
    ban_df = analyze_grid_cell_with_shared_guesses(image_metadata, prompts, submitters)
    # Filter to banned verdict rows; assume ban_df has columns "grid", "player", and "verdict".
    ban_df = ban_df[ban_df['verdict'] == 'ban'][['grid','player']].reset_index(drop=True)
    print("Ban DataFrame:")
    print(ban_df)
    
    illegal_dict = {}
    submitter_set = set(submitters)
    
    # Iterate over each banned player record.
    for idx, ban_row in ban_df.iterrows():
        ban_grid = ban_row['grid']
        banned_player = ban_row['player'].strip().lower()  # normalized for comparison
        
        # Look in image_metadata for any grid with grid_number > ban_grid and only include rows where submitter is in our list.
        subset = image_metadata[
            (image_metadata['grid_number'] > ban_grid) &
            (image_metadata['submitter'].isin(submitter_set))
        ]
        
        for _, row in subset.iterrows():
            responses = row['responses']
            matching_positions = []
            for pos, value in responses.items():
                if value.strip().lower() == banned_player:
                    matching_positions.append(pos)
            if matching_positions:
                key = (row['grid_number'], banned_player)
                if key not in illegal_dict:
                    illegal_dict[key] = {}
                sub = row['submitter']
                if sub not in illegal_dict[key]:
                    illegal_dict[key][sub] = set()
                illegal_dict[key][sub].update(matching_positions)
    
    # Now, compile the results into the desired output format.
    results = []
    for (grid_number, banned_player), sub_pos in illegal_dict.items():
        # Build newline-separated string: "submitter: pos1, pos2"
        submitter_positions_str = "\n".join(
            f"{sub}: {', '.join(sorted(list(positions)))}" for sub, positions in sub_pos.items()
        )
        # Get the union of all positions used.
        all_positions = set()
        for positions in sub_pos.values():
            all_positions.update(positions)
        all_positions = sorted(list(all_positions))
        
        # Look up the corresponding prompt from the prompts DataFrame.
        position_prompt_lines = []
        prompt_row = prompts[prompts['grid_id'] == grid_number]
        if not prompt_row.empty:
            for pos in all_positions:
                if pos in prompt_row.columns:
                    prompt_text = str(prompt_row.iloc[0][pos])
                    position_prompt_lines.append(f"{pos}: {prompt_text}")
        position_prompts_str = "\n".join(position_prompt_lines)
        
        results.append({
            "grid_number": grid_number,
            "player_used": banned_player.title(),
            "submitter_positions": submitter_positions_str,
            "position_prompts": position_prompts_str
        })
    
    return pd.DataFrame(results)



#--------------------------------------------------------------------------------------------------
# Plotting functions

def plot_summary_metrics(texts, color_map):
    """
    Create a 2x2 grid of subplots to visualize:
    1. Number of Immaculates
    2. Average Correct
    3. Average Score
    4. Average Rarity of Correct Square

    Args:
        texts (dict): Dictionary containing player data.
        color_map (dict): Dictionary mapping player names to colors.
    """
    # Initialize the figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing between subplots

    # Plot 1: Number of Immaculates
    counts = []
    for person in texts:
        data = [(1 if obj.correct == 9 else 0) for obj in texts[person]]
        counts.append(sum(data))
    axes[0, 0].bar([person for person in texts], counts, color=[color_map[person] for person in texts])
    axes[0, 0].set_title("Number of Immaculates")

    # Prepare data for the other plots
    texts_melted = make_texts_melted(texts)

    # Plot 2: Average Correct
    analysis_correct = texts_melted.groupby('name')['correct'].mean().reset_index()
    axes[0, 1].bar(
        analysis_correct['name'],
        analysis_correct['correct'],
        color=[color_map[person] for person in analysis_correct['name']]
    )
    axes[0, 1].set_title("Average Correct")

    # Plot 3: Average Score
    analysis_score = texts_melted.groupby('name')['score'].mean().reset_index()
    axes[1, 0].bar(
        analysis_score['name'],
        analysis_score['score'],
        color=[color_map[person] for person in analysis_score['name']]
    )
    axes[1, 0].set_title("Average Score")

    # Plot 4: Average Rarity of Correct Square
    analysis_rarity = texts_melted.groupby('name')['average_score_of_correct'].mean().reset_index()
    axes[1, 1].bar(
        analysis_rarity['name'],
        analysis_rarity['average_score_of_correct'],
        color=[color_map[person] for person in analysis_rarity['name']]
    )
    axes[1, 1].set_title("Average Rarity of Correct Square")

    # Set overall plot title
    fig.suptitle("Summary Metrics for Players", fontsize=16, fontweight='bold')
    
    # Show the plots
    plt.show()

    
# Graph distributions
def plot_correctness(texts, color_map):

    # Calculate the maximum value across all distributions
    max_height = 0
    for person in texts:
        distribution = [0 for _ in range(10)]
        for row in texts[person]:
            distribution[row.correct] += 1
        max_height = max(max_height, max(distribution))  # Update global max height

    HEIGHT = int(max_height * 1.2)  # Dynamically set height with some padding

    print("Max height:", HEIGHT)

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))  # Create a 3x2 grid for 5 plots
    
    # Flatten the axes array for easier indexing
    axs = axs.flatten()
    
    for i, person in enumerate(texts):
        distribution = [0 for _ in range(0, 10)]
        for row in texts[person]:
            distribution[row.correct] += 1
        
        # Plotting the distribution for each person
        axs[i].bar(range(0, 10), distribution, color=color_map[person])
        axs[i].set_xticks(range(0, 10))
        axs[i].set_title(person)
        axs[i].set_ylim(0, HEIGHT)

        print(f"{person} subplot ylim: {axs[i].get_ylim()}")

    
    # Hide the last subplot if it is not used
    if len(texts) < 6:
        axs[5].set_visible(False)
    
    fig.suptitle("Correctness Distribution", fontsize=16, fontweight='bold')
    plt.subplots_adjust(hspace=0.5)
    plt.show()


# Function to plot smoothed metrics using the smoothed DataFrame
def plot_smoothed_metrics(texts, metric, title, ylabel, color_map):
    """Plot the smoothed metrics (score, correct, or average score) over time."""
    smoothed_df = calculate_smoothed_metrics(texts, smoothness=28)
    
    plt.figure(figsize=(12, 6))

    # Plot smoothed metrics for each person
    for name in smoothed_df['name'].unique():
        person_data = smoothed_df[smoothed_df['name'] == name]
        
        # Plot line with proper date formatting for the selected metric
        plt.plot(person_data['date'], person_data[metric], label=name, color=color_map.get(name, 'blue'))

    # Formatting the plot
    plt.legend()
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)

    # Adjust x-axis date formatting and tick placement
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))  # Format to month and year
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))  # Limit number of x-ticks to avoid clutter

    plt.tight_layout()  # Adjust layout for better display
    plt.show()


def plot_win_rates(texts, color_map):
    """Plot win rates based on various criteria."""
    
    # Set a larger figure size to widen the graphs
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    criteria = ["overall", "correctness", "scores", "last_rate"]
    titles = ["Win Rates (Overall)", "Win Rates (Correctness Only)", "Win Rates (Scores Only)", "Last Rate (Overall)"]

    for ax, criterion, title in zip(axs.flat, criteria, titles):
        wins = calculate_win_rates(texts, criterion)
        ax.bar([person for person in wins], wins.values(), color=[color_map[person] for person in wins])
        ax.set_title(title)
        ax.set_yticks([i / 5 for i in range(6)])
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(FuncFormatter(to_percent))

    # Adjust the layout of the subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.suptitle("Win Rates", fontsize=16, fontweight='bold')
    plt.show()


def get_top_n_scores(texts, direction, cutoff):
    score_records = []
    # Prepare score records
    for person, games in texts.items():
        for game in games:
            grid_id = game.grid_number
            score_records.append((person, game.score, game.date, grid_id))
    
    # Sort records by score
    sorted_records = sorted(score_records, key=lambda x: x[1])
    
    # Extract best and worst scores
    if direction == "best":
        return sorted_records[:cutoff]
    else:
        return sorted_records[-cutoff:][::-1]
    

def plot_top_n_grids(image_metadata, texts, cutoff):
    """
    Plots the top N grids in a single Matplotlib figure with images arranged in a grid.

    Args:
        image_metadata (DataFrame): DataFrame containing grid_number, submitter, and image path.
        texts: Text data used to determine top grids.
        cutoff (int): Number of top grids to process.
    """

    # Create a summary page with results
    best_scores = get_top_n_scores(texts, "best", cutoff)

    # Determine the grid layout
    num_images = len(best_scores)
    grid_cols = 2  # Number of columns
    grid_rows = (num_images + grid_cols - 1) // grid_cols  # Calculate rows dynamically

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, 5 * grid_rows))
    fig.suptitle('Top Grids of All Time', fontsize=25, fontweight='bold', ha='center')
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for idx, row in enumerate(best_scores):
        submitter = row[0]
        score = row[1]
        grid_number = row[3]

        # Lookup the image path for the given grid number and submitter
        image_path = None
        for _, entry in image_metadata.iterrows():
            if entry['grid_number'] == grid_number and entry['submitter'] == submitter:
                image_path = entry.get('image_filename')
                break

        # Get the current subplot axis
        ax = axes[idx]
        ax.axis('off')

        if image_path is None:
            ax.text(0.5, 0.5, f"No image found... :(\nGrid {grid_number}\nSubmitter: {submitter}",
                    fontsize=10, ha='center', va='center', color='red')
            continue

        # First see if image is valid 
        image_path = IMAGES_PATH / image_path

        if not os.path.exists(image_path):
            ax.text(0.5, 0.5, f"Image filepath does not exist\nGrid {grid_number}\nSubmitter: {submitter}",
                    fontsize=10, ha='center', va='center', color='red')
            continue
        
        try:
            img = Image.open(image_path)
        except:
            image_processor = ImageProcessor(APPLE_TEXTS_DB_PATH, IMAGES_METADATA_PATH, IMAGES_PATH)
            img_cv2 = image_processor.read_heic_with_cv2(image_path)
            img = image_processor.convert_cv2_to_pil(img_cv2)

        ax.imshow(img, aspect='equal')
        ax.set_title(f"Grid {grid_number} by {submitter} (Score: {score})", fontsize=12, pad=10)

    # Hide any unused subplots
    for ax in axes[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()



def plot_best_worst_scores(texts, fig_title='Best and Worst Scores (All Time)'):
    """
    Creates a summary page in the PDF with the best and worst scores.

    Parameters:
    - texts: Data used for creating score records.
    """

    # Extract best and worst scores
    best_records = get_top_n_scores(texts, "best", 25)
    worst_records = get_top_n_scores(texts, "worst", 25)
    
    # Create a summary page with results
    plt.figure(figsize=(8.5, 11))
    plt.text(0.5, 0.97, fig_title, fontsize=25, ha='center', va='top', fontweight='bold')
        
    # Display best scores in a structured format with dynamic spacing
    plt.text(0, 0.85, 'Best Scores:', fontsize=16, ha='left', va='top', fontweight='bold')
    plt.text(0, 0.80, 'Rank | Name        | Score  | Date       | Game ID', fontsize=10, ha='left', va='top')
    
    for i, (name, score, date, grid_id) in enumerate(best_records):
        record_text = format_record(i + 1, name, score, date, grid_id)
        plt.text(0, 0.75 - i * 0.025, record_text, fontsize=10, ha='left', va='top', fontfamily='monospace')
    
    # Worst Scores Section
    plt.text(0.6, 0.85, 'Worst Scores:', fontsize=16, ha='left', va='top', fontweight='bold')
    plt.text(0.6, 0.80, 'Rank | Name        | Score  | Date       | Game ID', fontsize=10, ha='left', va='top')
    
    # Display worst scores in a structured format with dynamic spacing
    for i, (name, score, date, grid_id) in enumerate(worst_records):
        record_text = format_record(i + 1, name, score, date, grid_id)
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



def analyze_new_players_per_month(image_metadata, color_map=None):
    """
    Tracks and plots the number of new players used per month per submitter.
    
    The image_metadata DataFrame is assumed to have the following columns:
      - 'grid_number': numeric identifier of the grid/day.
      - 'submitter': the submitter's name.
      - 'date': the date (as a datetime or a string in YYYY-MM-DD format).
      - 'responses': a dictionary (or a JSON string) mapping positions (e.g., "top_left")
                     to a player's name.
    
    For each submitter, the function records the first time they use each player.
    Then, it groups these "first uses" by month (derived from the date) and counts the number
    of new players per submitter per month.
    
    Args:
        image_metadata (pd.DataFrame): DataFrame as described.
        color_map (dict, optional): Mapping from submitter name to a matplotlib color.
        
    The function then plots the result as a time series with one line per submitter.
    """
    # Ensure the date column is datetime.
    image_metadata['date'] = pd.to_datetime(image_metadata['date'])
    
    # Sort by date (earliest first).
    df_sorted = image_metadata.sort_values('date')
    
    # Dictionary to record the first use for each (submitter, player) pair.
    # Key: (submitter, player.lower()), Value: (date, grid_number, position, original_player)
    first_use = {}
    
    for _, row in df_sorted.iterrows():
        submitter = row['submitter']
        grid_number = row['grid_number']
        date = row['date']
        responses = row['responses']
        
        for pos, player in responses.items():
            player = player.strip()
            if not player:
                continue  # Skip empty answers.
            key = (submitter, player.lower())
            if key not in first_use:
                first_use[key] = (date, grid_number, pos, player)
    
    # Convert the first use records into a DataFrame.
    rows = []
    for (submitter, _), (date, grid_number, pos, original_player) in first_use.items():
        rows.append({
            "submitter": submitter,
            "player": original_player,
            "date": date,
            "grid_number": grid_number,
            "position": pos
        })
    first_use_df = pd.DataFrame(rows)
    
    # Create a 'month' column formatted as "YYYY-MM".
    first_use_df['month'] = first_use_df['date'].dt.strftime("%Y-%m")
    
    # Group by submitter and month to count new players.
    agg_df = first_use_df.groupby(['submitter', 'month']).agg(new_players_count=('player', 'count')).reset_index()
    agg_df = agg_df.sort_values(['submitter', 'month']).reset_index(drop=True)
    
    # Convert the month string to a datetime object (using the first day of the month) for plotting.
    agg_df['month_dt'] = pd.to_datetime(agg_df['month'] + "-01")
    
    # Plotting.
    plt.figure(figsize=(12, 6))
    
    # Plot a line for each submitter.
    for name in agg_df['submitter'].unique():
        person_data = agg_df[agg_df['submitter'] == name]
        if color_map and name in color_map:
            color = color_map[name]
        else:
            color = None  # Default color.
        plt.plot(person_data['month_dt'], person_data['new_players_count'],
                 marker='o', label=name, color=color)
    
    # Format the plot.
    plt.legend()
    plt.title("[NEW] New Players Used Per Month by Submitter", fontsize=16, fontweight='bold')
    plt.xlabel("Month")
    plt.ylabel("New Players Count")
    plt.xticks(rotation=45)
    
    # Use month-year formatting on the x-axis.
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
    
    plt.tight_layout()
    plt.show()


# ---------------- New metrics ----------------
def analyze_full_week_usage(texts_df: pd.DataFrame, images_df: pd.DataFrame, person: Optional[str] = None) -> pd.DataFrame:
    """For each submitter/week (Mon-Sun by grid id), list unique players with grid ids."""
    if images_df.empty or "grid_number" not in images_df.columns:
        return pd.DataFrame()
    df = images_df.copy()
    df["grid_number"] = pd.to_numeric(df["grid_number"], errors="coerce")
    df = df.dropna(subset=["grid_number"])
    if person:
        df = df[df["submitter"] == person]
    df["week_start"] = df["grid_number"].apply(week_start_from_grid).dt.strftime("%Y-%m-%d")

    def _players(resp):
        if isinstance(resp, dict):
            return [v for v in resp.values() if isinstance(v, str) and v.strip()]
        return []

    df["players"] = df["responses"].apply(_players)
    df = df.explode("players").dropna(subset=["players"])
    def _summarize(group):
        mapping = {}
        for _, row in group.iterrows():
            player = row["players"]
            gid = int(row["grid_number"])
            mapping.setdefault(player, set()).add(gid)
        players_formatted = [
            f"{player} ({', '.join(str(g) for g in sorted(gids))})"
            for player, gids in sorted(mapping.items(), key=lambda x: x[0])
        ]
        return pd.Series(
            {
                "players_used": ", ".join(players_formatted),
                "unique_players": len(mapping),
            }
        )

    grouped = df.groupby(["submitter", "week_start"]).apply(_summarize).reset_index()
    return grouped.sort_values(by=["week_start", "submitter"], ascending=False)


def analyze_shame_index(images_df: pd.DataFrame, person: Optional[str] = None) -> pd.DataFrame:
    """Shame index: repeated use of players within a Mon-Sun week per submitter (by grid date)."""
    if images_df.empty or "grid_number" not in images_df.columns:
        return pd.DataFrame()
    df = images_df.copy()
    df["grid_number"] = pd.to_numeric(df["grid_number"], errors="coerce")
    if person:
        df = df[df["submitter"] == person]
    df = df.dropna(subset=["grid_number"])
    df["week_start"] = df["grid_number"].apply(week_start_from_grid).dt.strftime("%Y-%m-%d")

    def _players(resp):
        if isinstance(resp, dict):
            return [v for v in resp.values() if isinstance(v, str) and v.strip()]
        return []

    df["players"] = df["responses"].apply(_players)
    df = df.explode("players").dropna(subset=["players"])

    rows = []
    for (sub, wk), group in df.groupby(["submitter", "week_start"]):
        counts = group["players"].value_counts()
        shame = int(sum(max(c - 1, 0) for c in counts))
        repeated = counts[counts > 1].index.tolist()
        repeated_fmt = []
        for player in repeated:
            grids = group.loc[group["players"] == player, "grid_number"].astype(int).unique().tolist()
            repeated_fmt.append(f"{player} ({', '.join(str(g) for g in sorted(grids))})")
        rows.append(
            {
                "submitter": sub,
                "week_start": wk,
                "shame_index": shame,
                "repeated_players": ", ".join(repeated_fmt),
                "repeated_count": len(repeated_fmt),
                "total_uses": int(counts.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(by=["week_start", "submitter"], ascending=False)


def analyze_adjusted_rarity(texts_df: pd.DataFrame, images_df: pd.DataFrame) -> pd.DataFrame:
    """Adjusted rarity = avg score * (1 + shame_index / max(total_uses,1)) per submitter/week."""
    if texts_df.empty or "date" not in texts_df.columns:
        return pd.DataFrame()
    texts = texts_df.copy()
    texts["date_dt"] = pd.to_datetime(texts["date"])
    texts["week_start"] = texts["date_dt"].apply(lambda d: d - timedelta(days=d.weekday())).dt.strftime("%Y-%m-%d")
    score_stats = (
        texts.groupby(["name", "week_start"])["score"]
        .agg(avg_score="mean", grids="count")
        .reset_index()
        .rename(columns={"name": "submitter"})
    )
    shame = analyze_shame_index(images_df)
    merged = score_stats.merge(shame, on=["submitter", "week_start"], how="left").fillna({"shame_index": 0, "total_uses": 0})
    merged["shame_factor"] = 1 + (merged["shame_index"] / merged["total_uses"].replace(0, 1))
    merged["adjusted_rarity"] = merged["avg_score"] * merged["shame_factor"]
    return merged.sort_values(by=["week_start", "submitter"], ascending=False)


def analyze_low_bit_high_reward(images_df: pd.DataFrame, prompts_df: pd.DataFrame) -> pd.DataFrame:
    """Low bit high reward: favor players with many applicable sub-prompts (parsed) and many uses."""
    if images_df.empty or prompts_df.empty:
        return pd.DataFrame()
    prompts_lookup = prompts_df.set_index("grid_id")
    try:
        prompts_lookup.index = prompts_lookup.index.astype(int)
    except Exception:
        pass
    records = []

    def _flatten_parts(val):
        """Extract individual categories from a prompt value (string/tuple/list)."""
        parts = set()
        if val is None:
            return []
        candidate = val
        if isinstance(candidate, str):
            candidate = candidate.strip()
            if " + " in candidate:
                for piece in candidate.split(" + "):
                    if piece.strip():
                        parts.add(piece.strip())
            else:
                try:
                    candidate = ast.literal_eval(candidate)
                except Exception:
                    candidate = candidate
        if isinstance(candidate, (list, tuple)):
            for item in candidate:
                if isinstance(item, (list, tuple)):
                    for sub in item:
                        if isinstance(sub, str) and sub.strip():
                            parts.add(sub.strip())
                elif isinstance(item, str) and item.strip():
                    parts.add(item.strip())
        elif isinstance(candidate, str) and candidate.strip():
            parts.add(candidate.strip())

        # Fallback to the canonical parser to catch any tuple strings
        first, second = get_categories_from_prompt(val)
        for p in (first, second):
            if isinstance(p, str) and p.strip():
                parts.add(p.strip())
        return sorted(parts)

    for _, row in images_df.iterrows():
        try:
            grid = int(row.get("grid_number"))
        except Exception:
            continue
        responses = row.get("responses", {})
        if grid not in prompts_lookup.index:
            continue
        prompt_row = prompts_lookup.loc[grid]
        for pos, player in (responses or {}).items():
            if not isinstance(player, str) or not player.strip():
                continue
            prompt_val = prompt_row.get(pos)
            for part in _flatten_parts(prompt_val):
                records.append({"player": player.strip(), "prompt": part})
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["prompt"] = df["prompt"].fillna("<unknown>")
    grouped = df.groupby("player").agg(
        appearances=("prompt", "count"),
        unique_prompts=("prompt", "nunique"),
        prompts_list=("prompt", lambda x: sorted(set(x))),
    )
    grouped["ratio"] = grouped["appearances"] / grouped["unique_prompts"].replace(0, 1)
    grouped["combined_score"] = grouped["appearances"] * grouped["unique_prompts"]
    grouped["unique_prompts_list"] = grouped["prompts_list"].apply(lambda lst: ", ".join(lst))
    grouped = grouped.drop(columns=["prompts_list"])
    return grouped.reset_index().sort_values(
        by=["unique_prompts", "combined_score", "appearances"], ascending=False
    )


def analyze_novelties(images_df: pd.DataFrame) -> pd.DataFrame:
    """
    Novelties: per-player first usage dates by submitter, sorted by global first usage (recent first).
    """
    if images_df.empty or "grid_number" not in images_df.columns:
        return pd.DataFrame()

    submitters = sorted(GRID_PLAYERS.keys())
    records = {}

    def _iter_players(row):
        grid_num = row.get("grid_number")
        try:
            grid_num = int(grid_num)
        except Exception:
            return []
        usage_date = grid_to_date(grid_num).strftime("%Y-%m-%d")
        responses = row.get("responses") or {}
        out = []
        for val in responses.values():
            if isinstance(val, str) and val.strip():
                out.append((val.strip(), usage_date, row.get("submitter")))
        return out

    for _, row in images_df.iterrows():
        for player, date_str, sub in _iter_players(row):
            if player not in records:
                records[player] = {"first_global_date": date_str}
            # global first date
            if date_str < records[player].get("first_global_date", date_str):
                records[player]["first_global_date"] = date_str
            # per submitter first date
            key = f"{sub}"
            prev = records[player].get(key)
            if prev is None or date_str < prev:
                records[player][key] = date_str

    rows = []
    for player, data in records.items():
        row = {"player": player, "first_global_date": data["first_global_date"]}
        for sub in submitters:
            row[sub] = data.get(sub, "")
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(by="first_global_date", ascending=False)
    return df.reset_index(drop=True)


def analyze_prompts_most_wrong(prompts_df: pd.DataFrame, texts_df: pd.DataFrame) -> pd.DataFrame:
    """Deprecated: kept for compatibility; returns empty."""
    return pd.DataFrame()


def analyze_prediction_future_grid(prompts_df: pd.DataFrame, texts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict how each submitter will perform on the most recent grid based on past performance and prompt components.
    Uses weighted blend of overall success, component success, and prompt-type success.
    """
    if prompts_df is None or texts_df is None:
        return pd.DataFrame()
    if prompts_df.empty or texts_df.empty:
        return pd.DataFrame()

    prompts_df = prompts_df.copy()
    if "grid_id" not in prompts_df.columns:
        return pd.DataFrame()
    prompts_df["grid_id"] = pd.to_numeric(prompts_df["grid_id"], errors="coerce")
    prompts_df = prompts_df.dropna(subset=["grid_id"])
    if prompts_df.empty:
        return pd.DataFrame()
    prompts_df["grid_id"] = prompts_df["grid_id"].astype(int)
    prompts_lookup = prompts_df.set_index("grid_id")

    positions = [
        "top_left", "top_center", "top_right",
        "middle_left", "middle_center", "middle_right",
        "bottom_left", "bottom_center", "bottom_right",
    ]

    def _cell_prompt_list(row: pd.Series) -> list:
        return [row.get(pos) for pos in positions]

    # Build training records from historical attempts
    records = []

    def _prompt_components(val) -> Tuple[str, str]:
        first, second = get_categories_from_prompt(val)
        return first or "", second or ""

    def _prompt_type(first: str, second: str) -> str:
        a_is_team = category_is_team(first)
        b_is_team = category_is_team(second)
        if a_is_team and b_is_team:
            return "team-team"
        if a_is_team or b_is_team:
            return "team-stat"
        return "stat-stat"

    for _, row in texts_df.iterrows():
        try:
            gid = int(row.get("grid_number"))
        except Exception:
            continue
        if gid not in prompts_lookup.index:
            continue
        matrix_raw = row.get("matrix")
        try:
            if isinstance(matrix_raw, str):
                matrix_flat = sum(json.loads(matrix_raw), [])
            else:
                matrix_flat = sum(matrix_raw, [])
        except Exception:
            try:
                matrix_flat = [v for sub in ast.literal_eval(str(matrix_raw)) for v in sub]
            except Exception:
                continue
        cell_prompts = _cell_prompt_list(prompts_lookup.loc[gid])
        if len(matrix_flat) != len(cell_prompts):
            continue
        for correct_flag, prompt_val in zip(matrix_flat, cell_prompts):
            f, s = _prompt_components(prompt_val)
            records.append(
                {
                    "submitter": row.get("name"),
                    "correct": bool(correct_flag),
                    "first": f,
                    "second": s,
                    "ptype": _prompt_type(f, s),
                }
            )

    hist = pd.DataFrame(records)
    if hist.empty:
        return pd.DataFrame()

    overall = hist.groupby("submitter")["correct"].mean().to_dict()
    comp_success: Dict[str, Dict[str, float]] = {}
    for submitter, grp in hist.groupby("submitter"):
        comp_rates = {}
        for c in pd.concat([grp["first"], grp["second"]]):
            if not c:
                continue
            sub_grp = grp[(grp["first"] == c) | (grp["second"] == c)]
            if not sub_grp.empty:
                comp_rates[c] = sub_grp["correct"].mean()
        comp_success[submitter] = comp_rates

    type_success: Dict[str, Dict[str, float]] = {}
    for submitter, grp in hist.groupby("submitter"):
        type_success[submitter] = grp.groupby("ptype")["correct"].mean().to_dict()

    if len(prompts_lookup.index) == 0:
        return pd.DataFrame()
    target_grid = int(prompts_lookup.index.max())
    target_row = prompts_lookup.loc[target_grid]
    target_prompts = _cell_prompt_list(target_row)

    pred_rows = []
    submitters = sorted(overall.keys())
    for submitter in submitters:
        base = overall.get(submitter, hist["correct"].mean())
        for pos_label, prompt_val in zip(positions, target_prompts):
            f, s = _prompt_components(prompt_val)
            ptype = _prompt_type(f, s)
            comp_rates = []
            for comp in (f, s):
                if comp:
                    val = comp_success.get(submitter, {}).get(comp)
                    if val is not None:
                        comp_rates.append(val)
            type_rate = type_success.get(submitter, {}).get(ptype)

            weights = []
            values = []
            # overall
            weights.append(0.4)
            values.append(base)
            # components
            if comp_rates:
                weights.append(0.4)
                values.append(sum(comp_rates) / len(comp_rates))
            # type
            if type_rate is not None:
                weights.append(0.2)
                values.append(type_rate)
            pred = sum(w * v for w, v in zip(weights, values)) / sum(weights)
            comp_str = ",".join(f"{c:.2f}" for c in comp_rates) if comp_rates else ""
            type_str = f"{type_rate:.2f}" if type_rate is not None else "n/a"
            pred_rows.append(
                {
                    "grid_id": target_grid,
                    "cell": pos_label,
                    "prompt": f"{f} | {s}",
                    "submitter": submitter,
                    "predicted_success_pct": round(pred * 100, 1),
                    "basis": f"overall={base:.2f}, comps={comp_str}, type={type_str}",
                }
            )

    pred_df = pd.DataFrame(pred_rows)
    if pred_df.empty:
        return pred_df
    # Wide view: cells x submitters
    wide = pred_df.pivot_table(index=["grid_id", "cell", "prompt"], columns="submitter", values="predicted_success_pct")
    wide = wide.reset_index()
    wide.columns.name = None
    wide = wide.sort_values(by=["grid_id", "cell"])
    return wide.reset_index(drop=True)
