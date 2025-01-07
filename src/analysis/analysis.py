# analysis.py
import re
import os
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from rapidfuzz import fuzz, process
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
    clean_image_parser_data,
    create_grid_cell_image_view,
    )
from utils.constants import (
    TEAM_LIST, 
    GRID_PLAYERS_RESTRICTED, 
    IMAGES_PATH, 
    APPLE_TEXTS_DB_PATH, 
    IMAGES_METADATA_PATH
)

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

    prompts_clean = prompts.copy()
    prompts_clean.columns = ["grid_id", "00", "01", "02", "10", "11", "12", "20", "21", "22"]

    result = StringIO()
    hardest_teams = {person: {team: [0, 0] for team in TEAM_LIST} for person in texts}

    # Analyze each person's games for hardest team intersections
    for person, games in texts.items():
        for game in games:
            id = game.grid_number
            prompt_rows = prompts_clean[prompts_clean["grid_id"] == id]
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

    prompts_clean = prompts.copy()
    prompts_clean.columns = ["grid_id", "00", "01", "02", "10", "11", "12", "20", "21", "22"]

    result = StringIO()
    hardest_team_stats = {person: {team: [0, 0] for team in TEAM_LIST} for person in texts}

    # Analyze each person's games for hardest team-to-stat intersections
    for person, games in texts.items():
        for game in games:
            id = game.grid_number
            prompt_rows = prompts_clean[prompts_clean["grid_id"] == id]
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


def analyze_person_to_category(person_to_category, threshold=25):
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

    # Process each person's performance data
    for person, category_data in person_to_category.items():
        for category, (correct, total) in category_data.items():
            # Only include categories that meet the attempt threshold
            if total > threshold:
                accuracy = (correct / total) * 100  # Calculate accuracy as percentage
                value = f"{round(accuracy, 2)}% ({total})"  # Format value
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
        table_data[category]["Consensus"] = f"{round(consensus, 2)}%"

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

    # # iterate through names and print each name's parser_data_aggregated
    # for name in parser_data_aggregated['submitter'].unique():
    #     print(f"\nParser results for {name}", file=output)
    #     person_specifc_aggregated = parser_data_aggregated[parser_data_aggregated['submitter'] == name]
    #     person_specifc_aggregated.to_string(buf=output, index=False)
    #     print("\n", file=output)

    return output.getvalue()


def analyze_top_players_by_month(image_metadata, cutoff):
    grid_month_mapping = [(entry['grid_number'], entry['date'][:7]) for (_, entry) in image_metadata.iterrows()]
    grid_month_dict = {}
    for grid_number, month in grid_month_mapping:
        if month not in grid_month_dict:
            grid_month_dict[month] = list()
        grid_month_dict[month].append(grid_number)
        
    result = StringIO()

    # For each month, get the list of grids and get consensus top players
    for month in dict(sorted(grid_month_dict.items())):
        players_of_month = analyze_top_players_by_submitter(image_metadata, "All", cutoff, grid_month_dict[month])
        print(f"Month: {month}", file=result)
        print(players_of_month, file=result)

    return result.getvalue()


# Analyze top X players (by frequency) from each submitter
def analyze_top_players_by_submitter(image_metadata, submitter, cutoff, grid_number_list=None):
    """
    Analyze the top players by frequency for each submitter.

    Args:
        image_metadata (pd.DataFrame): DataFrame containing image metadata.
        cutoff (int): The number of top players to analyze.
    """
    result = StringIO()

    player_frequency = {}

    # Preprocess image list based on grid numbers
    if grid_number_list is not None:
        image_metadata_preprocessed = image_metadata[image_metadata['grid_number'].isin(grid_number_list)]
    else:
        image_metadata_preprocessed = image_metadata

    # Analyze the top players by frequency for each submitter
    for row in image_metadata_preprocessed.iterrows():
        submitter_name = row[1]['submitter']
        if submitter_name != submitter and not (submitter == "All" and submitter_name in GRID_PLAYERS_RESTRICTED):
            continue
        responses = row[1]['responses'].values()
        for response in responses:
            if response != '':
                player_frequency[response] = player_frequency.get(response, 0) + 1

    df = pd.DataFrame(player_frequency.items(), columns=['Player', 'Frequency'])
    sorted_df = df.sort_values(by='Frequency', ascending=False).reset_index()

    # Output the top players by frequency for the submitter
    if submitter != "All":
        print(f"Top {cutoff} Players for {submitter}", file=result)
    else:
        print(f"Top {cutoff} Players Overall", file=result)
    for i in range(cutoff):
        if i < len(sorted_df):
            print(f"{i + 1}. {sorted_df['Player'][i]} ({sorted_df['Frequency'][i]})", file=result)
    print("\n", file=result)

    # Get the full output as a string
    result_string = result.getvalue()
    result.close()  # Close the StringIO object
    return result_string


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

        for _, row in filtered_rows.iterrows():
            grid_number = row['grid_number']
            position = row['position']
            prompt = row['prompt']

            for key, value in row['analyzed_response'].items():
                player = key.title()  # Capitalize the player's name
                count = value['count']
                
                if count < 3:
                    continue  # Skip players with less than 3 mentions
                
                if count >= 4:
                    # Banned player
                    result_data.append({
                        "grid_number": grid_number,
                        "position": position,
                        "prompt": prompt,
                        "player": player,
                        "verdict": "ban",
                        "saved_by": None
                    })
                elif count == 3:
                    # Saved player
                    excluded_submitters = [x for x in player_list if x not in value['submitters']]
                    result_data.append({
                        "grid_number": grid_number,
                        "position": position,
                        "prompt": prompt,
                        "player": player,
                        "verdict": "save",
                        "saved_by": ", ".join(excluded_submitters)  # Join the savers into a single string
                    })

        # Create a DataFrame from the result data
        result_df = pd.DataFrame(result_data, columns=[
            "grid_number", "prompt", "player", "verdict", "saved_by"
        ])

        return result_df.sort_values(by=['verdict', 'grid_number'])
    
    result_df = generate_analysis_dataframe(filtered_rows, player_list)

    return result_df


#--------------------------------------------------------------------------------------------------
# Plotting functions

# Graph number of immaculates
def plot_immaculates(texts, color_map):
    counts = []
    for person in texts:
        data = [(1 if obj.correct == 9 else 0) for obj in texts[person]]
        counts.append(sum(data))
    plt.bar([person for person in texts], counts, color=[color_map[person] for person in texts])
    plt.title("Number of Immaculates")
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
    
    fig.suptitle("Correctness Distribution")
    plt.subplots_adjust(hspace=0.5)
    plt.show()

def plot_avg(texts, color_map, title):
    texts_melted = make_texts_melted(texts)
    
    if title == "Average Correct":
        analysis_summary = texts_melted.groupby('name')['correct'].mean().reset_index().rename(columns={"correct": "value"})

    elif title == "Average Score":
        analysis_summary = texts_melted.groupby('name')['score'].mean().reset_index().rename(columns={"score": "value"})

    elif title == "Average Rarity of Correct Square":
        analysis_summary = texts_melted.groupby('name')['average_score_of_correct'].mean().reset_index().rename(columns={"average_score_of_correct" : "value"})

    else:
        return None
    
    plt.bar(
        analysis_summary.name, 
        analysis_summary.value, 
        color=[color_map[person] for person in analysis_summary.name])
    plt.title(title)
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
    plt.title(title)
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

