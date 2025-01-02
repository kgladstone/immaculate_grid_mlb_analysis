# analysis.py
import numpy as np
import pandas as pd
from io import StringIO

from data.data_prep import (
    to_percent,
    make_texts_melted,
    pivot_texts_by_grid_id,
    category_is_team,
    get_team_from_category,
    get_categories_from_prompt,
    build_intersection_structure_for_person,)
from utils.constants import TEAM_LIST, APPLE_TEXTS_DB_PATH, IMAGES_METADATA_PATH, IMAGES_PATH

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


# # Validate that performance data matches image data
# def compare_text_result_to_image_result(texts):
#     image_processor = ImageProcessor(APPLE_TEXTS_DB_PATH, IMAGES_METADATA_PATH, IMAGES_PATH)
#     image_metadata = image_processor.load_image_metadata()

#     print(image_metadata)

#     texts_melted = make_texts_melted(texts)
#     print(texts_melted)

#     for _, row in texts_melted.iterrows():
#         person = row['name']
#         grid_number = row['grid_number']
#         performance = row['matrix']

#         image_metadata = image_processor.get_image_metadata_entry(person, grid_number)

#         if image_metadata is not None:
#             print(person)
#             print(grid_number)
#             print(performance)
#             print(image_metadata['responses'])
#             quit()
