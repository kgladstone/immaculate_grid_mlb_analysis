# analysis.py
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta

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
    clean_image_parser_data
    )
from utils.constants import TEAM_LIST

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


def grid_numbers_with_matrix_image_nonmatches(texts, image_metadata, person):
    """
    """
    image_data_structure = build_results_image_structure(texts, image_metadata)

    grid_numbers = []

    for grid_number, results in image_data_structure[person].items():

        if results['image_metadata'] is None:
            # print(f"Skipping grid {grid_number} due to missing image metadata.")
            continue
        else:
            image_responses = list(results['image_metadata']['responses'].values())
            if len(results['performance']) != len(image_responses):
                print(f"Mismatch in lengths: performance={len(results['performance'])}, responses={len(image_responses)}")
                continue
            for i, matrix_element in enumerate(results['performance']):
                if matrix_element == True:
                    if image_responses[i] == '':
                        break
                elif matrix_element == False:
                    if image_responses[i] != '':
                        break
            grid_numbers.append(grid_number)
    
    return grid_numbers


def matrix_and_image_metadata_matches(matrix, image_metadata_day):
    image_metadata_responses_list = list(image_metadata_day['responses'].values())
    matches = 0

    # Get matches
    for i, value in enumerate(matrix):
        if value:
            if image_metadata_responses_list[i] != '':
                matches += 1

    # Get number of True values in matrix
    total = sum(matrix)

    return matches / total if total > 0 else 0


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
        image_text_matches = []

        for _, row in data.items():
            count_of_text_results += 1
            if row['image_metadata'] is not None:
                count_of_image_results += 1
                image_text_matches.append(matrix_and_image_metadata_matches(row['performance'], row['image_metadata']))

        average_image_text_matches = sum(image_text_matches) / len(image_text_matches) if len(image_text_matches) > 0 else 0

        results.append({
            'Person': person,
            'Text Results': count_of_text_results,
            'Parsed Image Results': count_of_image_results,
            'Avg. Accuracy of Results': average_image_text_matches,
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


# Analyze top X players (by frequency) from each submitter
def analyze_top_players_by_submitter(image_metadata, submitter, cutoff):
    """
    Analyze the top players by frequency for each submitter.

    Args:
        image_metadata (pd.DataFrame): DataFrame containing image metadata.
        cutoff (int): The number of top players to analyze.
    """
    result = StringIO()

    player_frequency = {}

    # Analyze the top 10 players by frequency for each submitter
    for row in image_metadata.iterrows():
        submitter_name = row[1]['submitter']
        if submitter_name != submitter:
            continue
        responses = row[1]['responses'].values()
        for response in responses:
            if response != '':
                player_frequency[response] = player_frequency.get(response, 0) + 1

    df = pd.DataFrame(player_frequency.items(), columns=['Player', 'Frequency'])
    sorted_df = df.sort_values(by='Frequency', ascending=False).reset_index()

    # Output the top 10 players by frequency for the submitter
    print(f"Top {cutoff} Players for {submitter}", file=result)
    for i in range(cutoff):
        if i < len(sorted_df):
            print(f"{i + 1}. {sorted_df['Player'][i]} ({sorted_df['Frequency'][i]})", file=result)
    print("\n", file=result)

    # Get the full output as a string
    result_string = result.getvalue()
    result.close()  # Close the StringIO object
    return result_string

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

