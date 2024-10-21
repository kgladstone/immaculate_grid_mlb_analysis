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

INPUT_GRID_RESULTS_FILE_PATH = './csv/results.csv'
INPUT_PROMPT_DATA_PATH = './csv/prompts.csv'
COLOR_MAP = {"Sam": "red", "Keith": "blue", "Will": "purple", "Rachel": "green", "Cliff": "orange"}
PDF_FILENAME = "./immaculate_grid_report.pdf"
TEAM_LIST = ["Cubs", "Cardinals", "Brewers", "Reds", "Pirates", "Nationals", "Mets", "Marlins", "Phillies", "Braves", "Dodgers", "Diamondbacks", "Rockies", "Giants", "Padres", "Royals", "White Sox", "Twins", "Guardians", "Tigers", "Red Sox", "Yankees", "Blue Jays", "Rays", "Orioles", "Angels", "Athletics", "Astros", "Mariners", "Rangers"]

#--------------------------------------------------------------------------------------------------

def _to_percent(y, position):
    """Convert a decimal to a percentage string."""
    return f"{100 * y:.0f}%"

# Function to calculate smoothed metrics (score, correct, average_score_of_correct) from analysis_df
def calculate_smoothed_metrics(analysis_df: pd.DataFrame, smoothness: int) -> pd.DataFrame:
    """Generate a DataFrame of smoothed scores, correct values, and average scores over time."""
    metric_table = []

    # Group the data by 'name' to process each person individually
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

# Function to plot smoothed metrics using the smoothed DataFrame
def plot_smoothed_metrics(smoothed_df: pd.DataFrame, metric: str, title: str, ylabel: str) -> None:
    """Plot the smoothed metrics (score, correct, or average score) over time."""
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


# ---------
def calculate_win_rates(reversed_dict, criterion):
    """
    Calculate win rates based on a given criterion.

    Args:
        reversed_dict (dict): The games data.
        criterion (str): The criterion to calculate win rates ("overall", "correctness", "scores", "last_rate").

    Returns:
        dict: A dictionary of win rates for each person.
    """
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

def plot_win_rates(reversed_dict):
    """Plot win rates based on various criteria."""
    # Set a larger figure size to widen the graphs
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    criteria = ["overall", "correctness", "scores", "last_rate"]
    titles = ["Win Rates (Overall)", "Win Rates (Correctness Only)", "Win Rates (Scores Only)", "Last Rate (Overall)"]

    for ax, criterion, title in zip(axs.flat, criteria, titles):
        wins = calculate_win_rates(reversed_dict, criterion)
        ax.bar([person for person in wins], wins.values(), color=[COLOR_MAP[person] for person in wins])
        ax.set_title(title)
        ax.set_yticks([i / 5 for i in range(6)])
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(FuncFormatter(_to_percent))

    # Adjust the layout of the subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()
    
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
def make_fig_3(analysis_df, COLOR_MAP):
    title = "Average Correct"
    analysis_summary = analysis_df.groupby('name')['correct'].mean().reset_index()
    
    plt.bar(
        analysis_summary.name, 
        analysis_summary.correct, 
        color=[COLOR_MAP[person] for person in analysis_summary.name])
    plt.title(title)
    plt.show()
    
# Graph average score
def make_fig_4(analysis_df, COLOR_MAP):
    title = "Average Score"
    analysis_summary = analysis_df.groupby('name')['score'].mean().reset_index()
    
    plt.bar(
        analysis_summary.name, 
        analysis_summary.score, 
        color=[COLOR_MAP[person] for person in analysis_summary.name])
    plt.title(title)
    plt.show()
    
# Graph average rarity of correct square
def make_fig_5(analysis_df, COLOR_MAP):
    title = "Average Rarity of Correct Square"
    analysis_summary = analysis_df.groupby('name')['average_score_of_correct'].mean().reset_index()
    
    plt.bar(
        analysis_summary.name, 
        analysis_summary.average_score_of_correct, 
        color=[COLOR_MAP[person] for person in analysis_summary.name])
    plt.title(title)
    plt.show()

# Plot each metric separately
def make_fig_6(smoothed_metrics_df):
    plot_smoothed_metrics(smoothed_metrics_df, 'smoothed_score', "Smoothed Scores Over Time", "Smoothed Score")

def make_fig_7(smoothed_metrics_df):
    plot_smoothed_metrics(smoothed_metrics_df, 'smoothed_correct', "Smoothed Correct Over Time", "Smoothed Correct")

def make_fig_8(smoothed_metrics_df):
    plot_smoothed_metrics(smoothed_metrics_df, 'smoothed_avg_score', "Smoothed Avg Score of Correct Over Time", "Smoothed Avg Score of Correct")

def make_fig_9(reversed_dict):
    plot_win_rates(reversed_dict)

# Function to format each record with proper alignment
def format_record(rank, name, score, date, game_id, name_width=7, score_width=2, date_width=10, game_id_width=4):
    formatted_rank = f'{rank:<2}'
    formatted_name = f'{name:<{name_width}}'
    formatted_score = f'{str(score):<{score_width}}'
    formatted_date = f'{str(date):<{date_width}}'
    formatted_game_id = f'{str(game_id):<{game_id_width}}'
    
    return f'{formatted_rank} | {formatted_name} | {formatted_score} | {formatted_date} | {formatted_game_id}'


def make_fig_10(texts, fig_title='Best and Worst Scores (All Time)'):
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

def make_fig_11(texts):

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
    make_fig_10(filtered_texts, fig_title)

#--------------------------------------------------------------------------------------------------

def read_prompt_data(filepath):
    with open(os.path.expanduser(filepath)) as f:
        prompt_df = pd.read_csv(f, header=None)
    prompt_df.columns = ["game_id", "00", "01", "02", "10", "11", "12", "20", "21", "22"]
    prompt_df = prompt_df.iloc[1:]
    
    new_rows = []
    for i, row in prompt_df.iterrows():
        new_row = {}
        for col, val in row.items():
            for char in ["(", "'", ")"]:
                val = val.replace(char, "")
            new_row[col] = val.replace(", ", " + ")
        new_rows.append(new_row)
             
    prompt_df = pd.DataFrame(new_rows)
    prompt_df['game_id'] = prompt_df['game_id'].astype(int)

    return prompt_df

def category_is_team(category):
    for team in TEAM_LIST:
        if team in category:
            return True
    return False

def get_team_from_category(category):
    for team in TEAM_LIST:
        if team in category:
            return team
    return ""

def get_categories_from_prompt(prompt):
    parts = prompt.split(" + ")
    return parts[0].strip(), parts[1].strip()

# Build up category data structure
def build_category_structure(prompt_df):
    categories = set()
    for person, games in texts.items():
        for game in games:
            id = game.grid_number
            prompt_rows = prompt_df[prompt_df["game_id"] == id]
            if len(prompt_rows) != 1:
                continue
            prompts = prompt_rows.iloc[0][1:]
            for prompt in prompts:
                part_one, part_two = get_categories_from_prompt(prompt)
                categories.add(part_one)
                categories.add(part_two)
    return categories

# Initialize person<>category data structure
# Dictionary where the key is the person name
# The value is a list of sub-dictionaries
# Each subdictionary's key is a category, and their value is a list of length=2
# The first element in the list is the number instances of the person getting the value in the category correct
# The second element in the list is the number of opportunities that the person had to answer for that category
def build_person_category_structure(texts, prompt_df, categories):
    person_to_category = {}
    for person, _ in texts.items():
        person_to_category[person] = {cat: [0, 0] for cat in categories}
    
    for person, games in texts.items():
        for game in games:
            id = game.grid_number
            prompt_rows = prompt_df[prompt_df["game_id"] == id]
            if len(prompt_rows) != 1:
                continue
            prompts = prompt_rows.iloc[0][1:]
    
            matrix = game.matrix
            for i in range(3):
                for j in range(3):
                    part_one, part_two = get_categories_from_prompt(prompts[f"{i}{j}"])
                    if matrix[i][j]:
                        person_to_category[person][part_one][0] += 1
                        person_to_category[person][part_two][0] += 1
                    person_to_category[person][part_one][1] += 1
                    person_to_category[person][part_two][1] += 1
    return person_to_category

def get_category_clearing_threshold(categories, person_to_category, threshold=25):
    categories_to_count = {}
    for category in categories:
        categories_to_count[category] = []
    for _, value in person_to_category.items():
        for category, (correct, total) in value.items():
            categories_to_count[category].append(total)
    categories_clearing_threshold = [cat for cat in filter(lambda x: sum(categories_to_count[x]) / len(categories_to_count[x]) > threshold, categories_to_count)]
    return categories_clearing_threshold

def get_person_to_type(texts, prompt_df, person_to_category):
    types = ["Team-Team", "Team-Stat", "Stat-Stat"]
    person_to_type = {person: {t: [0, 0] for t in types} for person in person_to_category}
    
    for person, games in texts.items():
        for game in games:
            id = game.grid_number
            prompt_rows = prompt_df[prompt_df["game_id"] == id]
            if len(prompt_rows) != 1:
                continue
            prompts = prompt_rows.iloc[0][1:]
    
            matrix = game.matrix
            for i in range(3):
                for j in range(3):
                    part_one, part_two = get_categories_from_prompt(prompts[f"{i}{j}"])
                    tag = ""
                    if category_is_team(part_one) and category_is_team(part_two):
                        tag = "Team-Team"
                    elif category_is_team(part_one) != category_is_team(part_two):
                        tag = "Team-Stat"
                    else:
                        tag = "Stat-Stat"
                    if matrix[i][j]:
                        person_to_type[person][tag][0] += 1
                    person_to_type[person][tag][1] += 1
    return person_to_type

def person_to_type_to_string(person_to_type):
    result = ""
    for person in person_to_type:
        result += person + "\n"
        for tag in person_to_type[person]:
            acc = person_to_type[person][tag][0] / person_to_type[person][tag][1]
            line = f"{tag}: {round(100 * acc)}% ({person_to_type[person][tag][1]})"
            result += line + "\n"
        result += "\n"
    return result

def person_to_category_to_string(person_to_category, threshold=25):
    result = ""
    for person, value in person_to_category.items():
        rankings = sorted([(cat, value[cat][0] / value[cat][1], value[cat][1]) for cat in value], key=lambda x: x[1], reverse=True)
    
        result += f"====={person}=====\n" 
        count = 1
        for i, (category, accuracy, total) in enumerate(rankings):
            if total > threshold:
                result += f"{count}. {category} ({round(accuracy, 2)}) ({total})\n"
                count += 1
        result += "\n\n"

    return result

def analyze_easiest_teams(categories, person_to_category):
    overall = []
    for category in categories:
        values = []
        counts = []
        for person in person_to_category:
            values.append(person_to_category[person][category][0] / person_to_category[person][category][1])
            counts.append(person_to_category[person][category][1])
        if category_is_team(category):
            overall.append((category, sum(values) / len(values)))

    result = ""
    result += "Consensus Easiest Teams\n"
    overall = sorted(overall, key=lambda x: x[1], reverse=True)
    for i, (category, avg) in enumerate(overall):
        result += f"{(i + 1)}. {category} ({round(100 * avg)}%)\n"

    return result

def analyze_team_std_dev(categories, person_to_category):
    overall = []
    for category in categories:
        values = []
        counts = []
        for person in person_to_category:
            values.append(person_to_category[person][category][0] / person_to_category[person][category][1])
            counts.append(person_to_category[person][category][1])
        if category_is_team(category):
            overall.append((category, np.std(values)))

    result = ""
    result += "Biggest Team Standard Deviations\n"
    overall = sorted(overall, key=lambda x: x[1], reverse=True)
    for i, (category, avg) in enumerate(overall):
        result += f"{(i + 1)}. {category} ({round(100 * avg)}%)\n"
    return result

def analyze_best_person_by_team(categories, person_to_category):
    overall = []
    for category in filter(category_is_team, categories):
    
        max_acc = 0
        for person in person_to_category:
            acc = person_to_category[person][category][0] / person_to_category[person][category][1]
            if acc > max_acc:
                max_acc = acc
    
        max_people = []
        for person in person_to_category:
            acc = person_to_category[person][category][0] / person_to_category[person][category][1]
            if abs(acc - max_acc) < 0.0001:
                max_people.append(person)
        
        overall.append((category, ", ".join(max_people)))

    result = ""
    result += "Best Person for Each Team\n"
    for category, people in sorted(overall, key=lambda x: x[0]):
        if len(category) > 15:
            result += category + "        "
        else:
            result += category + "            "
        result += people
        result += "\n"

    return result

def analyze_worst_person_by_team(categories, person_to_category):
    overall = []
    for category in filter(category_is_team, categories):
    
        min_acc = 101
        for person in person_to_category:
            acc = person_to_category[person][category][0] / person_to_category[person][category][1]
            if acc < min_acc:
                min_acc = acc
    
        min_people = []
        for person in person_to_category:
            acc = person_to_category[person][category][0] / person_to_category[person][category][1]
            if abs(acc - min_acc) < 0.0001:
                min_people.append(person)
        
        overall.append((category, ", ".join(min_people)))
    
    result = ""
    result += "Worst Person for Each Team\n"
    for category, people in sorted(overall, key=lambda x: x[0]):
        if len(category) > 15:
            result += category + "        "
        else:
            result += category + "            "
        result += people
        result += "\n"

    return result    

def analyze_best_person_by_category(categories, person_to_category, categories_clearing_threshold):
    overall = []
    for category in filter(lambda x: not category_is_team(x) and x in categories_clearing_threshold, categories):
    
        max_acc = 0
        for person in person_to_category:
            acc = person_to_category[person][category][0] / person_to_category[person][category][1]
            if acc > max_acc:
                max_acc = acc
    
        max_people = []
        for person in person_to_category:
            acc = person_to_category[person][category][0] / person_to_category[person][category][1]
            if abs(acc - max_acc) < 0.0001:
                max_people.append(person)
        
        overall.append((category, ", ".join(max_people)))

    result = ""
    result += "Best Person for Each Category\n"
    for category, people in sorted(overall, key=lambda x: x[0]):
        if len(category) > 15:
            result += category + "        "
        else:
            result += category + "            "
        result += people
        result += "\n"

    return result    

def analyze_worst_person_by_category(categories, person_to_category, categories_clearing_threshold):
    overall = []
    for category in filter(lambda x: not category_is_team(x) and x in categories_clearing_threshold, categories):
    
        min_acc = 101
        for person in person_to_category:
            acc = person_to_category[person][category][0] / person_to_category[person][category][1]
            if acc < min_acc:
                min_acc = acc
    
        min_people = []
        for person in person_to_category:
            acc = person_to_category[person][category][0] / person_to_category[person][category][1]
            if abs(acc - min_acc) < 0.0001:
                min_people.append(person)
        
        overall.append((category, ", ".join(min_people)))
    
    result = ""
    result += "Worst Person for Each Category\n"
    for category, people in sorted(overall, key=lambda x: x[0]):
        if len(category) > 15:
            result += category + "        "
        else:
            result += category + "            "
        result += people
        result += "\n"

    return result   

def analyze_person_prompt_performance(categories, person_to_category, categories_clearing_threshold, direction, category_type):
    if category_type == "Team" and direction == "Best":
        return analyze_best_person_by_team(categories, person_to_category)
    elif category_type == "Team" and direction == "Worst":
        return analyze_worst_person_by_team(categories, person_to_category)
    elif category_type == "Category" and direction == "Best":
        return analyze_best_person_by_category(categories, person_to_category, categories_clearing_threshold)
    elif category_type == "Category" and direction == "Worst":
        return analyze_worst_person_by_category(categories, person_to_category, categories_clearing_threshold)
    return ""

def analyze_hardest_teams(texts, prompt_df):
    result = StringIO()
    
    hardest_teams = {}
    
    for person, games in texts.items():
        hardest_teams[person] = {team: [0, 0] for team in TEAM_LIST}
        for game in games:
            id = game.grid_number
            prompt_rows = prompt_df[prompt_df["game_id"] == id]
            if len(prompt_rows) != 1:
                continue
            prompts = prompt_rows.iloc[0][1:]
    
            matrix = game.matrix
            for i in range(3):
                for j in range(3):
                    part_one, part_two = get_categories_from_prompt(prompts[f"{i}{j}"])
                    tag = ""
                    if category_is_team(part_one) and category_is_team(part_two):
                        team_one = get_team_from_category(part_one)
                        team_two = get_team_from_category(part_two)
                        if matrix[i][j]:
                            hardest_teams[person][team_one][0] += 1
                            hardest_teams[person][team_two][0] += 1
                        hardest_teams[person][team_one][1] += 1
                        hardest_teams[person][team_two][1] += 1
    
    print("Hardest Team-Team Intersections for Each Person", "\n\n", file=result)
    for person in hardest_teams:
        print(f"====={person}=====", file=result)
        for i, (team, res) in enumerate(sorted(hardest_teams[person].items(), key = lambda x: x[1][0] / x[1][1], reverse=True)):
            print(f"{i + 1}. {team} ({round(100 * res[0] / res[1])}%)", file=result)
        print("\n\n\n", file=result)
    
    consensus_intersection_difficulty = {}
    for team in TEAM_LIST:
        right = 0
        total = 0
        for person in hardest_teams:
            res = hardest_teams[person][team]
            right += res[0]
            total += res[1]
        consensus_intersection_difficulty[team] = right / total
        
    print("=====Consensus=====", file=result)
    for i, (team, pct) in enumerate(sorted(consensus_intersection_difficulty.items(), key=lambda x: x[1], reverse=True)):
        print(f"{i + 1}. {team} ({round(100 * pct)}%)", file=result)

    # Get the full output as a string
    result_string = result.getvalue()
    result.close()  # Close the StringIO object
    return result_string

def analyze_hardest_team_stats(texts, prompt_df):
    result = StringIO()

    hardest_team_stats = {}
    
    for person, games in texts.items():
        hardest_team_stats[person] = {team: [0, 0] for team in TEAM_LIST}
        for game in games:
            id = game.grid_number
            prompt_rows = prompt_df[prompt_df["game_id"] == id]
            if len(prompt_rows) != 1:
                continue
            prompts = prompt_rows.iloc[0][1:]
    
            matrix = game.matrix
            for i in range(3):
                for j in range(3):
                    part_one, part_two = get_categories_from_prompt(prompts[f"{i}{j}"])
                    tag = ""
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
    
    print("Hardest Team-Stats Intersections for Each Person", "\n\n", file=result)
    for person in hardest_team_stats:
        print(f"====={person}=====", file=result)
        for i, (team, res) in enumerate(sorted(hardest_team_stats[person].items(), key = lambda x: x[1][0] / x[1][1], reverse=True)):
            print(f"{i + 1}. {team} ({round(100 * res[0] / res[1])}%)", file=result)
        print("\n\n\n", file=result)
    
    consensus_intersection_difficulty = {}
    for team in TEAM_LIST:
        right = 0
        total = 0
        for person in hardest_team_stats:
            res = hardest_team_stats[person][team]
            right += res[0]
            total += res[1]
        consensus_intersection_difficulty[team] = right / total
        
    print("=====Consensus=====", file=result)
    for i, (team, pct) in enumerate(sorted(consensus_intersection_difficulty.items(), key=lambda x: x[1], reverse=True)):
        print(f"{i + 1}. {team} ({round(100 * pct)}%)", file=result)

    # Get the full output as a string
    result_string = result.getvalue()
    result.close()  # Close the StringIO object
    return result_string

def compute_most_common_exact_intersections(texts, prompt_df, name):
    most_common_exact_intersections = {}
    
    for game in texts[name]:
        id = game.grid_number
        prompt_rows = prompt_df[prompt_df["game_id"] == id]
        if len(prompt_rows) != 1:
            continue
        prompts = prompt_rows.iloc[0][1:]
        for i in range(3):
            for j in range(3):
                part_one, part_two = get_categories_from_prompt(prompts[f"{i}{j}"])
                key = " + ".join(sorted([part_one, part_two]))
                if key not in most_common_exact_intersections:
                    most_common_exact_intersections[key] = 0
                most_common_exact_intersections[key] += 1
    return most_common_exact_intersections

def analyze_most_common_exact_intersections(texts, prompt_df, name):
    result = StringIO()
    
    most_common_exact_intersections = compute_most_common_exact_intersections(texts, prompt_df, name)

    print("Most Common Exact Intersections for {}".format(name), file=result)
    for i, (combo, count) in enumerate(sorted(most_common_exact_intersections.items(), key = lambda x: x[1], reverse=True)):
        if count >= 5:
            print(f"{i + 1}. {combo} ({count})", file=result)

    # Get the full output as a string
    result_string = result.getvalue()
    result.close()  # Close the StringIO object
    return result_string

def analyze_empty_team_team_intersections(texts, prompt_df, name, categories):
    result = StringIO()

    most_common_exact_intersections = compute_most_common_exact_intersections(texts, prompt_df, name)
        
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
    print("Empty Team-Team Intersections for {}".format(name), file=result)
    for i, team in enumerate(sorted(TEAM_LIST)):
        for other in sorted(TEAM_LIST)[i + 1:]:
            key = " + ".join([team_to_full_names[team], team_to_full_names[other]])
            other_key =  " + ".join([team_to_full_names[other], team_to_full_names[team]])
            if key not in most_common_exact_intersections and other_key not in most_common_exact_intersections:
                print(key, file=result)
                missing += 1
                if team not in missing_maps:
                    missing_maps[team] = 0
                if other not in missing_maps:
                    missing_maps[other] = 0
                missing_maps[team] += 1
                missing_maps[other] += 1
            else:
                present += 1    
    
    print("\n\n\n\nTotal Missing for {}".format(name), file=result)
    for i, (team, count) in enumerate(sorted(missing_maps.items(), key=lambda x: x[1], reverse=True)):
        if count > 0:
            print(f"{i + 1}. {team} ({count})", file=result)

    # Get the full output as a string
    result_string = result.getvalue()
    result.close()  # Close the StringIO object
    return result_string

#--------------------------------------------------------------------------------------------------

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
    plt.text(0.0, 1, output, fontsize=10, ha='left', va='top')
    
    # Display the plot
    plt.show()
    
def create_pdf_with_graphs_cover_and_toc(texts, COLOR_MAP, analysis_df, smoothed_metrics_df, reversed_dict, pdf_filename):
    """
    Creates a PDF booklet with a cover page, table of contents, various graphs, 
    and a summary table of best and worst scores based on the provided data.

    Parameters:
    - texts: Data used for creating graphs.
    - COLOR_MAP: Color mapping for the graphs.
    - analysis_df: DataFrame containing analysis metrics.
    - smoothed_metrics_df: DataFrame for smoothed metrics over time.
    - reversed_dict: Dictionary for win rates.
    - pdf_filename: Name of the output PDF file.
    """
    # Use a non-interactive backend to prevent plots from rendering to the screen
    plt.switch_backend('Agg')

    # Get today's date in a readable format
    today_date = datetime.now().strftime('%B %d, %Y')

    # List of graph-making functions with their respective arguments and titles
    graph_functions = [
        (make_fig_1, (texts, COLOR_MAP), "Number of Immaculates"),
        (make_fig_2, (texts, COLOR_MAP), "Correctness Distribution"),
        (make_fig_3, (analysis_df, COLOR_MAP), "Average Correct"),
        (make_fig_4, (analysis_df, COLOR_MAP), "Average Score"),
        (make_fig_5, (analysis_df, COLOR_MAP), "Average Rarity of Correct Square"),
        (make_fig_6, (smoothed_metrics_df,), "Smoothed Scores Over Time"),
        (make_fig_7, (smoothed_metrics_df,), "Smoothed Correct Over Time"),
        (make_fig_8, (smoothed_metrics_df,), "Smoothed Avg Score of Correct Over Time"),
        (make_fig_9, (reversed_dict,), "Win Rates"),
        (make_fig_10, (texts, ), 'Best and Worst Scores (All Time)'),
        (make_fig_11, (texts, ), 'Best and Worst Scores (Last 30 Days)'),
        (make_generic_text_page, (person_to_type_to_string, (person_to_type, ), 'Type Performance Overview'), 'Type Performance Overview'),
        (make_generic_text_page, (person_to_category_to_string, (person_to_category, ), 'Category Performance Overview'), 'Category Performance Overview'),
        (make_generic_text_page, (analyze_easiest_teams, (categories, person_to_category, ), 'Easiest Teams Overview'), 'Easiest Teams Overview'),
        (make_generic_text_page, (analyze_team_std_dev, (categories, person_to_category, ), 'Easiest Teams Standard Deviation'), 'Easiest Teams Standard Deviation'),
        (make_generic_text_page, (analyze_person_prompt_performance, (categories, person_to_category, categories_clearing_threshold, "Best", "Team", ), 'Best Team Overview'), 'Best Team Overview'),
        (make_generic_text_page, (analyze_person_prompt_performance, (categories, person_to_category, categories_clearing_threshold, "Worst", "Team", ), 'Worst Team Overview'), 'Worst Team Overview'),
        (make_generic_text_page, (analyze_person_prompt_performance, (categories, person_to_category, categories_clearing_threshold, "Best", "Category", ), 'Best Category Overview'), 'Best Category Overview'),
        (make_generic_text_page, (analyze_person_prompt_performance, (categories, person_to_category, categories_clearing_threshold, "Worst", "Category", ), 'Worst Category Overview'), 'Worst Category Overview'),
        (make_generic_text_page, (analyze_hardest_teams, (texts, prompt_df, ), 'Hardest Teams Overview'), 'Hardest Teams Overview'),
        (make_generic_text_page, (analyze_hardest_team_stats, (texts, prompt_df, ), 'Hardest Teams Stats Overview'), 'Hardest Teams Stats Overview'),
        (make_generic_text_page, (analyze_most_common_exact_intersections, (texts, prompt_df, "Keith"), 'Common Intersections (Keith)'), 'Common Intersections (Keith)'),
        (make_generic_text_page, (analyze_most_common_exact_intersections, (texts, prompt_df, "Rachel"), 'Common Intersections (Rachel)'), 'Common Intersections (Rachel)'),
        (make_generic_text_page, (analyze_most_common_exact_intersections, (texts, prompt_df, "Sam"), 'Common Intersections (Sam)'), 'Common Intersections (Sam)'),
        (make_generic_text_page, (analyze_most_common_exact_intersections, (texts, prompt_df, "Will"), 'Common Intersections (Will)'), 'Common Intersections (Will)'),
        (make_generic_text_page, (analyze_most_common_exact_intersections, (texts, prompt_df, "Cliff"), 'Common Intersections (Cliff)'), 'Common Intersections (Cliff)'),
        (make_generic_text_page, (analyze_empty_team_team_intersections, (texts, prompt_df, "Keith", categories), 'Empty Intersections (Keith)'), 'Empty Intersections (Keith)'),
        (make_generic_text_page, (analyze_empty_team_team_intersections, (texts, prompt_df, "Rachel", categories), 'Empty Intersections (Rachel)'), 'Empty Intersections (Rachel)'),
        (make_generic_text_page, (analyze_empty_team_team_intersections, (texts, prompt_df, "Sam", categories), 'Empty Intersections (Sam)'), 'Empty Intersections (Sam)'),
        (make_generic_text_page, (analyze_empty_team_team_intersections, (texts, prompt_df, "Will", categories), 'Empty Intersections (Will)'), 'Empty Intersections (Will)'),
        (make_generic_text_page, (analyze_empty_team_team_intersections, (texts, prompt_df, "Cliff", categories), 'Empty Intersections (Cliff)'), 'Empty Intersections (Cliff)'),    
    ]

    try:
        # Create a PDF file with multiple pages
        with PdfPages(pdf_filename) as pdf:
            # Create the cover page
            plt.figure(figsize=(8.5, 11))  # Set the page size to standard A4 or letter
            plt.text(0.5, 0.7, 'Immaculate Grid Analysis Results', fontsize=24, ha='center', va='center', fontweight='bold')
            plt.text(0.5, 0.6, f'Date of Analysis: {today_date}', fontsize=16, ha='center', va='center')
            plt.axis('off')  # Hide axes for the cover page
            pdf.savefig()  # Save the cover page to the PDF
            plt.close()  # Close the figure for the cover page

            # Create the Table of Contents page
            plt.figure(figsize=(8.5, 11))
            plt.text(0.5, 0.9, 'Table of Contents', fontsize=20, ha='center', va='top', fontweight='bold')

            # Add the list of graphs to the Table of Contents
            toc_item_y_position = 0.85
            for i, (_, _, title) in enumerate(graph_functions, start=1):
                plt.text(0.1, toc_item_y_position, f'{i}. {title}', fontsize=10, ha='left', va='top')
                toc_item_y_position -= 0.02  # Adjust the position for the next line
         
            plt.axis('off')  # Hide axes for the Table of Contents page
            pdf.savefig()  # Save the Table of Contents page to the PDF
            plt.close()  # Close the figure for the Table of Contents page

            # Add each graph to a new page in the PDF
            for func, args, _ in graph_functions:
                plt.figure()
                func(*args)  # Call the graph-making function with its arguments
                pdf.savefig()  # Save the current figure to the PDF
                plt.close()  # Close the figure to free up memory

    except Exception as e:
        print(f"An error occurred: {e}")

    print(f"PDF file '{pdf_filename}' has been created with a cover page, table of contents, and all graphs.")

#--------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Main Execution
if __name__ == "__main__":

    # (0) Read data
    
    # Expand user directory and open the csv file
    data = pd.read_csv(INPUT_GRID_RESULTS_FILE_PATH, index_col=False)
    num_rows = len(data)
    
    # Keeping only the highest score for each grid_number and name combination
    data = data.loc[data.groupby(['grid_number', 'name'])['score'].idxmax()]
    num_rows_post = len(data)
    
    print("Data was trimmed to {} from original size of {}, handling instances of multiple entries per person per grid".format(num_rows_post, num_rows))
    
    #--------------------------------------------------------------------------------------------------
    # (1) Transform into base model format
    texts = ImmaculateGridUtils.df_to_immaculate_grid_objs(data)
    
    #--------------------------------------------------------------------------------------------------
    # (2) Make a dictionary called "reversed_dict" that is the reverse of texts
    
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
    
    ### --- Make a dataframe for score, correctness, and average_score_of_correct
    
    #--------------------------------------------------------------------------------------------------
    # (3) Store in a Pandas dataframe
    
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
    
    # # Calculate the smoothed metrics from the DataFrame
    smoothness = 28
    smoothed_metrics_df = calculate_smoothed_metrics(analysis_df, smoothness)
    
    # Execute
    prompt_df = read_prompt_data(INPUT_PROMPT_DATA_PATH)
    categories = build_category_structure(prompt_df)
    person_to_category = build_person_category_structure(texts, prompt_df, categories)
    categories_clearing_threshold = get_category_clearing_threshold(categories, person_to_category)
    person_to_type = get_person_to_type(texts, prompt_df, person_to_category)
    
    
    create_pdf_with_graphs_cover_and_toc(texts, COLOR_MAP, analysis_df, smoothed_metrics_df, reversed_dict, pdf_filename=PDF_FILENAME)