import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
from data_prep import make_texts_melted, to_percent, format_record

from analysis import calculate_smoothed_metrics, calculate_win_rates

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
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))  # Create a 3x2 grid for 5 plots
    top_bar = 130
    
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
        axs[i].set_ylim(0, 1.2*top_bar)
    
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