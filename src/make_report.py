# Imports
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
from datetime import datetime

from constants import GRID_PLAYERS
from data_prep import (
    preprocess_data_into_texts_structure, read_prompt_data, make_color_map,
    build_category_structure, build_person_category_structure,
    person_to_type_to_string, person_to_category_to_string
)
from analysis import (
    get_category_clearing_threshold, get_person_to_type, analyze_easiest_teams,
    analyze_team_std_dev, analyze_person_prompt_performance, analyze_hardest_teams,
    analyze_hardest_team_stats, analyze_most_successful_exact_intersections,
    analyze_empty_team_team_intersections
)
from plotting import (
    plot_immaculates, plot_correctness, plot_avg, plot_smoothed_metrics, plot_win_rates,
    plot_best_worst_scores, plot_best_worst_scores_30
)

class ReportGenerator:
    def __init__(self, messages_path, prompts_path, pdf_filename):
        self.messages_path = messages_path
        self.prompts_path = prompts_path
        self.pdf_filename = pdf_filename
        self.texts = None
        self.prompts = None
        self.color_map = None

    def load_data(self):
        print("Loading data...")
        self.texts = preprocess_data_into_texts_structure(pd.read_csv(self.messages_path))
        self.prompts = read_prompt_data(self.prompts_path)
        self.color_map = make_color_map(GRID_PLAYERS)

    def _make_generic_text_page(self, func, args, page_title):
        """
        This function prepares a generic page that includes the output from the function being executed
        """
        output = func(*args)
        MAX_LINES_PER_NORMAL_PAGE = 45
        page_len_scalar = max([int(math.floor(output.count('\n') / MAX_LINES_PER_NORMAL_PAGE)),1])
        fig, ax = plt.subplots(figsize=(8, 11*page_len_scalar))
        ax.axis('off')
        plt.text(0.5, 1.1, page_title, fontsize=24, ha='center', va='top', fontweight='bold')
        plt.text(0.0, 1, output, fontsize=8, ha='left', va='top')
        plt.show()
        return

    def _add_cover_page(self, pdf):
        """Helper function to create the cover page."""
        # Get today's date in a readable format
        today_date = datetime.now().strftime('%B %d, %Y')

        plt.figure(figsize=(8.5, 11))
        plt.text(0.5, 0.7, 'Immaculate Grid Analysis Results', fontsize=24, ha='center', va='center', fontweight='bold')
        plt.text(0.5, 0.6, f'Date of Analysis: {today_date}', fontsize=16, ha='center', va='center')
        plt.axis('off')
        pdf.savefig()
        plt.close()

    def _add_toc_page(self, pdf, graph_functions):
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

    def _add_graphs_to_pdf(self, pdf, graph_functions):
        """Helper function to generate graphs and add them to the PDF."""
        for func, args, _ in graph_functions:
            plt.figure()
            func(*args)
            pdf.savefig()
            plt.close()

    def prepare_graph_functions(self):
        """
        This function prepares a list of graph_functions that will be executed later in PDF generation
        """
        # Prepare various structures using "texts" and "prompts"
        categories = build_category_structure(self.texts, self.prompts)
        person_to_category = build_person_category_structure(self.texts, self.prompts, categories)
        categories_clearing_threshold = get_category_clearing_threshold(categories, person_to_category)
        person_to_type = get_person_to_type(self.texts, self.prompts, person_to_category)

        # List of graph-making functions with their respective arguments and titles
        graph_functions = [
            (plot_immaculates, (self.texts, self.color_map), "Number of Immaculates"),
            (plot_avg, (self.texts, self.color_map, 'Average Correct'), "Average Correct"),
            (plot_avg, (self.texts, self.color_map, 'Average Score'), "Average Score"),
            (plot_avg, (self.texts, self.color_map, 'Average Rarity of Correct Square'), "Average Rarity of Correct Square"),
            (plot_correctness, (self.texts, self.color_map), "Correctness Distribution"),
            (plot_smoothed_metrics, (self.texts, 'smoothed_score', "Smoothed Scores Over Time", "Smoothed Score", self.color_map), "Smoothed Scores Over Time"),
            (plot_smoothed_metrics, (self.texts, 'smoothed_correct', "Smoothed Correct Over Time", "Smoothed Correct", self.color_map), "Smoothed Correct Over Time"),
            (plot_smoothed_metrics, (self.texts, 'smoothed_avg_score', "Smoothed Avg Score of Correct Over Time", "Smoothed Avg Score of Correct", self.color_map), "Smoothed Avg Score of Correct Over Time"),
            (plot_win_rates, (self.texts, self.color_map), "Win Rates"),
            (plot_best_worst_scores, (self.texts,), 'Best and Worst Scores (All Time)'),
            (plot_best_worst_scores_30, (self.texts,), 'Best and Worst Scores (Last 30 Days)'),
            (self._make_generic_text_page, (person_to_type_to_string, (person_to_type,), 'Type Performance Overview'), 'Type Performance Overview'),
            (self._make_generic_text_page, (person_to_category_to_string, (person_to_category,), 'Category Performance Overview'), 'Category Performance Overview'),
            (self._make_generic_text_page, (analyze_easiest_teams, (categories, person_to_category,), 'Easiest Teams Overview'), 'Easiest Teams Overview'),
            (self._make_generic_text_page, (analyze_team_std_dev, (categories, person_to_category,), 'Easiest Teams Standard Deviation'), 'Easiest Teams Standard Deviation'),
            (self._make_generic_text_page, (analyze_person_prompt_performance, (categories, person_to_category, categories_clearing_threshold, "Best", "Team",), 'Best Team Overview'), 'Best Team Overview'),
            (self._make_generic_text_page, (analyze_person_prompt_performance, (categories, person_to_category, categories_clearing_threshold, "Worst", "Team",), 'Worst Team Overview'), 'Worst Team Overview'),
            (self._make_generic_text_page, (analyze_person_prompt_performance, (categories, person_to_category, categories_clearing_threshold, "Best", "Category",), 'Best Category Overview'), 'Best Category Overview'),
            (self._make_generic_text_page, (analyze_person_prompt_performance, (categories, person_to_category, categories_clearing_threshold, "Worst", "Category",), 'Worst Category Overview'), 'Worst Category Overview'),
            (self._make_generic_text_page, (analyze_hardest_teams, (self.texts, self.prompts), 'Hardest Teams Overview'), 'Hardest Teams Overview'),
            (self._make_generic_text_page, (analyze_hardest_team_stats, (self.texts, self.prompts), 'Hardest Teams Stats Overview'), 'Hardest Teams Stats Overview')
        ]

        person_specific_pages = [
            (self._make_generic_text_page, (analyze_most_successful_exact_intersections, (self.texts, self.prompts, person), f'Most Successful Intersections ({person})'), f'Most Successful Intersections ({person})')
            for person in GRID_PLAYERS.keys()
        ] + [
            (self._make_generic_text_page, (analyze_empty_team_team_intersections, (self.texts, self.prompts, person, categories), f'Never Shown Intersections ({person})'), f'Never Shown Intersections ({person})')
            for person in GRID_PLAYERS.keys()
        ]

        graph_functions.extend(person_specific_pages)

        return graph_functions

    def generate_report(self):
        print("Generating report...")
        graph_functions = self.prepare_graph_functions()

        # Use a non-interactive backend to prevent plots from rendering to the screen
        plt.switch_backend('Agg')

        with PdfPages(self.pdf_filename) as pdf:
            self._add_cover_page(pdf)
            self._add_toc_page(pdf, graph_functions)
            self._add_graphs_to_pdf(pdf, graph_functions)

        print(f"PDF report generated: {self.pdf_filename}")