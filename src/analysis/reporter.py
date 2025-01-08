# Imports
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
from datetime import datetime

from utils.constants import (
    GRID_PLAYERS, 
    GRID_PLAYERS_RESTRICTED,
    PROMPTS_CSV_PATH,
    MESSAGES_CSV_PATH,
    APPLE_TEXTS_DB_PATH, 
    IMAGES_METADATA_PATH, 
    IMAGES_PATH
)
from data.prompts_loader import PromptsLoader
from data.messages_loader import MessagesLoader
from data.image_processor import ImageProcessor
from data.data_prep import (
    preprocess_data_into_texts_structure, make_color_map,
    build_category_structure, build_person_category_structure,
)
from analysis.analysis import (
    get_category_clearing_threshold, get_person_to_type, 
    analyze_person_type_performance, analyze_team_performance, 
    analyze_person_prompt_performance, analyze_hardest_intersections, 
    analyze_most_successful_exact_intersections, analyze_empty_team_team_intersections, 
    plot_summary_metrics, plot_correctness, plot_smoothed_metrics, plot_win_rates,
    plot_best_worst_scores, plot_best_worst_scores_30,
    analyze_top_players_by_submitter,
    analyze_grid_cell_with_shared_guesses,
    analyze_top_players_by_month,
    plot_top_n_grids,
    analyze_person_to_category
)

class ReportGenerator:
    def __init__(self, pdf_filename):
        self.pdf_filename = pdf_filename
        self.texts = None
        self.prompts = None
        self.color_map = None
        self.image_metadata = None

    def load_data(self):
        print("Loading data...")
        texts_df = MessagesLoader(APPLE_TEXTS_DB_PATH, MESSAGES_CSV_PATH).load().get_data()
        self.texts = preprocess_data_into_texts_structure(texts_df)
        self.prompts = PromptsLoader(PROMPTS_CSV_PATH)._fetch_prompts_from_cache()
        self.image_metadata = ImageProcessor(APPLE_TEXTS_DB_PATH, IMAGES_METADATA_PATH, IMAGES_PATH).load_image_metadata()
        self.color_map = make_color_map(GRID_PLAYERS)

    def _make_generic_text_page(self, pdf, func, args, page_title):
        """
        This function prepares a generic page that includes the output from the function being executed.
        Handles large outputs by creating multiple pages if necessary, with tighter margins.
        """
        output = func(*args)
        if output is None:
            print(f"Warning: {func.__name__} returned None")
            output = "No data available."

        # Check if output is a DataFrame (table)
        if isinstance(output, pd.DataFrame):
            MAX_ROWS_PER_PAGE = 45  # Maximum rows to display per page
            total_pages = math.ceil(len(output) / MAX_ROWS_PER_PAGE)

            for page_num in range(total_pages):
                # Get the rows for the current page
                start_row = page_num * MAX_ROWS_PER_PAGE
                end_row = start_row + MAX_ROWS_PER_PAGE
                page_df = output.iloc[start_row:end_row]

                # Create a new figure for the page
                fig, ax = plt.subplots(figsize=(8.5, 11))  # Standard letter size
                ax.axis('off')

                # Add title
                if total_pages > 1:
                    ax.text(0.5, 1, f"{page_title} (Page {page_num + 1}/{total_pages})",
                            fontsize=14, ha='center', va='top', fontweight='bold')
                else:
                    ax.text(0.5, 1, page_title,
                            fontsize=14, ha='center', va='top', fontweight='bold')

                # Render the DataFrame as a table with adjusted column width
                table = ax.table(
                    cellText=page_df.values,
                    colLabels=page_df.columns,
                    cellLoc='center',
                    loc='center'
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)

                # Adjust column widths
                n_cols = len(page_df.columns)
                column_width = min(1.0 / n_cols, 0.1)  # Adjust this value for narrower columns
                for i in range(n_cols):
                    table.auto_set_column_width([i])  # Auto-resize column width
                    table[0, i].set_width(column_width)  # Explicitly set width for each column

                # Apply a green background to the header row and format text
                for (row, col), cell in table.get_celld().items():
                    if row == 0:  # Header row
                        # Format header text: Title case and replace underscores with spaces
                        cell_text = cell.get_text().get_text()
                        formatted_text = cell_text.replace("_", " ").title()
                        cell.get_text().set_text(formatted_text)      

                        # Apply styling
                        cell.set_facecolor("#32CD32")  # Lime Green color
                        cell.set_text_props(weight='bold', color='white')  # White text, bold font

                    # Check if the row contains "Consensus" and format it
                    else:
                        # Extract the cell's text content
                        cell_text = cell.get_text().get_text()

                        # Check if the cell belongs to a row containing "Consensus"
                        if "Consensus" in table[row, 0].get_text().get_text():  # Check the first column of the row
                            cell.set_text_props(weight='bold')  # Bold text for all cells in the "Consensus" row

                # Adjust layout
                plt.tight_layout()

                # Save the page to the PDF
                pdf.savefig(fig)
                plt.close(fig)

        else:
            MAX_LINES_PER_NORMAL_PAGE = 45

            # Split the output into lines
            lines = output.split('\n')
            total_pages = math.ceil(len(lines) / MAX_LINES_PER_NORMAL_PAGE)

            for page_num in range(total_pages):
                # Get the lines for the current page
                start_line = page_num * MAX_LINES_PER_NORMAL_PAGE
                end_line = start_line + MAX_LINES_PER_NORMAL_PAGE
                page_content = '\n'.join(lines[start_line:end_line])

                # Create a new figure for the page
                fig, ax = plt.subplots(figsize=(8, 11))  # Standard letter size
                ax.axis('off')

                # Adjust margins and spacing
                title_y_pos = 0.98  # Slightly below the top edge
                content_y_start = 0.92  # Start content closer to the title
                line_spacing = 0.02  # Decrease line spacing for tighter content placement

                # Add title
                if total_pages > 1:
                    ax.text(0.5, title_y_pos, f"{page_title} (Page {page_num + 1}/{total_pages})",
                            fontsize=14, ha='center', va='top', fontweight='bold')
                else:
                    ax.text(0.5, title_y_pos, page_title,
                            fontsize=14, ha='center', va='top', fontweight='bold')

                # Add content
                for i, line in enumerate(page_content.split('\n')):
                    line_y_pos = content_y_start - i * line_spacing
                    ax.text(-0.05, line_y_pos, line, fontsize=8, ha='left', va='top')

                # Save the page to the PDF
                pdf.savefig(fig)
                plt.close(fig)

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
            print("*" * 80)
            print(f"Running {func.__name__}")
            if func == self._make_generic_text_page:
                # Pass the pdf object to _make_generic_text_page
                self._make_generic_text_page(pdf, *args)
            else:
                plt.figure()
                func(*args)
                pdf.savefig()
                plt.close()

    def prepare_graph_functions(self):
        """
        Prepares a list of graph functions to be executed later during PDF generation.
        """
        # Prepare necessary structures
        categories = build_category_structure(self.texts, self.prompts)
        person_to_category = build_person_category_structure(self.texts, self.prompts, categories)
        categories_clearing_threshold = get_category_clearing_threshold(categories, person_to_category)
        person_to_type = get_person_to_type(self.texts, self.prompts, person_to_category)

        # Helper to create graph function tuples
        def make_graph_function(func, args, title):
            return (func, args, title)

        # Core graph functions
        core_graphs = [
            ("High Level Summary Metrics", plot_summary_metrics, (self.texts, self.color_map)),
            ("Correctness Distribution", plot_correctness, (self.texts, self.color_map)),
            ("Smoothed Scores Over Time", plot_smoothed_metrics, (self.texts, 'smoothed_score', "Smoothed Scores Over Time", "Smoothed Score", self.color_map)),
            ("Smoothed Correct Over Time", plot_smoothed_metrics, (self.texts, 'smoothed_correct', "Smoothed Correct Over Time", "Smoothed Correct", self.color_map)),
            ("Smoothed Avg Score of Correct Over Time", plot_smoothed_metrics, (self.texts, 'smoothed_avg_score', "Smoothed Avg Score of Correct Over Time", "Smoothed Avg Score of Correct", self.color_map)),
            ("Win Rates", plot_win_rates, (self.texts, self.color_map)),
            ("Best and Worst Scores (All Time)", plot_best_worst_scores, (self.texts,)),
            ("Best and Worst Scores (Last 30 Days)", plot_best_worst_scores_30, (self.texts,)),
            ("Top Grids of All Time", plot_top_n_grids, (self.image_metadata, self.texts, 10))
        ]

        graph_functions = [make_graph_function(func, args, title) for title, func, args in core_graphs]

        # Generic text page functions
        generic_pages = [
            ("Type Performance Overview", analyze_person_type_performance, (person_to_type,)),
            ("Category Performance Overview", analyze_person_to_category, (person_to_category,)),
            ("Easiest Teams Overview (Consensus)", analyze_team_performance, (categories, person_to_category)),
            ("Best/Worst Team Overview (Individual)", analyze_person_prompt_performance, (categories, person_to_category, categories_clearing_threshold, "Team")),
            ("Best/Worst Category Overview (Individual)", analyze_person_prompt_performance, (categories, person_to_category, categories_clearing_threshold, "Category")),
            ("Hardest Teams for Team-Team Intersections", analyze_hardest_intersections, (self.texts, self.prompts, "team")),
            ("Hardest Teams for Team-Stat Intersections", analyze_hardest_intersections, (self.texts, self.prompts, "stat")),
            ("Top Players Used", analyze_top_players_by_submitter, (self.image_metadata, 25)),
            ("Bans and Saves", analyze_grid_cell_with_shared_guesses, (self.image_metadata, self.prompts, GRID_PLAYERS_RESTRICTED)),
            ("Popular Players by Month", analyze_top_players_by_month, (self.image_metadata, 5)),
        ]

        graph_functions += [
            make_graph_function(self._make_generic_text_page, (func, args, title), title)
            for title, func, args in generic_pages
        ]

        # Person-specific pages
        person_specific_templates = [
            ("Most Successful Intersections", analyze_most_successful_exact_intersections, (self.texts, self.prompts, None)),
            ("Never Shown Intersections", analyze_empty_team_team_intersections, (self.texts, self.prompts, None, categories)),
        ]

        person_specific_pages = [
            make_graph_function(
                self._make_generic_text_page,
                (
                    func,
                    tuple(arg if arg is not None else person for arg in args),
                    f"{title} ({person})"
                ),
                f"{title} ({person})"
            )
            for title, func, args in person_specific_templates
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