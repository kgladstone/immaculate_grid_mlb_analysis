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
    build_category_structure
)
from analysis.analysis import (
    analyze_person_type_performance, analyze_team_performance, 
    analyze_person_prompt_performance, analyze_hardest_intersections, 
    analyze_most_successful_exact_intersections, analyze_empty_team_team_intersections, 
    plot_summary_metrics, plot_correctness, plot_smoothed_metrics, plot_win_rates,
    plot_best_worst_scores, plot_best_worst_scores_30,
    analyze_top_players_by_submitter,
    analyze_grid_cell_with_shared_guesses,
    analyze_top_players_by_month,
    plot_top_n_grids,
    analyze_person_to_category,
    analyze_submitter_specific_players,
    get_favorite_player_by_attribute,
    get_players_used_for_most_teams,
    analyze_immaculate_streaks,
    analyze_splits,
    analyze_everyone_missed,
    analyze_all_used_on_same_day,
    analyze_illegal_uses,
    analyze_new_players_per_month
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
        image_processor = ImageProcessor(APPLE_TEXTS_DB_PATH, IMAGES_METADATA_PATH, IMAGES_PATH)
        self.image_metadata, _ = image_processor.correct_typos_with_fuzzy_matching()
        self.color_map = make_color_map(GRID_PLAYERS)

    def _wrap_text(self, value, max_line_length=25):
        """
        Wrap text for a given value to ensure it doesn't exceed max_line_length.

        Args:
            value: The value to wrap (can be string or other types).
            max_line_length (int): Maximum characters in a single line.

        Returns:
            str: Wrapped text or the original value if not a string.
        """
        if isinstance(value, str) and len(value) > max_line_length:
            return '\n'.join([value[i:i + max_line_length] for i in range(0, len(value), max_line_length)])
        return value
    
    def _calculate_dynamic_max_line_length(self, page_df):
        """
        Dynamically calculate an appropriate max line length based on the table structure.

        Args:
            page_df (pd.DataFrame): The DataFrame for the current page.

        Returns:
            int: Calculated max line length.
        """
        total_columns = len(page_df.columns)
        avg_chars_per_cell = page_df.applymap(lambda val: len(str(val))).values.mean()

        # Adjust max line length based on column density and average characters per cell
        # Fewer columns + lower density = longer lines, More columns + higher density = shorter lines
        max_line_length = int(150 / total_columns * (50 / avg_chars_per_cell))
        return max(10, min(200, max_line_length))  # Ensure bounds for reasonable wrapping

    def _make_generic_text_page(self, pdf, func, args, page_title):
        """
        This function prepares a generic page that includes the output from the function being executed.
        Handles large outputs by creating multiple pages if necessary, with tighter margins.
        """

        output = func(*args)
        if output is None:
            print(f"Warning: {func.__name__} returned None")
            output = "No data available."

        if isinstance(output, pd.DataFrame):  # Handle table (DataFrame) output
            MAX_ROWS_PER_PAGE = 45
            # Compute the maximum number of lines in any cell.
            overall_max_newlines = max(
                output[column].astype(str).str.count('\n').max() for column in output.columns
            )
            max_lines = overall_max_newlines + 1  # convert newline count to total number of lines

            # Subtract header height (assume header takes ~max_lines lines) from MAX_ROWS_PER_PAGE.
            max_rows_per_page_adjusted = math.floor((MAX_ROWS_PER_PAGE - max_lines) / max_lines)
            total_rows = len(output)
            total_pages = math.ceil(total_rows / max_rows_per_page_adjusted)

            for page_num in range(total_pages):
                start_row = page_num * max_rows_per_page_adjusted
                end_row = start_row + max_rows_per_page_adjusted
                page_df = output.iloc[start_row:end_row]

                max_line_length = self._calculate_dynamic_max_line_length(page_df)
                page_df = page_df.applymap(lambda value: self._wrap_text(value, max_line_length=max_line_length))

                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                # Place title at the top.
                ax.text(
                    0.5, 0.98,
                    f"{page_title} (Page {page_num + 1}/{total_pages})" if total_pages > 1 else page_title,
                    fontsize=14, ha='center', va='top', fontweight='bold'
                )

                # Use a bounding box to place the table at the top (below the title).
                # bbox = [left, bottom, width, height] in axes coordinates.
                # Here, we set bottom=0.15 and height=0.75 so that the table spans from y=0.15 to y=0.90.
                table = ax.table(
                    cellText=page_df.values,
                    colLabels=page_df.columns,
                    cellLoc='center',
                    bbox=[0, 0.15, 1, 0.75]
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)

                col_widths = [max(len(str(val)) for val in page_df[col]) for col in page_df.columns]
                total_width = sum(col_widths)
                for i, col in enumerate(page_df.columns):
                    table.auto_set_column_width([i])
                    width_ratio = col_widths[i] / total_width
                    table[0, i].set_width(min(width_ratio, 0.2))

                row_line_counts = []
                for row in range(len(page_df) + 1):  # +1 to account for header
                    line_counts = [
                        cell.get_text().get_text().count('\n') + 1
                        for (cell_row, cell_col), cell in table.get_celld().items()
                        if cell_row == row
                    ]
                    row_line_counts.append(max(line_counts, default=1))

                scaling_factor = 0.015
                for (row, col), cell in table.get_celld().items():
                    if row == 0:
                        cell_text = cell.get_text().get_text()
                        formatted_text = cell_text.replace("_", " ").title()
                        cell.get_text().set_text(formatted_text)
                        cell.set_facecolor("#32CD32")
                        cell.set_text_props(weight='bold', color='white')
                        cell.set_height(scaling_factor)
                    else:
                        cell.set_height(scaling_factor * row_line_counts[row])
                    if row > 0 and "Consensus" in str(page_df.iloc[row - 1, 0]):
                        cell.set_text_props(weight='bold')
                    if row > 0 and "true" in cell.get_text().get_text().lower():
                        cell.set_facecolor("#69ffb4")
                        cell.set_text_props(weight='bold')
                    if row > 0 and "false" in cell.get_text().get_text().lower():
                        cell.set_facecolor("#ff6969")
                        cell.set_text_props(weight='bold')

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        else:
            MAX_LINES_PER_NORMAL_PAGE = 45

            # Split the output into lines
            lines = output.split('\n')
            total_pages = math.ceil(len(lines) / MAX_LINES_PER_NORMAL_PAGE)

            for page_num in range(total_pages):
                start_line = page_num * MAX_LINES_PER_NORMAL_PAGE
                end_line = start_line + MAX_LINES_PER_NORMAL_PAGE
                page_content = '\n'.join(lines[start_line:end_line])

                fig, ax = plt.subplots(figsize=(8, 11))
                ax.axis('off')

                title_y_pos = 0.98
                content_y_start = 0.92
                line_spacing = 0.02

                if total_pages > 1:
                    ax.text(0.5, title_y_pos, f"{page_title} (Page {page_num + 1}/{total_pages})",
                            fontsize=14, ha='center', va='top', fontweight='bold')
                else:
                    ax.text(0.5, title_y_pos, page_title,
                            fontsize=14, ha='center', va='top', fontweight='bold')

                for i, line in enumerate(page_content.split('\n')):
                    line_y_pos = content_y_start - i * line_spacing
                    ax.text(-0.05, line_y_pos, line, fontsize=8, ha='left', va='top')

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
                print(f"Specific function: {args[0].__name__}")
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
            ("Top Grids of All Time", plot_top_n_grids, (self.image_metadata, self.texts, 10)),
            ("[NEW] First Time Uses Over Time", analyze_new_players_per_month, (self.image_metadata, self.color_map,)),
        ]

        graph_functions = [make_graph_function(func, args, title) for title, func, args in core_graphs]

        # Generic text page functions
        generic_pages = [
            ("Type Performance Overview", analyze_person_type_performance, (self.texts, self.prompts, categories,)),
            ("Category Performance Overview", analyze_person_to_category, (self.texts, self.prompts, categories,)),
            ("Easiest Teams Overview (Consensus)", analyze_team_performance, (self.texts, self.prompts, categories)),
            ("Best/Worst Team Overview (Individual)", analyze_person_prompt_performance, (self.texts, self.prompts, categories, "Team")),
            ("Best/Worst Category Overview (Individual)", analyze_person_prompt_performance, (self.texts, self.prompts, categories, "Category")),
            ("Hardest Teams for Team-Team Intersections", analyze_hardest_intersections, (self.texts, self.prompts, "team")),
            ("Hardest Teams for Team-Stat Intersections", analyze_hardest_intersections, (self.texts, self.prompts, "stat")),
            ("Top Players Used", analyze_top_players_by_submitter, (self.image_metadata, 40)),
            ("Our Personal Favorites", analyze_submitter_specific_players, (self.image_metadata, 15)),
            ("Our Favorite Players by Team", get_favorite_player_by_attribute, (self.image_metadata, self.prompts, 'team', 0)),
            ("Our Favorite Players by Supercategory", get_favorite_player_by_attribute, (self.image_metadata, self.prompts, 'supercategory', 0)),
            ("Our Favorite Players by Position", get_favorite_player_by_attribute, (self.image_metadata, self.prompts, 'position', 0)),
            ("Players Used for Most Teams", get_players_used_for_most_teams, (self.image_metadata, self.prompts, 40, 0)),
            ("Bans and Saves", analyze_grid_cell_with_shared_guesses, (self.image_metadata, self.prompts, GRID_PLAYERS_RESTRICTED)),
            ("[NEW] Immaculate Streaks", analyze_immaculate_streaks, (self.texts,)),
            ("[NEW] Splits", analyze_splits, (self.image_metadata, self.prompts, GRID_PLAYERS_RESTRICTED)),
            ("[NEW] Everyone Missed", analyze_everyone_missed, (self.texts, self.prompts, GRID_PLAYERS_RESTRICTED)),
            ("[NEW] All Used on Same Day", analyze_all_used_on_same_day, (self.image_metadata, self.prompts, GRID_PLAYERS_RESTRICTED)),
            ("[NEW] Illegal Uses", analyze_illegal_uses, (self.image_metadata, self.prompts, GRID_PLAYERS_RESTRICTED)),
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