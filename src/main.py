import datetime
from data.messages_loader import MessagesLoader
from data.prompts_loader import PromptsLoader
from data.image_processor import ImageProcessor
from data.data_prep import create_disaggregated_results_df
from data.mlb_reference import correct_typos_with_fuzzy_matching
from analysis.reporter import ReportGenerator
from utils.constants import (
    APPLE_TEXTS_DB_PATH, 
    MESSAGES_CSV_PATH, 
    PROMPTS_CSV_PATH, 
    PDF_FILENAME,
    IMAGES_METADATA_PATH, 
    IMAGES_PATH,
    IMAGES_METADATA_CSV_PATH,
    IMAGES_METADATA_FUZZY_LOG_PATH
)

def refresh_data(image_dates_to_parse=None):
    print("Starting the data refresh process...")

    # Refresh Messages
    print("Refreshing messages...")
    messages_loader = MessagesLoader(APPLE_TEXTS_DB_PATH, MESSAGES_CSV_PATH)
    messages_loader.load().validate()
    messages_data = messages_loader.get_data()
    print(f"Messages refresh complete. Total rows: {len(messages_data)}")
    
    # Refresh Prompts
    print("Refreshing prompts...")
    prompts_loader = PromptsLoader(PROMPTS_CSV_PATH)
    prompts_loader.load().validate()
    prompts_data = prompts_loader.get_data()
    print(f"Prompts refresh complete. Total rows: {len(prompts_data)}")

    # Refresh Images
    print("Refreshing images...")
    image_processor = ImageProcessor(APPLE_TEXTS_DB_PATH, IMAGES_METADATA_PATH, IMAGES_PATH)
    image_processor.process_images(image_dates_to_parse)
    image_metadata = image_processor.load_image_metadata()
    image_parser_data = image_processor.load_parser_metadata()

    disaggregated_results_df = create_disaggregated_results_df(image_metadata, prompts_data, messages_data)
    disaggregated_results_df, typo_log = correct_typos_with_fuzzy_matching(disaggregated_results_df, "response")
    disaggregated_results_df.to_csv(IMAGES_METADATA_CSV_PATH, index=False)
    typo_log.to_csv(IMAGES_METADATA_FUZZY_LOG_PATH, index=False)

    print("Data refresh process completed successfully.")
    return messages_data, prompts_data, image_metadata, image_parser_data

def main(*args):
    """
    Main function to execute the data refresh and report generation.
    
    Args:
        *args: Optional flag for whether to process images (True/False).
    """

    image_dates_to_parse = None

    if args and args[0].strip():
        val = args[0].strip().lower()
        if val == "false":
            image_dates_to_parse = False
        else:
            # Split, strip, and filter out empty segments
            candidate_dates = [d.strip() for d in args[0].split(",") if d.strip()]
            valid_dates = []
            for d in candidate_dates:
                try:
                    # Validate format; will raise ValueError if wrong
                    datetime.datetime.strptime(d, "%Y-%m-%d")
                    valid_dates.append(d)
                except ValueError:
                    print(f"Invalid date format '{d}'. Please use YYYY-MM-DD.")
                    sys.exit(1)

            # If after filtering we have at least one date, keep it; otherwise None
            image_dates_to_parse = valid_dates if valid_dates else None

    # If args is empty or blank, image_dates_to_parse stays None
    print(f"Image dates to parse: {image_dates_to_parse}")

    # Step 1: Refresh data
    messages_data, prompts_data, image_metadata, image_parser_data = refresh_data(image_dates_to_parse)

    # Step 2: Generate report
    generator = ReportGenerator(PDF_FILENAME)
    
    generator.load_data()
    generator.generate_report()

    print("All processes completed successfully!")

# Optional CLI entrypoint
if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
