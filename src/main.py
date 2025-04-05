from data.messages_loader import MessagesLoader
from data.prompts_loader import PromptsLoader
from data.image_processor import ImageProcessor
from analysis.reporter import ReportGenerator
from utils.constants import (
    APPLE_TEXTS_DB_PATH, 
    MESSAGES_CSV_PATH, 
    PROMPTS_CSV_PATH, 
    PDF_FILENAME,
    IMAGES_METADATA_PATH, 
    IMAGES_PATH
)

def refresh_data(do_image_process=False):
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
    if do_image_process:
        image_processor.process_images()
    image_metadata = image_processor.load_image_metadata()
    image_parser_data = image_processor.load_parser_metadata()

    print("Data refresh process completed successfully.")
    return messages_data, prompts_data, image_metadata, image_parser_data

def main(*args):
    """
    Main function to execute the data refresh and report generation.
    
    Args:
        *args: Optional flag for whether to process images (True/False).
    """
    # Handle optional image processing flag
    do_image_process = args[0] if args else False
    if isinstance(do_image_process, str):
        do_image_process = do_image_process.lower() in ("true", "1", "yes")

    print(f"Image processing flag: {do_image_process}")

    # Step 1: Refresh data
    messages_data, prompts_data, image_metadata, image_parser_data = refresh_data(do_image_process)

    # Step 2: Generate report
    generator = ReportGenerator(PDF_FILENAME)
    
    generator.load_data()
    generator.generate_report()

    print("All processes completed successfully!")

# Optional CLI entrypoint
if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
