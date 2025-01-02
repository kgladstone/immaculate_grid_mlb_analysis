from messages_loader import MessagesLoader
from prompts_loader import PromptsLoader
from make_report import ReportGenerator
from constants import APPLE_TEXTS_DB_PATH, MESSAGES_CSV_PATH, PROMPTS_CSV_PATH, PDF_FILENAME, GRID_PLAYERS

def refresh_data():
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

    print("Data refresh process completed successfully.")
    return messages_data, prompts_data

if __name__ == "__main__":
    # Step 1: Refresh data
    messages_data, prompts_data = refresh_data()

    # Step 2: Generate report
    generator = ReportGenerator(MESSAGES_CSV_PATH, PROMPTS_CSV_PATH, PDF_FILENAME)
    generator.load_data()
    generator.generate_report()

    print("All processes completed successfully!")
