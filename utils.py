from datetime import datetime, timedelta
import json
from refresh_db import ImmaculateGridResult
import pandas as pd

class ImmaculateGridUtils:
    @staticmethod
    def df_to_immaculate_grid_objs(df):
        """
        Convert a DataFrame into a dictionary of ImmaculateGridResult objects grouped by name.
    
        Parameters:
            df (pd.DataFrame): The DataFrame containing the necessary columns to create ImmaculateGridResult objects.
    
        Returns:
            dict: A dictionary where the key is the player's name and the value is a list of ImmaculateGridResult objects.
        """
        results = {}
        for _, row in df.iterrows():
            try:
                # Extract values from the row
                grid_number = row['grid_number']
                correct = row['correct']
                score = row['score']
                date = row['date']
                matrix = json.loads(row['matrix']) if pd.notna(row['matrix']) else None
                name = row['name']
    
                # Add the result to the list for the specific player name
                result = ImmaculateGridResult(
                    grid_number=grid_number,
                    correct=correct,
                    score=score,
                    date=date,
                    matrix=matrix,
                    name=name
                )
    
                if name not in results:
                    results[name] = []
                results[name].append(result)
    
            except Exception as e:
                print(f"Error processing row: {row.to_dict()}\nField causing error: {e}")
                continue
    
        return results

    @staticmethod
    def _convert_timestamp(ts):
        """
        Convert Apple Messages timestamp to a human-readable date in YYYY-MM-DD format.
        
        Parameters:
            ts (int): The timestamp from Apple Messages database.
        
        Returns:
            str: The formatted date.
        """
        apple_timestamp_seconds = ts / 1e9
        unix_timestamp_seconds = apple_timestamp_seconds + 978307200
        return pd.to_datetime(unix_timestamp_seconds, unit='s').date().strftime('%Y-%m-%d')

    @staticmethod
    def _grid_number_from_text(text):
        try:
            match = re.search(r"Immaculate Grid (\d+) (\d)/\d", text)
            if not match:
                raise ValueError(f"No match found for text: '{text}'")
            else:
                parsed = match.groups()
                return int(parsed[0])
        except ValueError as e:
            print(e)
            print(text)
            return None

    @staticmethod
    def _fixed_date_from_grid_number(n):
        start_date = datetime(2023, 4, 2) # hardcoded start day of immaculate grid universe
        result_date = start_date + timedelta(days=n)
        return result_date.strftime('%Y-%m-%d')

    @staticmethod
    def _correct_from_text(text):
        parsed = re.search(r"Immaculate Grid (\d+) (\d)/\d", text).groups()
        return int(parsed[1])

    @staticmethod
    def _score_from_text(text):
        return int(re.search(r"Rarity: (\d{1,3})", text).groups()[0])

    @staticmethod
    def _matrix_from_text(text):
        """
        Extract matrix from the raw text of a message
        """
        matrix = []
        for text_row in text.split("\n"):
            current = []
            for char in text_row:
                if ord(char) == 11036:  # "⬜️":
                    current.append(False)
                elif ord(char) == 129001:  # "🟩":
                    current.append(True)
            if len(current) > 0:
                if len(current) != 3:
                    print(row.text)
                    assert len(current) == 3
                else:
                    matrix.append(current)
        assert len(matrix) == 3
        matrix = str(matrix).lower()
        return matrix

    @staticmethod
    def _row_to_name(phone_number, is_from_me):
        """
        Map phone numbers to known names or the user's own name.
        
        Parameters:
            phone_number (str): The phone number from the messages dataframe.
            is_from_me (bool): Flag indicating if the message is from the user.
        
        Returns:
            str: The name of the sender or recipient.
        """
        
        if is_from_me:
            return MY_NAME
        for name, details in GRID_PLAYERS.items():
            if details["phone_number"] == phone_number:
                return name
        return "Unknown"

    @staticmethod
    def _is_valid_message(name, text):
        """
        Validate messages based on specific content and exclusion criteria.
        
        Parameters:
            name (str): The name of the sender.
            text (str): The message text content.
        
        Returns:
            bool: True if the message is valid, False otherwise.
        """
        exclusion_keywords = [
            "Emphasized", "Laughed at", "Loved", 
            "Questioned", "Liked", "Disliked", "🏀"
        ]
    
        if text is not None:
            if name is not None:
                # Is not a reaction message
                if not any(keyword in text for keyword in exclusion_keywords):
                    # Has rarity
                    if "Rarity: " in text:                    
                        # Has the proper immaculate grid format
                        if "Immaculate Grid " in text:
                            return True
        return False