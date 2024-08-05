import pandas as pd
import sqlite3
import re
from pydantic import BaseModel
import json, pickle

# --------------------------------------------------------------------------------------
# Global Variables
DB_PATH = '/Users/samarnesen/Library/Messages/chat.db'
OUTPUT_PATH ='./output.p'
MY_NAME = 'Sam'

# --------------------------------------------------------------------------------------
# Classes
class ImmaculateGridResult(BaseModel):
    correct: int
    score: int
    date: str
    matrix: list[list[bool]] = None
    text: str

# --------------------------------------------------------------------------------------
# Private Methods
def _convert_timestamp(ts):
  """
  Convert Apple default timestamp (ts) to human readable
  """
  apple_timestamp_seconds = ts / 1e9
  unix_timestamp_seconds = apple_timestamp_seconds + 978307200
  return pd.to_datetime(unix_timestamp_seconds, unit='s').date().strftime('%m/%d/%Y')

def _row_to_name(row):
    if row.is_from_me:
        return MY_NAME
    elif row.phone_number == "+17736776982":
        return "Sam"
    elif row.phone_number == "+17736776717":
        return "Will"
    elif row.phone_number == "+17734281342":
        return "Rachel"
    elif row.phone_number == "+19087311244":
        return "Keith"

def _is_valid_message(name, text):
    exclusion_keywords = [
        "Emphasized", "Laughed at", "Loved", 
        "Questioned", "Liked", "Disliked", "🏀"
    ]
    
    if name and text and "Rarity:" in text:
        if not any(keyword in text for keyword in exclusion_keywords):
            return True
    return False

# --------------------------------------------------------------------------------------
# Public Methods
def extract_messages(db_path):
  """
  Extract Message contents from .db file
  """
  conn = sqlite3.connect(db_path)
  query = '''
  SELECT
      message.rowid, 
      message.handle_id, 
      message.text, 
      message.date, 
      message.is_from_me, 
      handle.id as phone_number
  FROM 
      message 
  LEFT JOIN 
      handle 
  ON 
      message.handle_id = handle.rowid
  '''
  messages_df = pd.read_sql_query(query, conn)
  conn.close()
  return messages_df

def process_immaculate_grid_results(messages_df):
  texts = {}
  reversed = {}
  current_grid_number = 0
  for idx, row in messages_df[["text", "date", "phone_number", "is_from_me"]].iterrows():
      name = _row_to_name(row)
      if _is_valid_message(name, row.text):
          try:
              parsed = re.search(r"Immaculate Grid (\d+) (\d)/\d", row.text).groups()
          except Exception as e:
              continue
          grid_number = int(parsed[0])
          correct = int(parsed[1])
          score = int(re.search(r"Rarity: (\d{1,3})", row.text).groups()[0])
          date = _convert_timestamp(row.date)
  
          # get specific correctness
          matrix = []
          for text_row in row.text.split("\n"):
              current = []
              for char in text_row:
                  if ord(char) == 11036: #"⬜️":
                      current.append(False)
                  elif ord(char) == 129001: #"🟩":
                      current.append(True)
              if len(current) > 0:
                  if len(current) != 3:
                      print(row.text)
                      assert len(current) == 3
                  else:
                      matrix.append(current)
          assert len(matrix) == 3
  
          obj = ImmaculateGridResult(correct=correct, score=score, date=date, matrix=matrix, text=row.text)
          if name not in texts or grid_number not in texts[name] or (name in texts and grid_number in texts[name] and texts[name][grid_number].correct == correct):
              texts.setdefault(name, {}).setdefault(grid_number, obj)
              reversed.setdefault(grid_number, {}).setdefault(name, obj)
          if grid_number >= current_grid_number:
              current_grid_number = grid_number
  return texts

def pickle_dump(file_path, data):
    # Convert pydantic models to dicts
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def main():
    messages_df = extract_messages(DB_PATH)
    texts = process_immaculate_grid_results(messages_df)
    pickle_dump(OUTPUT_PATH, texts)

main()
