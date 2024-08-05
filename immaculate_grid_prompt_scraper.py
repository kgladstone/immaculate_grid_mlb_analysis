# Immaculate Grid Prompt Extractor
import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime
import pandas as pd
import ast

IMM_GRID_START_DATE = datetime(2023, 4, 2)

def get_today_grid_id():
    """
    Calculate the Immaculate Grid ID for today.
    """
    today = datetime.now()
    return (today - IMM_GRID_START_DATE).days

def parse_raw_to_tuple(raw):
    """
    Parse a string in the format 'part1 + part2 + part3' into a tuple ('part1', 'part2 + part3').
    """
    parts = raw.split(" + ", 1)
    return tuple(part.strip() for part in parts) if len(parts) > 1 else (parts[0].strip(), '')

def _tokenize_tuples_list(tuples_list):
    """
    Flatten the list of tuples into a set of unique tokens
    """
    return set(token for tup in tuples_list for token in tup)

def get_tokenized_df(df):
    tokenized_df = pd.DataFrame(columns=['index', 'token'])
    for i, row in df.iterrows():
        token_dict = dict()
        token_dict['index'] = i
        token_dict['token'] = list(_tokenize_tuples_list(row[1:]))
        tokenized_df_i = pd.DataFrame(token_dict)
        tokenized_df = pd.concat([tokenized_df, tokenized_df_i], ignore_index=True, axis=0)
    return tokenized_df

def get_grid_as_list_online(grid_id):
    """
    Fetch the Immaculate Grid data from the web by grid ID.
    """
    print("Attempting to pull Immaculate Grid #{}".format(str(grid_id)))
    url = f"https://www.immaculategrid.com/grid-{grid_id}"
    response = requests.get(url)
    response.raise_for_status()  # Raises HTTPError for bad responses
    soup = BeautifulSoup(response.text, 'html.parser')
    buttons = soup.find_all("button", attrs={'aria-label': True})
    labels = [str(grid_id)]
    labels.extend(parse_raw_to_tuple(button['aria-label']) for button in buttons)
    return labels

def get_grids_as_df_online(min_index, max_index):
    """
    Fetch multiple Immaculate Grids and return as a DataFrame.
    """
    header = ["index"] + [f"cell{i}" for i in range(1, 10)]
    grids_data = [get_grid_as_list_online(i) for i in range(min_index, max_index + 1)]
    return pd.DataFrame(grids_data, columns=header)

def get_grids_as_df_csv(filename):
    df = pd.read_csv(filename)
    df_transformed = df.applymap(lambda cell: ast.literal_eval(cell) if isinstance(cell, str) else cell)
    return df_transformed

def create_freq_df(df):
    NUM_DAYS = df['index'].nunique()
    frequency_dict = {}
    for index, row in df.iterrows():
        token = row['token']
        if token in frequency_dict:
            frequency_dict[token] += 1
        else:
            frequency_dict[token] = 1
        
    df = pd.DataFrame(list(frequency_dict.items()), columns=['token', 'freq'])
    df['freq_pct'] = df['freq'].apply(lambda x: x / NUM_DAYS)

    df_sorted = df.sort_values(by='freq', ascending=False)        

    return df_sorted

def main():
    today_grid_id = get_today_grid_id()
    grids_df = get_grids_as_df_online(1, today_grid_id)
    grids_df.to_csv('immaculate_grid_prompt_data.csv', index=False)
    df = pd.read_csv('immaculate_grid_prompt_data.csv')
    df = df.applymap(lambda cell: ast.literal_eval(cell) if isinstance(cell, str) else cell)
    tokenized_df = get_tokenized_df(df)
    freq_df = create_freq_df(tokenized_df)
    freq_df.to_csv('immaculate_grid_token_freq.csv', index=False)
     
main()
