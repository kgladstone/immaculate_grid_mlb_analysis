# Immaculate Grid Prompt Extractor
import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime
import pandas as pd
import ast

# The Immaculate Grid started on April 3, 2023
# Extract the id of today's Immaculate Grid
def get_today_grid_id():
    # Calculate the number of days from April 3, 2023, to today
    today = datetime.now()
    START_DATE = datetime(2023, 4, 2)
    # Calculate the difference in days
    grid_id_today = (today - START_DATE).days
    return grid_id_today

# Function: part1 + part2 + part3 ==> (part1, part2 + part3)
def parse_raw_to_tuple(raw):
    split_result = raw.split(" + ", 1)
    trimmed_split_result = [part.strip() for part in split_result]
    if len(trimmed_split_result) == 1:
        trimmed_split_result.append('')
    return tuple(trimmed_split_result)

# Flattening the list of tuples and turning it into a set of each unique token
def _tokenize_tuples_list(tuples_list):
    return set(token for tuple_ in tuples_list for token in tuple_)

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
    print("Attempting to pull Immaculate Grid #{}".format(str(grid_id)))
    url = "https://www.immaculategrid.com/grid-" + str(grid_id)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    buttons_with_aria_label = soup.find_all("button", attrs={'aria-label': True})
    list_of_labels = list()
    list_of_labels.append(str(grid_id))
    for button in buttons_with_aria_label:
        button_content = button['aria-label']
        list_of_labels.append(parse_raw_to_tuple(button_content))
    return list_of_labels

def get_grids_as_df_online(min_index, max_index):
    header = ["index"] + [f"cell{i}" for i in range(1, 10)]
    grids_df = pd.DataFrame(columns=header)
    for i in range(min_index, max_index + 1):
        grids_row_i = pd.DataFrame([get_grid_as_list_online(i)], columns=header)
        grids_df = pd.concat([grids_df, grids_row_i], ignore_index=True, axis=0)
    return grids_df

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
    df = get_grids_as_df_online(1, today_grid_id)
    df.to_csv('immaculate_grid_prompt_data.csv', index=False)
    df = get_grids_as_df_csv("immaculate_grid_prompt_data.csv")
    tokenized_df = get_tokenized_df(df)
    freq_df = create_freq_df(tokenized_df)
    freq_df.to_csv('immaculate_grid_token_freq.csv', index=False)
     
main()
