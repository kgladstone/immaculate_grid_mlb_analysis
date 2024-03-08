# Immaculate Grid Prompt Extractor
import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime

def get_today_grid_id():
    # The Immaculate Grid started on April 3, 2023
    # Calculate the number of days from April 3, 2023, to today
    today = datetime.now()
    START_DATE = datetime(2023, 4, 2)
    # Calculate the difference in days
    grid_id_today = (today - START_DATE).days
    return grid_id_today

def get_grid_as_list(grid_id):
    print("Attempting to pull Immaculate Grid #{}".format(str(grid_id)))
    url = "https://www.immaculategrid.com/grid-" + str(grid_id)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    buttons_with_aria_label = soup.find_all("button", attrs={'aria-label': True})
    list_of_labels = list()
    list_of_labels.append(str(grid_id))
    for button in buttons_with_aria_label:
        list_of_labels.append(button['aria-label'])
    return list_of_labels

def write_lists_of_lists_csv(list_of_lists, output_fn):
    with open(output_fn, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(list_of_lists)
    print(f'Data written to {output_fn}')
    return

def main():
    grid_data = list()
    today_grid_id = get_today_grid_id()
    for i in range(1, today_grid_id + 1):
        grid_data.append(get_grid_as_list(i))
    write_lists_of_lists_csv(grid_data, "immaculate_grid_prompt_data.csv")
    
main()
