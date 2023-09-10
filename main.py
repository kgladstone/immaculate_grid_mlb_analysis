#-----------------------------------------------------------------------------------------------------------------------
# KG Immaculate Grid MLB
# My attempt at replicating the game Immaculate Grid (MLB)
# Filename: main.py
# Data source: pybaseball (public GitHub repo pulling from baseball reference)
#-----------------------------------------------------------------------------------------------------------------------
import pybaseball as pb
import pandas as pd
import unidecode
import random

def initialize_data():
    # Build team master data
    # primary key: franchID
    team_master = pd.DataFrame()
    THIS_YEAR = 2023
    
    print("(1) Loading team master data...")
    for i in range(1876,THIS_YEAR+1):
        teams_i = pb.team_ids(i)
        team_master = pd.concat([team_master, teams_i])
    team_master = team_master[["yearID","teamID", "franchID"]].drop_duplicates()
    print("Team master data loaded!")
        
    # Build player master data
    print("(2) Loading player master data...")
    player_master = pb.chadwick_register()
    player_master.name_last = player_master['name_last'].apply(lambda x: unidecode.unidecode(str(x)))
    player_master.name_first = player_master['name_first'].apply(lambda x: unidecode.unidecode(str(x)))
    print("Player master data loaded!")
    
    # Build player appearances master data
    print("(3) Loading player appearances master data...")
    appearances_master = pb.lahman.appearances()
    print("Player appearances master data loaded!")

    return(team_master, player_master, appearances_master)

#-----------------------------------------------------------------------------------------------------------------------

# Data Extraction
# TODO/Issues: 
# --- players with duplicate names
# --- no way to show players with fuzzy match names, need exact spelling
# --- data stops at 2021
def get_player_id(last, first):
    row = player_master.loc[(player_master['name_last'] == last.title()) & (player_master['name_first'] == first.title())]
    if len(row) > 1:
          raise Exception("Sorry, no players with duplicate names")
    return(row.key_bbref.iloc[0])

def get_appearances_from_player_id(player_id):
    rows = appearances_master.loc[appearances_master['playerID'] == player_id]
    return(rows)
    
def get_appearances_from_player(last, first):
    player_id = get_player_id(last, first)
    appearances = get_appearances_from_player_id(player_id)
    return(appearances)

def get_franchises_from_player(last, first):
    appearances = get_appearances_from_player(last, first)
    appearances = pd.DataFrame(appearances)
    teams = pd.merge(appearances, team_master, on=["teamID", "yearID"], how="left")["franchID"].unique()
    return(teams)

# Check Answers - Team Intersection
def is_player_in_db(last, first):
    row = player_master.loc[(player_master['name_last'] == last.title()) & (player_master['name_first'] == first.title())]
    return (len(row) != 0)
        
def is_player_team_intersection(team1, team2, last, first):
    if not is_player_in_db(last, first):
        print("Sorry, player does not exist in database (with this spelling of name)")
        return False
    team1 = team1.upper()
    team2 = team2.upper()
    player_franchises = get_franchises_from_player(last, first)
    return (team1 in player_franchises and team2 in player_franchises)

# Gameplay
# Team intersection only
def play_one_square():
    # Initialize random square
    MAX_YEAR = 2021
    current_teams = list(team_master.loc[team_master['yearID'] == MAX_YEAR]["franchID"])
    teams = random.sample(current_teams, 2)
    
    # User interface
    print("Choose a player who has made appearances in these two teams:")
    print(teams)
    first = input("Enter first name: ")
    last = input("Enter last name: ")
    check = is_player_team_intersection(teams[0], teams[1], last, first)
    if check:
        print("Yes! You got it!")
    else:
        print("Nope! Try again")

def main():
    team_master, player_master, appearances_master = initialize_data()
    while True:
        play_one_square()
        play_again = input("Play again? Type Y to continue: ")
        if play_again.upper() != "Y":
            break
