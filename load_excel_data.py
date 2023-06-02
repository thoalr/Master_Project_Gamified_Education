# Imports

import numpy as np
import pandas as pd


# load data
xls = pd.ExcelFile("../not code/Data/BDD_Thorsten - Copy.xlsx")#, engine="openpyxl")
games = pd.read_excel(xls, "Games", engine="openpyxl")
organic_players = pd.read_excel(xls, "Organic players")
campaign_players = pd.read_excel(xls, "Campaign players")
sessions_org = pd.read_excel(xls, "Sessions_organic")
sessions_camp = pd.read_excel(xls, "Sessions_campaign")

RenameMapping = {
    "GameName": "gamen_name",
    "Game Id": "gameid",
    "Game Subject": "subject",
    "Game mechanics": "game_mechanic",
    "School Level (junior, middle, high)": "school_level",
    "Registration date & time": "time",
    "Accumulated minutes per session": "duration",
    "Learnig Index per session": "learn_index",
    "Game Recomendations after session (6)": "game_recommend",
    "Click on recommendation": "click_recommend",
    "User Id": "userid",
    "Gender": "gender",
    "Age": "age",
    "Session Id": "sessionid",
    "Session login": "session_date",
    "Session logout": "session_logout_date",
}

games = games.rename(columns=RenameMapping)
organic_players = organic_players.rename(columns=RenameMapping)
campaign_players = campaign_players.rename(columns=RenameMapping)
sessions_org = sessions_org.rename(columns=RenameMapping)
sessions_camp = sessions_camp.rename(columns=RenameMapping)

# Create unions
players_total = pd.concat(
    [campaign_players, organic_players], ignore_index=True
)  # .drop_duplicates()
sessions_total = pd.concat(
    [sessions_camp, sessions_org], ignore_index=True
)  # .drop_duplicates()


# remove rows from sessions_total where game_id == 48. Because 48 is not part of the games list

print("After removing game id 48 the difference between the two sets of games is")
_before_games = sessions_total["gameid"].unique()
sessions_total = sessions_total[sessions_total["gameid"] != 48]

# print(_before_games not in sessions_total["gameid"].unique())

print(
    set(_before_games) - set(sessions_total["gameid"].unique())
)  # difference between two sets
