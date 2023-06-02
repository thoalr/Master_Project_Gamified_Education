
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pandas import DataFrame, Series
from load_excel_data import sessions_total, games, players_total
import matplotlib.pyplot as plt

list_games = [1,3,7,8,9,10,11,12,14,15,16,17,18,19,20,22,23,25,27,28,34,35,36,37,38,39,40,41,42,43,44,45,46,53,54,56,57,58,59,60,61,62,63,64,65,66,67,68,70,71,72,74,75,76,77,78,79,82,83,84,85,86,87,89,90,91,92,93,94,95,96,98,99]
list_games = [str(x) for x in list_games]

Data = pd.read_csv("Data_Processed.csv").drop("Unnamed: 0", axis=1)
print("\nData info:")
print(Data.info())

print()

df = players_total.groupby(["gender", "age"])

df = pd.merge(sessions_total, players_total, on="userid", how="left")
_player_copy = pd.merge(sessions_total, players_total[["userid", "age", "gender"]], on="userid", how="left").drop(columns=["gameid", "duration", "learn_index"])



df["duration"] = df["duration"].clip(0, 30) / 30

df = df.groupby(["gender", "age", "gameid"])[["learn_index", "duration"]].mean().reset_index()
df["gameid"] = df["gameid"].astype(str)


predt_y = pd.merge(df, _player_copy, how="inner", on=["gender", "age"]).drop(columns=["gender", "age", "userid", "session_date", "session_logout_date", "game_recommend", "click_recommend"])
print(predt_y.head())



_d = predt_y.pivot(index="sessionid", columns=["gameid"], values=["learn_index", "duration"]).reset_index()

print(_d.describe())
_d = _d.fillna(_d.mean())





from numpy.random import choice

def player_sim_plots(df:DataFrame, name="mean_player_sim", by=""):
    if by=="":
        by = "learn_index"
    else:
        name = name + "_duration"
        by = "duration"
    labels2index = dict(zip(list_games, range(len(list_games))))
    dict_of_labels = dict(zip(range(len(list_games)), list_games))
    output_highest_1 = dict.fromkeys(list_games, 0)
    output_highest_2 = dict.fromkeys(list_games, 0)
    output_weighted_1 = dict.fromkeys(list_games, 0)
    output_weighted_2 = dict.fromkeys(list_games, 0)
    for player in sessions_total.drop(columns=["click_recommend", "game_recommend", "learn_index", "duration", "session_date", "session_logout_date"])["userid"].unique():
        # print(player)
        # game_id = row["gameid"]
        _df = players_total[players_total["userid"]==player][["age", "gender"]]
        # print(_df["age"])
        age = _df["age"].iat[0]
        gender = _df["gender"].iat[0]
        # print("age:", age, "gender:", gender)
        values = df[(df["gender"]==gender) & (df["age"]==age)] # & (df["gameid"]==game_id)]
        # print(values)
        
        # by = "duration"
        learn_sorted = values.sort_values(by)
        values_unsorted = values[by]
        values_sorted = learn_sorted["gameid"]

        output_highest_1[values_sorted.iat[0]] = output_highest_1[values_sorted.iat[0]] + 1
        for r in values_sorted[:6]:
            output_highest_2[r] = output_highest_2[r] + 1
        norm_res = values_unsorted / (np.sum(values_unsorted))
        
        draw = choice(values["gameid"], 7, replace=False, p=(norm_res))
        
        output_weighted_1[draw[0]] = output_weighted_1[draw[0]] + 1
        for r in draw[:6]:
            output_weighted_2[r] = output_weighted_2[r] + 1

    df = Series(DataFrame(output_highest_1, index=[0]).transpose()[0])
    # print(df.head())
    plt.figure(figsize=(12, 3))
    ax = df.sort_values(ascending=False).plot.bar(figsize=(16, 4))
    plt.title(label="")
    plt.grid()
    ax.set_xlabel("Game Id's")
    ax.set_ylabel("Count")
    plt.savefig("./plots/" + name + "_highest" + "_plot1.png", bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 3))
    df = Series(DataFrame(output_highest_2, index=[0]).transpose()[0])
    ax = df.sort_values(ascending=False).plot.bar(figsize=(16, 4))#figsize=(16, 4))
    plt.title(label="")
    plt.grid()
    ax.set_xlabel("Game Id's")
    ax.set_ylabel("Count")
    plt.savefig("./plots/" + name + "_highest" + "_plot2.png", bbox_inches='tight')
    plt.close()

    df = Series(DataFrame(output_weighted_1, index=[0]).transpose()[0])
    plt.figure(figsize=(12, 3))
    ax=df.sort_values(ascending=False).plot.bar(figsize=(16, 4))#figsize=(16, 4))
    plt.title(label="")
    plt.grid()
    ax.set_xlabel("Game Id's")
    ax.set_ylabel("Count")
    plt.savefig("./plots/" + name + "_weighted" + "_plot1.png", bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12,3))
    df = Series(DataFrame(output_weighted_2, index=[0]).transpose()[0])
    ax = df.sort_values(ascending=False).plot.bar(figsize=(12, 3))#figsize=(16, 4))
    plt.title(label="")
    plt.grid()
    ax.set_xlabel("Game Id's")
    ax.set_ylabel("Count")
    plt.savefig("./plots/" + name + "_weighted" + "_plot2.png", bbox_inches='tight')
    plt.close()

player_sim_plots(df)
player_sim_plots(df, by="duration")
