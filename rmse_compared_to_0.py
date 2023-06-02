
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pandas import DataFrame, Series


def smape(actual, pred):
    # return round(
    #     np.mean(
    #         np.abs(pred - actual) / 
    #         ((np.abs(pred) + np.abs(actual))/2)
    #     )*100, 2 )

    return np.mean(100/len(actual) * np.sum(2 * np.abs(pred - actual) / (np.abs(actual) + np.abs(pred))))

Y = (
    pd.read_csv("./Y_mean_learn_index.csv").drop("Unnamed: 0", axis=1)
    # .drop("userid", axis=1)
)
Y_duration = (
    pd.read_csv("./Y_mean_duration.csv").drop("Unnamed: 0", axis=1)
    # .drop("userid", axis=1)
)

print("Y types", list(Y_duration.dtypes))


zero_series = Series([0] * len(Y))

_y = Y.drop(["userid", "sessionid"], axis=1)
_y_duration = Y_duration.drop(["userid", "sessionid"], axis=1)

zero_df = DataFrame({g:zero_series for g in _y.columns})

rmse = mean_squared_error(_y, zero_df, multioutput='raw_values', squared=False)
# rmse_dict[c] = rmse
# print("For game id: ", c, " RMSE: ", (rmse))
print("Zero learn_index RMSE:")
print(rmse)
print("Average rmse:", np.mean(rmse))
print("Avergae MAE:", mean_absolute_error(_y, zero_df))
print("Avergae MAPE:", mean_absolute_percentage_error(_y, zero_df))
print("Avergae SMAPE:", smape(_y, zero_df))

rmse = mean_squared_error(_y_duration, zero_df, multioutput='raw_values', squared=False)
# rmse_dict[c] = rmse
# print("For game id: ", c, " RMSE: ", (rmse))
print("Zero duration RMSE:")
print(rmse)
print("Average rmse:", np.mean(rmse))
print("Avergae MAE:", mean_absolute_error(_y_duration, zero_df))
print("Avergae MAPE:", mean_absolute_percentage_error(_y_duration, zero_df))
print("Avergae SMAPE:", smape(_y_duration, zero_df))


# compared to manual model
from load_excel_data import sessions_total, games, players_total

list_games = [1,3,7,8,9,10,11,12,14,15,16,17,18,19,20,22,23,25,27,28,34,35,36,37,38,39,40,41,42,43,44,45,46,53,54,56,57,58,59,60,61,62,63,64,65,66,67,68,70,71,72,74,75,76,77,78,79,82,83,84,85,86,87,89,90,91,92,93,94,95,96,98,99]
list_games = [str(x) for x in list_games]


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
# _d = _d.fillna(method="ffill").fillna(method="bfill")

def split_data_by_date(x, Y, date):
    date = pd.to_datetime(date)
    x_train = x[(x["session_date"] <= date)]
    x_test = x[(x["session_date"] > date)]
    y_train = pd.merge(x_train["sessionid"], Y, how="left", on="sessionid")
    y_test = pd.merge(x_test["sessionid"], Y, how="left", on="sessionid")
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = split_data_by_date(sessions_total, Y, "2021-09-01")
_y = pd.merge(_d["sessionid"], y_test, on="sessionid", how="inner") # .drop(["userid", "sessionid"], axis=1)

x_train, y_train, x_test, y_test = split_data_by_date(sessions_total, Y_duration, "2021-09-01")
_y_duration = pd.merge(_d["sessionid"], y_test, on="sessionid", how="inner").drop(["userid", "sessionid"], axis=1)

_d = _d[_d["sessionid"].isin(_y["sessionid"])]
_y = _y.drop(["userid", "sessionid"], axis=1)


_d["learn_index"] = _d["learn_index"] / 3
_d["duration"] = _d["duration"].clip(0, 30)

print("max y", _y.max().max(), "max model", _d["learn_index"].max().max())
print("max y", _y_duration.max().max(), "max model", _d["duration"].max().max())

print("Describe _d")
print(_d.describe())


rmse = mean_squared_error(_y, _d["learn_index"], multioutput='raw_values', squared=False)
# rmse_dict[c] = rmse
# print("For game id: ", c, " RMSE: ", (rmse))
print("Manual learn_index RMSE:")
print(rmse)
print("Average rmse:", np.mean(rmse))
print("Avergae MAE:", mean_absolute_error(_y, _d["learn_index"]))
print("Avergae MAPE:", mean_absolute_percentage_error(_y, _d["learn_index"]))
print("Avergae SMAPE:", smape(_y, _d["learn_index"]))


rmse = mean_squared_error(_y_duration, _d["duration"], multioutput='raw_values', squared=False)
# rmse_dict[c] = rmse
# print("For game id: ", c, " RMSE: ", (rmse))
print("Manual duration RMSE:")
print(rmse)
print("Average rmse:", np.mean(rmse))
print("Avergae MAE:", mean_absolute_error(_y_duration, _d["duration"]))
print("Avergae MAPE:", mean_absolute_percentage_error(_y_duration, _d["duration"]))
print("Avergae SMAPE:", smape(_y_duration, _d["duration"]))
