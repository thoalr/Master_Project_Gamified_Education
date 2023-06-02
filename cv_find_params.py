import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pandas import DataFrame, Series

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit


# Load data
Data = pd.read_csv("Data_Processed.csv").drop("Unnamed: 0", axis=1)
print("\nData info:")
print(Data.info())
Data["session_date"] = pd.to_datetime(Data["session_date"])

print()


Y = (
    pd.read_csv("./Y_mean_duration.csv").drop("Unnamed: 0", axis=1)
    # .drop("userid", axis=1)
)
print("Y\n")
# print(Y.info())
print()

def split_data_by_date(x, Y, date):
    date = pd.to_datetime(date)
    x_train = x[(x["session_date"] <= date)]
    x_test = x[(x["session_date"] > date)]
    y_train = pd.merge(x_train["sessionid"], Y, how="left", on="sessionid")
    y_test = pd.merge(x_test["sessionid"], Y, how="left", on="sessionid")
    return x_train, y_train, x_test, y_test

def model_with_multiout(model, cv_params: dict[str:list], Data: DataFrame, Y: DataFrame, important_col):

    x_train, y_train, x_test, y_test = split_data_by_date(Data, Y, "2022-03-01")
    
    rs_model = RandomizedSearchCV(
            model,
            param_distributions=cv_params,
            n_iter=10,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            cv=TimeSeriesSplit(n_splits=4, test_size=1),
            verbose=4
        )
    
    # model fitting
    print("Tuning parameters")
    rs_model.fit(x_train.drop(["session_date", "userid", "sessionid"], axis=1), y_train.drop(["userid", "sessionid"], axis=1), 
                 eval_set=[(x_test.drop(["session_date", "userid", "sessionid"], axis=1), y_test.drop(["userid", "sessionid"], axis=1))])
    print("Best parameters")
    print(rs_model.best_params_)
    print("Best score:")
    print(rs_model.best_score_)



xg_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        # max_depth=6,
        # learning_rate=0.15,
        # gamma=0.05,
        tree_method="gpu_hist",
        # early_stopping=20
    )
params = {
        "learning_rate": [0.14, 0.145, 0.15, 0.16, 0.165, 0.17],
        "max_depth": [5, 6, 7, 8],
        # "min_child_weight": [1, 3, 4],
        "gamma": [0.02, 0.3, 0.4, 0.5, 0.05, 0.065, 0.08, 0.1],
        # "colsample_bytree": [0.2, 0.25, 0.3, 0.4, 0.45],
    }

model_with_multiout(xg_model, params, Data, Y, ["3", "44"])

# Best parameters
# {'max_depth': 8, 'learning_rate': 0.15, 'gamma': 0.3}
# Best score:
# -0.007545902685950045