import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pandas import DataFrame, Series

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit



# Load data
Data = pd.read_csv("Data_Processed.csv").drop("Unnamed: 0", axis=1)
print("\nData info:")
print(Data.info())
Data["session_date"] = pd.to_datetime(Data["session_date"])

print()


Y = (
    pd.read_csv("./Y_mean_learn_index.csv").drop("Unnamed: 0", axis=1)
    # .drop("userid", axis=1)
)
Y_duration = (
    pd.read_csv("./Y_mean_duration.csv").drop("Unnamed: 0", axis=1)
    # .drop("userid", axis=1)
)

def smape(actual, pred):
    # return round(
    #     np.mean(
    #         np.abs(pred - actual) / 
    #         ((np.abs(pred) + np.abs(actual))/2)
    #     )*100, 2 )

    return np.mean(100/len(actual) * np.sum(2 * np.abs(pred - actual) / (np.abs(actual) + np.abs(pred))))

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

def plot_only_first_place_rec(df: DataFrame, save_name):
    r2 = df.idxmax(axis=1)
    # print(r2.describe())
    # print(r2.info())
    # print(r2.head(2))
    #r2.hist(bins=73)
    plt.figure(figsize=(12,3))
    ax = r2.value_counts(sort=True).plot.bar()#figsize=(16, 4))
    
    
    plt.title(label="")
    plt.grid()
    ax.set_xlabel("Game Id's")
    ax.set_ylabel("Count")

    plt.savefig("./plots/" + save_name + "_plot1.png", bbox_inches='tight')
    plt.close()

def plot_all_rec(df:DataFrame, save_name):
    r3 = (df.rank(1)>(df.shape[1]-6)).astype(int) # https://stackoverflow.com/questions/65937496/find-top-n-values-in-row-of-a-dataframe-python
    # print(r3.describe())
    # print(r3.head(10))
    r3 = r3.sum(axis=0)#.transpose()
    # print(r3.describe())
    # print(r3.head(10))
    plt.figure(figsize=(12,3))
    ax = r3.sort_values(ascending=False).plot.bar()#figsize=(16, 4))
    plt.title(label="")
    plt.grid()
    ax.set_xlabel("Game Id's")
    ax.set_ylabel("Count")
    
    plt.savefig("./plots/" + save_name + "_plot2.png", bbox_inches='tight')
    plt.close()

def model_with_multiout(model, Data: DataFrame, Y: DataFrame, important_col: list[str], name:str):

    x_train, y_train, x_test, y_test = split_data_by_date(Data, Y, "2021-09-01")
    
    _x_train, _y_train, _x_test, _y_test = (x_train.drop(["session_date", "userid", "sessionid"], axis=1), y_train.drop(["userid", "sessionid"], axis=1), 
                                            x_test.drop(["session_date", "userid", "sessionid"], axis=1), y_test.drop(["userid", "sessionid"], axis=1))

    # model fitting
    print(f"Fitting model for {name}")
    # Multioutput regression
    # https://xgboost.readthedocs.io/en/stable/python/examples/multioutput_regression.html#sphx-glr-python-examples-multioutput-regression-py
    
    model.fit(_x_train, _y_train, eval_set=[(_x_test, _y_test)])

    y_predt = model.predict(_x_test)
    print("prediction shape")
    print(y_predt.shape)
    rmse = mean_squared_error(_y_test, y_predt, multioutput='raw_values', squared=False)
    
    # rmse_dict[c] = rmse
    # print("For game id: ", c, " RMSE: ", (rmse))
    print("XGBoost multi-ouput RMSE:")
    print(rmse)
    print("Average rmse:", np.mean(rmse))
    print("Avergae MAE:", mean_absolute_error(_y_test, y_predt))
    # _y_test[_y_test == 0] = _y_test[_y_test == 0] + 0.01 
    print("Average MAPE:", mean_absolute_percentage_error(_y_test, y_predt))
    print("Average SMAPE:", smape(_y_test, y_predt))
    print("Pearson Correlation:", _y_test.corrwith(DataFrame(y_predt, columns=_y_test), axis = 0).mean())
    print()
    # input()

    explainer = shap.Explainer(model)
    shap_values = explainer(_x_test)  # or use either train or test
    print("Shap values shap:")
    print(shap_values.shape)
    # https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Multioutput%20Regression%20SHAP.html
    # shap.initjs()
    list_of_labels = _y_test.columns.to_list()
    tuple_of_labels = list(zip(list_of_labels, range(len(list_of_labels))))
    for current_label, current_index in tuple_of_labels:
        
        if current_label not in important_col:
            continue
        vals = np.abs(shap_values[:,:,current_index].values).mean(0)
        feature_names = _x_test.columns

        feature_importance = pd.DataFrame(
            list(zip(feature_names, vals)),
            columns=["col_name", "feature_importance_vals"],
        )
        feature_importance.sort_values(
            by=["feature_importance_vals"], ascending=False, inplace=True
        )
        # feature_importance.head()
        print("Most important features for game" + current_label)
        print(feature_importance.head(10))

        shap.summary_plot(shap_values = shap_values[:,:,current_index], features = _x_test, max_display=12, show=False)
        plt.savefig(f"./shap_cv/{name}/game_{current_label}.png")
        plt.close()

    df = pd.DataFrame(y_predt, columns=_y_test.columns)
    plot_only_first_place_rec(df, name)
    plot_all_rec(df, name)


xg_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            max_depth=6,
            learning_rate=0.15,
            gamma=0.05,
            tree_method="gpu_hist",
            # early_stopping=20
        )

model_with_multiout(xg_model, Data, Y, ["3", "44"], "mean_xg_multiout")

xg_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            max_depth=6,
            learning_rate=0.15,
            gamma=0.05,
            tree_method="gpu_hist",
            # early_stopping=20
        )

model_with_multiout(xg_model, Data, Y_duration, ["3", "44"], "mean_xg_multiout_duration")



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.optimizers import *
from sklearn.preprocessing import *
from sklearn.model_selection import *



def nn_model_with_multiout(model, Data: DataFrame, Y: DataFrame, important_col: list[str], name:str):

    x_train, y_train, x_test, y_test = split_data_by_date(Data, Y, "2021-09-01")
    
    _x_train, _y_train, _x_test, _y_test = (x_train.drop(["session_date", "userid", "sessionid"], axis=1), y_train.drop(["userid", "sessionid"], axis=1), 
                                            x_test.drop(["session_date", "userid", "sessionid"], axis=1), y_test.drop(["userid", "sessionid"], axis=1))

    # model fitting
    print(f"Fitting model for {name}")
    # Multioutput regression
    # https://xgboost.readthedocs.io/en/stable/python/examples/multioutput_regression.html#sphx-glr-python-examples-multioutput-regression-py
    
    model.fit(np.asarray(_x_train).astype('float32'), _y_train, batch_size=(128), epochs=820, verbose=0)
    
    y_predt = model.predict(np.asarray(_x_test).astype('float32'))
    print("prediction shape")
    print(y_predt.shape)
    rmse = mean_squared_error(_y_test, y_predt, multioutput='raw_values', squared=False)
    # rmse_dict[c] = rmse
    # print("For game id: ", c, " RMSE: ", (rmse))
    print("Neural net RMSE:")
    print(rmse)
    print("Average rmse:", np.mean(rmse))
    print("Avergae MAE:", mean_absolute_error(_y_test, y_predt))
    print("Average MAPE:", mean_absolute_percentage_error(_y_test, y_predt))
    print("Average SMAPE:", smape(_y_test, y_predt))
    print("Pearson Correlation:", _y_test.corrwith(DataFrame(y_predt, columns=_y_test), axis = 0).mean())
    print()
    # input()
    background = _x_train.iloc[np.random.choice(_x_train.shape[0], 1000, replace=False)]
    masker = shap.maskers.Independent(np.asarray(_x_train.head(100)).astype('float32'), 10)

    def f(X):
        return model.predict(X)

    explainer = shap.KernelExplainer( f, np.asarray(_x_train.head(100)).astype('float32') ) #, data = np.asarray(_x_train.head(100)).astype('float32'), link="identity")
    
    shap_values = explainer.shap_values(np.asarray(_x_test.head(20)).astype('float32'), nsamples=100)  # or use either train or test

    # shap.summary_plot(shap_values = shap_values, features = _x_test, max_display=12)

    print("Shap values shap:")
    # print(shap_values.shape)
    # https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Multioutput%20Regression%20SHAP.html
    # shap.initjs()
    list_of_labels = _y_test.columns.to_list()
    tuple_of_labels = list(zip(list_of_labels, range(len(list_of_labels))))
    for current_label, current_index in tuple_of_labels:
        
        if current_label not in important_col:
            continue
        vals = np.abs(shap_values[current_index]).mean(0)
        feature_names = _x_test.columns

        feature_importance = pd.DataFrame(
            list(zip(feature_names, vals)),
            columns=["col_name", "feature_importance_vals"],
        )
        feature_importance.sort_values(
            by=["feature_importance_vals"], ascending=False, inplace=True
        )
        # feature_importance.head()
        print("Most important features for game  " + current_label)
        print(feature_importance.head(10))

        shap.summary_plot(shap_values = shap_values[current_index], features = _x_test.columns, max_display=12, show=False)
        plt.savefig(f"./shap_cv/{name}/game_{current_label}.png")
        plt.close()

    df = pd.DataFrame(y_predt, columns=_y_test.columns)
    plot_only_first_place_rec(df, name)
    plot_all_rec(df, name)


def data_to_x(data: pd.DataFrame):
    _df = data.copy()
    _df["duration"] = _df["duration"].clip(0, 30) / 30
    _df["learn_index"] = _df["learn_index"] / 3
    _df["age"] = (_df["age"].clip(0, 16)-5) / 11
    _df["gender"] = (_df["gender"] - 1)

    return _df
_data = data_to_x(Data)


model = Sequential()
model.add(Dense(60, input_shape=(len(Data.drop(["session_date", "userid", "sessionid"], axis=1).columns),), activation="relu"))  # 97
model.add(Dense(30, activation="relu"))
model.add(Dense(40, activation="relu"))
model.add(Dense(len(Y.drop(["userid", "sessionid"], axis=1).columns), activation="linear"))  # 73
model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])

nn_model_with_multiout(model, _data, Y,  ["3", "44"], "mean_nn")

# _Y_duration = Y_duration.drop(["sessionid", "userid"], axis=1).clip(0, 30) / 30
# _Y_duration["userid"] = Y_duration["userid"]
# _Y_duration["sessionid"] = Y_duration["sessionid"]


model = Sequential()
model.add(Dense(60, input_shape=(len(Data.drop(["session_date", "userid", "sessionid"], axis=1).columns),), activation="relu"))  # 97
model.add(Dense(30, activation="relu"))
model.add(Dense(40, activation="relu"))
model.add(Dense(len(Y.drop(["userid", "sessionid"], axis=1).columns), activation="linear"))  # 73
# model.compile(
#     loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
# )  # optimizer=Adam(clipnorm=2.0)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])

nn_model_with_multiout(model, _data, Y_duration,  ["3", "44"], "mean_nn_duration")


from sklearn.metrics.pairwise import cosine_similarity

from numpy.random import choice

def cosine_sim_plots(Data:DataFrame):
    list_games = [1,3,7,8,9,10,11,12,14,15,16,17,18,19,20,22,23,25,27,28,34,35,36,37,38,39,40,41,42,43,44,45,46,53,54,56,57,58,59,60,61,62,63,64,65,66,67,68,70,71,72,74,75,76,77,78,79,82,83,84,85,86,87,89,90,91,92,93,94,95,96,98,99]
    list_games = [str(x) for x in list_games]

    # reverse one hot encoding
    gameid_col = [col for col in Data.columns if "gameid" in col]
    Data["gameid"] = Data[gameid_col].idxmax(1)
    Data = Data.drop(gameid_col, axis=1)
    game_col = [
        col
        for col in Data.columns
        if any(
            s in col for s in ["subject", "mechanic", "game", "high", "junior", "middle"]
        )
    ]
    # print(game_col)
    game_data = (
        Data[game_col]
        .drop_duplicates()
        .sort_values(by="gameid", ascending=True)
        .reset_index()
        .drop("index", axis=1)
    )

    # game_data.index = game_data["gameid"]
    print(game_data.drop("gameid", axis=1).info())
    print(game_data.head())
    print()


    cos_sim = cosine_similarity(game_data.drop("gameid", axis=1), game_data.drop("gameid", axis=1))

        
    labels2index = dict(zip(list_games, range(len(list_games))))
    dict_of_labels = dict(zip(range(len(list_games)), list_games))
    output_highest_1 = dict.fromkeys(list_games, 0)
    output_highest_2 = dict.fromkeys(list_games, 0)
    output_weighted_1 = dict.fromkeys(list_games, 0)
    output_weighted_2 = dict.fromkeys(list_games, 0)
    for i in range(100):
        for game_id in list_games:
            # print("Iterating: " + str(game_id))
            cosine_values = cos_sim[labels2index[game_id], :]

            # return 6 highest 
            highest = np.argsort(cosine_values)
            
            result = [dict_of_labels[rec] for rec in highest[:7] if str(rec) != labels2index[game_id]]
            output_highest_1[result[0]] = output_highest_1[result[0]] + 1
            for r in result[:6]:
                output_highest_2[r] = output_highest_2[r] + 1
            # return 6 games randomly weighted by the value
            # print(cosine_values)
            # print(cosine_values.shape)
            norm_res = cosine_values / (np.sum(cosine_values))
            # print(norm_res.shape)
            
            # print(norm_res)
            # print(np.sum(norm_res))
            draw = choice(list_games, 7, replace=False,
                        p=(norm_res))
            # weighted random draw
            # print(draw)
            # ouput = [x for x in draw if x != game_id]
            output_weighted_1[draw[0]] = output_weighted_1[draw[0]] + 1
            for r in draw[:6]:
                output_weighted_2[r] = output_weighted_2[r] + 1

    df = Series(DataFrame(output_highest_1, index=[0]).transpose()[0])
    # print(df.head())
    plt.figure(figsize=(12,3))
    ax = df.sort_values(ascending=False).plot.bar(figsize=(12,3))
    plt.title(label="")
    plt.grid()
    ax.set_xlabel("Game Id's")
    ax.set_ylabel("Count")

    plt.savefig("./plots/" + "mean_cosine_highest" + "_plot1.png", bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12,3))
    df = Series(DataFrame(output_highest_2, index=[0]).transpose()[0])
    ax = df.sort_values(ascending=False).plot.bar(figsize=(12,3))#figsize=(16, 4))
    plt.title(label="")
    plt.grid()
    ax.set_xlabel("Game Id's")
    ax.set_ylabel("Count")
    plt.savefig("./plots/" + "mean_cosine_highest" + "_plot2.png", bbox_inches='tight')
    plt.close()

    df = Series(DataFrame(output_weighted_1, index=[0]).transpose()[0])
    plt.figure(figsize=(12,3))
    ax = df.sort_values(ascending=False).plot.bar(figsize=(12,3))#figsize=(16, 4))
    plt.title(label="")
    plt.grid()
    ax.set_xlabel("Game Id's")
    ax.set_ylabel("Count")
    plt.savefig("./plots/" + "mean_cosine_weighted" + "_plot1.png", bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(12,3))
    df = Series(DataFrame(output_weighted_2, index=[0]).transpose()[0])
    ax = df.sort_values(ascending=False).plot.bar(figsize=(12,3))#figsize=(16, 4))
    plt.title(label="")
    plt.grid()
    ax.set_xlabel("Game Id's")
    ax.set_ylabel("Count")
    plt.savefig("./plots/" + "mean_cosine_weighted" + "_plot2.png", bbox_inches='tight')
    plt.close()

cosine_sim_plots(Data.copy())


list_games = [1,3,7,8,9,10,11,12,14,15,16,17,18,19,20,22,23,25,27,28,34,35,36,37,38,39,40,41,42,43,44,45,46,53,54,56,57,58,59,60,61,62,63,64,65,66,67,68,70,71,72,74,75,76,77,78,79,82,83,84,85,86,87,89,90,91,92,93,94,95,96,98,99]
list_games = [str(x) for x in list_games]

def create_xg_model_dict():
    return {
        g: xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            max_depth=6,
            learning_rate=0.15,
            gamma=0.05,
            tree_method="gpu_hist",
            # early_stopping=20
        )
        for g in list_games}

def xgboost_single_output(model_dict, Data: DataFrame, Y: DataFrame, important_col: list[str], name:str):
    list_games = [1,3,7,8,9,10,11,12,14,15,16,17,18,19,20,22,23,25,27,28,34,35,36,37,38,39,40,41,42,43,44,45,46,53,54,56,57,58,59,60,61,62,63,64,65,66,67,68,70,71,72,74,75,76,77,78,79,82,83,84,85,86,87,89,90,91,92,93,94,95,96,98,99]
    list_games = [str(x) for x in list_games]

    
    x_train, y_train, x_test, y_test = split_data_by_date(Data, Y, "2021-09-01")
    
    _x_train, _y_train, _x_test, _y_test = (x_train.drop(["session_date", "userid", "sessionid"], axis=1), y_train.drop(["userid", "sessionid"], axis=1), 
                                            x_test.drop(["session_date", "userid", "sessionid"], axis=1), y_test.drop(["userid", "sessionid"], axis=1))

    y_predt_dict = {}
    rmse_dict = {}
    mae_dict = {}
    mape_dict = {}
    smape_dict = {}
    corr_dict = {}
    print("Traning " + name + " models")
    for g in list_games:
        users_that_played_game = Data[Data["gameid_" + str(g)] == 1]["userid"].drop_duplicates()
        # _x = x_train[x_train["userid"].isin(x_train[x_train["gameid_" + str(g)] == 1]["userid"].unique())]
        _x = pd.merge(users_that_played_game, Data, how="left", on="userid")
        _y = pd.merge(_x["sessionid"], Y, how="left", on="sessionid")

        # print("shape", _x.shape)
        _x = _x.drop(["session_date", "userid", "sessionid"], axis=1)
        _y = _y.drop(["userid", "sessionid"], axis=1)
        model_dict[g].fit(_x, _y[g] ) # , eval_set=[(_x_test, _y_test[g])])
        y_predt_dict[g] = model_dict[g].predict(_x_test)
        rmse_dict[g] = mean_squared_error(_y_test[g], y_predt_dict[g], squared=False)
        mae_dict[g] = mean_absolute_error(_y_test[g], y_predt_dict[g])
        mape_dict[g] = mean_absolute_percentage_error(_y_test[g], y_predt_dict[g])
        smape_dict[g] = smape(_y_test[g], y_predt_dict[g])
        corr_dict[g] = _y_test[g].corrwith(y_predt_dict[g])
    
    print(name + " RMSE:")
    print(rmse_dict)
    print("Average rmse:", np.mean(list(rmse_dict.values())))
    print("Avergae MAE:", np.mean(list(mae_dict.values())))
    print("Avergae MAPE:", np.mean(list(mape_dict.values())))
    print("Average SMAPE:", np.mean(list(smape_dict.values())))
    print("Pearson Correlation:", np.mean(list(corr_dict.values())))
    print()
    # input()
    print("Shap plots for " + name)
    for g in important_col:
        explainer = shap.Explainer(model_dict[g])
        shap_values = explainer(_x_test)  # or use either train or test
        
        vals = np.abs(shap_values.values).mean(0)
        feature_names = _x_test.columns

        feature_importance = pd.DataFrame(
            list(zip(feature_names, vals)),
            columns=["col_name", "feature_importance_vals"],
        )
        feature_importance.sort_values(
            by=["feature_importance_vals"], ascending=False, inplace=True
        )
        # feature_importance.head()
        print("Most important features for game: " + g)
        print(feature_importance.head(15))
        # set a display version of the data to use for plotting (has string values)
        # shap_values.display_data = shap.datasets.adult(display=True)[0].values

        # shap.plots.bar(shap_values)
        # print("shap", shap_values)
        plt.figure()
        shap.summary_plot(shap_values, _x_test, max_display=12, show=False)
        plt.savefig(f"./plots/{name}_shap_game_{g}.png")
        plt.close()

    df = pd.DataFrame(y_predt_dict)
    plot_only_first_place_rec(df, name)
    plot_all_rec(df, name)




xgboost_single_output(create_xg_model_dict(), Data, Y, ["3", "44"], "mean_xgtree")
xgboost_single_output(create_xg_model_dict(),Data, Y_duration, ["3", "44"], "mean_xgtree_duration")

def create_rf_model_dict():
    return {
        g: xgb.XGBRFRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            max_depth=6,
            learning_rate=1.0,
            gamma=0.05,
            num_parallel_tree=50,
            tree_method= 'gpu_hist',
            )
            for g in list_games}


xgboost_single_output(create_rf_model_dict(), Data, Y, ["3", "44"], "mean_rf")
xgboost_single_output(create_rf_model_dict(),Data, Y_duration, ["3", "44"], "mean_rf_duration")



