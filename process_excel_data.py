from load_excel_data import sessions_total, players_total, games
import pandas as pd

Data = sessions_total[
    ["userid", "sessionid", "duration", "learn_index", "gameid", "session_date"]
].copy()


# Data["session_date"] = Data["session_date"].values.astype("int64")

Data = pd.merge(
    Data, players_total[["userid", "age", "gender"]], on="userid", how="left"
)

# One column for each subject and mechanic and schoollevel
Data = pd.merge(
    Data,
    games[["gameid", "subject", "game_mechanic", "school_level"]],
    on="gameid",
    how="left",
)

# No one hot encoding instead set type as categorical
Categorical_Data = Data.astype(
    {
        "gameid": "category",
        "gender": "category",
        "subject": "category",
        "game_mechanic": "category",
        "school_level": "category",
    }
)


Data = pd.get_dummies(Data, columns=["gameid"], prefix="gameid", dtype=int)

Data = pd.get_dummies(Data, columns=["subject"], prefix="subject", dtype=int)
Data = pd.get_dummies(Data, columns=["game_mechanic"], prefix="game_mechanic", dtype=int)
# Data = pd.get_dummies(Data, columns=["school_level"], prefix="school_level")

school_ll = Data["school_level"].str.get_dummies(", ")
school_ll = school_ll.rename(columns={"junior" : "school_level_junior", "middle": "school_level_middle", "high": "school_level_high"}).astype(int)
Data = Data.join(school_ll, how="left")
Data = Data.drop("school_level", axis=1)

print("Data\n")
# print("Description\n", Data.describe(), "\n")
# print("Head\n", Data.head(), "\n")
# print("Info\n", Data.info(), "\n")

# print("Columns\n", Data.columns)
# print("types\n", Data.dtypes)
# print("object\n", Data.select_dtypes(include=['object']))

print("Any nans?", Data.isna().any().any())
print("Info")
print(Data.info())

# save data
print("Data saved to Data_Provessed.csv")
Data.to_csv("./Data_Processed.csv")

