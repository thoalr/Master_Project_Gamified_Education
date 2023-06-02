import pandas as pd
import numpy as np

# from process_excel_data import Data
from load_excel_data import sessions_total, players_total, games

# Load data
Data = pd.read_csv("Data_Processed.csv").drop("Unnamed: 0", axis=1)
print("\nData info:")
print(Data.info())

print()

# Y

tmp_df = sessions_total
tmp_df["duration"] = sessions_total["duration"].clip(0, 30)

result = tmp_df.groupby(["userid", "gameid"]).agg(
    {"learn_index": np.average, "duration": np.average}
)
result["learn_index"] = result["learn_index"] / 3  # get percentage of learning index
result["duration"] = result["duration"] / 30  # get percentage of duration

rmerge = pd.merge(sessions_total["userid"], result, on="userid", how="left")


rpivot = result["duration"].unstack("gameid")
rpivot = rpivot.fillna(rpivot.mean())

final_y = pd.merge(
    sessions_total[["userid", "sessionid"]], rpivot, on="userid", how="left"
)
final_y.columns = final_y.columns.astype(str)
final_y.to_csv("./Y_mean_duration.csv")


rpivot = result["learn_index"].unstack("gameid")
rpivot = rpivot.fillna(rpivot.mean())
final_y = pd.merge(
    sessions_total[["userid", "sessionid"]], rpivot, on="userid", how="left"
)
final_y.columns = final_y.columns.astype(str)
final_y.to_csv("./Y_mean_learn_index.csv")
