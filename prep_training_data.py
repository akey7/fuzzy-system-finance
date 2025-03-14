import os
import glob
import json
import pandas as pd


def make_long_df():
    """
    Create a DataFrame with the following columns on each row:

    - date
    - symbol
    - adj_open
    - adj_high
    - adj_low
    - adj_close,
    - adj_volume

    Walk over the *_history.json files in the input/ folder
    to get the data.

    Finally, write the data to output/stocks_long.csv
    """
    rows = []
    input_path = "input"
    pattern = "*_history.json"
    for file_path in glob.glob(os.path.join(input_path, pattern)):
        with open(file_path, "r") as f:
            days = json.load(f)
            for day in days:
                rows.append({
                    "date": day["date"],
                    "symbol": day["symbol"],
                    "adj_open": day["adj_open"],
                    "adj_high": day["adj_high"],
                    "adj_low": day["adj_low"],
                    "adj_close": day["adj_close"]
                })
    df = pd.DataFrame(rows)
    df_filename = os.path.join("output", "stocks_long.csv")
    df.to_csv(df_filename, index=False)


if __name__ == "__main__":
    make_long_df()
