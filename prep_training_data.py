import os
import glob
import pandas as pd


def make_melted_df():
    input_path = "input"
    pattern = "*_history.json"
    for file_path in glob.glob(os.path.join(input_path, pattern)):
        print(file_path)


if __name__ == "__main__":
    make_melted_df()
