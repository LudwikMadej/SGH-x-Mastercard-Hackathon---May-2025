import pandas as pd
import numpy as np
import os

def load_data(path_to_directory="../Data/Cleaned_data/"):
    dfs = []
    for filename in os.listdir(path_to_directory):
        if filename[-3:] != "csv":
            continue
        dfs.append(pd.read_csv(path_to_directory+filename))

    return pd.concat(dfs, axis=0)