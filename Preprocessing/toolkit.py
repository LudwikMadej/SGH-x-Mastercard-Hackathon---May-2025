import pandas as pd
import numpy as np
import os

def load_data(path_to_directory="../Data/Cleaned_data/") -> pd.DataFrame:
    """
    Wczytuje i łączy wszystkie pliki CSV z podanego katalogu w jeden DataFrame.

    Parameters:
    -----------
    path_to_directory : str
        Ścieżka do katalogu zawierającego pliki CSV.

    Returns:
    --------
    pd.DataFrame
        Połączony DataFrame ze wszystkimi plikami CSV.
    """
    dfs = []
    for filename in os.listdir(path_to_directory):
        if not filename.endswith(".csv"):
            continue
        filepath = os.path.join(path_to_directory, filename)
        dfs.append(pd.read_csv(filepath))

    return pd.concat(dfs, axis=0, ignore_index=True)