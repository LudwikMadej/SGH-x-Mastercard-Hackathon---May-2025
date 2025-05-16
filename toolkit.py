import pandas as pd
import numpy as np
import os
import pickle as pkl

def load_data(path_to_directory="../data/"):
    dfs = []
    for filename in os.listdir(path_to_directory):
        if filename[-3:] != "csv":
            continue
        
        dfs.append(pd.read_csv(path_to_directory+filename))

    return pd.concat(dfs, axis=0)

def fetch_fraud_fraction(year : int, month : int, user_country : str, merchant_country : str, transaction_country : str, default = 0.0, path="../data/country_combinations/") -> tuple:
    """
    Returns a tuple with two numbers:
        first_item : fraud fraction for countries combination in the previous month;
        second_item : fraud fraction for countries combination in the past months;
    WARNING : If the combination of countries does not exist then the default value is returned.
    """
    if year == 2022 and month == 1:
        return default, default

    if month == 1:
        year -= 1
        month = 12

    else:
        month -= 1

    
    with open(path+"year_month_countires_combinations.pkl", "rb") as file:
        dictionary = pkl.load(file)

    with open(path+f"cumulative_year_month_countries_combinations/{year}_{month:02}.pkl", "rb") as file:
        dictionary_cumulative = pkl.load(file)

    print()
    key = f"{year}_{month:02}_{user_country}_{merchant_country}_{transaction_country}"
    key_2 = f"{user_country}_{merchant_country}_{transaction_country}"
    
    return dictionary.get(key, default), dictionary_cumulative.get(key_2, default)
