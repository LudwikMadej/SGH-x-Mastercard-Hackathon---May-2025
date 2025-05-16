import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

def load_data(train=True):
    """
    Loads and concatenates CSV files from either the training or test data directory.

    Parameters
    ----------
    train : bool, optional (default=True)
        If True, loads data from the training directory.
        If False, loads data from the test directory.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing all concatenated CSV files from the specified directory.

    Notes
    -----
    - Assumes a directory structure with CSV files located in:
        "../Data/Preprocessed_data/train/" or "../Data/Preprocessed_data/test/"
    - Skips any non-CSV files in the directory.
    """
    path_to_directory = "../Data/Preprocessed_data/train/" if train else "../Data/Preprocessed_data/test/"
    dfs = []
    
    for filename in os.listdir(path_to_directory):
        if not filename.endswith(".csv"):
            continue
        full_path = os.path.join(path_to_directory, filename)
        dfs.append(pd.read_csv(full_path))

    return pd.concat(dfs, axis=0)


    
def make_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Creates a ColumnTransformer for preprocessing a DataFrame with numeric, binary, 
    and categorical features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame used to infer column types and generate appropriate 
        preprocessing pipelines.

    Returns
    -------
    ColumnTransformer
        A fitted ColumnTransformer that applies:
        - Standard scaling to numeric features
        - One-hot encoding to binary and multi-class categorical features
        - Pass-through for binary numeric features (0/1)
    
    Column classification logic:
    ----------------------------
    - Numeric columns (more than 2 unique values): scaled using StandardScaler
    - Numeric binary columns (exactly two values: 0 and 1): passed through
    - Non-numeric binary columns (e.g., 'Yes'/'No'): one-hot encoded (drop='if_binary')
    - Non-numeric categorical columns with more than 2 levels: one-hot encoded with
      'handle_unknown=ignore'
    
    Notes
    -----
    This function assumes that the DataFrame contains a mix of numeric and categorical
    features and automatically determines appropriate transformations based on unique 
    values and data types.
    """
    num_cols, num_bin_cols = [], []
    bin_to_convert, multi_cat_cols = [], []

    for col in df.columns:
        s = df[col]
        uniques = s.dropna().unique()

        if pd.api.types.is_numeric_dtype(s):
            if len(uniques) == 2 and set(uniques) == {0, 1}:
                num_bin_cols.append(col)  # binary numeric (already 0/1)
            else:
                num_cols.append(col)  # continuous numeric
        else:
            if len(uniques) == 2:
                bin_to_convert.append(col)  # binary categorical
            else:
                multi_cat_cols.append(col)  # multi-class categorical

    return ColumnTransformer(
        transformers=[
            ('num',               StandardScaler(),              num_cols),
            ('binary_to_convert', OneHotEncoder(drop='if_binary',
                                                sparse_output=False,
                                                dtype='int64'),
                                                bin_to_convert),
            ('multi_cat',         OneHotEncoder(handle_unknown='ignore',
                                                sparse_output=False),
                                                multi_cat_cols),
            ('already_binary',    'passthrough',                 num_bin_cols)
        ],
        remainder='drop'
    )