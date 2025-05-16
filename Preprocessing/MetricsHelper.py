from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse
import numpy as np
import pandas as pd

def expanding_cosine_similarity(
        df: pd.DataFrame,
        user_col: str,
        time_col: str,
        numeric_cols: list[str],
        categorical_cols: list[str],
        min_history: int = 1) -> pd.DataFrame:
    """
     Calculate an expanding historical cosine similarity between current and past features per user.

     This function computes, for each transaction (row) in the input DataFrame, the cosine similarity
     between the feature vector of the current transaction and the aggregated historical feature vector
     of all previous transactions for the same user. Historical aggregation uses:
       - Numeric features: expanding mean of past values.
       - Categorical features: expanding mode (most frequent) of past values.

     Parameters
     ----------
     df : pandas.DataFrame
         Input data containing transactions or events.
     user_col : str
         Column name identifying each user (groups rows for historical aggregation).
     time_col : str
         Column name indicating the timestamp or ordering within each user group.
     numeric_cols : list[str]
         List of numeric column names to include in similarity computation.
     categorical_cols : list[str]
         List of categorical column names to include in similarity computation.
     min_history : int, default=1
         Minimum number of past records required to compute similarity; rows with fewer will get similarity=0.

     Returns
     -------
     pandas.DataFrame
         A DataFrame with two columns:
         - `transaction_id`: carried over from the input for identification.
         - `cosine_similarity_hist_all`: the calculated cosine similarity score between current and historical features.
     """

    df = df.sort_values([user_col, time_col]).reset_index(drop=True)

    for col in numeric_cols:
        df[f"{col}_hist_mean"] = (
            df.groupby(user_col)[col]
              .shift()
              .expanding()
              .mean()
              .fillna(0)
              .reset_index(level=0, drop=True)
        )

    for col in categorical_cols:
        modes = (
            df.groupby(user_col)[col]
              .apply(lambda s: (
                  pd.get_dummies(s)
                    .cumsum()
                    .idxmax(axis=1)
              ))
              .reset_index(level=0, drop=True)
        )
        df[f"{col}_hist_mode"] = modes.shift().fillna(0)


    enc = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    enc.fit(df[categorical_cols])


    X_num = df[numeric_cols].to_numpy(dtype=float)
    X_cat = enc.transform(df[categorical_cols])


    X_hist_num_all = df[[f"{c}_hist_mean" for c in numeric_cols]].to_numpy(dtype=float)
    hist_cat_all = df[[f"{c}_hist_mode" for c in categorical_cols]].copy()
    hist_cat_all.columns = categorical_cols
    X_hist_cat_all = enc.transform(hist_cat_all)



    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)
    X_hist_num_all = scaler.fit_transform(X_hist_num_all)



    X_current   = sparse.hstack([sparse.csr_matrix(X_num),       X_cat],          format="csr")
    X_past_all  = sparse.hstack([sparse.csr_matrix(X_hist_num_all), X_hist_cat_all],  format="csr")



    def _cosine(A, B):
        num   = A.multiply(B).sum(axis=1).A1
        normA = np.sqrt(A.multiply(A).sum(axis=1)).A1
        normB = np.sqrt(B.multiply(B).sum(axis=1)).A1
        with np.errstate(divide="ignore", invalid="ignore"):
            cs = np.divide(num, normA * normB, out=np.zeros_like(num), where=(normA*normB)!=0)
        return cs

    cs_all = _cosine(X_current, X_past_all)



    hist_len_all = df.groupby(user_col).cumcount()

    cs_all[hist_len_all < min_history] = 0

    df["cosine_similarity_hist_all"]      = cs_all

    return df[['transaction_id','cosine_similarity_hist_all']]


def expanding_euclidia_metric(
    df: pd.DataFrame,
    user_col: str,
    time_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    min_history: int = 1 ) -> pd.DataFrame:
    """
        Compute an expanding historical Euclidean distance between current and past features per user.

        This function calculates, for each entry in the DataFrame, the Euclidean distance
        between the feature vector of the current record and the aggregated historical feature vector
        of all previous records for the same user. Historical aggregation uses:
          - Numeric features: expanding mean of past values (excluding current).
          - Categorical features: expanding mode (most frequent) of past values (excluding current).
        Parameters
        ----------
        df : pandas.DataFrame
            Input data containing events or transactions.
        user_col : str
            Column name identifying each user for grouping.
        time_col : str
            Column name with timestamps to order each user's records.
        numeric_cols : list[str]
            Names of numeric feature columns.
        categorical_cols : list[str]
            Names of categorical feature columns.
        min_history : int, default=1
            Minimum number of prior records required to compute distance; otherwise distance=0.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing:
            - `transaction_id`: identifier from input.
            - `euclidean_similarity_hist_all`: Euclidean distance between current and historical features.
        """


    df = df.sort_values([user_col, time_col]).reset_index(drop=True)

    for col in numeric_cols:
        df[f"{col}_hist_mean"] = (
            df.groupby(user_col)[col]
              .shift()
              .expanding()
              .mean().fillna(0)
              .reset_index(level=0, drop=True)
        )

    for col in categorical_cols:
        modes = (
        df.groupby(user_col)[col]
          .apply(lambda s: (
              pd.get_dummies(s)
                .cumsum()
                .idxmax(axis=1)
          ))
          .reset_index(level=0, drop=True)
        )
        df[f"{col}_hist_mode"] = (
        modes.shift()
             .fillna(0)
        )


    enc = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    enc.fit(df[categorical_cols])

    X_num = df[numeric_cols].to_numpy(dtype=float)
    X_cat = enc.transform(df[categorical_cols])

    X_hist_num_all = df[[f"{c}_hist_mean" for c in numeric_cols]].to_numpy(dtype=float)
    hist_cat_all = df[[f"{c}_hist_mode" for c in categorical_cols]].copy()
    hist_cat_all.columns = categorical_cols
    X_hist_cat_all = enc.transform(hist_cat_all)



    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)
    X_hist_num_all = scaler.fit_transform(X_hist_num_all)


    X_current = sparse.hstack([sparse.csr_matrix(X_num), X_cat], format="csr")
    X_past_all = sparse.hstack([sparse.csr_matrix(X_hist_num_all), X_hist_cat_all], format="csr")


    def EuclidianMetric(A,B):
        diff = A - B
        sqsum = diff.multiply(diff).sum(axis=1).A1
        dist = np.sqrt(sqsum)
        return dist

    cs_all = EuclidianMetric(X_current, X_past_all)
    hist_len_all = df.groupby(user_col).cumcount()
    cs_all[hist_len_all < min_history] = 0
    df["euclidian_similarity_hist_all"] = cs_all

    return df[['transaction_id', 'euclidian_similarity_hist_all']]
