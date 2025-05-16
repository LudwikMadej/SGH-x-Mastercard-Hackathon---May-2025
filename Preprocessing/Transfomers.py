from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import copy
from typing import List
from collections import deque
from MetricsHelper import *
from scipy.special import stdtrit
from math import sqrt
from geopy.distance import geodesic
from shapely.wkt import loads
import pickle as pkl
from tqdm import tqdm


# Abstract Base Class
from abc import ABC, abstractmethod
from typing import List
import pandas as pd

class Transformer(ABC):
    """
    Abstract base class for data transformers that operate on pandas DataFrames.

    Subclasses must define a list of required columns (`required_columns`)
    and implement the `transform` method that performs a transformation
    on the input DataFrame.

    Attributes:
        required_columns (List[str]): List of column names that must be present in the input DataFrame.
    """

    required_columns: List[str] = []

    @abstractmethod
    def transform(self, df_base: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a transformation to the given DataFrame.

        Args:
            df_base (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        pass

    def __init_subclass__(cls, **kwargs):
        """
        Automatically checks if a subclass defines `required_columns` as a list.

        Raises:
            TypeError: If `required_columns` is not defined or is not a list.
        """
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'required_columns'):
            raise TypeError(f"{cls.__name__} must define `required_columns`.")
        if not isinstance(cls.required_columns, list):
            raise TypeError(f"{cls.__name__}.required_columns must be a list.")

    def check_columns(self, df: pd.DataFrame):
        """
        Validates that all required columns are present in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to validate.

        Raises:
            ValueError: If any required columns are missing.
        """
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"{self.__class__.__name__}: Missing columns: {missing}")


# ScoreTransformer
class ScoreTransformer(Transformer):
    """
    A concrete implementation of the Transformer class that computes 
    various score-related features based on risk and trust scores.

    This transformer requires the following columns in the input DataFrame:
    - 'transaction_id'
    - 'risk_score'
    - 'trust_score'

    It adds squared versions of the input scores and calculates a weighted 
    risk score adjusted by trust, along with its squared version.
    """

    required_columns = ['transaction_id', 'risk_score', 'trust_score']

    def transform(self, df_base: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by computing additional score-based features.

        This includes:
        - Square of 'risk_score' and 'trust_score'
        - A composite 'score' = risk_score * (1 - trust_score)
        - A composite 'score_sq' = risk_score^2 * (1 - trust_score^2)

        Args:
            df_base (pd.DataFrame): Input DataFrame containing at least the required columns.

        Returns:
            pd.DataFrame: A new DataFrame with the following columns:
                - 'transaction_id'
                - 'risk_score_sq'
                - 'trust_score_sq'
                - 'score'
                - 'score_sq'

        Raises:
            ValueError: If any of the required columns are missing.
        """
        self.check_columns(df_base)
        df = df_base.copy()

        df['risk_score_sq'] = df['risk_score'] ** 2
        df['trust_score_sq'] = df['trust_score'] ** 2
        df['score'] = df['risk_score'] * (1 - df['trust_score'])
        df['score_sq'] = df['risk_score_sq'] * (1 - df['trust_score_sq'])

        return df[['transaction_id', 'risk_score_sq', 'trust_score_sq', 'score', 'score_sq']]


# TimeTransformer
class TimeTransformer(Transformer):
    """
    A concrete transformer that extracts and encodes temporal features from a timestamp column.

    This transformer requires the following columns in the input DataFrame:
    - 'transaction_id'
    - 'timestamp'

    It extracts components of the timestamp (day, month, hour, weekday) and applies
    cyclical encoding using sine and cosine transformations. It also adds binary flags
    for night-time and weekend transactions, along with corresponding angular features.
    """

    required_columns = ['transaction_id', 'timestamp']

    def transform(self, df_base: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by extracting and encoding time-based features.

        Operations performed:
        - Converts 'timestamp' to datetime.
        - Extracts day, month, hour, and weekday from timestamp.
        - Applies cyclical encoding to day, month, and hour (sine/cosine + arctangent angle).
        - Flags transactions occurring at night or on weekends.

        Args:
            df_base (pd.DataFrame): Input DataFrame with timestamp information.

        Returns:
            pd.DataFrame: A new DataFrame with:
                - 'transaction_id'
                - Sine and cosine encodings for day, month, and hour
                - Angle encodings for day, month, and hour
                - Binary flags: 'IsNight', 'IsWeekend'

        Raises:
            ValueError: If any required columns are missing.
        """
        self.check_columns(df_base)
        df = df_base.copy()

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['transaction_day'] = df['timestamp'].dt.day
        df['transaction_month'] = df['timestamp'].dt.month
        df['transaction_weekday'] = df['timestamp'].dt.weekday
        df['transaction_hour'] = df['timestamp'].dt.hour

        # Cyclical encoding for day of month
        df['day_rad'] = 2 * np.pi * df['transaction_day'] / 31
        df['day_sin'] = np.sin(df['day_rad'])
        df['day_cos'] = np.cos(df['day_rad'])
        df['day_angle'] = np.arctan2(df['day_sin'], df['day_cos'])

        # Cyclical encoding for month
        df['month_rad'] = 2 * np.pi * df['transaction_month'] / 12
        df['month_sin'] = np.sin(df['month_rad'])
        df['month_cos'] = np.cos(df['month_rad'])
        df['month_angle'] = np.arctan2(df['month_sin'], df['month_cos'])

        # Cyclical encoding for hour of day
        df['hour_rad'] = 2 * np.pi * df['transaction_hour'] / 24
        df['hour_sin'] = np.sin(df['hour_rad'])
        df['hour_cos'] = np.cos(df['hour_rad'])
        df['hour_angle'] = np.arctan2(df['hour_sin'], df['hour_cos'])

        # Binary time-based flags
        df['IsNight'] = np.where((df['transaction_hour'] >= 22) | (df['transaction_hour'] <= 6), 1, 0)
        df['IsWeekend'] = np.where(df['transaction_weekday'] >= 5, 1, 0)

        return df[['transaction_id', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                   'hour_sin', 'hour_cos', 'IsNight', 'IsWeekend', 'hour_angle', 'month_angle', 'day_angle']]


# DeviceTransformer
class DeviceTransformer(Transformer):
    """
    A transformer that analyzes device and payment method patterns over time 
    for each user to detect repeated usage and potential fraud signals.

    This transformer requires the following columns:
    - 'transaction_id'
    - 'user_id'
    - 'timestamp'
    - 'device'
    - 'payment_method'

    It generates rolling similarity features, tracks if the user has previously used 
    the same device or payment method, and flags these usages separately for fraud 
    and non-fraud transactions.
    """

    required_columns = ['transaction_id', 'user_id', 'timestamp', 'device', 'payment_method']

    def transform(self, df_base: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by generating features that capture 
        device and payment method reuse behavior over time.

        Key operations:
        - Sorts transactions by user and timestamp.
        - Applies rolling window checks (1-9 previous) to see if device or 
          payment method was reused by the same user.
        - Combines device and payment method reuse into a joint flag.
        - Flags whether the user has historically used a device/payment combination,
          including fraud vs. non-fraud differentiation based on `is_fraud` column.

        Args:
            df_base (pd.DataFrame): Input DataFrame with required columns and optionally 'is_fraud'.

        Returns:
            pd.DataFrame: Transformed DataFrame including:
                - Rolling match features (e.g., device_prev1_same)
                - Combined match features (e.g., deviceAndPayment_prev1_same_one)
                - Historic usage flags (all/fraud/non-fraud versions)
                - 'transaction_id' for tracking

        Raises:
            ValueError: If required columns are missing.
        """
        self.check_columns(df_base)
        df = df_base.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        def rolling_match(series: pd.Series, window: int) -> np.ndarray:
            """
            Checks whether the current value exists within the rolling window 
            of previous values for the same user.
            """
            out = np.zeros(len(series), dtype='int8')
            prev = deque(maxlen=window)
            for i, val in enumerate(series):
                out[i] = int(val in prev)
                prev.append(val)
            return out

        n_vector = range(1, 10)
        new_columns = []

        # Rolling match for device and payment_method
        for col in ['device', 'payment_method']:
            for n in n_vector:
                new_col = f'{col}_prev{n}_same'
                df[new_col] = df.groupby('user_id')[col].transform(lambda s: rolling_match(s, n))
                new_columns.append(new_col)

        # Combined indicator where both device and payment_method match in history
        for n in n_vector:
            col = f'deviceAndPayment_prev{n}_same_one'
            device_col = f'device_prev{n}_same'
            payment_col = f'payment_method_prev{n}_same'
            df[col] = np.where((df[device_col] == df[payment_col]) & (df[payment_col] == 1), 1, 0)
            new_columns.append(col)

        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        # Historic usage flags
        df['historic_device'] = df.duplicated(subset=['user_id', 'device'], keep='first').astype(int)
        df['historic_payment'] = df.duplicated(subset=['user_id', 'payment_method'], keep='first').astype(int)
        df['historic_device_payment'] = df.duplicated(subset=['user_id', 'device', 'payment_method'], keep='first').astype(int)

        
        def compute_seen_flag(group: pd.DataFrame, columns: list[str], fraud_bool: bool) -> pd.Series:
            """
            Sprawdza, czy dana wartość/kombinacja wartości pojawiła się wcześniej
            w transakcjach typu `fraud_bool`.

            Parameters:
            -----------
            group : pd.DataFrame
                Dane jednej grupy użytkownika, posortowane po czasie.
            columns : list of str
                Kolumny tworzące kombinację, którą śledzimy (np. ['device'], ['payment_method'], ['device', 'payment_method']).
            fraud_bool : bool
                Typ transakcji, które aktualizują historię (0 = non-fraud, 1 = fraud).

            Returns:
            --------
            pd.Series
                Wektor True/False.
            """
            seen = set()
            result = []

            for _, row in group.iterrows():
                key = tuple(row[col] for col in columns)
                result.append(key in seen)
                if row['is_fraud'] == fraud_bool:
                    seen.add(key)

            return pd.Series(result, index=group.index)
     
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        df['historic_device_nonfraud'] = df.groupby('user_id', group_keys=False)\
            .apply(lambda g: compute_seen_flag(g, ['device'], fraud_bool=0)).astype(int)

        df['historic_payment_nonfraud'] = df.groupby('user_id', group_keys=False)\
            .apply(lambda g: compute_seen_flag(g, ['payment_method'], fraud_bool=0)).astype(int)

        df['historic_device_payment_nonfraud'] = df.groupby('user_id', group_keys=False)\
            .apply(lambda g: compute_seen_flag(g, ['device', 'payment_method'], fraud_bool=0)).astype(int)

        df['historic_device_fraud'] = df.groupby('user_id', group_keys=False)\
            .apply(lambda g: compute_seen_flag(g, ['device'], fraud_bool=1)).astype(int)

        df['historic_payment_fraud'] = df.groupby('user_id', group_keys=False)\
            .apply(lambda g: compute_seen_flag(g, ['payment_method'], fraud_bool=1)).astype(int)

        df['historic_device_payment_fraud'] = df.groupby('user_id', group_keys=False)\
            .apply(lambda g: compute_seen_flag(g, ['device', 'payment_method'], fraud_bool=1)).astype(int)
   
        df["historic_device_nonfraud_+_historic_payment_nonfraud"] = df["historic_device_nonfraud"] + df["historic_payment_nonfraud"]

        # Final column selection
        new_columns += [
            'transaction_id',
            'historic_device',
            'historic_payment',
            'historic_device_payment',
            'historic_device_nonfraud',
            'historic_payment_nonfraud',
            'historic_device_payment_nonfraud',
            'historic_device_fraud',
            'historic_payment_fraud',
            'historic_device_payment_fraud',
            "historic_device_nonfraud_+_historic_payment_nonfraud"
        ]

        return df[new_columns]

# MetricsTransformer
class MetricsTransformer(Transformer):
    """
    A transformer that computes expanding window similarity metrics (cosine and Euclidean)
    for selected numerical features, grouped by user over time.

    This transformer requires the following columns:
    - 'transaction_id'
    - 'channel'
    - 'device'
    - 'payment_method'
    - 'amount'
    - 'session_length_seconds'
    - 'is_international'
    - 'user_id'
    - 'timestamp'

    It uses two similarity functions:
    - `expanding_cosine_similarity`: Computes cosine similarity between the current
      and previous user transactions, expanding over time.
    - `expanding_euclidia_metric`: Computes Euclidean distance in a similar fashion.

    The results of both metrics are merged on `transaction_id`.
    """

    required_columns = ['transaction_id', 'channel', 'device', 'payment_method',
                        'amount', 'session_length_seconds', 'is_international',
                        'user_id', 'timestamp']

    def transform(self, df_base: pd.DataFrame) -> pd.DataFrame:
        """
        Applies expanding cosine similarity and Euclidean distance metrics for each user.

        Steps performed:
        1. Ensures all required columns are present.
        2. Converts 'timestamp' to datetime format.
        3. Applies `expanding_cosine_similarity` grouped by 'user_id' and ordered by 'timestamp'
           using selected numeric features and categorical groupings.
        4. Applies `expanding_euclidia_metric` with the same setup.
        5. Merges both result sets on 'transaction_id'.

        Args:
            df_base (pd.DataFrame): Input DataFrame containing transaction and user data.

        Returns:
            pd.DataFrame: Merged DataFrame with additional similarity metrics.

        Raises:
            ValueError: If any required columns are missing.
        """
        self.check_columns(df_base)
        df_1 = df_base.copy()
        df_2 = df_base.copy()

        df_1['timestamp'] = pd.to_datetime(df_1['timestamp'], errors='coerce')
        df_2['timestamp'] = pd.to_datetime(df_2['timestamp'], errors='coerce')

        # Apply expanding cosine similarity
        df_1 = expanding_cosine_similarity(
            df_1, 'user_id', 'timestamp',
            ['amount', 'session_length_seconds', 'is_international'],
            ['channel', 'device', 'payment_method']
        )

        # Apply expanding Euclidean distance
        df_2 = expanding_euclidia_metric(
            df_2, 'user_id', 'timestamp',
            ['amount', 'session_length_seconds', 'is_international'],
            ['channel', 'device', 'payment_method']
        )

        # Merge both sets of metrics
        result = df_1.merge(df_2, how='inner', on='transaction_id')
        return result


# CountryTransformer
class CountryTransformer(Transformer):
    """
    A transformer that derives features based on country relationships between the user,
    the transaction, and the merchant. It also tracks historic combinations of these countries,
    split by fraud and non-fraud history.

    Required columns:
    - 'transaction_id'
    - 'country_user'
    - 'country_transaction'
    - 'country_merchant'
    - 'user_id'
    - 'timestamp'

    Feature groups:
    1. **Same-country flags**:
       - Whether the user's country matches the transaction's country.
       - Whether the transaction's country matches the merchant's country.

    2. **Historical combination flags**:
       - Has the same (user, user_country, merchant_country) pair occurred before?
       - Has the same (user, user_country, transaction_country) pair occurred before?
       - Has the same triplet (user, user_country, transaction_country, merchant_country)
         occurred before?

    3. **Fraud-aware versions**:
       - Each of the above historical features is duplicated and split by fraud context:
         - Seen before only in non-fraud transactions.
         - Seen before only in fraud transactions.
    """

    required_columns = ['transaction_id', 'country_user', 'country_transaction',
                        'country_merchant', 'user_id', 'timestamp']

    def transform(self, df_base: pd.DataFrame) -> pd.DataFrame:
        """
        Generate country-based comparison features and history-based duplication flags,
        both globally and split by fraud status.

        Steps:
        1. Ensure all required columns exist.
        2. Convert timestamp to datetime and sort by user/time.
        3. Create binary flags for:
            - Same country between user and transaction.
            - Same country between transaction and merchant.
        4. Track if specific user-country patterns have appeared before.
        5. Separate tracking based on whether past occurrences were in fraudulent
           or non-fraudulent transactions.

        Args:
            df_base (pd.DataFrame): DataFrame with transaction data.

        Returns:
            pd.DataFrame: DataFrame with additional engineered features.
        """
        self.check_columns(df_base)
        df = df_base.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        # Binary same-country indicators
        df['same_CouUser_CouTrans'] = (df['country_user'] == df['country_transaction']).astype(int)
        df['same_CouTrans_CouMerch'] = (df['country_transaction'] == df['country_merchant']).astype(int)

        # General historical duplicates (regardless of fraud)
        df['pair_couUser_couMerchant'] = df.duplicated(
            subset=['user_id', 'country_user', 'country_merchant'], keep='first'
        )
        df['pair_couUser_couTrans'] = df.duplicated(
            subset=['user_id', 'country_user', 'country_transaction'], keep='first'
        )
        df['triplet_couUser_couTrans_couMerchant'] = df.duplicated(
            subset=['user_id', 'country_user', 'country_transaction', 'country_merchant'], keep='first'
        )


        df.sort_values("timestamp", ascending=True, inplace=True)
        
        pair_couUser_couMerchant_nonfraud = []
        pair_couUser_couTrans_nonfraud = []
        triplet_couUser_couTrans_couMerchant_nonfraud = []
        
        pair_couUser_couMerchant_fraud = []
        pair_couUser_couTrans_fraud = []
        triplet_couUser_couTrans_couMerchant_fraud = []

        set_fraud = set()
        set_legit = set()
        for index, row in tqdm(df.iterrows(), total = len(df)):
            user_id = row["user_id"]
            country_user = row["country_user"]
            country_merchant = row["country_merchant"]
            country_transaction = row["country_transaction"]

            key_pair_couUser_couMerchant = f"{user_id}_{country_user}_{country_merchant}"
            key_pair_couUser_couTrans = f"{user_id}_{country_user}_{country_transaction}"
            key_triplet_couUser_couTrans_couMerchant_fraud = f"{user_id}_{country_user}_{country_merchant}_{country_transaction}"

            if key_pair_couUser_couMerchant in set_fraud:
                pair_couUser_couMerchant_fraud.append(True)
            elif row["is_fraud"] == 1:
                set_fraud.add(key_pair_couUser_couMerchant)
                pair_couUser_couMerchant_fraud.append(False)
            else:
                pair_couUser_couMerchant_fraud.append(False)

            
            if key_pair_couUser_couMerchant in set_legit:
                pair_couUser_couMerchant_nonfraud.append(True)
            elif row["is_fraud"] == 0:
                set_legit.add(key_pair_couUser_couMerchant)
                pair_couUser_couMerchant_nonfraud.append(False)
            else:
                pair_couUser_couMerchant_nonfraud.append(False)



                
            if key_pair_couUser_couTrans in set_fraud:
                pair_couUser_couTrans_fraud.append(True)
            elif row["is_fraud"] == 1:
                set_fraud.add(key_pair_couUser_couTrans)
                pair_couUser_couTrans_fraud.append(False)
            else:
                pair_couUser_couTrans_fraud.append(False)

            
            if key_pair_couUser_couTrans in set_legit:
                pair_couUser_couTrans_nonfraud.append(True)
            elif row["is_fraud"] == 0:
                set_legit.add(key_pair_couUser_couTrans)
                pair_couUser_couTrans_nonfraud.append(False)
            else:
                pair_couUser_couTrans_nonfraud.append(False)

            
            
            if key_triplet_couUser_couTrans_couMerchant_fraud in set_fraud:
                triplet_couUser_couTrans_couMerchant_fraud.append(True)
            elif row["is_fraud"] == 1:
                set_fraud.add(key_triplet_couUser_couTrans_couMerchant_fraud)
                triplet_couUser_couTrans_couMerchant_fraud.append(False)
            else:
                triplet_couUser_couTrans_couMerchant_fraud.append(False)

            
            if key_triplet_couUser_couTrans_couMerchant_fraud in set_legit:
                triplet_couUser_couTrans_couMerchant_nonfraud.append(True)
            elif row["is_fraud"] == 0:
                set_legit.add(key_triplet_couUser_couTrans_couMerchant_fraud)
                triplet_couUser_couTrans_couMerchant_nonfraud.append(False)
            else:
                triplet_couUser_couTrans_couMerchant_nonfraud.append(False)
            
            
            
        # Non-fraud-specific history
        df["pair_couUser_couMerchant_nonfraud"] = pair_couUser_couMerchant_nonfraud
        df["pair_couUser_couMerchant_fraud"] = pair_couUser_couMerchant_fraud

        df["pair_couUser_couTrans_nonfraud"] = pair_couUser_couTrans_fraud
        df["pair_couUser_couTrans_fraud"] = pair_couUser_couTrans_fraud

        df["triplet_couUser_couTrans_couMerchant_nonfraud"] = triplet_couUser_couTrans_couMerchant_nonfraud
        df["triplet_couUser_couTrans_couMerchant_fraud"] = triplet_couUser_couTrans_couMerchant_fraud

        # Select final columns to return
        new_columns = [
            'transaction_id',
            'same_CouUser_CouTrans',
            'same_CouTrans_CouMerch',
            'pair_couUser_couMerchant',
            'pair_couUser_couTrans',
            'triplet_couUser_couTrans_couMerchant',
            'pair_couUser_couMerchant_nonfraud',
            'pair_couUser_couTrans_nonfraud',
            'triplet_couUser_couTrans_couMerchant_nonfraud',
            'pair_couUser_couMerchant_fraud',
            'pair_couUser_couTrans_fraud',
            'triplet_couUser_couTrans_couMerchant_fraud'
        ]
        return df[new_columns]


# Transformer for history with consideration of fraud status
class FraudWiseHistoryTransformer(Transformer):
    required_columns = ['transaction_id', 'user_id', 'is_fraud', 'timestamp', 'amount']

    def transform(self, df_base: pd.DataFrame) -> pd.DataFrame:
        self.check_columns(df_base)
        df = df_base.copy()
        
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        df['prev_transaction_is_fraud'] = (
            df
            .groupby('user_id')['is_fraud']
            .shift(1, fill_value=0)
        )

        df['num_of_prev_transactions'] = df.groupby('user_id')['timestamp'].transform(lambda s: s.shift(fill_value=0).expanding().count()).astype(int) - 1
        df['num_of_prev_frauds'] = df.groupby('user_id')['is_fraud'].transform(lambda s: s.shift(fill_value=0).cumsum()).astype(int)

        def compute_stat(group: pd.DataFrame, column_name: str, fraud_bool: bool, type: str) -> pd.Series:
            values = []
            result = []
            for is_fraud, val in zip(group['is_fraud'], group[column_name]):
                if values:
                    if type == 'avg':
                        x = sum(values) / len(values)
                    elif type == 'sd':
                        if len(values) == 1:
                            x = 0
                        else:
                            x = np.std(values, ddof = 1)
                    else: raise ValueError
                else:
                    x = 0 # bez historii transakcji
                result.append(x)
                if is_fraud == fraud_bool:
                    values.append(val)
            return pd.Series(result, index=group.index)

        df['non_fraud_amount_avg'] = df.groupby('user_id')[['is_fraud', 'amount']].apply(lambda g: compute_stat(g, 'amount', 0, 'avg')).reset_index(level=0, drop=True)
        df['fraud_amount_avg'] = df.groupby('user_id')[['is_fraud', 'amount']].apply(lambda g: compute_stat(g, 'amount', 1, 'avg')).reset_index(level=0, drop=True)

        df['amount_avg_closer_to_fraud'] = df[['is_fraud', 'amount','fraud_amount_avg','non_fraud_amount_avg']
                                                          ].apply(lambda row: 0 if row['fraud_amount_avg'] == 0 else np.abs(row['amount']-row['fraud_amount_avg']) < np.abs(row['amount']-row['non_fraud_amount_avg']), axis = 1).reset_index(level=0, drop=True)

        df['non_fraud_amount_sd'] = df.groupby('user_id')[['is_fraud', 'amount']].apply(lambda g: compute_stat(g, 'amount', 0, 'sd')).reset_index(level=0, drop=True)
        df['fraud_amount_sd'] = df.groupby('user_id')[['is_fraud', 'amount']].apply(lambda g: compute_stat(g, 'amount', 1, 'sd')).reset_index(level=0, drop=True)

        confidence_level = 0.95
        alpha = 1 - confidence_level

        def compute_ci_lower(row: pd.Series) -> float:
            n = row['num_of_prev_transactions'] - row['num_of_prev_frauds']
            if n > 1:
                t = stdtrit(n - 1, 1 - alpha / 2)
                return row['non_fraud_amount_avg'] - t * row['non_fraud_amount_sd'] / sqrt(n)
            else:
                return 0
            
        def compute_ci_upper(row: pd.Series) -> float:
            n = row['num_of_prev_transactions'] - row['num_of_prev_frauds']
            if n > 1:
                t = stdtrit(n - 1, 1 - alpha / 2)
                return row['non_fraud_amount_avg'] + t * row['non_fraud_amount_sd'] / sqrt(n)
            else:
                return 0

        df['conf_int_mean_amount_lower'] = df.apply(compute_ci_lower, axis=1)
        df['conf_int_mean_amount_upper'] = df.apply(compute_ci_upper, axis=1)

        df['amount_outside_ci'] = df[['amount', 'conf_int_mean_amount_lower', 'conf_int_mean_amount_upper']].apply(lambda row: 0 if row['conf_int_mean_amount_lower'] == row['conf_int_mean_amount_upper'] else (row['amount'] < row['conf_int_mean_amount_lower'] or row['amount'] > row['conf_int_mean_amount_upper']), axis=1).astype(int)

        df['mean_sess_len_non_fraud'] = df.groupby(['user_id'])[['is_fraud','session_length_seconds']].apply(lambda x: compute_stat(x, 'session_length_seconds', 0, 'avg')).reset_index(level=0, drop=True)
        df['mean_sess_len_fraud'] = df.groupby(['user_id'])[['is_fraud','session_length_seconds']].apply(lambda x: compute_stat(x, 'session_length_seconds', 1, 'avg')).reset_index(level=0, drop=True)

        df['sess_len_avg_closer_to_fraud'] = df[['is_fraud', 'session_length_seconds','mean_sess_len_non_fraud','mean_sess_len_fraud']
                                                          ].apply(lambda row: 0 if row['mean_sess_len_fraud'] == 0 else np.abs(row['session_length_seconds']-row['mean_sess_len_fraud']) < np.abs(row['session_length_seconds']-row['mean_sess_len_non_fraud']), axis = 1).reset_index(level=0, drop=True)

        return df[['transaction_id', 'num_of_prev_transactions', 'prev_transaction_is_fraud', 'num_of_prev_frauds',
                   'amount_avg_closer_to_fraud', 'amount_outside_ci', 'sess_len_avg_closer_to_fraud']]


# Distance Transformer
class DistanceTransformer(Transformer):
    """
    Transformer to calculate distance-based features for user transactions.

    This transformer expects a DataFrame with columns:
    - 'transaction_id': unique identifier for each transaction
    - 'geometry': WKT string representing the transaction location
    - 'user_id': identifier for the user
    - 'timestamp': datetime of the transaction

    It computes:
    - Distance between the last two transactions per user ('dist_last_two')
    - Time difference in hours from the last transaction ('hours_from_last_transaction')
    - Flag indicating if travel speed between transactions is suspiciously high ('too_fast_travel')
    - Rolling average and standard deviation of distance changes per user ('dist_change_avg', 'dist_change_sd')
    - Z-score based flags for distance anomalies ('dist_z_score_1', 'dist_z_score_2', 'dist_z_score_3')
    """

    required_columns = ['transaction_id', 'geometry', 'user_id', 'timestamp']

    def transform(self, df_base: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by adding distance-related features per transaction.

        Parameters
        ----------
        df_base : pd.DataFrame
            Input DataFrame containing the required columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - 'transaction_id'
            - 'dist_last_two': distance in kilometers between current and previous transaction for the same user
            - 'hours_from_last_transaction': hours elapsed since previous transaction for the same user
            - 'too_fast_travel': binary flag indicating suspiciously high travel speed (>180 km/h)
            - 'dist_change_avg': rolling mean of previous distance changes for the user
            - 'dist_change_sd': rolling standard deviation of previous distance changes for the user
            - 'dist_z_score_1': flag if z-score of distance change > 1
            - 'dist_z_score_2': flag if z-score of distance change > 2
            - 'dist_z_score_3': flag if z-score of distance change > 3
        """
        self.check_columns(df_base)
        df = df_base.copy()
        
        # Convert WKT geometry string to shapely Point objects
        df['geometry'] = df['geometry'].apply(loads)

        def compute_distances(group: pd.Series) -> pd.Series:
            # Extract (latitude, longitude) tuples from geometry Points
            coords = group.apply(lambda point: (point.y, point.x))
            
            distances = [0]  # No previous transaction distance for the first record
            for i in range(1, len(coords)):
                # Calculate geodesic distance (in km) between consecutive points
                dist = geodesic(coords.iloc[i], coords.iloc[i-1]).kilometers
                distances.append(dist)
            
            return pd.Series(distances, index=group.index)
        
        df.reset_index(drop=True, inplace=True)
        
        # Compute distances between consecutive transactions per user
        df['dist_last_two'] = df.groupby('user_id')['geometry'].apply(compute_distances).reset_index(level=0, drop=True)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate time difference in hours between transactions per user
        df['hours_from_last_transaction'] = df.groupby('user_id')['timestamp'] \
            .transform(lambda x: (x - x.shift(1)).dt.total_seconds() / 3600) \
            .fillna(0).astype(float)

        # Flag cases where implied travel speed exceeds 180 km/h (possible fraud)
        df['too_fast_travel'] = df[['dist_last_two', 'hours_from_last_transaction']].apply(
            lambda x: 0 if x['hours_from_last_transaction'] == 0 else (x['dist_last_two'] / x['hours_from_last_transaction'] > 180),
            axis=1
        ).astype(int)

        # Sort by user and timestamp for rolling calculations
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        # Rolling average and std deviation of distance changes (excluding current)
        df['dist_change_avg'] = df.groupby('user_id')['dist_last_two'] \
            .apply(lambda x: x.shift().expanding().mean()) \
            .fillna(0).astype(float).reset_index(drop=True)
        
        df['dist_change_sd'] = df.groupby('user_id')['dist_last_two'] \
            .apply(lambda x: x.shift().expanding().std()) \
            .fillna(0).astype(float).reset_index(drop=True)

        # Z-score flags for distance anomalies at thresholds 1, 2, and 3 std deviations
        def compute_z_score_flag(group):
            dist = group['dist_last_two']
            avg = group['dist_change_avg']
            sd = group['dist_change_sd'].replace(0, np.nan)  # avoid division by zero
            
            z = (dist - avg) / sd
            return pd.DataFrame({
                'dist_z_score_1': (z > 1).astype(int),
                'dist_z_score_2': (z > 2).astype(int),
                'dist_z_score_3': (z > 3).astype(int),
            }, index=group.index)

        z_scores = df.groupby('user_id')[['dist_last_two', 'dist_change_avg', 'dist_change_sd']] \
            .apply(compute_z_score_flag) \
            .reset_index(level=0, drop=True)

        df = pd.concat([df, z_scores], axis=1)
        
        return df[['transaction_id',
                   'dist_last_two', 'hours_from_last_transaction',
                   'too_fast_travel', 'dist_change_avg',
                   'dist_change_sd', 'dist_z_score_1', 'dist_z_score_2', 'dist_z_score_3']]

# General stats from Time Series Transformer
import pandas as pd
import numpy as np
from typing import Any

class GeneralHistoryTransformer(Transformer):
    """
    Transformer obliczający historyczne statystyki dotyczące transakcji użytkowników,
    takie jak średnie, mediany, odchylenia standardowe kwot transakcji oraz różnice czasowe
    między transakcjami.
    """

    required_columns = ['transaction_id', 'amount', 'user_id', 'timestamp', 'signup_date']

    def transform(self, df_base: pd.DataFrame) -> pd.DataFrame:
        """
        Transformuje bazowy DataFrame z transakcjami, dodając cechy historyczne na poziomie użytkownika.

        Parameters:
        -----------
        df_base : pd.DataFrame
            Dane transakcyjne zawierające wymagane kolumny.

        Returns:
        --------
        pd.DataFrame
            DataFrame z kolumnami cechami historycznymi dla każdej transakcji.
        """
        self.check_columns(df_base)
        df = df_base.copy()

        # Sortowanie po użytkowniku i czasie transakcji
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        # Średnia kwota poprzednich transakcji użytkownika (bez bieżącej)
        df['avg_amount_prev'] = (
            df
            .groupby('user_id')['amount']
            .transform(lambda s: s.shift().expanding().mean())
            .fillna(0)  # Brak poprzednich transakcji = 0
        )

        # Mediana kwoty poprzednich transakcji
        df['median_amount_prev'] = (
            df
            .groupby('user_id')['amount']
            .transform(lambda s: s.shift().expanding().median())
            .fillna(0)
        )

        # Odchylenie standardowe kwoty poprzednich transakcji, by unikać dzielenia przez 0 w z-score
        df['std_amount_prev'] = (
            df
            .groupby('user_id')['amount']
            .transform(lambda s: s.shift().expanding().std())
            .fillna(1)  # Jeśli brak odchylenia, zakładamy 1 by nie dzielić przez zero
        )

        # Z-score kwoty względem historycznej średniej i odchylenia
        df['z_score_amount'] = (df['amount'] - df['avg_amount_prev']) / df['std_amount_prev']
        # Jeśli odchylenie to 0, ustawiamy z-score na 0
        df.loc[df['std_amount_prev'] == 0, 'z_score_amount'] = 0

        # Flaga, czy z-score przekracza 1 (duże odchylenie)
        df['z_score_greater_amount'] = np.where(abs(df['z_score_amount']) > 1, 1, 0)

        # Data pierwszej rejestracji użytkownika (statyczna w całym zbiorze)
        df['first_signup'] = df.groupby('user_id')['signup_date'].transform('first')

        # Konwersja kolumn dat na datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['first_signup'] = pd.to_datetime(df['first_signup'])

        # Czas poprzedniego zdarzenia — albo poprzednia transakcja albo data rejestracji
        df['prev_event'] = (
            df.groupby('user_id')['timestamp']
            .shift()
            .fillna(df['first_signup'])
        )

        # Różnica w dniach między aktualną a poprzednią transakcją (lub rejestracją)
        df['last_transaction_diff'] = ((df['timestamp'] - df['prev_event']).dt.days).astype(int)

        # Usuwamy kolumny pomocnicze
        df.drop(columns=['first_signup', 'prev_event'], inplace=True)

        # Średnia różnica czasu między poprzednimi transakcjami
        df['avg_last_transaction_diff'] = (
            df.groupby('user_id')['last_transaction_diff']
            .transform(lambda s: s.shift().expanding().mean())
            .fillna(0)
        )

        # Mediana różnicy czasu między poprzednimi transakcjami
        df['median_last_transaction_diff'] = (
            df.groupby('user_id')['last_transaction_diff']
            .transform(lambda s: s.shift().expanding().median())
            .fillna(0)
        )

        # Odchylenie standardowe różnicy czasu między poprzednimi transakcjami
        df['std_last_transaction_diff'] = (
            df.groupby('user_id')['last_transaction_diff']
            .transform(lambda s: s.shift().expanding().std())
            .fillna(1)
        )

        # Flaga, czy mediana poprzednich różnic jest większa niż bieżąca różnica czasu
        df['GreaterPrevMedlast_transaction_diff'] = np.where(
            df['median_last_transaction_diff'] > df['last_transaction_diff'], 1, 0
        )

        # Z-score różnicy czasu względem historycznej średniej i odchylenia
        df['z_score_last_transaction_diff'] = (
            (df['last_transaction_diff'] - df['avg_last_transaction_diff']) / df['std_last_transaction_diff']
        )
        # Jeśli odchylenie to 0, ustawiamy z-score na 0
        df.loc[df['std_last_transaction_diff'] == 0, 'z_score_last_transaction_diff'] = 0

        # Flaga, czy z-score różnicy czasu przekracza 1
        df['z_score_greater_last_transaction_diff'] = np.where(
            abs(df['z_score_last_transaction_diff']) > 1, 1, 0
        )

        # Zwracamy tylko wybrane kolumny cech
        return df[[
            'transaction_id',
            'avg_amount_prev', 'median_amount_prev', 'std_amount_prev', 'z_score_amount', 'z_score_greater_amount',
            'last_transaction_diff', 'avg_last_transaction_diff', 'median_last_transaction_diff', 'std_last_transaction_diff',
            'GreaterPrevMedlast_transaction_diff', 'z_score_last_transaction_diff', 'z_score_greater_last_transaction_diff'
        ]]



import pandas as pd
import numpy as np
import pickle as pkl

class FraudFractionTransformer(Transformer):
    """
    Transformer obliczający statystyki dotyczące ułamka fraudów na podstawie
    historycznych danych z podziałem na kombinacje krajów i okresy czasowe.

    Wczytuje słowniki z preliczonymi danymi, a następnie dla każdej transakcji
    zwraca liczbę legalnych i fraudalnych transakcji oraz ich ułamek z poprzednich miesięcy.
    """

    required_columns = ["timestamp", "country_user", "country_merchant", "country_transaction"]

    def __init__(self, path: str = "./dictionaries/"):
        """
        Inicjalizuje transformer, ładując słowniki z plików pickle.

        Parameters:
        -----------
        path : str
            Ścieżka do folderu ze słownikami.
        """
        with open(path + "year_month_countires_combinations_counted.pkl", "rb") as file:
            self.dictionary = pkl.load(file)

        with open(path + "year_month_countries_combinations_cumulatiove_counted.pkl", "rb") as file:
            self.cumulative_dictionary = pkl.load(file)

    def fetch_fraud_fraction(
        self, year: int, month: int,
        user_country: str, merchant_country: str, transaction_country: str,
        default: np.ndarray = np.array([-1, -1], dtype=np.int16),
        fill: int = -1
    ) -> np.ndarray:
        """
        Pobiera statystyki fraudów i transakcji legalnych z poprzednich trzech miesięcy.

        Parameters:
        -----------
        year, month : int
            Rok i miesiąc transakcji.
        user_country, merchant_country, transaction_country : str
            Kody krajów użytkownika, sprzedawcy i transakcji.
        default : np.ndarray
            Wartość domyślna, gdy brak danych.
        fill : int
            Wartość do wypełnienia, gdy nie ma danych dla ułamka fraudów.

        Returns:
        --------
        np.ndarray
            Tablica z liczbą legalnych, fraudalnych i ułamkiem fraudów dla 6 okresów
            (3 miesiące i ich kumulatywne dane).
        """
        counts = []

        for _ in range(3):
            # Cofamy się o miesiąc
            if month == 1:
                month = 12
                year -= 1
            else:
                month -= 1

            # Jeśli rok jest przed 2022, brak danych - wypełniamy defaultami
            if year < 2022:
                counts.extend(list(default) + [fill] + list(default) + [fill])
                continue

            key = f"{year}_{month:02}_{user_country}_{merchant_country}_{transaction_country}"

            # Pobieramy dane ze słownika zwykłego
            scores = self.dictionary.get(key, default)
            counts.extend(list(scores))
            if scores.sum() != -2:
                counts.append(scores[1] / scores.sum())  # ułamek fraudów
            else:
                counts.append(fill)

            # Pobieramy dane ze słownika kumulatywnego
            scores = self.cumulative_dictionary.get(key, default)
            counts.extend(list(scores))
            if scores.sum() != -2:
                counts.append(scores[1] / scores.sum())
            else:
                counts.append(fill)

        return np.array(counts)

    def transform(self, df: pd.DataFrame,
                  default: np.ndarray = np.array([-1, -1], dtype=np.int16),
                  fill: int = -1) -> pd.DataFrame:
        """
        Transformuje DataFrame, dodając kolumny z liczbą i ułamkiem fraudów dla kombinacji krajów.

        Parameters:
        -----------
        df : pd.DataFrame
            Dane wejściowe zawierające kolumny z wymaganymi krajami i timestamp.
        default : np.ndarray
            Wartość domyślna, gdy brak danych.
        fill : int
            Wartość do uzupełniania w przypadku braku ułamka fraudów.

        Returns:
        --------
        pd.DataFrame
            DataFrame z kolumnami cech związanych z frakcją fraudów na poziomie transakcji.
        """
        columns = [
            "legit_transactions_last_month",
            "frauds_last_mont",
            "fraud_fraction_last_month",

            "legit_transactions_before_this_month",
            "frauds_before_this_month",
            "fraud_fraction_before_last_month",

            "legit_transactions_two_months_before",
            "frauds_two_months_before",
            "fraud_fraction_two_months_before",

            "legit_transactions_before_the_previous_month",
            "frauds_before_the_previous_month",
            "fraud_fraction_before_the_previous_month",

            "legit_transactions_three_months_before",
            "frauds_three_months_before",
            "fraud_fraction_three_months_before",

            "legit_transactions_before_the_previous_previous_month",
            "frauds_before_the_previous_previous_month",
            "fraud_fraction_before_the_previous_previous_month"
        ]

        scores = []

        # Iterujemy po każdej transakcji
        for _, row in df.iterrows():
            year = int(row["timestamp"][:4])
            month = int(row["timestamp"][5:7])
            country_user = row["country_user"]
            country_merchant = row["country_merchant"]
            country_transaction = row["country_transaction"]

            scores.append(self.fetch_fraud_fraction(
                year=year, month=month,
                user_country=country_user,
                merchant_country=country_merchant,
                transaction_country=country_transaction,
                default=default,
                fill=fill
            ))

        # Tworzymy DataFrame z wynikami
        df_scores = pd.DataFrame(scores, columns=columns)
        df_result = df[['transaction_id']].copy()
        for col in df_scores.columns:
            df_result[col] = df_scores[col].to_numpy()

        return df_result

class CountryEncoderTransformer(Transformer):
    required_columns = ['transaction_id', 'country_user', 'country_merchant', 'country_transaction']

    def transform(self, df_base: pd.DataFrame) -> pd.DataFrame:
        self.check_columns(df_base)
        df = df_base[['transaction_id', 'country_user', 'country_merchant', 'country_transaction']].copy()

        transaction_id=df[['transaction_id']].copy()
        df=df.drop(columns=['transaction_id'],axis=1)
        countries_col = ['country_user', 'country_merchant', 'country_transaction']
        OH = OneHotEncoder(handle_unknown='error', drop='if_binary', sparse_output=False, dtype='int64')

        x = OH.fit_transform(df[countries_col])
        df = pd.DataFrame(x, columns=OH.get_feature_names_out(countries_col)).reset_index(drop=True)

        df[['transaction_id']] = transaction_id.reset_index(drop=True)

        return df
