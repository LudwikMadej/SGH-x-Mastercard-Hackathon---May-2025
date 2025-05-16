import pandas as pd
from typing import List
from Transfomers import *
import copy

class Merger:
    def __init__(self, df_base: pd.DataFrame):
        """
        Initialize the Merger with a base DataFrame and instantiate all transformers.

        Parameters
        ----------
        df_base : pd.DataFrame
            The base DataFrame containing the original transaction data.
        """
        self.DeviceTransformer = DeviceTransformer()
        self.ScoreTransformer = ScoreTransformer()
        self.TimeTransformer = TimeTransformer()
        self.MetricsTransformer = MetricsTransformer()
        self.CountryTransformer = CountryTransformer()
        self.FraudWiseHistoryTransformer = FraudWiseHistoryTransformer()
        self.DistanceTransformer = DistanceTransformer()
        self.GeneralHistoryTransformer = GeneralHistoryTransformer()
        self.FraudFractionTransformer = FraudFractionTransformer()

        self.transformerList = [
            self.MetricsTransformer,
            self.ScoreTransformer,
            self.TimeTransformer,
            self.DeviceTransformer,
            self.CountryTransformer,
            self.FraudWiseHistoryTransformer,
            self.DistanceTransformer,
            self.GeneralHistoryTransformer,
            self.FraudFractionTransformer
        ]

        self.df_base = df_base

    def mergeAndTransform(self) -> pd.DataFrame:
        """
        Apply all transformers to the base DataFrame and merge their outputs
        on 'transaction_id'.

        Returns
        -------
        pd.DataFrame
            The merged DataFrame including all transformed features.
        """
        transformed_dfs = []
        for i, transformer in enumerate(self.transformerList):
            new_df = transformer.transform(self.df_base)
            print(f"Success: Transformer {i} applied.")
            transformed_dfs.append(new_df)

        result_df = self.df_base.copy()
        for df in transformed_dfs:
            result_df = result_df.merge(df, how='inner', on='transaction_id')

        return result_df




