"""Grouped mean/median/mode using Dask for very large datasets."""
from typing import Dict
import dask.dataframe as dd
import pandas as pd


def run(df: pd.DataFrame, params: Dict):
    stat = params.get("stat", "median")
    group_col = params.get("group_col")  # optional

    ddf = dd.from_pandas(df, npartitions=16)
    df_filled = df.copy()

    for col in df.columns:
        if df[col].isna().any():
            if group_col and group_col in df.columns:
                grouped = ddf.groupby(group_col)[col]
                if stat == "mean":
                    fill_values = grouped.mean().compute()
                elif stat == "median":
                    fill_values = grouped.median().compute()
                else:
                    fill_values = grouped.apply(lambda x: x.mode().iloc[0], meta=(col, df[col].dtype)).compute()

                df_filled[col] = df[col].fillna(df[group_col].map(fill_values))
            else:
                if stat == "mean":
                    value = df[col].mean()
                elif stat == "median":
                    value = df[col].median()
                else:
                    value = df[col].mode().iloc[0]
                df_filled[col] = df[col].fillna(value)
    return df_filled
