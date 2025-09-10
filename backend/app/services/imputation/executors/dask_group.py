"""Grouped mean/median/mode using Dask for very large datasets."""
from typing import Dict
import dask.dataframe as dd
import pandas as pd


def run(df: pd.DataFrame, params: Dict):
    stat = params.get("stat", "median")
    group_col = params.get("group_col")  # optional

    ddf = dd.from_pandas(df, npartitions=16)
    df_filled = df.copy()

    numeric_cols = set(df.select_dtypes(include=["number"]).columns)

    for col in df.columns:
        if not df[col].isna().any():
            continue

        is_numeric = col in numeric_cols

        if group_col and group_col in df.columns:
            grouped = ddf.groupby(group_col)[col]
            if is_numeric and stat == "mean":
                fill_values = grouped.mean().compute()
            elif is_numeric and stat == "median":
                fill_values = grouped.median().compute()
            else:
                # Fallback to mode for non-numeric or when 'mode' requested
                def _mode(s: pd.Series):
                    m = s.mode(dropna=True)
                    return m.iloc[0] if not m.empty else None

                fill_values = grouped.apply(
                    _mode,
                    meta=(col, df[col].dtype),
                ).compute()

            mapped = df[group_col].map(fill_values)
            df_filled[col] = df[col].fillna(mapped)
        else:
            if is_numeric and stat == "mean":
                value = df[col].mean()
            elif is_numeric and stat == "median":
                value = df[col].median()
            else:
                mode_series = df[col].mode(dropna=True)
                value = mode_series.iloc[0] if not mode_series.empty else None

            if value is not None:
                df_filled[col] = df[col].fillna(value)

    return df_filled
