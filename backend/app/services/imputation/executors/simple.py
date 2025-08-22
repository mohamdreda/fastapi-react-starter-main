"""Simple mean/median/mode imputation."""
from typing import Dict
import pandas as pd


def run(df: pd.DataFrame, params: Dict):
    strategy = params.get("strategy", "mean")
    df_filled = df.copy()

    for col in df.columns:
        if df[col].isna().any():
            if strategy == "mean" and df[col].dtype.kind in "if":
                value = df[col].mean()
            elif strategy == "median" and df[col].dtype.kind in "if":
                value = df[col].median()
            else:
                value = df[col].mode().iloc[0] if not df[col].mode().empty else None
            df_filled[col] = df[col].fillna(value)
    return df_filled
