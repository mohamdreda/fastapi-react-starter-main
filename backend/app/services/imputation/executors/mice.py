from typing import Dict
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


def run(df: pd.DataFrame, params: Dict):
    max_iter = params.get("max_iter", 10)
    imputer = IterativeImputer(max_iter=max_iter, random_state=0)
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df_numeric = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)
    df_filled = df.copy()
    df_filled[numeric_cols] = df_numeric
    return df_filled
