"""Missing-value profiling & mechanism detection (MCAR/MAR/MNAR)."""
from typing import Dict, Any
import pandas as pd
import numpy as np
from scipy import stats
try:
    from statsmodels.stats.missing import LittleMCAR  # available in dev builds
except ImportError:  # pragma: no cover
    LittleMCAR = None
import logging

logger = logging.getLogger(__name__)


def profile_missing(df: pd.DataFrame, sample_size: int = 50000) -> Dict[str, Any]:
    """Return a dict with percentage missing per column and mechanism hint."""
    total_rows = len(df)
    pct_per_col = df.isna().mean().round(4).to_dict()
    overall_pct = float(np.mean(list(pct_per_col.values())))

    # Sample for statistical tests to save time on big data
    if total_rows > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df

    mechanism = _detect_mechanism(df_sample)
    return {
        "pct_missing": pct_per_col,
        "overall_pct": overall_pct,
        "mechanism": mechanism,
    }


def _detect_mechanism(df: pd.DataFrame) -> str:
    """Rudimentary mechanism detection using Little's MCAR test and simple logistic masks."""
    if LittleMCAR is not None:
        try:
            mcar_p = LittleMCAR(df).pvalue
            if mcar_p > 0.05:
                return "MCAR"
        except Exception as exc:  # noqa: BLE001
            logger.debug("Little MCAR failed: %s", exc)

    # Simple MAR heuristic: for each column, logistic regression of mask against others
    try:
        import sklearn.linear_model  # lazy import
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        cat = df.select_dtypes(exclude=["number"]).columns.tolist()

        transformers = []
        if numeric:
            transformers.append(("num", "passthrough", numeric))
        if cat:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
        if not transformers:
            return "MNAR"  # degenerate dataframe
        pre = ColumnTransformer(transformers)

        for col in df.columns:
            mask = df[col].isna().astype(int)
            if mask.sum() in (0, len(df)):
                continue
            X = df.drop(columns=[col])
            pipe = Pipeline([("pre", pre), ("lr", sklearn.linear_model.LogisticRegression(max_iter=1000))])
            try:
                pipe.fit(X, mask)
                score = pipe.score(X, mask)
                if score > 0.6:
                    return "MAR"
            except Exception:  # pylint: disable=broad-except
                continue
    except ImportError:
        logger.debug("sklearn not available for MAR heuristic")

    return "MNAR"
