"""Strategy selection depending on dataset size, mechanism, user preference, and resources."""
from typing import Dict, Tuple


def pick_strategy(size_rows: int, mechanism: str, user_choice: str, has_gpu: bool) -> Tuple[str, Dict]:
    """Return (strategy_name, default_params)"""
    if user_choice and user_choice != "auto":
        return user_choice, {}

    # Auto selection rules based on professor guide
    if size_rows < 10_000:
        # small → allow complex
        return ("knn", {"n_neighbors": 5}) if mechanism == "MCAR" else ("mice", {"max_iter": 10})

    if size_rows < 1_000_000:
        # medium
        if mechanism == "MCAR":
            return "simple", {"strategy": "median"}
        if mechanism == "MAR":
            if has_gpu:
                return "lightgbm", {}
            return "mice", {"max_iter": 10}
        return "mice", {"max_iter": 10}

    # large datasets
    if mechanism == "MCAR":
        return "dask_group", {"stat": "median"}
    # For MAR / MNAR on big data fall back to grouped statistics
    return "dask_group", {"stat": "median"}
