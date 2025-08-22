"""High-level orchestration for the data-imputation workflow."""
from __future__ import annotations

from typing import Dict, Any, List, Tuple
import importlib
import pandas as pd
from app.services.upload import read_dataframe  # existing helper to read dataset path
from app.services.imputation import profiling, strategy, validation, utils

EXECUTOR_MODULE_PATH = "app.services.imputation.executors"


class ImputationError(Exception):
    """Domain exception for imputation failures."""


def _load_executor(name: str):
    try:
        return importlib.import_module(f"{EXECUTOR_MODULE_PATH}.{name}")
    except ModuleNotFoundError as exc:  # noqa: BLE001
        raise ImputationError(f"Executor '{name}' not implemented") from exc


def _postprocess_dataframe(
    df_orig: pd.DataFrame,
    df_new: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """Apply basic sanity corrections to the imputed dataframe.

    1. Clip negative values when the original column contained only non-negative values.
    2. Round and cast to *int* when the original non-missing values were all integers.
    3. For percentage/ratio style columns (heuristic: column name starts with *discount* or
       original max ≤ 1), clip values to the [0, 1] interval.

    Returns the corrected dataframe and a list of human-readable warnings describing
    each modification performed.
    """
    df_out = df_new.copy()
    warnings: List[str] = []

    numeric_cols = df_out.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        orig_non_na = df_orig[col].dropna()
        if orig_non_na.empty:
            continue

        # 1. Integer-like columns – round values
        if (orig_non_na % 1 == 0).all():
            rounded = df_out[col].round().astype(int)
            diff_mask = rounded != df_out[col]
            if diff_mask.any():
                warnings.append(
                    f"Column '{col}' rounded to integer for {int(diff_mask.sum())} rows."
                )
                df_out[col] = rounded

        # 2. Clip negatives when original had only non-negative values
        if orig_non_na.min() >= 0:
            neg_mask = df_out[col] < 0
            if neg_mask.any():
                warnings.append(
                    f"Negative values clipped to 0 in column '{col}' for {int(neg_mask.sum())} rows."
                )
                df_out.loc[neg_mask, col] = 0

        # 3. Discount / percentage columns clipping
        if col.lower().startswith("discount") or orig_non_na.max() <= 1:
            clip_mask = (df_out[col] < 0) | (df_out[col] > 1)
            if clip_mask.any():
                warnings.append(
                    f"Values clipped to [0,1] in column '{col}' for {int(clip_mask.sum())} rows."
                )
                df_out[col] = df_out[col].clip(lower=0, upper=1)

    return df_out, warnings



def run_imputation(
    dataset: "Dataset",  # <-- On passe l'objet entier
    req: "ImputationRunRequest",  # forward ref; actual class in schemas
    user_id: int,
) -> Dict[str, Any]:
    """Top-level callable used by API route and Celery task."""
    df = read_dataframe(dataset.file_path, dataset.file_type)
    stats = profiling.profile_missing(df)
    strat, default_params = strategy.pick_strategy(
        len(df), stats["mechanism"], req.strategy, has_gpu=False  # TODO: detect GPU
    )
    params = {**default_params, **(req.params or {})}

    # ---
    # Performance timing starts
    import time
    import numpy as np
    t0 = time.perf_counter()

    executor = _load_executor(strat)
    df_imputed_raw = executor.run(df, params)

    runtime_sec = time.perf_counter() - t0

    # Benchmark metrics (5% artificial mask)
    def _benchmark_imputer(df_orig: pd.DataFrame) -> Dict[str, float]:
        frac = 0.05
        df_masked = df_orig.copy()
        mask_info: Dict[str, pd.Index] = {}
        rng = np.random.default_rng(42)
        for col in df_masked.columns:
            available_idx = df_masked[df_masked[col].notna()].index
            if len(available_idx) == 0:
                continue
            sample_size = max(1, int(len(available_idx) * frac))
            sampled_idx = rng.choice(available_idx, size=sample_size, replace=False)
            mask_info[col] = sampled_idx
            df_masked.loc[sampled_idx, col] = np.nan
        # Run imputation on the masked copy
        df_imputed_bench = executor.run(df_masked, params)
        # Compute metrics
        num_rmses, num_maes = [], []
        cat_correct = cat_total = 0
        for col, idx in mask_info.items():
            true_vals = df_orig.loc[idx, col]
            pred_vals = df_imputed_bench.loc[idx, col]
            if df_orig[col].dtype.kind in "if":
                diff = pred_vals.astype(float) - true_vals.astype(float)
                num_rmses.append(np.sqrt(np.mean(np.square(diff))))
                num_maes.append(np.mean(np.abs(diff)))
            else:
                cat_correct += (pred_vals == true_vals).sum()
                cat_total += len(idx)
        metrics: Dict[str, float] = {}
        if num_rmses:
            metrics["rmse"] = float(np.mean(num_rmses))
            metrics["mae"] = float(np.mean(num_maes))
        if cat_total:
            metrics["cat_accuracy"] = float(cat_correct) / cat_total
        return metrics

    bench_metrics = _benchmark_imputer(df)

    # Post-processing: clip negatives, round integer-like columns, etc.
    df_imputed, warnings = _postprocess_dataframe(df, df_imputed_raw)

    val = validation.validate_and_save(df, df_imputed, dataset.id, user_id)

    # Save cleaned dataset (preserve format & name provided by user)
    cleaned_path = utils.get_artifact_path(dataset.id, user_id, req.output_name)
    _save_dataframe(df_imputed, cleaned_path)

    preview = df_imputed.head(10).to_dict(orient="records")

    missing_before = {col: int(df[col].isna().sum()) for col in df.columns}
    missing_after = {col: int(df_imputed[col].isna().sum()) for col in df.columns}
    filled_counts = {col: missing_before[col] - missing_after[col] for col in df.columns}

    total_missing_before = int(sum(missing_before.values()))
    total_missing_after = int(sum(missing_after.values()))
    total_filled = total_missing_before - total_missing_after
    rows_with_missing_before = int(df.isna().any(axis=1).sum())

    return {
        "status": "success",
        "message": "Imputation completed",
        "summary": {
            "strategy": strat,
            "mechanism": stats["mechanism"],
            "missing_before": missing_before,
            "missing_after": missing_after,
            "filled_count": filled_counts,
            "total_missing_before": total_missing_before,
            "total_missing_after": total_missing_after,
            "total_filled": total_filled,
            "performance": {"runtime_seconds": round(runtime_sec, 4), **bench_metrics},
        },
        "imputed_dataset_path": cleaned_path,
        "validation_report_path": val["path"],
        "preview": preview,
        "imputed_values_count": total_filled,
        "imputed_rows_count": rows_with_missing_before,
        "warnings": warnings,
    }


def _save_dataframe(df: pd.DataFrame, path: str):
    if path.endswith(".csv"):
        df.to_csv(path, index=False)
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        df.to_excel(path, index=False)
    elif path.endswith(".json"):
        df.to_json(path, orient="records")
    elif path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        raise ImputationError(f"Unsupported output format: {path}")
