# backend/app/utils/preprocess_utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import os
from typing import Tuple, List, Any, Optional

def load_dataset_to_df(file_path: str) -> pd.DataFrame:
    """Loads a dataset file into a pandas DataFrame."""
    print(f"UTILS: Original file path: {file_path}")
    
    # Check if the path is absolute or relative
    if not os.path.isabs(file_path):
        # Get the project root directory (assuming the structure is consistent)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        # Combine with the relative path
        full_path = os.path.join(project_root, file_path)
        print(f"UTILS: Converting relative path to absolute: {full_path}")
    else:
        full_path = file_path
    
    print(f"UTILS: Loading dataset from: {full_path}")
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Dataset file not found: {full_path}. Original path: {file_path}")
        
    file_path = full_path  # Use the full path for loading

    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_extension == ".csv":
        df = pd.read_csv(file_path)
    elif file_extension in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    print(f"UTILS: Dataset loaded. Shape: {df.shape}, Columns: {df.columns.tolist()}")
    df = df.reset_index(drop=True) 
    return df

def prepare_numerical_data(
    df: pd.DataFrame, 
    scaler_path: Optional[str] = None, 
    fit_scaler: bool = True,
    scaler_type: str = "minmax", 
) -> Tuple[pd.DataFrame, Any, List[str]]:
    """
    Selects numerical columns, optionally drops ID-like columns, handles NaNs, and scales the data.
    Saves the scaler if fit_scaler is True and scaler_path is provided.
    Returns scaled numerical DataFrame, the scaler object, and numerical column names used for scaling.
    The index of the returned DataFrame matches the input numerical_df.
    """
    numerical_df_full_index = df.select_dtypes(include=np.number)
    if numerical_df_full_index.empty:
        raise ValueError("No numerical columns found in the dataset to begin with.")
    
    numerical_df = numerical_df_full_index.copy()

    # --- Logic to drop ID-like columns ---
    id_like_keywords = ['id', 'index', 'key', 'identifier', 'number', 'num', 'no', 'record', 'unnamed:']
    potential_id_cols = set()

    for col in numerical_df.columns:
        col_lower = col.lower()
        
        # Exact matches for common ID names
        if col_lower == 'id' or col_lower == 'id_col' or col_lower == 'index': # Explicitly add 'id_col'
            potential_id_cols.add(col)
            continue # Already added, move to next column

        # Check for endings like _id, _key, _no, _number
        if any(col_lower.endswith(f"_{keyword}") for keyword in id_like_keywords):
            potential_id_cols.add(col)
            continue
            
        # Check for beginnings like id_, index_, key_, unnamed:
        if any(col_lower.startswith(keyword) for keyword in id_like_keywords): # Check full keyword for startswith
            potential_id_cols.add(col)
            continue

        # Check if the column name *is* one of the keywords (e.g. a column just named "number")
        if col_lower in id_like_keywords:
            potential_id_cols.add(col)
            continue

    actual_cols_to_drop = [col for col in potential_id_cols if col in numerical_df.columns]

    if actual_cols_to_drop:
        # Remove duplicates from the list before printing and dropping
        unique_actual_cols_to_drop = sorted(list(set(actual_cols_to_drop)))
        print(f"UTILS: Attempting to drop potential ID/index columns before scaling: {unique_actual_cols_to_drop}")
        numerical_df = numerical_df.drop(columns=unique_actual_cols_to_drop, errors='ignore')
        if numerical_df.empty:
            raise ValueError("No numerical columns left after dropping ID-like columns. Check drop heuristics or input data.")
        print(f"UTILS: Columns after dropping ID-like ones: {numerical_df.columns.tolist()}")
    else:
        print("UTILS: No ID-like columns identified for dropping based on heuristics.")
    # --- END: Logic to drop ID-like columns ---
    
    original_numerical_cols_for_scaling = numerical_df.columns.tolist()
    print(f"UTILS: Numerical columns to be used for scaling: {original_numerical_cols_for_scaling}")

    if not original_numerical_cols_for_scaling:
        raise ValueError("No numerical columns selected for scaling after potential drops.")

    if numerical_df.isnull().values.any():
        print("UTILS: NaNs found in numerical data. Imputing with column medians.")
        for col_name in numerical_df.columns[numerical_df.isnull().any()]: # Iterate over column names
            median_val = numerical_df[col_name].median()
            numerical_df[col_name].fillna(median_val, inplace=True)
        if numerical_df.isnull().values.any():
            print("UTILS: Persistent NaNs after median. Filling remaining with 0.")
            numerical_df.fillna(0, inplace=True)
    
    data_values = numerical_df.values
    
    if scaler_type.lower() == "standard":
        selected_scaler = StandardScaler()
    elif scaler_type.lower() == "minmax":
        selected_scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler_type: {scaler_type}. Choose 'minmax' or 'standard'.")

    if fit_scaler:
        scaler_instance = selected_scaler
        scaled_values = scaler_instance.fit_transform(data_values)
        if scaler_path:
            if os.path.dirname(scaler_path) and not os.path.exists(os.path.dirname(scaler_path)):
                 os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler_instance, scaler_path)
            print(f"UTILS: Scaler ({scaler_type}) fitted and saved to: {scaler_path}")
    else:
        if not scaler_path or not os.path.exists(scaler_path):
            raise ValueError("Scaler path must be provided and exist if fit_scaler is False.")
        scaler_instance = joblib.load(scaler_path)
        print(f"UTILS: Scaler ({scaler_type}) loaded from: {scaler_path}")
        scaled_values = scaler_instance.transform(data_values)
        
    scaled_numerical_df = pd.DataFrame(scaled_values, columns=original_numerical_cols_for_scaling, index=numerical_df.index)
    print(f"UTILS: Numerical data prepared and scaled. Shape: {scaled_numerical_df.shape}")
    return scaled_numerical_df, scaler_instance, original_numerical_cols_for_scaling


def split_data_for_autoencoder(
    scaled_numerical_df: pd.DataFrame, 
    validation_split: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not (0 < validation_split < 1):
        raise ValueError("validation_split must be strictly between 0 and 1.")
        
    X_train, X_val = train_test_split(
        scaled_numerical_df, 
        test_size=validation_split, 
        random_state=random_state
    )
    print(f"UTILS: Data split for AE: Train shape: {X_train.shape}, Validation shape: {X_val.shape}")
    return X_train, X_val