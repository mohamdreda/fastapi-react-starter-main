"""
Preprocessing service for data deduplication.

This module handles the standardization of different field types:
- Text fields: Basic cleaning (lowercase, strip)
- Numeric fields: Min-Max scaling
- Categorical fields: Label encoding
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import os
import json
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import logging

from app.config.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

def _get_preprocessing_artifact_path(
    base_path: str,
    dataset_id: int,
    user_id: int,
    artifact_name: str
) -> str:
    """Create and return the path for preprocessing artifacts."""
    dir_path = os.path.join(base_path, f"user_{user_id}", f"dataset_{dataset_id}", "deduplication", "preprocessing")
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, artifact_name)

def preprocess_text(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
    """
    Apply basic text preprocessing: lowercase and strip whitespace.
    
    Args:
        df: Input DataFrame
        text_columns: List of text column names
        
    Returns:
        DataFrame with preprocessed text columns
    """
    df_processed = df.copy()
    
    for col in text_columns:
        if col in df.columns:
            # Convert to string (in case of non-string values)
            df_processed[col] = df_processed[col].astype(str)
            # Apply lowercase and strip
            df_processed[col] = df_processed[col].str.lower().str.strip()
            
    return df_processed

def preprocess_numeric(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """
    Apply Min-Max scaling to numeric columns.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric column names
        
    Returns:
        DataFrame with preprocessed numeric columns
    """
    df_processed = df.copy()
    
    # Create a scaler
    scaler = MinMaxScaler()
    
    for col in numeric_columns:
        if col in df.columns:
            # Handle missing values
            if df_processed[col].isna().any():
                # Fill missing values with median
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            
            # Reshape for scaler
            values = df_processed[col].values.reshape(-1, 1)
            
            try:
                # Scale the values
                scaled_values = scaler.fit_transform(values)
                # Create a new column with scaled values
                df_processed[f"{col}_scaled"] = scaled_values
            except Exception as e:
                logger.error(f"Error scaling column {col}: {str(e)}")
                # Keep original values if scaling fails
                df_processed[f"{col}_scaled"] = df_processed[col]
    
    return df_processed

def preprocess_categorical(df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    """
    Apply Label Encoding to categorical columns.
    
    Args:
        df: Input DataFrame
        categorical_columns: List of categorical column names
        
    Returns:
        DataFrame with preprocessed categorical columns
    """
    df_processed = df.copy()
    encoders = {}
    
    for col in categorical_columns:
        if col in df.columns:
            # Create an encoder
            encoder = LabelEncoder()
            
            # Handle missing values
            df_processed[col] = df_processed[col].fillna('missing')
            
            # Convert to string (in case of numeric categories)
            df_processed[col] = df_processed[col].astype(str)
            
            try:
                # Encode the values
                encoded_values = encoder.fit_transform(df_processed[col])
                # Create a new column with encoded values
                df_processed[f"{col}_encoded"] = encoded_values
                # Store the encoder for later use
                encoders[col] = encoder
            except Exception as e:
                logger.error(f"Error encoding column {col}: {str(e)}")
                # Keep original values if encoding fails
                df_processed[f"{col}_encoded"] = df_processed[col]
    
    return df_processed, encoders

async def run_preprocessing(
    df: pd.DataFrame,
    text_columns: List[str],
    numeric_columns: List[str],
    categorical_columns: List[str],
    dataset_id: int,
    user_id: int,
    output_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the preprocessing pipeline on a dataset.
    
    Args:
        df: Input DataFrame
        text_columns: List of text column names
        numeric_columns: List of numeric column names
        categorical_columns: List of categorical column names
        dataset_id: ID of the dataset
        user_id: ID of the user
        output_name: Optional output filename base
        
    Returns:
        Dictionary with preprocessing results and metadata
    """
    try:
        # Process text columns
        df_processed = preprocess_text(df, text_columns)
        
        # Process numeric columns
        df_processed = preprocess_numeric(df_processed, numeric_columns)
        
        # Process categorical columns
        df_processed, encoders = preprocess_categorical(df_processed, categorical_columns)
        
        # Save the preprocessed data
        filename_base = output_name or "preprocessed_data"
        output_path = _get_preprocessing_artifact_path(
            settings.DATASET_UPLOAD_DIR,
            dataset_id,
            user_id,
            f"{filename_base}.csv"
        )
        
        df_processed.to_csv(output_path, index=False)
        
        # Save the encoders
        encoders_path = _get_preprocessing_artifact_path(
            settings.DATASET_UPLOAD_DIR,
            dataset_id,
            user_id,
            "encoders.json"
        )
        
        # Convert encoders to a serializable format
        encoders_dict = {}
        for col, encoder in encoders.items():
            encoders_dict[col] = {
                "classes": encoder.classes_.tolist()
            }
        
        with open(encoders_path, 'w') as f:
            json.dump(encoders_dict, f)
        
        # Create a summary of preprocessing
        # Convert filesystem paths to web paths (ensure leading slash)
        web_output_path = output_path if output_path.startswith(('/', '\\')) else f"/{output_path}"
        web_encoders_path = encoders_path if encoders_path.startswith(('/', '\\')) else f"/{encoders_path}"

        numeric_scaled_cols = [f"{col}_scaled" for col in numeric_columns if f"{col}_scaled" in df_processed.columns]
        categorical_encoded_cols = [f"{col}_encoded" for col in categorical_columns if f"{col}_encoded" in df_processed.columns]

        summary = {
            "rows_processed": len(df),
            "columns_processed": len(df_processed.columns),
            "text_columns_cleaned": text_columns,
            "numeric_columns_scaled": numeric_columns,
            "numeric_scaled_columns": numeric_scaled_cols,
            "categorical_columns_encoded": categorical_columns,
            "categorical_encoded_columns": categorical_encoded_cols,
            "output_path": web_output_path,
            "encoders_path": web_encoders_path
        }
        
        return {
            "status": "success",
            "message": "Preprocessing completed successfully",
            "summary": summary,
            "preprocessed_data_path": web_output_path,
            "encoders_path": web_encoders_path,
            "filename_base": filename_base
        }
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error in preprocessing: {str(e)}",
            "error": str(e)
        }
