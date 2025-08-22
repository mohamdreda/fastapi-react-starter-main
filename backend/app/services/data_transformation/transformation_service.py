"""
Data Transformation Service

This module provides a service for transforming data through various encoding and scaling operations.
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

# Local imports
from app.db import models
from app.schemas.transformation import (
    CategoricalEncodingConfig, FeatureScalingConfig,
    CategoricalEncodingMethod, FeatureScalingMethod
)
from app.utils.file_utils import save_dataframe
from app.services.data_transformation.categorical_encoding.one_hot_encoder import OneHotEncoder
from app.services.data_transformation.categorical_encoding.label_encoder import LabelEncoder
from app.services.data_transformation.feature_scaling.robust_scaler import RobustScaler
from app.services.data_transformation.feature_scaling.standard_scaler import StandardScaler

# Configure logging
logger = logging.getLogger(__name__)

from app.config.config import get_settings

settings = get_settings()

class DataTransformationService:
    """
    Service for applying data transformations including encoding and scaling.
    """
    
    def __init__(
        self,
        dataset_id: int,
        user_id: int,
        config: Dict[str, Any],
        db: AsyncSession
    ):
        """
        Initialize the transformation service.
        
        Args:
            dataset_id: ID of the dataset to transform
            user_id: ID of the user performing the transformation
            config: Transformation configuration dictionary
            db: Database session
        """
        self.dataset_id = dataset_id
        self.user_id = user_id
        self.config = config
        self.db = db
        self.transformers = {}  # Store transformers for reuse
        self.data = None  # Will store the loaded dataset
        self.df = None  # DataFrame to store the loaded dataset
        self._initialize_transformers()
        
    def _initialize_transformers(self):
        """Initialize the transformers based on the configuration."""
        if self.config is None:
            self.config = {}
            logger.info("No configuration provided, using empty config")
            return
            
        try:
            # Initialize categorical encoders if configured
            if 'categorical_encoding' in self.config and isinstance(self.config['categorical_encoding'], dict):
                if 'methods' in self.config['categorical_encoding'] and isinstance(self.config['categorical_encoding']['methods'], list):
                    for method_config in self.config['categorical_encoding']['methods']:
                        if not isinstance(method_config, dict):
                            continue
                            
                        method = method_config.get('method')
                        columns = method_config.get('columns', [])
                        
                        if not isinstance(columns, list) or not columns:
                            logger.debug("No valid columns specified for method: %s", method)
                            continue
                            
                        if method == 'one_hot':
                            drop = method_config.get('drop', 'first')
                            for col in columns:
                                if not isinstance(col, str):
                                    logger.warning("Skipping invalid column name: %s", col)
                                    continue
                                self.transformers[f'one_hot_{col}'] = OneHotEncoder(columns=[col], drop=drop)
                                logger.debug("Initialized OneHotEncoder for column: %s (drop=%s)", col, drop)
                                
                        elif method == 'label':
                            categories = method_config.get('categories', {})
                            if not isinstance(categories, dict):
                                categories = {}
                                
                            for col in columns:
                                if not isinstance(col, str):
                                    logger.warning("Skipping invalid column name: %s", col)
                                    continue
                                    
                                col_categories = categories.get(col, [])
                                if not isinstance(col_categories, list):
                                    logger.warning("Categories for column %s should be a list, got %s", 
                                                col, type(col_categories))
                                    col_categories = []
                                    
                                self.transformers[f'label_{col}'] = LabelEncoder(
                                    columns=[col],
                                    categories={col: col_categories}
                                )
                                logger.debug("Initialized LabelEncoder for column: %s with %d categories", 
                                           col, len(col_categories))
            
            # Initialize feature scaling if configured
            if 'feature_scaling' in self.config and isinstance(self.config['feature_scaling'], dict):
                methods = self.config['feature_scaling'].get('methods', [])
                if not isinstance(methods, list):
                    logger.warning("feature_scaling.methods should be a list, got %s", type(methods))
                    methods = []
                    
                for method_config in methods:
                    if not isinstance(method_config, dict):
                        logger.warning("Skipping invalid scaling method config: %s", method_config)
                        continue
                        
                    method = method_config.get('method')
                    columns = method_config.get('columns', [])
                    
                    if not isinstance(columns, list) or not columns or not method:
                        logger.debug("Skipping scaling method with no valid columns or method")
                        continue
                        
                    for col in columns:
                        if not isinstance(col, str):
                            logger.warning("Skipping invalid column name: %s", col)
                            continue
                            
                        if method == 'standard':
                            self.transformers[f'standard_{col}'] = StandardScaler(
                                columns=[col],
                                with_mean=bool(method_config.get('with_mean', True)),
                                with_std=bool(method_config.get('with_std', True))
                            )
                            logger.debug("Initialized StandardScaler for column: %s", col)
                            
                        elif method == 'robust':
                            quantile = method_config.get('quantile_range', [25.0, 75.0])
                            # Ensure quantile_range is a tuple of floats
                            try:
                                if isinstance(quantile, (list, tuple)) and len(quantile) == 2:
                                    quantile = tuple(float(x) for x in quantile)
                                else:
                                    raise ValueError("Invalid format")
                            except (ValueError, TypeError):
                                logger.warning("Invalid quantile_range %s, using default (25.0, 75.0)", quantile)
                                quantile = (25.0, 75.0)
                                
                            self.transformers[f'robust_{col}'] = RobustScaler(
                                columns=[col],
                                with_centering=bool(method_config.get('with_centering', True)),
                                with_scaling=bool(method_config.get('with_scaling', True)),
                                quantile_range=quantile
                            )
                            logger.debug("Initialized RobustScaler for column: %s", col)
            
            logger.info("Successfully initialized %d transformers", len(self.transformers))
            
        except Exception as e:
            logger.error("Error initializing transformers: %s", str(e), exc_info=True)
            raise
    
    async def _load_dataset(self) -> pd.DataFrame:
        """
        Load the dataset from the database asynchronously.
        
        Returns:
            Loaded DataFrame
        """
        try:
            # Get dataset from database using async session
            result = await self.db.execute(
                select(models.Dataset).filter(
                    models.Dataset.id == self.dataset_id,
                    models.Dataset.user_id == self.user_id
                )
            )
            dataset = result.scalars().first()
            
            if not dataset:
                raise ValueError(f"Dataset {self.dataset_id} not found or access denied")
                
            # Load the file into a DataFrame (synchronous operation)
            file_extension = os.path.splitext(dataset.file_path)[1].lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(dataset.file_path)
            elif file_extension == '.parquet':
                df = pd.read_parquet(dataset.file_path)
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(dataset.file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
            # Store the DataFrame in both df and data for backward compatibility
            self.df = df
            self.data = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    async def transform(self) -> pd.DataFrame:
        """
        Apply all configured transformations to the dataset asynchronously.
        
        Returns:
            Transformed DataFrame
        """
        try:
            # Load the dataset if not already loaded
            if self.df is None:
                await self._load_dataset()
            
            # Make a copy of the original data to avoid modifying it directly
            transformed_df = self.df.copy()
            
            # Log the shape and columns before transformations
            logger.info(f"Starting transformations on dataset with shape: {transformed_df.shape}")
            logger.debug(f"Original columns: {transformed_df.columns.tolist()}")
            
            # Apply transformations (these are synchronous operations)
            if self.config.get('categorical_encoding'):
                logger.info("Applying categorical encoding...")
                transformed_df = self._apply_categorical_encoding(transformed_df, self.config['categorical_encoding'])
                logger.debug(f"After categorical encoding - shape: {transformed_df.shape}")
                
            if self.config.get('feature_scaling'):
                logger.info("Applying feature scaling...")
                transformed_df = self._apply_feature_scaling(transformed_df, self.config['feature_scaling'])
                logger.debug(f"After feature scaling - shape: {transformed_df.shape}")
            
            # Log the final shape and columns
            logger.info(f"Transformations completed. Final shape: {transformed_df.shape}")
            logger.debug(f"Final columns: {transformed_df.columns.tolist()}")
            
            # Update the instance variable with the transformed data
            self.df = transformed_df
            return transformed_df
            
        except Exception as e:
            logger.error(f"Error in transform: {str(e)}", exc_info=True)
            raise
    
    def _apply_categorical_encoding(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Apply categorical encoding to specified columns using pre-initialized transformers.
        
        Args:
            data: Input DataFrame
            config: Configuration for encoding (unused, here for compatibility)
            
        Returns:
            DataFrame with encoded categorical columns
        """
        try:
            # Make a deep copy of the data to avoid modifying the original
            result = data.copy(deep=True)
            
            # First, handle label encoding as it doesn't change the number of columns
            for name, transformer in list(self.transformers.items()):
                if name.startswith('label_'):
                    col_name = name.replace('label_', '')
                    if col_name in result.columns:
                        try:
                            # Get the column to encode
                            col_to_encode = result[[col_name]].copy()
                            
                            # Log the unique values before encoding
                            unique_vals = col_to_encode[col_name].unique()
                            logger.debug(f"Unique values in {col_name} before encoding: {unique_vals}")
                            
                            # Convert to string and fill NA values with a placeholder
                            col_to_encode[col_name] = col_to_encode[col_name].fillna('__missing__').astype(str)
                            
                            # Log the categories the encoder was trained on
                            if hasattr(transformer, 'category_mappings_') and col_name in transformer.category_mappings_:
                                logger.debug(f"Category mapping for {col_name}: {transformer.category_mappings_[col_name]}")
                            
                            # Fit and transform the column
                            logger.debug(f"Fitting and transforming {col_name} with LabelEncoder")
                            encoded = transformer.fit_transform(col_to_encode)
                            
                            # Handle both DataFrame and Series returns
                            if isinstance(encoded, pd.DataFrame):
                                result[col_name] = encoded[col_name]
                            else:
                                result[col_name] = encoded
                                
                            logger.debug(f"Applied label encoding to column: {col_name}")
                            logger.debug(f"Unique encoded values: {result[col_name].unique()}")
                            
                        except Exception as e:
                            logger.error(f"Error applying label encoding to {col_name}: {str(e)}", exc_info=True)
                            raise
            
            # Then handle one-hot encoding
            for name, transformer in list(self.transformers.items()):
                if name.startswith('one_hot_'):
                    try:
                        col_name = name.replace('one_hot_', '')
                        if col_name in result.columns:
                            logger.debug(f"Starting one-hot encoding for column: {col_name}")
                            
                            # Get unique values before encoding for debugging
                            unique_vals = result[col_name].unique()
                            logger.debug(f"Unique values in {col_name}: {unique_vals}")
                            
                            # Make a copy of just the column we want to encode
                            col_to_encode = result[[col_name]].copy()
                            
                            try:
                                logger.debug("Fitting and transforming with OneHotEncoder")
                                
                                # Ensure we're working with a DataFrame with the column to encode
                                if not isinstance(col_to_encode, pd.DataFrame):
                                    col_to_encode = pd.DataFrame({col_name: col_to_encode})
                                
                                # Get unique values before encoding for debugging
                                unique_vals = col_to_encode[col_name].unique()
                                logger.debug(f"Unique values in {col_name} before encoding: {unique_vals}")
                                
                                # Convert column to string type if it's not already
                                if not pd.api.types.is_string_dtype(col_to_encode[col_name]):
                                    col_to_encode[col_name] = col_to_encode[col_name].astype(str)
                                
                                # Apply one-hot encoding to just this column
                                logger.debug(f"Data types before encoding: {col_to_encode.dtypes}")
                                logger.debug(f"Sample values: {col_to_encode[col_name].head()}")
                                
                                # Fit the transformer first to check for any issues
                                transformer.fit(col_to_encode[[col_name]])
                                logger.debug("Successfully fit the transformer")
                                
                                # Now transform the data
                                encoded = transformer.transform(col_to_encode[[col_name]])
                                logger.debug("Successfully transformed the data")
                                
                                # Handle different return types from OneHotEncoder
                                if hasattr(encoded, 'toarray'):  # For sparse matrices
                                    encoded_data = encoded.toarray()
                                    n_features = encoded_data.shape[1]
                                else:
                                    encoded_data = encoded
                                    n_features = encoded_data.shape[1] if len(encoded_data.shape) > 1 else 1
                                
                                logger.debug(f"Encoded data shape: {encoded_data.shape}")
                                logger.debug(f"Sample encoded data: {encoded_data[:5]}")
                                
                                # Generate feature names
                                try:
                                    # Try to get feature names from the transformer
                                    feature_names = transformer.get_feature_names_out([col_name])
                                    logger.debug(f"Got feature names from get_feature_names_out: {feature_names}")
                                except (AttributeError, IndexError) as e:
                                    try:
                                        # Fallback for older scikit-learn versions
                                        feature_names = transformer.get_feature_names([col_name])
                                        logger.debug(f"Got feature names from get_feature_names: {feature_names}")
                                    except (AttributeError, IndexError) as e:
                                        # If no feature names method is available, create generic names
                                        logger.warning(f"Could not get feature names: {str(e)}. Using generic names.")
                                        feature_names = [f"{col_name}_{i}" for i in range(n_features)]
                                
                                logger.debug(f"Generated {len(feature_names)} feature names")
                                
                                # Ensure we have the correct number of feature names
                                if len(feature_names) != n_features:
                                    logger.warning(f"Mismatch in number of features. Expected {n_features}, got {len(feature_names)}. Using generic names.")
                                    feature_names = [f"{col_name}_{i}" for i in range(n_features)]
                                
                                # Create a DataFrame with the encoded values
                                encoded_df = pd.DataFrame(
                                    encoded_data,
                                    columns=feature_names,
                                    index=result.index
                                )
                                
                                logger.debug(f"Encoded DataFrame shape: {encoded_df.shape}")
                                logger.debug(f"Encoded DataFrame columns: {encoded_df.columns.tolist()}")
                                logger.debug(f"Original result shape before drop: {result.shape}")
                                
                                # Drop the original column if it exists
                                if col_name in result.columns:
                                    result = result.drop(columns=[col_name])
                                
                                # Concatenate the encoded columns
                                logger.debug(f"Result shape after drop: {result.shape}")
                                logger.debug(f"Encoded DataFrame columns to add: {encoded_df.columns.tolist()}")
                                result = pd.concat([result, encoded_df], axis=1)
                                logger.debug(f"Final result shape after concat: {result.shape}")
                                logger.debug(f"Final columns: {result.columns.tolist()}")
                                
                            except Exception as e:
                                logger.error(f"Error during one-hot encoding of column {col_name}: {str(e)}", exc_info=True)
                                # Re-raise the exception to see the full traceback
                                raise
                            
                            logger.debug(f"Applied one-hot encoding to column: {col_name}")
                            
                    except Exception as e:
                        logger.error(f"Error applying one-hot encoding to {col_name}: {str(e)}", exc_info=True)
                        raise
            
            return result
            
        except Exception as e:
            logger.error(f"Error in categorical encoding: {str(e)}", exc_info=True)
            raise ValueError(f"Error in categorical encoding: {str(e)}")
    
    def _apply_feature_scaling(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Apply feature scaling to specified columns using pre-initialized transformers.
        
        Args:
            data: Input DataFrame
            config: Configuration for scaling (unused, here for compatibility)
            
        Returns:
            DataFrame with scaled features
        """
        try:
            transformed_data = data.copy()
            
            # Apply all pre-initialized scalers
            for name, transformer in self.transformers.items():
                if name.startswith(('standard_', 'robust_')):
                    try:
                        # Extract column name from transformer name
                        col_name = name.split('_', 1)[1]
                        
                        if col_name in transformed_data.columns:
                            logger.debug(f"Applying {name} to column: {col_name}")
                            
                            # Make sure we're working with a DataFrame with the column to scale
                            col_to_scale = transformed_data[[col_name]].copy()
                            
                            # Log values before scaling for debugging
                            logger.debug(f"Values before scaling - Min: {col_to_scale[col_name].min()}, "
                                       f"Max: {col_to_scale[col_name].max()}, "
                                       f"Mean: {col_to_scale[col_name].mean():.2f}")
                            
                            # Apply the transformation
                            scaled_values = transformer.fit_transform(col_to_scale)
                            
                            # Handle different return types
                            if isinstance(scaled_values, np.ndarray):
                                if scaled_values.ndim == 2:
                                    scaled_values = scaled_values.flatten()
                                transformed_data[col_name] = scaled_values
                            elif isinstance(scaled_values, pd.DataFrame):
                                transformed_data[col_name] = scaled_values.values.flatten()
                            
                            # Log values after scaling for debugging
                            logger.debug(f"Values after scaling - Min: {transformed_data[col_name].min():.2f}, "
                                       f"Max: {transformed_data[col_name].max():.2f}, "
                                       f"Mean: {transformed_data[col_name].mean():.2f}")
                                
                    except Exception as e:
                        logger.error(f"Error applying {name} to column {col_name}: {str(e)}", exc_info=True)
                        raise
            
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error in _apply_feature_scaling: {str(e)}", exc_info=True)
            raise
