"""
Label Encoder for categorical variables.

This module provides functionality for label encoding of categorical variables.
"""
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder
import logging

logger = logging.getLogger(__name__)

class LabelEncoder:
    """
    A label encoder that handles both nominal and ordinal categorical variables.
    
    For ordinal data, you can specify the category order.
    For nominal data, it will use lexicographical order by default.
    """
    def __init__(self, columns: List[str], categories: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the LabelEncoder.
        
        Args:
            columns: List of column names to encode
            categories: Optional dictionary mapping column names to ordered category lists
                     Example: {"Size": ["Small", "Medium", "Large"]}
        """
        self.columns = columns
        self.categories = categories or {}
        self.encoders: Dict[str, SklearnLabelEncoder] = {}
        self.fitted_columns_: List[str] = []
        self.category_mappings_: Dict[str, Dict[Any, int]] = {}
    
    def _get_ordered_categories(self, X: pd.Series) -> List[str]:
        """Get ordered categories for a column, either from user input or sorted unique values."""
        col_name = X.name
        if col_name in self.categories:
            return self.categories[col_name]
        return sorted(X.astype(str).dropna().unique())
    
    def fit(self, X: Union[pd.DataFrame, pd.Series]) -> 'LabelEncoder':
        """
        Fit the LabelEncoder to the data.
        
        Args:
            X: Input DataFrame or Series containing the columns to encode
            
        Returns:
            self: Fitted encoder
        """
        if not self.columns:
            return self
            
        # Handle Series input
        if isinstance(X, pd.Series):
            X = X.to_frame()
            
        for col in self.columns:
            if col in X.columns:
                # Convert to string and replace NaN with a placeholder
                col_data = X[col].fillna('__missing__').astype(str)
                
                # Get categories in the correct order
                categories = self._get_ordered_categories(X[col])
                
                # Create and fit the encoder
                encoder = SklearnLabelEncoder()
                encoder.fit(categories)
                
                # Store the encoder and mapping
                self.encoders[col] = encoder
                self.category_mappings_[col] = dict(zip(categories, range(len(categories))))
                self.fitted_columns_.append(col)
                
                logger.debug(f"Fitted encoder for {col} with categories: {categories}")
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Transform the data using the fitted encoder.
        
        Args:
            X: Input DataFrame or Series to transform
            
        Returns:
            Transformed DataFrame with label encoded columns
        """
        if not self.encoders:
            return X
            
        # Handle Series input
        if isinstance(X, pd.Series):
            X = X.to_frame()
            
        X_transformed = X.copy()
        
        for col, encoder in self.encoders.items():
            if col in X_transformed.columns:
                try:
                    # Convert to string and handle NaN values
                    col_data = X_transformed[col].fillna('__missing__').astype(str)
                    
                    # Get the categories this encoder was trained on
                    known_categories = set(encoder.classes_)
                    
                    # Check for unknown categories
                    unknown_categories = set(col_data.unique()) - known_categories
                    if unknown_categories:
                        logger.warning(f"Found {len(unknown_categories)} unknown categories in column {col}: {unknown_categories}")
                        
                        # Re-fit the encoder with all categories to handle unknown values
                        all_categories = list(known_categories) + list(unknown_categories)
                        new_encoder = SklearnLabelEncoder()
                        new_encoder.fit(all_categories)
                        
                        # Update the encoder and mappings
                        self.encoders[col] = new_encoder
                        self.category_mappings_[col] = dict(zip(all_categories, range(len(all_categories))))
                        encoder = new_encoder
                    
                    # Transform all values with the updated encoder
                    X_transformed[col] = encoder.transform(col_data)
                    
                    logger.debug(f"Transformed column {col} with {len(col_data)} values")
                except Exception as e:
                    logger.error(f"Error transforming column {col}: {str(e)}")
                    # Fallback to -1 for error cases
                    X_transformed[col] = -1
        
        return X_transformed
    
    def fit_transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Fit the encoder and transform the data.
        
        Args:
            X: Input DataFrame or Series to fit and transform
            
        Returns:
            Transformed DataFrame with label encoded columns
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Transform labels back to original encoding.
        
        Args:
            X: Input DataFrame or Series with encoded values
            
        Returns:
            DataFrame with original categorical values
        """
        if not self.encoders:
            return X
            
        # Handle Series input
        if isinstance(X, pd.Series):
            X = X.to_frame()
            
        X_transformed = X.copy()
        
        for col, encoder in self.encoders.items():
            if col in X_transformed.columns:
                # Initialize with None for missing values
                X_transformed[col] = None
                
                # Only transform valid encoded values
                valid_mask = X_transformed[col].notna() & X_transformed[col].astype(str).str.isdigit()
                if valid_mask.any():
                    # Convert to int for inverse_transform
                    encoded_values = X_transformed.loc[valid_mask, col].astype(int)
                    X_transformed.loc[valid_mask, col] = encoder.inverse_transform(encoded_values)
                
                # Replace the missing value placeholder with actual None
                X_transformed[col] = X_transformed[col].replace('__missing__', None)
        
        return X_transformed
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the encoder.
        
        Returns:
            Dictionary of parameters and category mappings
        """
        return {
            'columns': self.columns,
            'fitted_columns_': self.fitted_columns_,
            'category_mappings': self.category_mappings_,
            'categories': self.categories
        }
    
    def set_params(self, **params) -> 'LabelEncoder':
        """
        Set the parameters of the encoder.
        
        Args:
            **params: Parameters to set
            
        Returns:
            self: Updated encoder
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
