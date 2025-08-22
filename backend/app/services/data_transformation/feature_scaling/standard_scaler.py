"""
Standard Scaler for numerical features.

This module provides functionality for standard scaling of numerical features.
"""
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

class StandardScaler:
    """
    Standard Scaler for numerical features.
    
    This scaler standardizes features by removing the mean and scaling to unit variance.
    """
    
    def __init__(
        self, 
        columns: List[str],
        with_mean: bool = True,
        with_std: bool = True,
        copy: bool = True
    ):
        """
        Initialize the StandardScaler.
        
        Args:
            columns: List of column names to be scaled
            with_mean: If True, center the data before scaling
            with_std: If True, scale the data to unit variance
            copy: If False, try to avoid a copy and do inplace scaling instead
        """
        self.columns = columns
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy
        self.scaler = None
        self.fitted_columns_ = []
    
    def fit(self, X: pd.DataFrame) -> 'StandardScaler':
        """
        Fit the StandardScaler to the data.
        
        Args:
            X: Input DataFrame containing the columns to scale
            
        Returns:
            self: Fitted scaler
        """
        if not self.columns:
            return self
            
        # Only fit on columns that exist in the DataFrame
        self.fitted_columns_ = [col for col in self.columns if col in X.columns]
        
        if not self.fitted_columns_:
            return self
            
        self.scaler = SklearnStandardScaler(
            with_mean=self.with_mean,
            with_std=self.with_std,
            copy=self.copy
        )
        
        # Fit the scaler on the specified columns
        self.scaler.fit(X[self.fitted_columns_])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted scaler.
        
        Args:
            X: Input DataFrame to transform
            
        Returns:
            Transformed DataFrame with standardized features
        """
        if self.scaler is None or not self.fitted_columns_:
            return X
            
        # Create a copy to avoid modifying the original DataFrame
        X_transformed = X.copy()
        
        # Transform the specified columns
        X_transformed[self.fitted_columns_] = self.scaler.transform(
            X_transformed[self.fitted_columns_]
        )
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the scaler and transform the data.
        
        Args:
            X: Input DataFrame to fit and transform
            
        Returns:
            Transformed DataFrame with standardized features
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale back the data to the original representation.
        
        Args:
            X: Input DataFrame with standardized values
            
        Returns:
            DataFrame with original scale
        """
        if self.scaler is None or not self.fitted_columns_:
            return X
            
        X_transformed = X.copy()
        
        # Inverse transform the specified columns
        X_transformed[self.fitted_columns_] = self.scaler.inverse_transform(
            X_transformed[self.fitted_columns_]
        )
        
        return X_transformed
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Get output feature names for transformation.
        
        Args:
            input_features: Input features
            
        Returns:
            Output feature names
        """
        if input_features is None:
            return self.fitted_columns_
        return input_features
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the scaler.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'columns': self.columns,
            'with_mean': self.with_mean,
            'with_std': self.with_std,
            'copy': self.copy,
            'fitted_columns_': self.fitted_columns_,
            'mean_': getattr(self.scaler, 'mean_', None) if self.with_mean else None,
            'scale_': getattr(self.scaler, 'scale_', None) if self.with_std else None,
            'n_features_in_': getattr(self.scaler, 'n_features_in_', None),
            'n_samples_seen_': getattr(self.scaler, 'n_samples_seen_', None)
        }
    
    def set_params(self, **params) -> 'StandardScaler':
        """
        Set the parameters of the scaler.
        
        Args:
            **params: Parameters to set
            
        Returns:
            self: Scaler with updated parameters
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self
