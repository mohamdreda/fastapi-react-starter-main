"""
Robust Scaler for numerical features.

This module provides functionality for robust scaling of numerical features.
"""
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler as SklearnRobustScaler

class RobustScaler:
    """
    Robust Scaler for numerical features.
    
    This scaler removes the median and scales the data according to the 
    quantile range (defaults to IQR: Interquartile Range).
    """
    
    def __init__(
        self, 
        columns: List[str],
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: Tuple[float, float] = (25.0, 75.0),
        copy: bool = True
    ):
        """
        Initialize the RobustScaler.
        
        Args:
            columns: List of column names to be scaled
            with_centering: If True, center the data before scaling
            with_scaling: If True, scale the data to interquartile range
            quantile_range: Quantile range used to calculate scale_
            copy: If False, try to avoid a copy and do inplace scaling instead
        """
        self.columns = columns
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy
        self.scaler = None
        self.fitted_columns_ = []
        
    def fit(self, X: pd.DataFrame) -> 'RobustScaler':
        """
        Fit the RobustScaler to the data.
        
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
            
        self.scaler = SklearnRobustScaler(
            with_centering=self.with_centering,
            with_scaling=self.with_scaling,
            quantile_range=self.quantile_range,
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
            Transformed DataFrame with scaled features
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
            Transformed DataFrame with scaled features
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale back the data to the original representation.
        
        Args:
            X: Input DataFrame with scaled values
            
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
            'with_centering': self.with_centering,
            'with_scaling': self.with_scaling,
            'quantile_range': self.quantile_range,
            'copy': self.copy,
            'fitted_columns_': self.fitted_columns_,
            'scale_': getattr(self.scaler, 'scale_', None),
            'center_': getattr(self.scaler, 'center_', None) if self.with_centering else None
        }
    
    def set_params(self, **params) -> 'RobustScaler':
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
