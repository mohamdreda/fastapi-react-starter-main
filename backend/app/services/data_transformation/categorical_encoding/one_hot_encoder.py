"""
One-Hot Encoder for categorical variables.

This module provides functionality for one-hot encoding of categorical variables.
"""
import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

# Set up logger
logger = logging.getLogger(__name__)

class SimpleOneHotEncoder:
    """
    A simplified one-hot encoder that doesn't rely on get_feature_names
    """
    def __init__(self, drop=None):
        self.drop = drop
        self.categories_ = None
        self.feature_names_ = None
        
    def fit(self, X):
        # Flatten the input array and get unique categories
        X_flat = np.ravel(X)
        self.categories_ = np.unique(X_flat[~pd.isna(X_flat)])
        
        # Generate feature names based on actual categories
        if self.drop == 'first' and len(self.categories_) > 0:
            self.feature_names_ = [str(cat) for cat in self.categories_[1:]]
        else:
            self.feature_names_ = [str(cat) for cat in self.categories_]
            
        return self
        
    def transform(self, X):
        if self.categories_ is None:
            raise ValueError("Must call fit before transform")
            
        # Flatten the input array
        X_flat = np.ravel(X)
        
        # Initialize output array
        n_samples = len(X_flat)
        n_features = len(self.categories_) - 1 if self.drop == 'first' and len(self.categories_) > 1 else len(self.categories_)
        
        encoded = np.zeros((n_samples, n_features), dtype=int)
        
        # Create a mapping from category to index
        cat_to_idx = {cat: idx for idx, cat in enumerate(self.categories_)}
        
        # Fill in the encoded array
        for i, x in enumerate(X_flat):
            if pd.isna(x):
                continue
                
            if x in cat_to_idx:
                idx = cat_to_idx[x]
                if self.drop == 'first' and len(self.categories_) > 1:
                    if idx > 0:  # Skip the first category if drop='first'
                        encoded[i, idx-1] = 1
                else:
                    encoded[i, idx] = 1
        
        return encoded

class OneHotEncoder:
    """
    A custom one-hot encoder that handles pandas DataFrames and provides
    consistent behavior across different scikit-learn versions.
    """
    def __init__(self, drop: Optional[str] = None, columns: Optional[Union[str, List[str]]] = None):
        self.drop = drop
        self.columns = columns
        self.encoders = {}
        self.feature_names_ = []
        self.categories_ = {}
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.column_name_mapping_ = {}  # Store original column to encoded columns mapping
        
    def fit(self, X: pd.DataFrame, y: Any = None) -> 'OneHotEncoder':
        """
        Fit the OneHotEncoder to X.
        
        Args:
            X: Input DataFrame with categorical columns to encode
            y: Ignored. This parameter exists only for compatibility.
            
        Returns:
            self: Returns the instance itself.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        
        # Determine which columns to encode
        if self.columns is None:
            self.columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        elif isinstance(self.columns, str):
            self.columns = [self.columns]
            
        if not self.columns:
            return self
            
        self.encoders = {}
        self.feature_names_ = []
        
        for col in self.columns:
            if col not in X.columns:
                continue
                
            try:
                # Initialize and fit the encoder for this column
                self.encoders[col] = SimpleOneHotEncoder(drop=self.drop)
                self.encoders[col].fit(X[[col]].values)
                
                # Store categories
                self.categories_[col] = self.encoders[col].categories_
                
                # Generate feature names with actual category values
                if self.drop == 'first' and len(self.categories_[col]) > 0:
                    feature_names = [f"{col}_{str(cat).replace(' ', '_').replace('.', '_')}" 
                                   for cat in self.categories_[col][1:]]
                else:
                    feature_names = [f"{col}_{str(cat).replace(' ', '_').replace('.', '_')}" 
                                   for cat in self.categories_[col]]
                
                self.feature_names_.extend(feature_names)
                self.column_name_mapping_[col] = feature_names
                
            except Exception as e:
                logger.error(f"Error fitting column '{col}': {str(e)}")
                continue
                
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform X using one-hot encoding.
        
        Args:
            X: Input DataFrame with categorical columns to encode
            
        Returns:
            DataFrame with categorical columns replaced by their one-hot encoded versions
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        if not hasattr(self, 'feature_names_in_'):
            raise NotFittedError("This OneHotEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
            
        # Make a copy of the input data to avoid modifying the original
        X_transformed = X.copy()
        
        # Process each column that has an encoder
        for col, encoder in self.encoders.items():
            if col not in X.columns:
                continue
                
            try:
                # Get the column data
                col_data = X[[col]].values
                
                # Transform the data
                encoded = encoder.transform(col_data)
                
                # Create a DataFrame with the encoded data and proper column names
                if encoded.size > 0:  # Only proceed if we have data
                    encoded_df = pd.DataFrame(
                        encoded,
                        columns=self.column_name_mapping_.get(col, []),
                        index=X.index
                    )
                    
                    # Convert to int for better compatibility
                    encoded_df = encoded_df.astype(int)
                    
                    # Drop the original column and concatenate the encoded ones
                    X_transformed = X_transformed.drop(columns=[col])
                    X_transformed = pd.concat([X_transformed, encoded_df], axis=1)
                
            except Exception as e:
                logger.error(f"Error encoding column '{col}': {str(e)}")
                logger.warning(f"Skipping column '{col}' due to error")
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the encoder and transform the data.
        
        Args:
            X: Input DataFrame to fit and transform
            
        Returns:
            Transformed DataFrame with one-hot encoded columns
        """
        return self.fit(X).transform(X)
    
    def get_feature_names(self, input_features=None):
        """
        Get feature names from the encoder.
        
        Args:
            input_features: Not used, for scikit-learn compatibility
            
        Returns:
            List of output feature names
        """
        return self.feature_names_
    
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.
        
        Args:
            input_features: Not used, for scikit-learn compatibility
            
        Returns:
            List of output feature names
        """
        return self.feature_names_
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the encoder.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'columns': self.columns,
            'drop': self.drop,
            'feature_names_': self.feature_names_,
            'fitted_columns_': self.fitted_columns_
        }
    
    def set_params(self, **params) -> 'OneHotEncoder':
        """
        Set the parameters of the encoder.
        
        Args:
            **params: Parameters to set
            
        Returns:
            self: Encoder with updated parameters
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self
