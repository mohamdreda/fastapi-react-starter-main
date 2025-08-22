# backend/app/services/pca_service.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os
import joblib
from typing import Dict, Any, Optional, Tuple

from app.config.config import get_settings
settings = get_settings()

def _get_pca_artifact_path(
    base_path: str,
    dataset_id: int,
    user_id: int,
    artifact_name: str
) -> str:
    dir_path = os.path.join(base_path, f"user_{user_id}", f"dataset_{dataset_id}", "pca_outputs")
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, artifact_name)

class PCAService:
    def __init__(
        self,
        dataset_id: int,
        user_id: int,
        n_components: Optional[int] = None,
        random_state: int = 42
    ):
        self.dataset_id = dataset_id
        self.user_id = user_id
        self.n_components = n_components
        self.random_state = random_state
        
        self.pca_model = None
        
        self.ARTIFACT_BASE = settings.OUTLIER_ARTIFACTS_BASE_PATH
        
        # Define paths for artifacts
        self.pca_model_path = _get_pca_artifact_path(
            self.ARTIFACT_BASE, dataset_id, user_id, "pca_model.joblib"
        )
        self.latent_features_save_path = _get_pca_artifact_path(
            self.ARTIFACT_BASE, dataset_id, user_id, "pca_latent_features.parquet"
        )
        self.latent_features_csv_path = _get_pca_artifact_path(
            self.ARTIFACT_BASE, dataset_id, user_id, "pca_latent_features.csv"
        )
        self.explained_variance_path = _get_pca_artifact_path(
            self.ARTIFACT_BASE, dataset_id, user_id, "pca_explained_variance.json"
        )
    
    def extract_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts latent features using PCA.
        
        Args:
            features_df: DataFrame with features (may include non-numeric columns)
        Returns:
            DataFrame with latent features
        """
        if not isinstance(features_df, pd.DataFrame) or features_df.empty:
            raise ValueError("PCA_SERVICE: Input features_df must be a non-empty Pandas DataFrame.")
        # Drop non-numeric columns
        orig_cols = features_df.columns.tolist()
        features_df = features_df.select_dtypes(include=["number"])
        dropped_cols = list(set(orig_cols) - set(features_df.columns))
        if dropped_cols:
            print(f"PCA_SERVICE WARNING: Dropping non-numeric columns for PCA: {dropped_cols}")
        
        # Handle NaNs
        if features_df.isnull().values.any():
            print(f"PCA_SERVICE: Warning - NaNs found in {features_df.isnull().sum().sum()} cells. Imputing with column means.")
            for col in features_df.columns[features_df.isnull().any()]:
                features_df[col] = features_df[col].fillna(features_df[col].mean())
            # If any column was entirely NaN, fill with 0
            if features_df.isnull().values.any():
                features_df = features_df.fillna(0)
        
        # Dynamically determine the appropriate number of components based on dataset dimensions
        n_samples, n_features = features_df.shape
        max_possible_components = min(n_samples, n_features)
        if self.n_components is not None:
            n_components = min(self.n_components, max_possible_components)
        else:
            # Default: use 80% of possible components or 10, whichever is smaller
            n_components = min(max(1, int(max_possible_components * 0.8)), 10)
        
        # Ensure we have at least 1 component
        n_components = max(1, n_components)
        
        print(f"PCA_SERVICE: Using {n_components} components for a dataset with {n_samples} samples and {n_features} features")
        
        # Create and fit PCA model
        self.pca_model = PCA(n_components=n_components, random_state=self.random_state)
        
        try:
            # Transform data
            latent_features = self.pca_model.fit_transform(features_df.values)
            
            # Create DataFrame with latent features
            latent_features_df = pd.DataFrame(
                latent_features,
                index=features_df.index,
                columns=[f"latent_{i}" for i in range(n_components)]
            )
            
            # Save PCA model
            joblib.dump(self.pca_model, self.pca_model_path)
            print(f"PCA_SERVICE: PCA model saved to {self.pca_model_path}")
            
            # Save latent features
            latent_features_df.to_parquet(self.latent_features_save_path)
            print(f"PCA_SERVICE: Latent features saved to {self.latent_features_save_path}")
            # Save as CSV for download
            latent_features_df.to_csv(self.latent_features_csv_path, index=False)
            print(f"PCA_SERVICE: Latent features CSV saved to {self.latent_features_csv_path}")
            
            # Save explained variance
            import json
            explained_variance_data = {
                "explained_variance_ratio": self.pca_model.explained_variance_ratio_.tolist(),
                "cumulative_explained_variance": np.cumsum(self.pca_model.explained_variance_ratio_).tolist(),
                "total_explained_variance": sum(self.pca_model.explained_variance_ratio_)
            }
            with open(self.explained_variance_path, 'w') as f:
                json.dump(explained_variance_data, f, indent=4)
            print(f"PCA_SERVICE: Explained variance saved to {self.explained_variance_path}")

            # Generate and save a scatter plot of the first two latent features
            import matplotlib.pyplot as plt
            scatter_plot_path = _get_pca_artifact_path(self.ARTIFACT_BASE, self.dataset_id, self.user_id, "scatter_plot.png")
            if latent_features.shape[1] >= 2:
                plt.figure(figsize=(6, 5))
                plt.scatter(latent_features[:, 0], latent_features[:, 1], alpha=0.6, s=12)
                plt.xlabel('Latent 1')
                plt.ylabel('Latent 2')
                plt.title('PCA Latent Feature Scatter Plot')
                plt.tight_layout()
                plt.savefig(scatter_plot_path)
                plt.close()
                print(f"PCA_SERVICE: Scatter plot saved to {scatter_plot_path}")
                self.scatter_plot_path = scatter_plot_path
            else:
                self.scatter_plot_path = None
            
            print(f"PCA_SERVICE: PCA completed. Reduced dimensions from {features_df.shape[1]} to {n_components}.")
            print(f"PCA_SERVICE: Total explained variance: {explained_variance_data['total_explained_variance']:.4f}")
            
            return latent_features_df
            
        except Exception as e:
            print(f"PCA_SERVICE: Error during PCA: {e}")
            raise
    
    def get_artifact_paths(self) -> Dict[str, Optional[str]]:
        """Returns paths to artifacts generated by this service."""
        return {
            "pca_model_path": self.pca_model_path if os.path.exists(self.pca_model_path) else None,
            "latent_features_path": self.latent_features_save_path if os.path.exists(self.latent_features_save_path) else None,
            "latent_features_csv_path": self.latent_features_csv_path if os.path.exists(self.latent_features_csv_path) else None,
            "explained_variance_path": self.explained_variance_path if os.path.exists(self.explained_variance_path) else None
        }
