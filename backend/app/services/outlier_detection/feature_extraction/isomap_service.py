# backend/app/services/isomap_service.py
import pandas as pd
import numpy as np
from sklearn.manifold import Isomap
import os
import joblib
import time
import threading
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError


class TimeoutException(Exception):
    """Exception raised when a function times out."""
    pass

from app.config.config import get_settings
settings = get_settings()

def _get_isomap_artifact_path(
    base_path: str,
    dataset_id: int,
    user_id: int,
    artifact_name: str
) -> str:
    dir_path = os.path.join(base_path, f"user_{user_id}", f"dataset_{dataset_id}", "isomap_outputs")
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, artifact_name)

class IsomapService:
    def __init__(
        self,
        dataset_id: int,
        user_id: int,
        n_components: Optional[int] = None,
        n_neighbors: int = 15,
        random_state: int = 42
    ):
        self.dataset_id = dataset_id
        self.user_id = user_id
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        
        self.isomap_model = None
        self.scatter_plot_path = None
        
        self.ARTIFACT_BASE = settings.OUTLIER_ARTIFACTS_BASE_PATH
        
        # Define paths for artifacts
        self.isomap_model_path = _get_isomap_artifact_path(
            self.ARTIFACT_BASE, dataset_id, user_id, "isomap_model.joblib"
        )
        self.latent_features_save_path = _get_isomap_artifact_path(
            self.ARTIFACT_BASE, dataset_id, user_id, "isomap_latent_features.parquet"
        )
        self.latent_features_csv_path = _get_isomap_artifact_path(
            self.ARTIFACT_BASE, dataset_id, user_id, "isomap_latent_features.csv"
        )
    
    def extract_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts latent features using Isomap.
        
        Args:
            features_df: DataFrame with features (may include non-numeric columns)
        Returns:
            DataFrame with latent features
        """
        if not isinstance(features_df, pd.DataFrame) or features_df.empty:
            raise ValueError("ISOMAP_SERVICE: Input features_df must be a non-empty Pandas DataFrame.")
        
        # Drop non-numeric columns
        orig_cols = features_df.columns.tolist()
        features_df = features_df.select_dtypes(include=["number"])
        dropped_cols = list(set(orig_cols) - set(features_df.columns))
        if dropped_cols:
            print(f"ISOMAP_SERVICE WARNING: Dropping non-numeric columns for Isomap: {dropped_cols}")
        
        # Handle NaNs
        if features_df.isnull().values.any():
            print(f"ISOMAP_SERVICE: Warning - NaNs found in {features_df.isnull().sum().sum()} cells. Imputing with column means.")
            for col in features_df.columns[features_df.isnull().any()]:
                features_df[col] = features_df[col].fillna(features_df[col].mean())
            # If any column was entirely NaN, fill with 0
            if features_df.isnull().values.any():
                features_df = features_df.fillna(0)
        
        # Check dataset size and sample if needed
        MAX_SAMPLES = 5000  # Reduced to 5000 samples to make it faster
        original_shape = features_df.shape
        sampled = False
        
        if features_df.shape[0] > MAX_SAMPLES:
            print(f"ISOMAP_SERVICE: Dataset is large ({features_df.shape[0]} samples). Sampling {MAX_SAMPLES} samples to avoid memory issues.")
            # Force random sampling for speed and simplicity
            features_df = features_df.sample(n=MAX_SAMPLES, random_state=self.random_state)
            sampled = True
            print(f"ISOMAP_SERVICE: Sampled dataset shape: {features_df.shape}")
        
        # Determine number of components
        n_components = self.n_components
        if n_components is None:
            n_components = min(2, features_df.shape[1])  # Default to 2D
        n_components = min(n_components, features_df.shape[1])  # Can't have more components than features
        
        # Adjust n_neighbors if needed
        n_neighbors = min(self.n_neighbors, features_df.shape[0] - 1)
        if n_neighbors < self.n_neighbors:
            print(f"ISOMAP_SERVICE: Warning - Reducing n_neighbors from {self.n_neighbors} to {n_neighbors} due to small dataset.")
        
        try:
            # Try with memory-efficient algorithm first
            print(f"ISOMAP_SERVICE: Using memory-efficient algorithm with {n_components} components and {n_neighbors} neighbors")
            
            # Increase neighbors if we have disconnected components
            if features_df.shape[0] > 100:  # Only for reasonably sized datasets
                n_neighbors = max(n_neighbors, 10)  # Use at least 10 neighbors to avoid disconnected components
                print(f"ISOMAP_SERVICE: Increased neighbors to {n_neighbors} to avoid disconnected components")
            
            # Create model with optimized parameters
            self.isomap_model = Isomap(
                n_components=n_components,
                n_neighbors=n_neighbors,
                path_method='D',  # Use Dijkstra's algorithm which is more memory efficient
                neighbors_algorithm='ball_tree',  # More memory efficient
                eigen_solver='arpack'  # Faster for large datasets
            )
            
            # Use timeout to prevent hanging
            print("ISOMAP_SERVICE: Starting ISOMAP computation with 120 second timeout")
            start_time = time.time()
            
            # Function to run in thread
            def run_isomap(df, model):
                return model.fit_transform(df.values)
            
            try:
                # Use ThreadPoolExecutor for timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_isomap, features_df, self.isomap_model)
                    latent_features = future.result(timeout=120)  # 2 minute timeout
            except TimeoutError:
                print("ISOMAP_SERVICE: ISOMAP computation timed out after 120 seconds. Trying with fewer samples.")
                # If timeout occurs, try with even fewer samples
                if features_df.shape[0] > 1000:
                    features_df = features_df.sample(n=1000, random_state=self.random_state)
                    print(f"ISOMAP_SERVICE: Reduced to 1000 samples due to timeout")
                    
                    # Try again with reduced dataset
                    self.isomap_model = Isomap(
                        n_components=n_components,
                        n_neighbors=min(n_neighbors, 8),  # Reduce neighbors too
                        path_method='D',
                        neighbors_algorithm='ball_tree',
                        eigen_solver='arpack'
                    )
                    # Try one more time with reduced dataset
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(run_isomap, features_df, self.isomap_model)
                        try:
                            latent_features = future.result(timeout=60)  # 1 minute timeout for the reduced dataset
                        except TimeoutError:
                            raise ValueError("ISOMAP_SERVICE: ISOMAP computation timed out even with reduced dataset")
                else:
                    raise ValueError("ISOMAP_SERVICE: ISOMAP computation timed out even with reduced dataset")
            
            end_time = time.time()
            print(f"ISOMAP_SERVICE: ISOMAP computation completed in {end_time - start_time:.2f} seconds")
            
            # Create DataFrame with latent features
            latent_features_df = pd.DataFrame(
                latent_features,
                index=features_df.index,
                columns=[f"latent_{i}" for i in range(n_components)]
            )
            
            # If we sampled the dataset, we need to explain this in the output
            if sampled:
                print(f"ISOMAP_SERVICE: Note - Results are based on a sample of {features_df.shape[0]} out of {original_shape[0]} data points")
            
            # Save Isomap model
            joblib.dump(self.isomap_model, self.isomap_model_path)
            print(f"ISOMAP_SERVICE: Isomap model saved to {self.isomap_model_path}")
            
            # Save latent features
            latent_features_df.to_parquet(self.latent_features_save_path)
            print(f"ISOMAP_SERVICE: Latent features saved to {self.latent_features_save_path}")
            # Save as CSV for download
            latent_features_df.to_csv(self.latent_features_csv_path, index=False)
            print(f"ISOMAP_SERVICE: Latent features CSV saved to {self.latent_features_csv_path}")
            
            print(f"ISOMAP_SERVICE: Isomap completed. Reduced dimensions from {features_df.shape[1]} to {n_components}.")

            # Generate and save a scatter plot of the first two latent features
            import matplotlib.pyplot as plt
            scatter_plot_path = _get_isomap_artifact_path(self.ARTIFACT_BASE, self.dataset_id, self.user_id, "scatter_plot.png")
            if latent_features.shape[1] >= 2:
                plt.figure(figsize=(6, 5))
                plt.scatter(latent_features[:, 0], latent_features[:, 1], alpha=0.6, s=12)
                plt.xlabel('Latent 1')
                plt.ylabel('Latent 2')
                plt.title('Isomap Latent Feature Scatter Plot')
                plt.tight_layout()
                plt.savefig(scatter_plot_path)
                plt.close()
                print(f"ISOMAP_SERVICE: Scatter plot saved to {scatter_plot_path}")
                self.scatter_plot_path = scatter_plot_path
            else:
                self.scatter_plot_path = None

            return latent_features_df
            
        except Exception as e:
            print(f"ISOMAP_SERVICE: Error during Isomap: {str(e)}")
            raise
    
    def get_artifact_paths(self) -> Dict[str, Optional[str]]:
        """Returns paths to artifacts generated by this service."""
        return {
            "isomap_model_path": self.isomap_model_path if os.path.exists(self.isomap_model_path) else None,
            "latent_features_path": self.latent_features_save_path if os.path.exists(self.latent_features_save_path) else None,
            "latent_features_csv_path": self.latent_features_csv_path if os.path.exists(self.latent_features_csv_path) else None
        }
