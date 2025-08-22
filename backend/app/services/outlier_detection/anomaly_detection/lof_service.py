# backend/app/services/lof_service.py
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import os
import json
from typing import Dict, Any, Optional, Tuple

from app.config.config import get_settings
settings = get_settings()

def _get_lof_artifact_path(
    base_path: str,
    dataset_id: int,
    user_id: int,
    artifact_name: str,
    cluster_id_or_suffix: Optional[Any] = None
) -> str:
    sub_dir = f"cluster_{cluster_id_or_suffix}" if cluster_id_or_suffix is not None else "overall_lof"
    dir_path = os.path.join(base_path, f"user_{user_id}", f"dataset_{dataset_id}", "lof", sub_dir)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, artifact_name)

class LOFService:
    def __init__(
        self,
        dataset_id: int,
        user_id: int,
        n_neighbors: int = 20,
        contamination: Any = 0.1,
        random_state: Optional[int] = 42
    ):
        self.dataset_id = dataset_id
        self.user_id = user_id
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.random_state = random_state
        
        self.ARTIFACT_BASE = settings.OUTLIER_ARTIFACTS_BASE_PATH
        
        # Define paths for artifacts
        self.lof_results_summary_path = _get_lof_artifact_path(
            self.ARTIFACT_BASE, dataset_id, user_id, "lof_summary_results.json"
        )
        
        # Store artifact paths
        self.artifact_paths = {
            "lof_results_summary_path": self.lof_results_summary_path
        }
    
    def get_artifact_paths(self) -> Dict[str, str]:
        """Return a dictionary of artifact paths."""
        return self.artifact_paths
    
    def _fit_predict_lof_on_group(
        self,
        features_group_np: np.ndarray,
        group_identifier: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits LOF to a group of features and predicts outliers.
        
        Args:
            features_group_np: NumPy array of features for this group
            group_identifier: Identifier for this group (e.g., cluster ID)
            
        Returns:
            Tuple of (outlier_labels, negative_outlier_factors)
            outlier_labels: -1 for outliers, 1 for inliers
            negative_outlier_factors: Higher values indicate more outlier-ness
        """
        if features_group_np.shape[0] == 0:
            print(f"LOF_SERVICE: Group '{group_identifier}' is empty. No LOF processing.")
            return np.array([]), np.array([])
        
        # Handle small groups
        current_n_samples = features_group_np.shape[0]
        if current_n_samples <= 1:
            print(f"LOF_SERVICE: Group '{group_identifier}' has {current_n_samples} sample(s). Marking all as inliers by default.")
            return np.ones(current_n_samples, dtype=int), np.zeros(current_n_samples)
        
        # Adjust n_neighbors if needed
        current_n_neighbors = min(self.n_neighbors, current_n_samples - 1)
        if current_n_neighbors < self.n_neighbors:
            print(f"LOF_SERVICE: Reducing n_neighbors from {self.n_neighbors} to {current_n_neighbors} for group '{group_identifier}' due to small group size.")
        
        # Contamination handling
        current_contamination = self.contamination
        if isinstance(self.contamination, float):
            if not (0 < self.contamination <= 0.5):
                print(f"LOF_SERVICE: Warning - Contamination {self.contamination} for group '{group_identifier}' is outside (0, 0.5]. Using 'auto'.")
                current_contamination = 'auto'
        
        try:
            # Initialize and fit LOF
            lof = LocalOutlierFactor(
                n_neighbors=current_n_neighbors,
                contamination=current_contamination,
                novelty=False,  # Fit and predict in one step
                n_jobs=-1  # Use all cores
            )
            
            # Fit and predict
            outlier_labels = lof.fit_predict(features_group_np)
            
            # Get negative outlier factors (higher = more outlier-ness)
            negative_outlier_factors = lof.negative_outlier_factor_
            
            return outlier_labels, negative_outlier_factors
            
        except Exception as e:
            print(f"LOF_SERVICE: Error in LOF for group '{group_identifier}': {e}")
            # Return default values (all inliers) in case of error
            outlier_labels = np.ones(current_n_samples, dtype=int)
            negative_outlier_factors = np.zeros(current_n_samples)
            
            return outlier_labels, negative_outlier_factors
    
    def detect_outliers_per_cluster(
        self,
        all_features_df: pd.DataFrame,
        cluster_labels_series: pd.Series
    ) -> pd.DataFrame:
        """
        Detects outliers in each cluster using LOF.
        
        Args:
            all_features_df: DataFrame with features
            cluster_labels_series: Series with cluster labels
            
        Returns:
            DataFrame with outlier detection results
        """
        if not all_features_df.index.equals(cluster_labels_series.index):
            raise ValueError("LOF_SERVICE: Indices of features DataFrame and cluster labels Series do not match.")
        
        # Prepare results DataFrame
        final_results_df = pd.DataFrame(index=all_features_df.index)
        final_results_df['is_outlier'] = False  # Default to not outlier
        final_results_df['lof_score'] = 0.0  # Default neutral score
        final_results_df['final_cluster_label'] = cluster_labels_series
        
        # Get unique clusters
        unique_clusters = sorted(cluster_labels_series.unique())
        print(f"LOF_SERVICE: Processing {len(unique_clusters)} unique cluster labels: {unique_clusters}")
        
        for cluster_id in unique_clusters:
            current_cluster_indices = cluster_labels_series[cluster_labels_series == cluster_id].index
            
            if cluster_id == -1:  # Noise points
                print(f"LOF_SERVICE: {len(current_cluster_indices)} noise points (cluster_id = -1) will be analyzed with LOF.")
                # Use a higher contamination rate for noise points
                noise_contamination = 0.05  # Fixed higher contamination rate for noise points
                if isinstance(self.contamination, float):
                    # Use at least 5% contamination for noise points, or higher if the original setting was higher
                    noise_contamination = max(0.05, self.contamination)
                
                features_noise_points_df = all_features_df.loc[current_cluster_indices]
                
                # Apply LOF to noise points
                noise_outlier_labels_np, noise_negative_outlier_factors_np = self._fit_predict_lof_on_group(
                    features_noise_points_df.values,
                    group_identifier="noise_points"
                )
                
                # Update results for noise points
                if len(noise_outlier_labels_np) > 0:
                    final_results_df.loc[current_cluster_indices, 'is_outlier'] = (noise_outlier_labels_np == -1)
                    final_results_df.loc[current_cluster_indices, 'lof_score'] = noise_negative_outlier_factors_np
                continue
            
            if current_cluster_indices.empty:
                print(f"LOF_SERVICE: Cluster {cluster_id} has no points. Skipping.")
                continue
            
            # Get features for this cluster
            features_this_cluster_df = all_features_df.loc[current_cluster_indices]
            
            # Apply LOF to this cluster
            outlier_labels_np, negative_outlier_factors_np = self._fit_predict_lof_on_group(
                features_this_cluster_df.values,
                group_identifier=cluster_id
            )
            
            # Update results for this cluster
            if len(outlier_labels_np) > 0:
                final_results_df.loc[current_cluster_indices, 'is_outlier'] = (outlier_labels_np == -1)
                final_results_df.loc[current_cluster_indices, 'lof_score'] = negative_outlier_factors_np
        
        # Add original_index as a column
        final_results_df['original_index'] = final_results_df.index
        final_results_df = final_results_df.reset_index(drop=True)  # Reset index
        
        # Reorder columns for clarity
        final_results_df = final_results_df[['original_index', 'is_outlier', 'lof_score', 'final_cluster_label']]
        
        print(f"LOF_SERVICE: LOF per cluster processing completed. Total outliers: {final_results_df['is_outlier'].sum()}")
        return final_results_df
