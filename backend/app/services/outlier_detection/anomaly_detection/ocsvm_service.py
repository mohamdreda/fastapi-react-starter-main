# backend/app/services/ocsvm_service.py
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
import os
import json
from typing import Dict, Any, Optional, Tuple

from app.config.config import get_settings
settings = get_settings()

def _get_ocsvm_artifact_path(
    base_path: str,
    dataset_id: int,
    user_id: int,
    artifact_name: str,
    cluster_id_or_suffix: Optional[Any] = None
) -> str:
    sub_dir = f"cluster_{cluster_id_or_suffix}" if cluster_id_or_suffix is not None else "overall_ocsvm"
    dir_path = os.path.join(base_path, f"user_{user_id}", f"dataset_{dataset_id}", "ocsvm", sub_dir)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, artifact_name)

class OCSVMService:
    def __init__(
        self,
        dataset_id: int,
        user_id: int,
        nu: float = 0.1,
        kernel: str = 'rbf',
        gamma: Any = 'scale',
        random_state: Optional[int] = 42
    ):
        self.dataset_id = dataset_id
        self.user_id = user_id
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state
        
        self.ARTIFACT_BASE = settings.OUTLIER_ARTIFACTS_BASE_PATH
        
        # Define paths for artifacts
        self.ocsvm_results_summary_path = _get_ocsvm_artifact_path(
            self.ARTIFACT_BASE, dataset_id, user_id, "ocsvm_summary_results.json"
        )
        
        # Store artifact paths
        self.artifact_paths = {
            "ocsvm_results_summary_path": self.ocsvm_results_summary_path
        }
    
    def get_artifact_paths(self) -> Dict[str, str]:
        """Return a dictionary of artifact paths."""
        return self.artifact_paths
    
    def _fit_predict_ocsvm_on_group(
        self,
        features_group_np: np.ndarray,
        group_identifier: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits One-Class SVM to a group of features and predicts outliers.
        
        Args:
            features_group_np: NumPy array of features for this group
            group_identifier: Identifier for this group (e.g., cluster ID)
            
        Returns:
            Tuple of (outlier_labels, decision_scores)
            outlier_labels: -1 for outliers, 1 for inliers
            decision_scores: Higher negative values indicate more outlier-ness
        """
        if features_group_np.shape[0] == 0:
            print(f"OCSVM_SERVICE: Group '{group_identifier}' is empty. No OCSVM processing.")
            return np.array([]), np.array([])
        
        # Handle small groups
        current_n_samples = features_group_np.shape[0]
        if current_n_samples <= 1:
            print(f"OCSVM_SERVICE: Group '{group_identifier}' has {current_n_samples} sample(s). Marking all as inliers by default.")
            return np.ones(current_n_samples, dtype=int), np.zeros(current_n_samples)
        
        # Adjust nu if needed
        current_nu = min(self.nu, 0.99)  # Ensure nu is not too close to 1
        
        try:
            # Initialize and fit One-Class SVM
            ocsvm = OneClassSVM(
                nu=current_nu,
                kernel=self.kernel,
                gamma=self.gamma
                # Note: OneClassSVM doesn't support random_state parameter
            )
            
            # Fit and predict
            outlier_labels = ocsvm.fit_predict(features_group_np)
            
            # Get decision scores (negative values are outliers)
            decision_scores = ocsvm.decision_function(features_group_np)
            
            # Convert decision scores to outlier scores (higher = more outlier-ness)
            # Negate the decision scores so that higher values indicate more outlier-ness
            outlier_scores = -decision_scores
            
            return outlier_labels, outlier_scores
            
        except Exception as e:
            print(f"OCSVM_SERVICE: Error in OCSVM for group '{group_identifier}': {e}")
            # Return default values (all inliers) in case of error
            outlier_labels = np.ones(current_n_samples, dtype=int)
            outlier_scores = np.zeros(current_n_samples)
            
            return outlier_labels, outlier_scores
    
    def detect_outliers_per_cluster(
        self,
        all_features_df: pd.DataFrame,
        cluster_labels_series: pd.Series
    ) -> pd.DataFrame:
        """
        Detects outliers in each cluster using One-Class SVM.
        
        Args:
            all_features_df: DataFrame with features
            cluster_labels_series: Series with cluster labels
            
        Returns:
            DataFrame with outlier detection results
        """
        if not all_features_df.index.equals(cluster_labels_series.index):
            raise ValueError("OCSVM_SERVICE: Indices of features DataFrame and cluster labels Series do not match.")
        
        # Prepare results DataFrame
        final_results_df = pd.DataFrame(index=all_features_df.index)
        final_results_df['is_outlier'] = False  # Default to not outlier
        final_results_df['ocsvm_score'] = 0.0  # Default neutral score
        final_results_df['final_cluster_label'] = cluster_labels_series
        
        # Get unique clusters
        unique_clusters = sorted(cluster_labels_series.unique())
        print(f"OCSVM_SERVICE: Processing {len(unique_clusters)} unique cluster labels: {unique_clusters}")
        
        for cluster_id in unique_clusters:
            current_cluster_indices = cluster_labels_series[cluster_labels_series == cluster_id].index
            
            if cluster_id == -1:  # Noise points
                print(f"OCSVM_SERVICE: {len(current_cluster_indices)} noise points (cluster_id = -1) will be analyzed with OCSVM.")
                # Use a higher nu for noise points
                noise_nu = min(0.05, 1.0)  # Fixed higher nu for noise points
                if isinstance(self.nu, float):
                    # Use at least 5% nu for noise points, or higher if the original setting was higher
                    noise_nu = max(0.05, self.nu)
                
                features_noise_points_df = all_features_df.loc[current_cluster_indices]
                
                # Apply OCSVM to noise points
                noise_outlier_labels_np, noise_outlier_scores_np = self._fit_predict_ocsvm_on_group(
                    features_noise_points_df.values,
                    group_identifier="noise_points"
                )
                
                # Update results for noise points
                if len(noise_outlier_labels_np) > 0:
                    final_results_df.loc[current_cluster_indices, 'is_outlier'] = (noise_outlier_labels_np == -1)
                    final_results_df.loc[current_cluster_indices, 'ocsvm_score'] = noise_outlier_scores_np
                continue
            
            if current_cluster_indices.empty:
                print(f"OCSVM_SERVICE: Cluster {cluster_id} has no points. Skipping.")
                continue
            
            # Get features for this cluster
            features_this_cluster_df = all_features_df.loc[current_cluster_indices]
            
            # Apply OCSVM to this cluster
            outlier_labels_np, outlier_scores_np = self._fit_predict_ocsvm_on_group(
                features_this_cluster_df.values,
                group_identifier=cluster_id
            )
            
            # Update results for this cluster
            if len(outlier_labels_np) > 0:
                final_results_df.loc[current_cluster_indices, 'is_outlier'] = (outlier_labels_np == -1)
                final_results_df.loc[current_cluster_indices, 'ocsvm_score'] = outlier_scores_np
        
        # Add original_index as a column
        final_results_df['original_index'] = final_results_df.index
        final_results_df = final_results_df.reset_index(drop=True)  # Reset index
        
        # Reorder columns for clarity
        final_results_df = final_results_df[['original_index', 'is_outlier', 'ocsvm_score', 'final_cluster_label']]
        
        print(f"OCSVM_SERVICE: OCSVM per cluster processing completed. Total outliers: {final_results_df['is_outlier'].sum()}")
        return final_results_df
