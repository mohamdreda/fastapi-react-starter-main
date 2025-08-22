# backend/app/services/isolation_forest_service.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple, Optional, Any # Added Any
import os

from app.config.config import get_settings # MODIFIED IMPORT
settings = get_settings()

def _get_if_artifact_path(
    base_path: str, 
    dataset_id: int, 
    user_id: int, 
    artifact_name: str,
    cluster_id_or_suffix: Optional[Any] = None
) -> str:
    sub_dir = f"cluster_{cluster_id_or_suffix}" if cluster_id_or_suffix is not None else "overall_if" # Suffix can also be 'noise'
    dir_path = os.path.join(base_path, f"user_{user_id}", f"dataset_{dataset_id}", "isolation_forest", sub_dir)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, artifact_name)

class IsolationForestService:
    def __init__(
        self,
        dataset_id: int,
        user_id: int,
        n_estimators: int = 100,
        contamination: Any = 'auto', # Can be 'auto' or a float
        max_samples: Any = 'auto', # Can be 'auto', int or float
        random_state: Optional[int] = 42,
    ):
        self.dataset_id = dataset_id
        self.user_id = user_id
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state

        self.ARTIFACT_BASE = settings.OUTLIER_ARTIFACTS_BASE_PATH
        # IF models are not typically saved/loaded, but parameters/results can be.
        self.if_results_summary_path = _get_if_artifact_path(
            self.ARTIFACT_BASE, dataset_id, user_id, "if_summary_results.json" # Overall summary
        )
        
        # Store artifact paths
        self.artifact_paths = {
            "if_results_summary_path": self.if_results_summary_path
        }


    def get_artifact_paths(self) -> Dict[str, str]:
        """Return a dictionary of artifact paths."""
        return self.artifact_paths
        
    def _fit_predict_if_on_group(
        self, 
        features_group_np: np.ndarray, 
        group_identifier: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        if features_group_np.shape[0] == 0:
             print(f"IF_SERVICE: Group '{group_identifier}' is empty. No IF processing.")
             return np.array([]), np.array([])

        # Handle max_samples relative to current group size
        current_n_samples = features_group_np.shape[0]
        if current_n_samples <= 1: # IF cannot run on 0 or 1 sample
            print(f"IF_SERVICE: Group '{group_identifier}' has {current_n_samples} sample(s). Marking all as inliers by default.")
            return np.ones(current_n_samples, dtype=int), np.zeros(current_n_samples) # Default: inlier, score 0

        current_max_samples = self.max_samples
        if self.max_samples == 'auto':
            current_max_samples = min(current_n_samples, 256)
        elif isinstance(self.max_samples, float):
            current_max_samples = int(self.max_samples * current_n_samples)
        elif isinstance(self.max_samples, int):
            current_max_samples = self.max_samples # Use as is
        
        current_max_samples = max(1, min(current_max_samples, current_n_samples)) # Ensure valid range


        # Contamination handling: if float, it must be in (0, 0.5] for scikit-learn IF
        # 'auto' is also valid.
        current_contamination = self.contamination
        if isinstance(self.contamination, float):
            if not (0 < self.contamination <= 0.5):
                print(f"IF_SERVICE: Warning - Contamination {self.contamination} for group '{group_identifier}' is outside (0, 0.5]. Using 'auto'.")
                current_contamination = 'auto'
        
        # print(f"IF_SERVICE: Applying IF to group '{group_identifier}' (size: {current_n_samples}) with contamination='{current_contamination}', max_samples={current_max_samples}")

        isolation_forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=current_contamination,
            max_samples=current_max_samples,
            random_state=self.random_state,
            n_jobs=-1 # Use all cores
        )
        
        try:
            outlier_labels = isolation_forest.fit_predict(features_group_np) # -1 for outliers, 1 for inliers
            anomaly_scores = isolation_forest.decision_function(features_group_np) # Lower is more anomalous
        except ValueError as e:
            print(f"IF_SERVICE: Error fitting IF for group '{group_identifier}': {e}. Marking all as inliers.")
            outlier_labels = np.ones(current_n_samples, dtype=int)
            anomaly_scores = np.zeros(current_n_samples) # Assign neutral score

        return outlier_labels, anomaly_scores

    def detect_outliers_per_cluster(
        self, 
        all_features_df: pd.DataFrame, 
        cluster_labels_series: pd.Series
    ) -> pd.DataFrame:
        if not all_features_df.index.equals(cluster_labels_series.index):
            raise ValueError("IF_SERVICE: Indices of features DataFrame and cluster labels Series do not match.")

        results_list = [] # To store dicts for each original point
        unique_clusters = sorted(cluster_labels_series.unique())
        
        # Prepare a DataFrame to hold all results, initialized with original index
        # This ensures all points are covered, even if a cluster is skipped or IF fails
        final_results_df = pd.DataFrame(index=all_features_df.index)
        final_results_df['is_outlier'] = False # Default to not outlier
        final_results_df['if_score'] = 0.0     # Default neutral score
        final_results_df['final_cluster_label'] = cluster_labels_series

        print(f"IF_SERVICE: Processing {len(unique_clusters)} unique cluster labels: {unique_clusters}")

        for cluster_id in unique_clusters:
            current_cluster_indices = cluster_labels_series[cluster_labels_series == cluster_id].index
            
            if cluster_id == -1: # Noise points from DBSCAN
                print(f"IF_SERVICE: {len(current_cluster_indices)} noise points (cluster_id = -1) will be analyzed with Isolation Forest.")
                # Use a higher contamination rate for noise points to be more aggressive in detecting outliers
                noise_contamination = 0.05  # Fixed higher contamination rate for noise points
                if isinstance(self.contamination, float):
                    # Use at least 5% contamination for noise points, or higher if the original setting was higher
                    noise_contamination = max(0.05, self.contamination)
                
                features_noise_points_df = all_features_df.loc[current_cluster_indices]
                
                # Apply Isolation Forest to noise points
                noise_outlier_labels_np, noise_anomaly_scores_np = self._fit_predict_if_on_group(
                    features_noise_points_df.values,
                    group_identifier="noise_points"
                )
                
                # Update results for noise points
                final_results_df.loc[current_cluster_indices, 'is_outlier'] = (noise_outlier_labels_np == -1)
                final_results_df.loc[current_cluster_indices, 'if_score'] = noise_anomaly_scores_np
                continue

            if current_cluster_indices.empty:
                print(f"IF_SERVICE: Cluster {cluster_id} has no points. Skipping.")
                continue
                
            features_this_cluster_df = all_features_df.loc[current_cluster_indices]
            
            outlier_labels_np, anomaly_scores_np = self._fit_predict_if_on_group(
                features_this_cluster_df.values, # Pass NumPy array
                group_identifier=cluster_id
            )
            
            # Update the final_results_df for points in this cluster
            final_results_df.loc[current_cluster_indices, 'is_outlier'] = (outlier_labels_np == -1)
            final_results_df.loc[current_cluster_indices, 'if_score'] = anomaly_scores_np
        
        # Add original_index as a column from the index
        final_results_df['original_index'] = final_results_df.index
        final_results_df = final_results_df.reset_index(drop=True) # Make original_index a regular column and reset index

        # Reorder columns for clarity
        final_results_df = final_results_df[['original_index', 'is_outlier', 'if_score', 'final_cluster_label']]

        print(f"IF_SERVICE: Isolation Forest per cluster processing completed. Total outliers: {final_results_df['is_outlier'].sum()}")
        return final_results_df