# backend/app/services/outlier_detection/clustering/denclue_service.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import json
import joblib
from typing import Dict, Any, Optional, Tuple, List, Union

# Helper function to convert NumPy types to Python native types for JSON serialization
def convert_to_serializable(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(i) for i in obj)
    else:
        return obj

from app.config.config import get_settings
settings = get_settings()

def _get_denclue_artifact_path(
    base_path: str,
    dataset_id: int,
    user_id: int,
    artifact_name: str
) -> str:
    dir_path = os.path.join(base_path, f"user_{user_id}", f"dataset_{dataset_id}", "denclue_outputs")
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, artifact_name)

class DenclueService:
    def __init__(
        self,
        dataset_id: int,
        user_id: int,
        h: float = 0.1,  # Kernel bandwidth
        eps: float = 1e-4,  # Convergence threshold
        random_state: int = 42
    ):
        self.dataset_id = dataset_id
        self.user_id = user_id
        self.h = h
        self.eps = eps
        self.random_state = random_state
        
        self.fitted_scaler_ = None
        
        self.ARTIFACT_BASE = settings.OUTLIER_ARTIFACTS_BASE_PATH
        
        # Define paths for artifacts
        self.clustering_metadata_path = _get_denclue_artifact_path(
            self.ARTIFACT_BASE, dataset_id, user_id, "denclue_metadata.json"
        )
        self.cluster_labels_save_path = _get_denclue_artifact_path(
            self.ARTIFACT_BASE, dataset_id, user_id, "denclue_cluster_labels.parquet"
        )
        self.latent_features_scaler_path = _get_denclue_artifact_path(
            self.ARTIFACT_BASE, dataset_id, user_id, "denclue_latent_features_scaler.joblib"
        )
    
    def _gaussian_kernel(self, x, y, h):
        """Gaussian kernel function for density estimation"""
        d = np.linalg.norm(x - y)
        return np.exp(-(d**2) / (2 * h**2))
    
    def _gaussian_kernel_vectorized(self, x, data, h):
        """Vectorized Gaussian kernel for multiple points"""
        # Calculate squared distances efficiently using broadcasting
        diff = data - x  # Broadcasting to create differences for all points
        squared_distances = np.sum(diff * diff, axis=1)  # Sum of squared differences
        return np.exp(-squared_distances / (2 * h**2))
    
    def _estimate_density(self, x, data, h):
        """Estimate density at point x using vectorized Gaussian kernel"""
        n = data.shape[0]
        # Use vectorized kernel calculation for better performance
        kernel_values = self._gaussian_kernel_vectorized(x, data, h)
        return np.sum(kernel_values) / (n * h)
    
    def _mean_shift(self, x, data, h, eps):
        """Optimized mean shift algorithm to find density attractors"""
        max_iter = 50  # Reduced from 100 to improve performance
        x = np.array(x, dtype=np.float64)  # Ensure x is a numpy array with proper dtype
        
        # Use early stopping if movement is small
        for _ in range(max_iter):
            # Vectorized kernel calculation
            kernel_values = self._gaussian_kernel_vectorized(x, data, h)
            
            # Calculate weighted sum of points
            weighted_sum = np.sum(kernel_values[:, np.newaxis] * data, axis=0)
            kernel_sum = np.sum(kernel_values)
            
            if kernel_sum < 1e-10:  # Avoid division by very small numbers
                return x  # No significant movement
            
            new_x = weighted_sum / kernel_sum
            if np.linalg.norm(new_x - x) < eps:
                return new_x  # Converged
            
            x = new_x
        
        return x  # Return after max iterations
    
    def _scale_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Scale features using StandardScaler"""
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df)
        self.scaler = scaler
        return scaled_features
    
    def perform_denclue_clustering(self, features_df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Performs DENCLUE clustering on the provided features.
        Features are scaled (StandardScaler) before clustering.
        Saves cluster labels, metadata, and the scaler used.
        
        For large datasets, sampling is used to improve performance.
        
        Args:
            features_df: DataFrame with features to cluster
            
        Returns:
            Tuple of (cluster labels series, metadata dictionary)
        """
        try:
            print("DENCLUE_SERVICE: Starting DENCLUE clustering")
            
            if not isinstance(features_df, pd.DataFrame) or features_df.empty:
                raise ValueError("DENCLUE_SERVICE: Input features_df must be a non-empty Pandas DataFrame.")
            
            # Ensure all columns are numeric
            if not all(features_df.dtypes.apply(pd.api.types.is_numeric_dtype)):
                non_numeric_cols = features_df.columns[~features_df.dtypes.apply(pd.api.types.is_numeric_dtype)].tolist()
                raise ValueError(f"DENCLUE_SERVICE: All input features must be numeric. Non-numeric columns found: {non_numeric_cols}")
            
            # Handle NaN values
            if features_df.isnull().values.any():
                print(f"DENCLUE_SERVICE: Warning - NaNs found in {features_df.isnull().sum().sum()} cells. Imputing with column means.")
                for col in features_df.columns[features_df.isnull().any()]:
                    features_df[col] = features_df[col].fillna(features_df[col].mean())
                # If any column was entirely NaN, fill with 0
                if features_df.isnull().values.any():
                    features_df = features_df.fillna(0)
            
            # Scale features
            scaled_feature_values = self._scale_features(features_df)
            print(f"DENCLUE_SERVICE: Features scaled, shape: {scaled_feature_values.shape}")
            
            # Save scaler
            with open(self.latent_features_scaler_path, 'wb') as f:
                joblib.dump(self.scaler, f)
            print(f"DENCLUE_SERVICE: Latent features scaler saved to {self.latent_features_scaler_path}")
            
            # Set random seed
            np.random.seed(self.random_state)
            
            # Get data points
            data = scaled_feature_values
            original_n_samples = data.shape[0]
            
            # Define sampling thresholds
            LARGE_DATASET_THRESHOLD = 10000
            VERY_LARGE_DATASET_THRESHOLD = 20000
            EXTREME_DATASET_THRESHOLD = 30000
            
            # Initialize variables
            sample_data = data
            sample_indices = None
            sample_size = None
            current_n_samples = original_n_samples
            use_dbscan_fallback = False
            sample_cluster_labels = None
            
            # For extremely large datasets, use DBSCAN as a fallback
            if original_n_samples > EXTREME_DATASET_THRESHOLD:
                print(f"DENCLUE_SERVICE: Dataset is extremely large ({original_n_samples} samples). Using DBSCAN fallback.")
                use_dbscan_fallback = True
                
                # Import DBSCAN
                from sklearn.cluster import DBSCAN
                
                # Estimate eps parameter based on kernel bandwidth
                dbscan_eps = self.h * 3  # Rule of thumb: 3 times the kernel bandwidth
                dbscan_min_samples = 5    # Default min_samples
                
                print(f"DENCLUE_SERVICE: Running DBSCAN with eps={dbscan_eps}, min_samples={dbscan_min_samples}")
                
                # Run DBSCAN
                dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, n_jobs=-1)
                sample_cluster_labels = dbscan.fit_predict(data)
                
                # Skip the rest of the DENCLUE algorithm
                print(f"DENCLUE_SERVICE: DBSCAN completed. Found {len(np.unique(sample_cluster_labels)) - (1 if -1 in sample_cluster_labels else 0)} clusters")
            
            # Handle sampling for large datasets
            elif original_n_samples > VERY_LARGE_DATASET_THRESHOLD:
                # For very large datasets, use more aggressive sampling
                sample_size = 5000
                print(f"DENCLUE_SERVICE: Dataset is very large ({original_n_samples} samples). Using aggressive sampling for DENCLUE.")
                sample_indices = np.random.choice(original_n_samples, sample_size, replace=False)
                sample_data = data[sample_indices]
                current_n_samples = sample_size
                print(f"DENCLUE_SERVICE: Sampled {sample_size} points for DENCLUE clustering")
            elif original_n_samples > LARGE_DATASET_THRESHOLD:
                # For large datasets, use moderate sampling
                sample_size = 10000
                print(f"DENCLUE_SERVICE: Dataset is large ({original_n_samples} samples). Using sampling for DENCLUE.")
                sample_indices = np.random.choice(original_n_samples, sample_size, replace=False)
                sample_data = data[sample_indices]
                current_n_samples = sample_size
                print(f"DENCLUE_SERVICE: Sampled {sample_size} points for DENCLUE clustering")
            
            # Only run the density attractor calculation if not using DBSCAN fallback
            if not use_dbscan_fallback:
                print(f"DENCLUE_SERVICE: Finding density attractors for {current_n_samples} points")
                attractors = []
                for i in range(current_n_samples):
                    attractor = self._mean_shift(sample_data[i], sample_data, self.h, self.eps)
                    attractors.append(attractor)
                
                attractors = np.array(attractors)
            
                # Only perform DENCLUE clustering if not using DBSCAN fallback
                # Use a more efficient approach for clustering based on attractors
                from sklearn.neighbors import NearestNeighbors
                
                # Cluster points based on their attractors using nearest neighbors
                sample_cluster_labels = np.full(current_n_samples, -1)  # Start with all points as noise
                current_cluster = 0
                
                print(f"DENCLUE_SERVICE: Clustering points based on density attractors using nearest neighbors")
                
                # Use nearest neighbors to find points with similar attractors more efficiently
                nn = NearestNeighbors(radius=self.eps, algorithm='auto', n_jobs=-1)
                nn.fit(attractors)
                
                # Find neighbors for each attractor within epsilon radius
                neighbors_indices = nn.radius_neighbors(attractors, return_distance=False)
                
                # Process points in batches to improve performance
                batch_size = 500  # Process attractors in batches
                for batch_start in range(0, current_n_samples, batch_size):
                    batch_end = min(batch_start + batch_size, current_n_samples)
                    
                    for i in range(batch_start, batch_end):
                        if sample_cluster_labels[i] != -1:
                            continue  # Already assigned
                        
                        # Get points with similar attractors
                        cluster_points = neighbors_indices[i]
                        
                        # If density is high enough, form a cluster
                        density = self._estimate_density(attractors[i], sample_data, self.h)
                        min_density_threshold = 1e-5  # Minimum density to form a cluster
                        
                        if density > min_density_threshold and len(cluster_points) >= 2:
                            for point in cluster_points:
                                sample_cluster_labels[point] = current_cluster
                            current_cluster += 1
                            
                            # Early stopping check - if we've found enough clusters, we can stop
                            if current_cluster >= 20:  # Arbitrary limit for number of clusters
                                print(f"DENCLUE_SERVICE: Early stopping - found {current_cluster} clusters")
                                break
                    
                    # Another early stopping check after each batch
                    if current_cluster >= 20:
                        break
            
            # Handle cluster label assignment based on whether we used DBSCAN fallback or sampling
            if use_dbscan_fallback:
                # For DBSCAN fallback, we already have cluster labels for the full dataset
                cluster_labels = sample_cluster_labels
                print(f"DENCLUE_SERVICE: Using DBSCAN cluster labels directly for full dataset")
            elif sample_indices is not None:
                # If we used sampling with DENCLUE, assign cluster labels to all points in the original dataset
                print(f"DENCLUE_SERVICE: Assigning cluster labels to full dataset using nearest neighbors")
                
                # Use nearest neighbors to assign cluster labels to all points
                from sklearn.neighbors import NearestNeighbors
                
                # Get points with assigned clusters (non-noise points)
                clustered_indices = np.where(sample_cluster_labels >= 0)[0]
                if len(clustered_indices) > 0:
                    clustered_points = sample_data[clustered_indices]
                    clustered_labels = sample_cluster_labels[clustered_indices]
                    
                    # Fit nearest neighbors model on clustered points
                    nn = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1).fit(clustered_points)
                    
                    # Assign labels to all points in the original dataset
                    # Process in batches to avoid memory issues
                    batch_size = 10000
                    cluster_labels = np.full(original_n_samples, -1)  # Start with all points as noise
                    
                    for i in range(0, original_n_samples, batch_size):
                        end_idx = min(i + batch_size, original_n_samples)
                        batch_data = data[i:end_idx]
                        
                        # Find nearest neighbor for each point
                        distances, indices = nn.kneighbors(batch_data)
                        
                        # Assign cluster label of nearest neighbor
                        # Use vectorized assignment for better performance
                        batch_labels = clustered_labels[indices.flatten()]
                        cluster_labels[i:end_idx] = batch_labels
                else:
                    # No clusters found in sample, all points are noise
                    cluster_labels = np.full(original_n_samples, -1)
            else:
                # No sampling was used, use the sample_cluster_labels directly
                cluster_labels = sample_cluster_labels
            
            # Create a Series for cluster labels
            cluster_labels_series = pd.Series(cluster_labels, index=features_df.index, name="cluster_label")
            
            # Calculate number of clusters and noise points
            unique_labels = set(cluster_labels)
            num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            num_noise = int(np.sum(cluster_labels == -1))
            
            print(f"DENCLUE_SERVICE: DENCLUE completed. Clusters found: {num_clusters}, Noise points: {num_noise}")
            
            # Save metadata
            metadata = {
                "algorithm": "DENCLUE",
                "params_input": {"h": self.h, "eps": self.eps},
                "input_features_shape": list(features_df.shape),
                "scaled_features_shape": list(scaled_feature_values.shape),
                "num_clusters_found": num_clusters,
                "num_noise_points": num_noise,
                "artifacts": {
                    "labels_path": self.cluster_labels_save_path,
                    "metadata_path": self.clustering_metadata_path,
                    "latent_features_scaler_path": self.latent_features_scaler_path
                },
                "sampling": {
                    "used": sample_indices is not None or use_dbscan_fallback,
                    "original_size": original_n_samples,
                    "sampled_size": current_n_samples,
                    "dbscan_fallback": use_dbscan_fallback
                }
            }
            
            # Save metadata to file
            with open(self.clustering_metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            print(f"DENCLUE_SERVICE: Metadata saved to {self.clustering_metadata_path}")
            
            # Save cluster labels - convert Series to DataFrame before saving to parquet
            cluster_labels_df = cluster_labels_series.to_frame()
            cluster_labels_df.to_parquet(self.cluster_labels_save_path)
            print(f"DENCLUE_SERVICE: Cluster labels saved to {self.cluster_labels_save_path}")
            
            return cluster_labels_series, metadata
            
        except Exception as e:
            print(f"DENCLUE_SERVICE: Error during DENCLUE clustering: {e}")
            import traceback
            print(traceback.format_exc())
            raise
    
    def get_artifact_paths(self) -> Dict[str, Optional[str]]:
        """Returns paths to artifacts generated by this service."""
        return {
            "clustering_metadata_path": self.clustering_metadata_path if os.path.exists(self.clustering_metadata_path) else None,
            "cluster_labels_path": self.cluster_labels_save_path if os.path.exists(self.cluster_labels_save_path) else None,
            "latent_features_scaler_path": self.latent_features_scaler_path if os.path.exists(self.latent_features_scaler_path) else None
        }


def run_denclue_clustering(df: pd.DataFrame, parameters: dict, true_labels=None) -> dict:
    """
    Wrapper function to run DENCLUE clustering on a DataFrame.
    
    Args:
        df: DataFrame to cluster
        parameters: Dictionary of parameters for DENCLUE
        
    Returns:
        Dictionary with clustering results, evaluation metrics, and visualizations
    """
    # Extract parameters
    h = parameters.get('h', 0.1)  # Kernel bandwidth
    eps = parameters.get('eps', 1e-4)  # Convergence threshold
    
    # Create a temporary service instance
    # Using 0 for dataset_id and user_id as they're not used for plotting
    service = DenclueService(
        dataset_id=0,
        user_id=0,
        h=h,
        eps=eps
    )
    
    # Run clustering
    cluster_labels, metadata = service.perform_denclue_clustering(df)
    
    # Add cluster labels to the DataFrame
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels
    
    # Calculate evaluation metrics
    from sklearn import metrics
    
    # Only calculate silhouette score if there's more than one cluster and no noise points (-1)
    silhouette_score = None
    if len(set(cluster_labels)) > 1 and -1 not in cluster_labels:
        try:
            silhouette_score = metrics.silhouette_score(df, cluster_labels)
        except:
            silhouette_score = None
    
    # Calculate Davies-Bouldin index if there's more than one cluster
    davies_bouldin_score = None
    if len(set(cluster_labels)) > 1 and -1 not in cluster_labels:
        try:
            davies_bouldin_score = metrics.davies_bouldin_score(df, cluster_labels)
        except:
            davies_bouldin_score = None
            
    # Calculate supervised metrics if true labels are provided
    f1_score = None
    precision_score = None
    recall_score = None
    accuracy_score = None
    
    if true_labels is not None:
        # For clustering evaluation with ground truth, we need to handle label matching
        # since cluster labels might not match true labels numerically
        from sklearn.metrics import accuracy_score, f1_score as f1, precision_score as precision, recall_score as recall
        from sklearn.metrics.cluster import contingency_matrix
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        
        try:
            # Create contingency matrix
            contingency = contingency_matrix(true_labels, cluster_labels)
            # Find optimal one-to-one mapping between cluster labels and true labels
            row_ind, col_ind = linear_sum_assignment(-contingency)
            
            # Create a mapping from cluster labels to true labels
            cluster_to_true = {}
            unique_clusters = np.unique(cluster_labels)
            for i, cluster in enumerate(unique_clusters):
                if cluster != -1:  # Skip noise points
                    idx = np.where(unique_clusters == cluster)[0][0]
                    if idx < len(row_ind):
                        cluster_to_true[cluster] = col_ind[idx]
            
            # Map cluster labels to matched true labels
            mapped_labels = np.array([cluster_to_true.get(label, -1) for label in cluster_labels])
            
            # Calculate metrics (ignoring noise points)
            mask = cluster_labels != -1
            if np.any(mask):
                accuracy_score = accuracy_score(true_labels[mask], mapped_labels[mask])
                precision_score = precision(true_labels[mask], mapped_labels[mask], average='weighted')
                recall_score = recall(true_labels[mask], mapped_labels[mask], average='weighted')
                f1_score = f1(true_labels[mask], mapped_labels[mask], average='weighted')
        except Exception as e:
            print(f"Error calculating supervised metrics: {e}")
            # Keep metrics as None if calculation fails
    
    # Calculate cluster statistics
    unique_labels = set(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(cluster_labels).count(-1) if -1 in unique_labels else 0
    cluster_sizes = {label: list(cluster_labels).count(label) for label in unique_labels if label != -1}
    
    # Generate enhanced visualizations
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import matplotlib.cm as cm
    import tempfile
    import io
    import base64
    from sklearn.decomposition import PCA
    
    # Create visualization paths
    scatter_plot_path = None
    distribution_plot_path = None
    pca_plot_path = None
    density_plot_path = None
    
    # 1. Basic scatter plot for the first two dimensions
    if df.shape[1] >= 2:
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        x_col, y_col = df.columns[0], df.columns[1]
        scatter = ax.scatter(df[x_col], df[y_col], c=cluster_labels, cmap='viridis', alpha=0.6)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'DENCLUE Clustering (h={h}, eps={eps})')
        fig.colorbar(scatter, ax=ax, label='Cluster')
        
        # Save the plot to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            scatter_plot_path = tmp.name
            fig.savefig(scatter_plot_path)
    
    # 2. Cluster size distribution plot
    if n_clusters > 0:
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        cluster_labels_list = [label for label in unique_labels if label != -1]
        cluster_sizes_list = [cluster_sizes[label] for label in cluster_labels_list]
        
        bars = ax.bar(cluster_labels_list, cluster_sizes_list, color='skyblue')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Points')
        ax.set_title('Cluster Size Distribution')
        
        # Add the exact count on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            distribution_plot_path = tmp.name
            fig.savefig(distribution_plot_path)
    
    # 3. PCA visualization for high-dimensional data
    if df.shape[1] > 2:
        # Apply PCA to reduce to 2 dimensions for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df)
        
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title(f'PCA Visualization of DENCLUE Clusters (h={h}, eps={eps})')
        fig.colorbar(scatter, ax=ax, label='Cluster')
        
        # Add explained variance ratio
        explained_variance = pca.explained_variance_ratio_
        ax.text(0.02, 0.98, f'Explained variance: PC1={explained_variance[0]:.2f}, PC2={explained_variance[1]:.2f}',
                transform=ax.transAxes, verticalalignment='top')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            pca_plot_path = tmp.name
            fig.savefig(pca_plot_path)
    
    # 4. Density estimation visualization (specific to DENCLUE)
    if df.shape[1] == 2:  # Only for 2D data for simplicity
        try:
            # Create a mesh grid for density estimation
            x_min, x_max = df[df.columns[0]].min() - 0.1, df[df.columns[0]].max() + 0.1
            y_min, y_max = df[df.columns[1]].min() - 0.1, df[df.columns[1]].max() + 0.1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            
            # Estimate density at each point in the grid
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            densities = np.zeros(grid_points.shape[0])
            
            # Use the service's kernel function to estimate density
            for i, point in enumerate(grid_points):
                densities[i] = service._estimate_density(point, df.values, h)
            
            # Reshape for plotting
            densities = densities.reshape(xx.shape)
            
            # Plot the density map
            fig = Figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            
            contour = ax.contourf(xx, yy, densities, cmap='viridis', alpha=0.8)
            ax.scatter(df[df.columns[0]], df[df.columns[1]], c=cluster_labels, cmap='viridis', 
                       edgecolor='k', s=50, alpha=0.7)
            ax.set_xlabel(df.columns[0])
            ax.set_ylabel(df.columns[1])
            ax.set_title(f'DENCLUE Density Estimation (h={h})')
            fig.colorbar(contour, ax=ax, label='Density')
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                density_plot_path = tmp.name
                fig.savefig(density_plot_path)
        except Exception as e:
            print(f"Could not generate density plot: {e}")
            density_plot_path = None
    
    # Create a summary of the analysis
    analysis_summary = {
        "algorithm": "DENCLUE",
        "parameters": {
            "h": h,  # Kernel bandwidth
            "eps": eps  # Convergence threshold
        },
        "results": {
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "cluster_sizes": cluster_sizes,
            "data_shape": df.shape
        },
        "evaluation": {
            "silhouette_score": silhouette_score,
            "davies_bouldin_score": davies_bouldin_score,
            "f1_score": f1_score,
            "precision_score": precision_score,
            "recall_score": recall_score,
            "accuracy_score": accuracy_score
        }
    }
    
    # Convert all NumPy types to Python native types for JSON serialization
    analysis_summary = convert_to_serializable(analysis_summary)
    
    # Convert absolute file paths to URL paths for visualization images
    def convert_to_url_path(file_path):
        if file_path is None:
            return None
        # For Windows paths, extract just the filename
        import os
        filename = os.path.basename(file_path)
        return f"/tmp/{filename}"
    
    # Return enhanced results
    result = {
        "clustered_data": df_with_clusters,
        "n_clusters": n_clusters,
        "cluster_labels": cluster_labels.tolist(),
        "analysis_summary": analysis_summary,
        "visualizations": {
            "scatter_plot_path": convert_to_url_path(scatter_plot_path),
            "distribution_plot_path": convert_to_url_path(distribution_plot_path),
            "pca_plot_path": convert_to_url_path(pca_plot_path) if df.shape[1] > 2 else None,
            "density_plot_path": convert_to_url_path(density_plot_path)
        },
        "performance": {
            "sampling_used": metadata.get("sampling", {}).get("used", False),
            "original_dataset_size": metadata.get("sampling", {}).get("original_size", len(df)),
            "sampled_size": metadata.get("sampling", {}).get("sampled_size", len(df))
        }
    }
    
    # Convert any remaining NumPy types in the result
    return convert_to_serializable(result)
