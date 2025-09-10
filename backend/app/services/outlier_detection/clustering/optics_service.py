# backend/app/services/outlier_detection/clustering/optics_service.py
import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
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

def _get_optics_artifact_path(
    base_path: str,
    dataset_id: int,
    user_id: int,
    artifact_name: str
) -> str:
    dir_path = os.path.join(base_path, f"user_{user_id}", f"dataset_{dataset_id}", "optics_outputs")
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, artifact_name)

class OpticsService:
    def __init__(
        self,
        dataset_id: int,
        user_id: int,
        min_samples: int = 5,
        max_eps: float = float('inf'),
        xi: float = 0.05,
        random_state: int = 42
    ):
        self.dataset_id = dataset_id
        self.user_id = user_id
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.xi = xi
        self.random_state = random_state
        
        self.cluster_model = None
        self.fitted_scaler_ = None
        
        self.ARTIFACT_BASE = settings.OUTLIER_ARTIFACTS_BASE_PATH
        
        # Define paths for artifacts
        self.clustering_metadata_path = _get_optics_artifact_path(
            self.ARTIFACT_BASE, dataset_id, user_id, "optics_metadata.json"
        )
        self.cluster_labels_save_path = _get_optics_artifact_path(
            self.ARTIFACT_BASE, dataset_id, user_id, "optics_cluster_labels.parquet"
        )
        self.latent_features_scaler_path = _get_optics_artifact_path(
            self.ARTIFACT_BASE, dataset_id, user_id, "optics_latent_features_scaler.joblib"
        )
        self.reachability_path = _get_optics_artifact_path(
            self.ARTIFACT_BASE, dataset_id, user_id, "optics_reachability.parquet"
        )
    
    def perform_optics_clustering(self, features_df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Performs OPTICS clustering on the provided features.
        Features are scaled (StandardScaler) before clustering.
        Saves cluster labels, metadata, and the scaler used.
        
        Args:
            features_df: DataFrame with features to cluster
            
        Returns:
            Tuple of (cluster_labels_series, metadata_dict)
        """
        if not isinstance(features_df, pd.DataFrame) or features_df.empty:
            raise ValueError("OPTICS_SERVICE: Input features_df must be a non-empty Pandas DataFrame.")
        
        # Ensure all columns are numeric
        if not all(features_df.dtypes.apply(pd.api.types.is_numeric_dtype)):
            non_numeric_cols = features_df.columns[~features_df.dtypes.apply(pd.api.types.is_numeric_dtype)].tolist()
            raise ValueError(f"OPTICS_SERVICE: All input features must be numeric. Non-numeric columns found: {non_numeric_cols}")
        
        # Handle NaNs
        if features_df.isnull().values.any():
            print(f"OPTICS_SERVICE: Warning - NaNs found in {features_df.isnull().sum().sum()} cells. Imputing with column means.")
            for col in features_df.columns[features_df.isnull().any()]:
                features_df[col] = features_df[col].fillna(features_df[col].mean())
            # If any column was entirely NaN, fill with 0
            if features_df.isnull().values.any():
                features_df = features_df.fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        scaled_feature_values = scaler.fit_transform(features_df.values)
        self.fitted_scaler_ = scaler
        
        # Save the scaler
        joblib.dump(self.fitted_scaler_, self.latent_features_scaler_path)
        print(f"OPTICS_SERVICE: Latent features scaler saved to {self.latent_features_scaler_path}")
        
        # Perform OPTICS clustering
        try:
            print(f"OPTICS_SERVICE: Starting OPTICS clustering with min_samples={self.min_samples}, max_eps={self.max_eps}, xi={self.xi}")
            
            # Check if dataset is very large and sample if needed
            MAX_SAMPLES = 10000  # Maximum number of samples for full OPTICS
            use_sampling = False
            sampled_indices = None
            needs_knn_assignment = False
            
            if scaled_feature_values.shape[0] > MAX_SAMPLES:
                print(f"OPTICS_SERVICE: Dataset is large ({scaled_feature_values.shape[0]} samples). Using sampling for OPTICS.")
                use_sampling = True
                needs_knn_assignment = True
                # Use stratified sampling if we have true labels, otherwise random sampling
                np.random.seed(42)  # For reproducibility
                sampled_indices = np.random.choice(scaled_feature_values.shape[0], MAX_SAMPLES, replace=False)
                sampled_data = scaled_feature_values[sampled_indices]
                print(f"OPTICS_SERVICE: Sampled {MAX_SAMPLES} points for initial OPTICS clustering")
            else:
                sampled_data = scaled_feature_values
                # Track indices so we can support secondary sub-sampling consistently
                sampled_indices = np.arange(scaled_feature_values.shape[0])
            
            # Initialize OPTICS with optimized parameters
            optics = OPTICS(
                min_samples=self.min_samples,
                max_eps=self.max_eps,
                xi=self.xi,
                metric='euclidean',
                cluster_method='xi',
                n_jobs=-1,  # Use all available cores
                algorithm='auto',  # Let scikit-learn choose the best algorithm
                # Note: OPTICS doesn't support random_state parameter
            )
            
            # Fit the model on the sampled data
            print(f"OPTICS_SERVICE: Fitting OPTICS on {sampled_data.shape[0]} samples")
            
            # Use a more aggressive sampling if the dataset is still large
            if sampled_data.shape[0] > 5000:
                print(f"OPTICS_SERVICE: Dataset still large, using more aggressive sampling")
                # Further reduce to 5000 samples for very large datasets
                np.random.seed(42)  # For reproducibility
                subsample_indices = np.random.choice(sampled_data.shape[0], 5000, replace=False)
                subsample_data = sampled_data[subsample_indices]
                print(f"OPTICS_SERVICE: Further reduced to {subsample_data.shape[0]} samples")
                
                # Update mapping to reflect secondary sub-sampling
                sampled_indices = sampled_indices[subsample_indices]
                sampled_data = subsample_data
                needs_knn_assignment = True
                
                # Use a faster algorithm for initial clustering
                from sklearn.cluster import DBSCAN
                print("OPTICS_SERVICE: Using DBSCAN for initial clustering (faster)")
                dbscan = DBSCAN(eps=0.5, min_samples=self.min_samples)
                initial_labels = dbscan.fit_predict(subsample_data)
                
                # Only use OPTICS if DBSCAN fails to find meaningful clusters
                if len(set(initial_labels)) <= 2:  # Only noise or one real cluster
                    print("OPTICS_SERVICE: DBSCAN found few clusters, running OPTICS synchronously (no timeout)")
                    try:
                        sampled_labels = optics.fit_predict(subsample_data)
                    except Exception as e:
                        print(f"OPTICS_SERVICE: Error during OPTICS fit_predict: {e}, using DBSCAN results")
                        sampled_labels = initial_labels
                else:
                    print("OPTICS_SERVICE: Using DBSCAN results (found good clusters)")
                    sampled_labels = initial_labels
                    optics = dbscan  # Use DBSCAN as our model
                    
                    # When using double sampling, we need to make sure the data and labels match
                    # Map subsample data back to the original sampled data
                    if use_sampling:
                        # We're using the subsample data and labels for the KNN classifier
                        # so we need to update sampled_data to match
                        sampled_data = subsample_data
                        # And update the indices mapping
                        sampled_indices = sampled_indices[subsample_indices]
            else:
                # For smaller datasets, use OPTICS directly
                sampled_labels = optics.fit_predict(sampled_data)
            
            self.cluster_model = optics
            
            # If we used sampling, we need to assign labels to the full dataset
            if needs_knn_assignment:
                # Use a nearest neighbors classifier to assign labels to all points
                from sklearn.neighbors import KNeighborsClassifier
                print(f"OPTICS_SERVICE: Assigning cluster labels to all {scaled_feature_values.shape[0]} points")
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(sampled_data, sampled_labels)
                cluster_labels = knn.predict(scaled_feature_values)
            else:
                cluster_labels = sampled_labels
            
            # Create a Series for cluster labels
            cluster_labels_series = pd.Series(cluster_labels, index=features_df.index, name="cluster_label")
            
            # Calculate number of clusters and noise points
            unique_labels = set(cluster_labels)
            num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            num_noise = int(np.sum(cluster_labels == -1))
            
            print(f"OPTICS_SERVICE: OPTICS completed. Clusters found: {num_clusters}, Noise points: {num_noise}")
            
            # Save reachability plot data if available (only for OPTICS, not for DBSCAN)
            if hasattr(optics, 'reachability_') and optics.reachability_ is not None:
                # If we used sampling, the reachability_ is only for the sampled points
                if needs_knn_assignment and hasattr(optics, 'ordering_'):
                    # We can only save reachability for the sampled points
                    reachability_df = pd.DataFrame({
                        'ordering': np.arange(len(optics.reachability_)),
                        'reachability': optics.reachability_,
                        'original_index': features_df.index[sampled_indices][optics.ordering_]
                    })
                    print(f"OPTICS_SERVICE: Saving reachability data for {len(optics.reachability_)} sampled points")
                elif hasattr(optics, 'ordering_'):
                    reachability_df = pd.DataFrame({
                        'ordering': np.arange(len(optics.reachability_)),
                        'reachability': optics.reachability_,
                        'original_index': features_df.index[optics.ordering_]
                    })
                    print(f"OPTICS_SERVICE: Saving full reachability data")
                else:
                    # If ordering_ is not available, use a simple index
                    reachability_df = pd.DataFrame({
                        'ordering': np.arange(len(optics.reachability_)),
                        'reachability': optics.reachability_,
                        'original_index': np.arange(len(optics.reachability_))
                    })
                    print(f"OPTICS_SERVICE: Saving reachability data without ordering")
                
                reachability_df.to_parquet(self.reachability_path)
                print(f"OPTICS_SERVICE: Reachability data saved to {self.reachability_path}")
            else:
                # For DBSCAN, create a dummy reachability file
                print("OPTICS_SERVICE: Using DBSCAN, no reachability data available")
                # Create a simple dummy reachability dataframe
                dummy_reachability = np.ones(min(1000, len(cluster_labels_series))) * -1  # -1 indicates no reachability
                reachability_df = pd.DataFrame({
                    'ordering': np.arange(len(dummy_reachability)),
                    'reachability': dummy_reachability,
                    'original_index': np.arange(len(dummy_reachability))
                })
                reachability_df.to_parquet(self.reachability_path)
                print(f"OPTICS_SERVICE: Dummy reachability data saved to {self.reachability_path}")
            
            # Save metadata
            metadata = {
                "algorithm": "OPTICS",
                "params_input": {
                    "min_samples": self.min_samples,
                    "max_eps": str(self.max_eps) if self.max_eps == float('inf') else self.max_eps,
                    "xi": self.xi
                },
                "input_features_shape": list(features_df.shape),
                "scaled_features_shape": list(scaled_feature_values.shape),
                "num_clusters_found": num_clusters,
                "num_noise_points": num_noise,
                "artifacts": {
                    "labels_path": self.cluster_labels_save_path,
                    "metadata_path": self.clustering_metadata_path,
                    "latent_features_scaler_path": self.latent_features_scaler_path,
                    "reachability_path": self.reachability_path
                }
            }
            
            # Convert all NumPy types to Python native types for JSON serialization
            metadata = convert_to_serializable(metadata)
            
            try:
                with open(self.clustering_metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                print(f"OPTICS_SERVICE: Clustering metadata saved to {self.clustering_metadata_path}")
            except IOError as e:
                print(f"OPTICS_SERVICE: Error saving clustering metadata: {e}")
            
            # Save cluster labels
            try:
                cluster_labels_df_to_save = cluster_labels_series.to_frame()
                cluster_labels_df_to_save.to_parquet(self.cluster_labels_save_path)
                print(f"OPTICS_SERVICE: Cluster labels saved to {self.cluster_labels_save_path}")
            except IOError as e:
                print(f"OPTICS_SERVICE: Error saving cluster labels: {e}")
            
            return cluster_labels_series, metadata
            
        except Exception as e:
            print(f"OPTICS_SERVICE: Error during OPTICS clustering: {e}")
            import traceback
            print(traceback.format_exc())
            raise
    
    def get_artifact_paths(self) -> Dict[str, Optional[str]]:
        """Returns paths to artifacts generated by this service."""
        return {
            "clustering_metadata_path": self.clustering_metadata_path if os.path.exists(self.clustering_metadata_path) else None,
            "cluster_labels_path": self.cluster_labels_save_path if os.path.exists(self.cluster_labels_save_path) else None,
            "latent_features_scaler_path": self.latent_features_scaler_path if os.path.exists(self.latent_features_scaler_path) else None,
            "reachability_path": self.reachability_path if os.path.exists(self.reachability_path) else None
        }


def run_optics_clustering(df: pd.DataFrame, parameters: dict, true_labels=None) -> dict:
    """
    Wrapper function to run OPTICS clustering on a DataFrame.
    
    Args:
        df: DataFrame to cluster
        parameters: Dictionary of parameters for OPTICS
        
    Returns:
        Dictionary with clustering results, evaluation metrics, and visualizations
    """
    # Extract parameters
    min_samples = parameters.get('min_samples', 5)
    max_eps = parameters.get('max_eps', float('inf'))
    xi = parameters.get('xi', 0.05)
    
    # Create a temporary service instance
    # Using 0 for dataset_id and user_id as they're not used for plotting
    service = OpticsService(
        dataset_id=0,
        user_id=0,
        min_samples=min_samples,
        max_eps=max_eps,
        xi=xi
    )
    
    # Run clustering
    cluster_labels, metadata = service.perform_optics_clustering(df)
    
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
    reachability_plot_path = None
    
    # 1. Basic scatter plot for the first two dimensions
    if df.shape[1] >= 2:
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        x_col, y_col = df.columns[0], df.columns[1]
        scatter = ax.scatter(df[x_col], df[y_col], c=cluster_labels, cmap='viridis', alpha=0.6)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'OPTICS Clustering (min_samples={min_samples}, xi={xi})')
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
        ax.set_title(f'PCA Visualization of OPTICS Clusters (min_samples={min_samples}, xi={xi})')
        fig.colorbar(scatter, ax=ax, label='Cluster')
        
        # Add explained variance ratio
        explained_variance = pca.explained_variance_ratio_
        ax.text(0.02, 0.98, f'Explained variance: PC1={explained_variance[0]:.2f}, PC2={explained_variance[1]:.2f}',
                transform=ax.transAxes, verticalalignment='top')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            pca_plot_path = tmp.name
            fig.savefig(pca_plot_path)
    
    # 4. Reachability plot (specific to OPTICS)
    try:
        # Check if we're using OPTICS or DBSCAN
        is_optics = hasattr(service.cluster_model, 'reachability_') and service.cluster_model.reachability_ is not None
        
        if is_optics:
            # OPTICS reachability plot
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            # Get reachability values
            reachability = service.cluster_model.reachability_
            
            # For large datasets, sample the reachability plot to improve performance
            MAX_POINTS_TO_PLOT = 5000
            if len(reachability) > MAX_POINTS_TO_PLOT:
                print(f"Sampling reachability plot to {MAX_POINTS_TO_PLOT} points for better performance")
                # Use systematic sampling to preserve the shape of the plot
                step = len(reachability) // MAX_POINTS_TO_PLOT
                indices = np.arange(0, len(reachability), step)
                reachability_sampled = reachability[indices]
                space = np.arange(len(reachability_sampled))
                
                # Plot sampled reachability
                ax.plot(space, np.sort(reachability_sampled))
                ax.set_xlabel(f'Points (sampled from {len(reachability)} total points)')
            else:
                # Plot full reachability
                space = np.arange(len(reachability))
                ax.plot(space, np.sort(reachability))
                ax.set_xlabel('Points (ordered by distance)')
            
            ax.set_ylabel('Reachability Distance')
            ax.set_title('OPTICS Reachability Plot')
            
            # Add horizontal line at xi * max reachability distance
            if xi > 0:
                # Use finite values only to calculate max
                finite_reach = reachability[np.isfinite(reachability)]
                if len(finite_reach) > 0:
                    max_reach = np.max(finite_reach)
                    ax.axhline(y=xi * max_reach, color='r', linestyle='--', 
                               label=f'xi cutoff ({xi:.2f} * max reachability)')
                    ax.legend()
        else:
            # DBSCAN alternative visualization (cluster distribution)
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            # Count points in each cluster
            unique_labels = np.unique(cluster_labels)
            cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
            
            # Create a bar chart of cluster sizes
            bars = ax.bar(unique_labels, cluster_sizes)
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Number of Points')
            ax.set_title('DBSCAN Cluster Distribution (OPTICS fallback)')
            
            # Add the exact count on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        # Save the plot
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            reachability_plot_path = tmp.name
            fig.savefig(reachability_plot_path)
    except Exception as e:
        print(f"Could not generate reachability plot: {e}")
        import traceback
        print(traceback.format_exc())
        reachability_plot_path = None
    
    # Create a summary of the analysis
    analysis_summary = {
        "algorithm": "OPTICS",
        "parameters": {
            "min_samples": min_samples,
            "max_eps": max_eps if max_eps != float('inf') else "infinity",
            "xi": xi
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
            "reachability_plot_path": convert_to_url_path(reachability_plot_path)
        }
    }
    
    # Convert any remaining NumPy types in the result
    return convert_to_serializable(result)
