# backend/app/utils/visualization_utils.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server-side plotting
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from typing import Dict, Optional, Any, List # Added List

from app.config.config import get_settings # Ensures settings are fresh if called multiple times
settings = get_settings() # Get settings once at module level

def _get_viz_artifact_path(
    # base_path: str, # No longer needed as settings.OUTLIER_ARTIFACTS_BASE_PATH is used
    user_id: int, 
    dataset_id: int, 
    plot_name: str,
    run_id: Optional[int] = None
) -> str:
    # Use settings directly for the base path
    base_dir_for_viz = settings.OUTLIER_ARTIFACTS_BASE_PATH
    sub_dir = f"run_{run_id}" if run_id is not None else "general_viz"
    dir_path = os.path.join(base_dir_for_viz, f"user_{user_id}", f"dataset_{dataset_id}", "visualizations", sub_dir)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, plot_name)

def generate_and_save_outlier_visualizations(
    dataset_id: int,
    user_id: int,
    outlier_run_id: int,
    # base_artifact_path: str, # <<< REMOVED THIS PARAMETER
    latent_features_df: pd.DataFrame,
    outlier_results_df: pd.DataFrame, # Contains 'original_index', 'is_outlier', 'final_cluster_label'
    num_pca_components: Optional[int] = 2
) -> Dict[str, Optional[str]]:
    """
    Generates and saves basic visualizations for outlier detection results.
    Returns a dictionary of plot names to their saved paths.
    """
    print(f"VIZ_UTILS: Starting visualization generation for dataset {dataset_id}, run {outlier_run_id}")
    saved_plot_paths: Dict[str, Optional[str]] = {}

    if latent_features_df.empty or outlier_results_df.empty:
        print("VIZ_UTILS: Latent features or outlier results are empty. Skipping visualization.")
        return saved_plot_paths

    # --- Prepare data for plotting ---
    plot_df = pd.merge(
        latent_features_df.reset_index(drop=True), # Ensure 0-based index for merge if not already
        outlier_results_df, # This should have 'original_index' as a column
        left_index=True,    # Merge latent_features_df's new 0-based index
        right_on='original_index', # with this column from outlier_results_df
        how='inner'
    )

    if plot_df.empty:
        print("VIZ_UTILS: Merged data for plotting is empty. Check indices/original_index alignment. Skipping visualization.")
        print(f"VIZ_UTILS: Latent features index sample: {latent_features_df.index[:5]}")
        print(f"VIZ_UTILS: Outlier results original_index sample: {outlier_results_df['original_index'][:5] if 'original_index' in outlier_results_df else 'Column Missing'}")
        return saved_plot_paths

    # --- 1. Scatter Plot of Latent Features (PCA if > 2D) ---
    # Look for latent feature columns with various possible prefixes
    feature_prefixes = ['latent_', 'pca_feature_', 'isomap_feature_', 'ae_latent_']
    
    # Try to find columns with any of the known prefixes
    feature_cols_for_pca = []
    for prefix in feature_prefixes:
        cols = [col for col in plot_df.columns if col.startswith(prefix)]
        if cols:
            feature_cols_for_pca = cols
            print(f"VIZ_UTILS: Found {len(cols)} feature columns with prefix '{prefix}'")
            break
    
    # If no features found with known prefixes, try to identify numerical columns that might be latent features
    if not feature_cols_for_pca:
        # Look for columns that might be numerical features (exclude known metadata columns)
        excluded_cols = ['original_index', 'is_outlier', 'cluster_label', 'final_cluster_label', 
                         'if_score', 'lof_score', 'ocsvm_score', 'ensemble_score', 'vote_ratio']
        potential_feature_cols = [col for col in plot_df.columns 
                               if col not in excluded_cols and pd.api.types.is_numeric_dtype(plot_df[col])]
        
        # If we found some potential feature columns, use the first two
        if len(potential_feature_cols) >= 2:
            feature_cols_for_pca = potential_feature_cols[:2]
            print(f"VIZ_UTILS: No standard feature columns found. Using {len(feature_cols_for_pca)} numerical columns as features.")
    
    if not feature_cols_for_pca:
        print("VIZ_UTILS: No suitable feature columns found for visualization. Skipping scatter plot.")
        return saved_plot_paths
        
    features_for_pca_data = plot_df[feature_cols_for_pca]
    
    plot_features_np: np.ndarray
    plot_x_label: str = "Component 1"
    plot_y_label: str = "Component 2"

    if features_for_pca_data.shape[1] > 2 and num_pca_components is not None and num_pca_components > 0:
        actual_pca_components = min(num_pca_components, features_for_pca_data.shape[1], features_for_pca_data.shape[0]) # Cannot have more components than samples or features
        if actual_pca_components < 1 : # Need at least 1 component
            print(f"VIZ_UTILS: Not enough samples/features for PCA ({actual_pca_components} components). Using raw features if possible.")
            # Fallback to using first 1 or 2 raw features if actual_pca_components is < 1
            if features_for_pca_data.shape[1] >= 2:
                plot_features_np = features_for_pca_data.iloc[:, :2].fillna(0).values
                plot_x_label = features_for_pca_data.columns[0]
                plot_y_label = features_for_pca_data.columns[1]
            elif features_for_pca_data.shape[1] == 1:
                plot_features_np = np.column_stack((features_for_pca_data.iloc[:, 0].fillna(0).values, np.arange(len(features_for_pca_data))))
                plot_x_label = features_for_pca_data.columns[0]
                plot_y_label = "Index"
            else: # Should be caught earlier
                 return saved_plot_paths
        else:
            print(f"VIZ_UTILS: Performing PCA to reduce to {actual_pca_components} components for plotting...")
            pca = PCA(n_components=actual_pca_components, random_state=42)
            try:
                plot_features_np = pca.fit_transform(features_for_pca_data.fillna(0)) # fillna for robustness
                plot_x_label = f"PCA Comp 1 (Expl. Var: {pca.explained_variance_ratio_[0]:.2f})"
                if actual_pca_components > 1:
                    plot_y_label = f"PCA Comp 2 (Expl. Var: {pca.explained_variance_ratio_[1]:.2f})"
                    if actual_pca_components == 1: # If PCA results in 1D, make it plottable on 2D
                        plot_features_np = np.column_stack((plot_features_np[:,0], np.zeros(len(plot_features_np))))
                else: # Only 1 PCA component
                    plot_y_label = "N/A (1D PCA)"
                    plot_features_np = np.column_stack((plot_features_np[:,0], np.zeros(len(plot_features_np))))
            except Exception as e:
                print(f"VIZ_UTILS: PCA failed: {e}. Using first two latent features if available.")
                # Fallback if PCA fails
                if features_for_pca_data.shape[1] >= 2:
                    plot_features_np = features_for_pca_data.iloc[:, :2].fillna(0).values
                    plot_x_label = features_for_pca_data.columns[0]
                    plot_y_label = features_for_pca_data.columns[1]
                elif features_for_pca_data.shape[1] == 1:
                    plot_features_np = np.column_stack((features_for_pca_data.iloc[:, 0].fillna(0).values, np.arange(len(features_for_pca_data))))
                    plot_x_label = features_for_pca_data.columns[0]
                    plot_y_label = "Index"
                else: return saved_plot_paths
    elif features_for_pca_data.shape[1] == 2:
        plot_features_np = features_for_pca_data.fillna(0).values
        plot_x_label = features_for_pca_data.columns[0]
        plot_y_label = features_for_pca_data.columns[1]
    elif features_for_pca_data.shape[1] == 1:
        plot_features_np = np.column_stack((features_for_pca_data.iloc[:, 0].fillna(0).values, np.arange(len(features_for_pca_data))))
        plot_x_label = features_for_pca_data.columns[0]
        plot_y_label = "Index"
    else:
        print("VIZ_UTILS: Not enough latent features for scatter plot. Skipping.")
        return saved_plot_paths

    plt.figure(figsize=(12, 10))
    
    if 'final_cluster_label' not in plot_df.columns or 'is_outlier' not in plot_df.columns:
        print("VIZ_UTILS: 'final_cluster_label' or 'is_outlier' missing from plot_df. Skipping scatter plot.")
        plt.close() # Close figure if not used
        return saved_plot_paths

    # Ensure hue data is appropriate for palette
    hue_data = plot_df['final_cluster_label'].astype(str) # Convert to string for discrete palette
    unique_hues = sorted(hue_data.unique())
    palette = sns.color_palette("deep", n_colors=len(unique_hues))
    
    sns.scatterplot(
        x=plot_features_np[:, 0],
        y=plot_features_np[:, 1],
        hue=hue_data,
        hue_order=unique_hues, # Ensure consistent color mapping
        size=plot_df['is_outlier'], 
        sizes=(40, 120),
        style=plot_df['is_outlier'],
        markers={False: 'o', True: 'X'},
        palette=palette,
        alpha=0.8,
        legend="full" 
    )
    plt.title(f"Outlier Detection: Latent Space (Dataset {dataset_id}, Run {outlier_run_id})", fontsize=16)
    plt.xlabel(plot_x_label, fontsize=12)
    plt.ylabel(plot_y_label, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    scatter_file_name = f"scatter_latent_pca_clusters_run_{outlier_run_id}.png"
    # Use the internal _get_viz_artifact_path which now uses global settings
    scatter_plot_save_path = _get_viz_artifact_path(
        user_id, # Corrected order for internal helper
        dataset_id, 
        scatter_file_name, 
        run_id=outlier_run_id
    )
    try:
        plt.savefig(scatter_plot_save_path, dpi=150)
        saved_plot_paths["scatter_plot_pca_path"] = scatter_plot_save_path
        print(f"VIZ_UTILS: Scatter plot saved to {scatter_plot_save_path}")
    except Exception as e:
        print(f"VIZ_UTILS: Error saving scatter plot: {e}")
        saved_plot_paths["scatter_plot_pca_path"] = None
    finally:
        plt.close() # Always close the figure
    
    print(f"VIZ_UTILS: Visualization generation finished for run {outlier_run_id}.")
    return saved_plot_paths