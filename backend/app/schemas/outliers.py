# backend/app/schemas/outliers.py
from pydantic import BaseModel, Field, ConfigDict # Added ConfigDict for Pydantic v2
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- Existing Schemas (largely unchanged) ---

class DetectOutliersRequest(BaseModel):
    # Algorithm selection for each step
    feature_extraction_method: str = Field("autoencoder", description="Feature extraction method to use. Options: 'autoencoder', 'pca', 'isomap'")
    clustering_method: str = Field("dbscan", description="Clustering method to use. Options: 'dbscan', 'denclue', 'optics'")
    outlier_detection_method: str = Field("isolation_forest", description="Outlier detection method to use. Options: 'isolation_forest', 'lof', 'ocsvm'")
    evaluation_type: str = Field("auto", description="Type of evaluation metrics to use. Options: 'auto', 'classification', 'regression'. If 'auto', the system will automatically determine the best type based on the data.")
    
    # Common parameters
    random_state: int = Field(42, description="Random seed for reproducibility")
    
    # 1. Feature Extraction Parameters
    # 1.1 Autoencoder parameters
    latent_dim: int = Field(16, ge=2, le=128, description="Dimension of the autoencoder's latent space.")
    autoencoder_epochs: int = Field(50, ge=10, le=500, description="Number of training epochs for the autoencoder.")
    autoencoder_batch_size: int = Field(32, ge=8, le=256, description="Batch size for autoencoder training.")
    
    # 1.2 PCA parameters
    pca_n_components: Optional[int] = Field(None, description="Number of components for PCA. If None, same as latent_dim will be used.")
    
    # 1.3 ISOMAP parameters
    isomap_n_components: Optional[int] = Field(None, description="Number of components for ISOMAP. If None, same as latent_dim will be used.")
    isomap_n_neighbors: int = Field(5, ge=2, le=100, description="Number of neighbors for ISOMAP.")
    
    # 2. Clustering Parameters
    # 2.1 DBSCAN parameters
    clustering_eps: float = Field(0.5, gt=0.001, description="DBSCAN: Epsilon parameter.")
    clustering_min_samples: int = Field(5, ge=1, description="DBSCAN: Minimum samples parameter.")
    
    # 2.2 DENCLUE parameters
    denclue_h: float = Field(0.1, gt=0.0, description="DENCLUE: Kernel bandwidth parameter.")
    denclue_eps: float = Field(1e-4, gt=0.0, description="DENCLUE: Convergence threshold.")
    
    # 2.3 OPTICS parameters
    optics_min_samples: int = Field(5, ge=1, description="OPTICS: Minimum samples parameter.")
    optics_max_eps: float = Field(1e10, description="OPTICS: Maximum distance between points. Uses a large value (1e10) instead of infinity for database compatibility.")
    optics_xi: float = Field(0.05, ge=0.0, le=1.0, description="OPTICS: Determines the minimum steepness on the reachability plot.")
    
    # 3. Outlier Detection Parameters
    # 3.1 Isolation Forest parameters
    if_contamination: float = Field(0.05, ge=0.001, le=0.5, description="Isolation Forest: Expected proportion of outliers.")
    if_n_estimators: int = Field(100, ge=10, le=500, description="Isolation Forest: Number of trees.")
    
    # 3.2 Local Outlier Factor parameters
    lof_n_neighbors: int = Field(20, ge=1, le=100, description="LOF: Number of neighbors to consider.")
    lof_contamination: float = Field(0.05, ge=0.001, le=0.5, description="LOF: Expected proportion of outliers.")
    
    # 3.3 One-Class SVM parameters
    ocsvm_nu: float = Field(0.05, ge=0.001, le=0.5, description="OCSVM: An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.")
    ocsvm_kernel: str = Field("rbf", description="OCSVM: Kernel type. Options: 'linear', 'poly', 'rbf', 'sigmoid'.")
    ocsvm_gamma: str = Field("scale", description="OCSVM: Kernel coefficient. Options: 'scale', 'auto' or float value.")
    
    # 4. Output and Visualization Parameters
    save_visualizations: bool = Field(True, description="Whether to generate and save visualizations.")
    include_pca_plot: bool = Field(True, description="Whether to include PCA visualization for high-dimensional data.")
    max_samples_for_visualization: int = Field(10000, ge=100, description="Maximum number of samples to use for visualization to avoid performance issues.")
    export_results: bool = Field(True, description="Whether to export results as downloadable files.")

class DetectOutliersResponse(BaseModel):
    message: str
    task_id: Optional[str] = None # task_id from background_task_status_dict, can be None if not set immediately
    outlier_run_id: int
    run_id: Optional[int] = None  # Added for frontend compatibility

class OutlierResultDetailSchema(BaseModel):
    original_index: int
    is_outlier: bool
    if_score: Optional[float] = Field(None, alias="outlier_score") # if_score is more specific, aliasing for compatibility
    final_cluster_label: Optional[int] = Field(None, alias="cluster_label") # final_cluster_label from orchestrator
    reconstruction_error: Optional[float] = None

    model_config = ConfigDict(populate_by_name=True) # For Pydantic v2


# --- New Schemas for Evaluation ---

class GroundTruthDataPoint(BaseModel):
    original_index: int
    true_is_outlier: bool # Or int (0 or 1) - bool is generally clearer

class EvaluateOutliersRequest(BaseModel):
    ground_truth_data: List[GroundTruthDataPoint]

class ConfusionMatrixSchema(BaseModel):
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

class EvaluationMetricsSchema(BaseModel):
    # Classification metrics
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    confusion_matrix: Optional[ConfusionMatrixSchema] = None
    
    # Regression metrics
    mae: Optional[float] = None  # Mean Absolute Error
    mse: Optional[float] = None  # Mean Squared Error
    rmse: Optional[float] = None  # Root Mean Squared Error
    r2_score: Optional[float] = None  # Coefficient of Determination
    
    # General
    evaluation_type: Optional[str] = None  # 'classification' or 'regression'
    message: Optional[str] = None  # For any info/warnings during evaluation
    
    # Source of the metrics
    source: Optional[str] = None  # 'original_dataset' or 'provided_ground_truth'
    accuracy: Optional[float] = None  # Overall accuracy

class EvaluateOutliersResponse(BaseModel):
    message: str
    run_id: int
    evaluation_metrics: Dict[str, Any]
    has_score_metrics: bool = False

# --- Modified OutlierDetectionRunSchema and TaskStatusResponse ---

class OutlierDetectionRunSchema(BaseModel):
    id: int
    dataset_id: int
    user_id: int
    task_id: Optional[str] = None
    status: str
    parameters_json: Optional[Dict[str, Any]] = Field(None, alias="parameters")
    
    latent_features_path: Optional[str] = None
    reconstruction_errors_path: Optional[str] = None
    autoencoder_model_path: Optional[str] = None
    encoder_model_path: Optional[str] = None
    scaler_path: Optional[str] = None # Path to the scaler for original numerical data
    cluster_labels_path: Optional[str] = None
    # ADDED: path for scaler used on latent features before clustering (if different)
    clustering_latent_features_scaler_path: Optional[str] = None 
    
    scatter_plot_pca_path: Optional[str] = None
    outlier_distribution_path: Optional[str] = None
    pca_plot_path: Optional[str] = None

    total_points_processed: Optional[int] = None
    num_numerical_features_used: Optional[int] = None
    num_clusters_found: Optional[int] = None
    num_noise_points: Optional[int] = None
    total_outliers_detected: Optional[int] = None
    
    outlier_results_json: Optional[List[OutlierResultDetailSchema]] = Field(None, alias="outlier_results")
    
    # ADDED: Field to store evaluation metrics if calculated
    evaluation_metrics_json: Optional[EvaluationMetricsSchema] = Field(None, alias="evaluation_metrics")

    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    # For Pydantic v2, use model_config. For Pydantic v1, use class Config.
    # Assuming you are using Pydantic v2 or a version that supports model_config
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
    
    # If using Pydantic v1:
    # class Config:
    #     from_attributes = True  # Replaces orm_mode = True
    #     populate_by_name = True


class TaskStatusResponse(BaseModel):
    task_id: Optional[str] = None # task_id can be None if run record doesn't have it yet
    status: str
    run_details: Optional[OutlierDetectionRunSchema] = None