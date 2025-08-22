"""
Schemas for the modular deduplication pipeline.

This module defines the request and response models for each step of the pipeline:
1. Preprocessing
2. Blocking
3. Similarity Calculation
4. Classification
5. Clustering
6. Resolution
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union

# Common models

class PipelineStepResponse(BaseModel):
    """Base response model for all pipeline steps."""
    status: str
    message: str
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    session_step_id: Optional[str] = None

# Preprocessing models

class PreprocessingRequest(BaseModel):
    """Request model for the preprocessing step."""
    dataset_id: int
    text_columns: List[str] = Field(default_factory=list)
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    output_name: Optional[str] = None

class PreprocessingResponse(PipelineStepResponse):
    """Response model for the preprocessing step."""
    preprocessed_data_path: Optional[str] = None

# Blocking models

class BlockingRequest(BaseModel):
    """Request model for the blocking step."""
    dataset_id: int
    method: str = "minhash_lsh"  # 'minhash_lsh' or 'simhash'
    key_fields: List[str]
    params: Dict[str, Any] = Field(default_factory=dict)
    output_name: Optional[str] = None

class BlockingResponse(PipelineStepResponse):
    """Response model for the blocking step."""
    candidate_pairs_json_path: Optional[str] = None
    candidate_pairs_csv_path: Optional[str] = None
    preview: Optional[List[Dict[str, Any]]] = None

# Similarity models

class FieldConfig(BaseModel):
    """Configuration for a field in similarity calculation."""
    type: str  # 'text', 'numeric', 'categorical'
    method: Optional[str] = None
    weight: float = 1.0

class SimilarityRequest(BaseModel):
    """Request model for the similarity calculation step."""
    dataset_id: int
    candidate_pairs_path: str
    field_configs: Dict[str, Dict[str, Any]]
    threshold: float = 0.7

class SimilarityResponse(PipelineStepResponse):
    """Response model for the similarity calculation step."""
    similarity_results_path: Optional[str] = None
    preview: Optional[List[Dict[str, Any]]] = None

# Classification models

class ClassificationRequest(BaseModel):
    """Request model for the classification step."""
    dataset_id: int
    similarity_results_path: str
    method: str = "random_forest"  # 'random_forest', 'xgboost', 'siamese_network'
    params: Dict[str, Any] = Field(default_factory=dict)

class ClassificationResponse(PipelineStepResponse):
    """Response model for the classification step."""
    classification_results_path: Optional[str] = None
    preview: Optional[List[Dict[str, Any]]] = None

# Clustering models

class ClusteringRequest(BaseModel):
    """Request model for the clustering step."""
    dataset_id: int
    classification_results_path: str
    method: str = "graph_connected_components"  # 'graph_connected_components', 'graph_community_detection', 'dbscan', 'optics'
    params: Dict[str, Any] = Field(default_factory=dict)

class ClusteringResponse(PipelineStepResponse):
    """Response model for the clustering step."""
    clustering_results_path: Optional[str] = None
    visualization_path: Optional[str] = None
    preview: Optional[List[Dict[str, Any]]] = None

# Resolution models

class ResolutionRequest(BaseModel):
    """Request model for the resolution step."""
    dataset_id: int
    clustering_results_path: str
    method: str = "keep_first"  # 'keep_first', 'keep_most_complete', 'merge', 'manual'
    params: Dict[str, Any] = Field(default_factory=dict)

class ResolutionResponse(PipelineStepResponse):
    """Response model for the resolution step."""
    resolved_dataset_path: Optional[str] = None
    records_kept: Optional[int] = None
    records_removed: Optional[int] = None
    manual_review_path: Optional[str] = None
    manual_review_data: Optional[Dict[str, Any]] = None

# Manual resolution models

class ManualResolutionRequest(BaseModel):
    """Request model for manual resolution."""
    dataset_id: int
    manual_review_path: str
    resolution_decisions: Dict[int, List[int]]  # Mapping of cluster_id to list of record_ids to keep

class ManualResolutionResponse(PipelineStepResponse):
    """Response model for manual resolution."""
    resolved_dataset_path: Optional[str] = None
    records_kept: Optional[int] = None
    records_removed: Optional[int] = None

# Legacy models (for backward compatibility)

class DeduplicationRequest(BaseModel):
    """Legacy request model for the old deduplication endpoint."""
    dataset_id: int
    algorithm: str
    params: Dict[str, Any]
    remove_duplicates: bool = False
    output_filename: Optional[str] = None

class DeduplicationResponse(BaseModel):
    """Legacy response model for the old deduplication endpoint."""
    status: str
    message: str
    num_duplicates: Optional[int] = None
    result_preview: Optional[List[Dict[str, Any]]] = None
    duplicates_removed: Optional[bool] = None
    cleaned_dataset_path: Optional[str] = None

# Legacy algorithm integration in pipeline

class LegacyDeduplicationRequest(BaseModel):
    """Request model for running legacy deduplication algorithms in the pipeline."""
    dataset_id: int
    algorithm: str  # 'fuzzy' or 'deep_er'
    params: Dict[str, Any] = Field(default_factory=dict)

class LegacyDeduplicationResponse(PipelineStepResponse):
    """Response model for legacy deduplication algorithms in the pipeline."""
    results_path: Optional[str] = None
    num_duplicates: Optional[int] = None
    num_clusters: Optional[int] = None
    preview: Optional[List[Dict[str, Any]]] = None
