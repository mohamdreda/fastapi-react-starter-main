from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

class CategoricalEncodingMethod(str, Enum):
    ONE_HOT = "one_hot"
    LABEL = "label"

class FeatureScalingMethod(str, Enum):
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"

class CategoricalEncodingConfig(BaseModel):
    method: CategoricalEncodingMethod
    columns: List[str]
    drop: Optional[str] = None
    mapping: Optional[Dict[str, Any]] = None

class FeatureScalingConfig(BaseModel):
    method: FeatureScalingMethod
    columns: List[str]
    with_mean: Optional[bool] = True
    with_std: Optional[bool] = True
    quantile_range: Optional[tuple[float, float]] = (25.0, 75.0)

class TransformationConfig(BaseModel):
    categorical_encoding: Optional[Dict[str, Any]] = None
    feature_scaling: Optional[Dict[str, Any]] = None

class TransformationRequest(BaseModel):
    dataset_id: Optional[int] = None
    file: Optional[str] = None
    config: TransformationConfig

class TransformationResponse(BaseModel):
    status: str
    message: str
    original_dataset_id: Optional[int] = None
    transformed_dataset_id: Optional[int] = None
    download_url: Optional[str] = None
    transformation_id: Optional[int] = None
