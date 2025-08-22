from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class FeatureSetCreate(BaseModel):
    name: str = Field(..., description="User-defined name for the feature set")
    dataset_id: int = Field(..., description="ID of the dataset this feature set is derived from")
    feature_type: str = Field(..., description="Type of feature engineering (autoencoder, pca, isomap)")
    path: str = Field(..., description="Path to the saved feature set CSV file")
    description: Optional[str] = Field(None, description="Optional description of the feature set")

class FeatureSetUpdate(BaseModel):
    name: Optional[str] = Field(None, description="New name for the feature set")
    description: Optional[str] = Field(None, description="Description of the feature set")

class FeatureSetOut(BaseModel):
    id: int
    user_id: int
    dataset_id: int
    name: str
    path: str
    feature_type: str
    description: Optional[str]
    created_at: datetime

    class Config:
        orm_mode = True
