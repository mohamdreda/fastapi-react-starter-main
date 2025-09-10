from typing import Any, Dict, List, Optional
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field

class TemplateStep(BaseModel):
    tool: Optional[str] = None
    step: str
    algorithm: str
    substep: Optional[str] = None
    params: Optional[Dict[str, Any]] = None

class WorkflowTemplateCreate(BaseModel):
    name: str
    description: Optional[str]
    steps: List[TemplateStep]

class WorkflowTemplateUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    steps: Optional[List[TemplateStep]] = None

class WorkflowRunCreate(BaseModel):
    dataset_id: int
    template_id: Optional[UUID]

class WorkflowStepRunRead(BaseModel):
    order: int
    step: str
    substep: Optional[str]
    algorithm: str
    params: Optional[Dict[str, Any]]
    status: str
    elapsed_ms: Optional[int]
    metrics: Optional[Dict[str, Any]]
    visuals: Optional[List[Dict[str, Any]]]
    error: Optional[str]

    class Config:
        orm_mode = True

class WorkflowRunRead(BaseModel):
    id: UUID
    dataset_id: int
    template_id: Optional[UUID]
    status: str
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    steps: List[WorkflowStepRunRead] = Field(default_factory=list)

    class Config:
        orm_mode = True

class WorkflowTemplateRead(BaseModel):
    id: UUID
    name: str
    version: int
    description: Optional[str]
    steps: List[TemplateStep]
    created_at: Optional[datetime]

    class Config:
        orm_mode = True
