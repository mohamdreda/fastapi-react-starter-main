from typing import Any, Dict, List, Optional
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class SessionCreate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None

class SessionStepCreate(BaseModel):
    tool: str
    step: str
    substep: Optional[str] = None
    algorithm: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class SessionStepRead(BaseModel):
    id: UUID
    order: int
    tool: str
    step: str
    substep: Optional[str] = None
    algorithm: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    status: str
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    run_ref_type: Optional[str] = None
    run_ref_id: Optional[str] = None

    # Pydantic v2: allow serialization from ORM objects
    model_config = ConfigDict(from_attributes=True)


class SessionRead(BaseModel):
    id: UUID
    title: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    steps: List[SessionStepRead] = Field(default_factory=list)

    # Pydantic v2: allow serialization from ORM objects
    model_config = ConfigDict(from_attributes=True)


class SaveWorkflowFromSessionRequest(BaseModel):
    name: str
    description: Optional[str] = None
    selected_step_ids: Optional[List[UUID]] = None  # if None, include all; else include only these
