from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, validator
from .session import SessionStepRead


class ImputationRunRequest(BaseModel):
    dataset_id: int
    strategy: Literal[
        "auto",
        "simple",
        "knn",
        "mice",
        "missforest",
        "lightgbm",
        "dask_group",
    ] = "auto"
    params: Dict[str, Any] = Field(default_factory=dict)
    output_name: str
    async_job: bool = False

    @validator("output_name")
    def _must_have_extension(cls, v):  # noqa: N805
        if "." not in v:
            raise ValueError("output_name must include extension, e.g. sales_clean.csv")
        return v


class ImputationResponse(BaseModel):
    status: Literal["success", "error", "queued"]
    message: str
    summary: Optional[Dict[str, Any]] = None
    imputed_dataset_path: Optional[str] = None
    validation_report_path: Optional[str] = None
    preview: Optional[List[Dict[str, Any]]] = None
    warnings: Optional[List[str]] = None

    task_id: Optional[str] = None
    session_step_id: Optional[str] = None


class PerformanceItem(BaseModel):
    strategy: str
    performance: Dict[str, float]


class PerformanceChartRequest(BaseModel):
    chart_type: Literal["radar", "scatter"]
    history: List[PerformanceItem]


class ImputationTaskStatusResponse(BaseModel):
    task_id: Optional[str] = None
    status: str
    session_step: Optional[SessionStepRead] = None
    result: Optional[Dict[str, Any]] = None
