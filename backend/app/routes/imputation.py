"""Routes for data imputation – mirrors style of other route modules."""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from ..schemas.imputation import (
    ImputationRunRequest,
    ImputationResponse,
    PerformanceChartRequest,
    ImputationTaskStatusResponse,
)
from ..services.imputation.service import run_imputation
from ..db.database import get_db
from ..db.models import User
from ..services.auth import get_current_user
from sqlalchemy import select
from ..db.models import Dataset
from typing import Optional
from uuid import UUID
from datetime import datetime
from celery.result import AsyncResult

from ..services import sessions as session_service

from ..celery_app import celery_app

router = APIRouter(
    prefix="/imputation",
    tags=["Data Imputation"],
    responses={
        404: {"description": "Not found"},
        403: {"description": "Not authorized"},
    },
)


@router.post("/run", response_model=ImputationResponse)
async def run_imputation_endpoint(
    req: ImputationRunRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    session_id: Optional[UUID] = Query(None, description="Optional session to record this run as a step"),
):
    # Fetch dataset and verify ownership
    result = await db.execute(
        select(Dataset).where(
            Dataset.id == req.dataset_id,
            Dataset.user_id == current_user.id
        )
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    dataset_path = dataset.file_path  # model stores file path

    # Optionally create a session step prior to dispatch
    run_params_dict = req.dict()
    session_step_id = None
    if session_id is not None:
        try:
            step_row = await session_service.add_step(
                db,
                current_user,
                session_id,
                tool="imputation",
                step="imputation",
                algorithm=req.strategy,
                params=run_params_dict,
            )
            session_step_id = step_row.id
        except PermissionError:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Session not found or access denied")
        except Exception as e:
            print(f"WARNING: Unable to create session step: {e}")

    if req.async_job:
        task = celery_app.send_task(
            "tasks.imputation.run",
            args=[req.dict(), dataset.id, current_user.id, str(session_step_id) if session_step_id else None],
        )
        # Link session step to the queued task and mark running
        if session_step_id is not None:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    step_id=session_step_id,
                    status="running",
                    run_ref_type="imputation",
                    run_ref_id=str(task.id),
                )
            except Exception as e:
                print(f"WARNING: Failed to update session step after task queued: {e}")
        return {
            "status": "queued",
            "message": "Job queued",
            "task_id": task.id,
            "session_step_id": str(session_step_id) if session_step_id else None,
        }

    # Synchronous execution path
    if session_step_id is not None:
        try:
            await session_service.update_step(
                db,
                current_user,
                step_id=session_step_id,
                status="running",
                run_ref_type="imputation",
            )
        except Exception as e:
            print(f"WARNING: Failed to mark session step running: {e}")

    try:
        result = run_imputation(dataset, req, current_user.id)
    except Exception as e:
        if session_step_id is not None:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    step_id=session_step_id,
                    status="failed",
                    error=str(e),
                    finished_at=datetime.utcnow(),
                )
            except Exception as upd_e:
                print(f"WARNING: Failed to update session step on error: {upd_e}")
        raise

    if session_step_id is not None:
        try:
            await session_service.update_step(
                db,
                current_user,
                step_id=session_step_id,
                status="success",
                finished_at=datetime.utcnow(),
            )
        except Exception as upd_e:
            print(f"WARNING: Failed to update session step on success: {upd_e}")

    # Ensure response includes session_step_id when available
    if isinstance(result, dict):
        result.setdefault("session_step_id", str(session_step_id) if session_step_id else None)
    return result


# -----------------------------------------------------------------------------
# Task status
# -----------------------------------------------------------------------------
@router.get("/task-status/{task_id}", response_model=ImputationTaskStatusResponse)
async def get_imputation_task_status(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return Celery task status and associated session step (if any)."""
    try:
        async_result = AsyncResult(task_id, app=celery_app)
        status_str = async_result.status
    except Exception as e:
        # If Celery backend is unreachable or task unknown
        status_str = f"unknown: {e}"

    # Try to attach session step by run_ref
    session_step = None
    try:
        session_step = await session_service.get_step_by_run_ref(db, current_user, "imputation", task_id)
    except PermissionError:
        # Hide details but still return task status
        session_step = None

    # Attach result only if ready and successful
    result_payload = None
    try:
        if hasattr(async_result, "successful") and async_result.successful():
            # Avoid blocking; .result is non-blocking when ready
            res = async_result.result
            if isinstance(res, dict):
                result_payload = res
    except Exception:
        result_payload = None

    return ImputationTaskStatusResponse(
        task_id=task_id,
        status=status_str,
        session_step=session_step,
        result=result_payload,
    )


# -----------------------------------------------------------------------------
# Performance chart generation
# -----------------------------------------------------------------------------
from io import BytesIO
import matplotlib

matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
from fastapi.responses import StreamingResponse


def _radar_factory(num_vars, frame="circle"):
    """Create a radar chart with `num_vars` axes."""
    from matplotlib.path import Path
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection

    # Compute angle of each axis in the plot (we divide the plot / number of
    # variables in full circle)
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = "radar"
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                line.set_clip_on(False)
            return lines

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

    register_projection(RadarAxes)
    return theta


def _generate_radar(history):
    metrics = ["runtime_seconds", "rmse", "mae"]
    labels = {"runtime_seconds": "Speed", "rmse": "RMSE", "mae": "MAE"}

    # compute max for each metric
    max_vals = {m: max(item["performance"].get(m, 0) for item in history) for m in metrics}

    theta = _radar_factory(len(metrics))
    fig, ax = plt.subplots(subplot_kw=dict(projection="radar"))

    for item in history:
        algo = item["strategy"].upper()
        perf = item["performance"]
        # Normalize: bigger = better => invert
        values = []
        for m in metrics:
            val = perf.get(m, 0)
            norm = (max_vals[m] - val) / max_vals[m] if max_vals[m] else 0
            values.append(norm)
        values += values[:1]  # close loop
        ax.plot(theta.tolist() + theta[:1].tolist(), values, label=algo, linewidth=2)
        ax.fill(theta.tolist() + theta[:1].tolist(), values, alpha=0.25)

    ax.set_varlabels([labels[m] for m in metrics])
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _generate_scatter(history):
    fig, ax = plt.subplots()
    for item in history:
        algo = item["strategy"].upper()
        perf = item["performance"]
        x = perf.get("runtime_seconds", 0)
        y = perf.get("rmse", 0)
        size = (perf.get("mae", 0) or 1) * 20  # scale bubble
        ax.scatter(x, y, s=size, label=algo, alpha=0.7)
        ax.annotate(algo, (x, y), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("RMSE")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    fig.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


@router.post("/chart", response_class=StreamingResponse)
async def performance_chart(
    req: PerformanceChartRequest,
    current_user: User = Depends(get_current_user),
):
    """Return radar or scatter chart PNG based on provided history."""
    history_dicts = [item.dict() for item in req.history]
    if len(history_dicts) < 2:
        raise HTTPException(status_code=400, detail="At least two runs required for chart")

    if req.chart_type == "radar":
        buf = _generate_radar(history_dicts)
    else:
        buf = _generate_scatter(history_dicts)

    return StreamingResponse(buf, media_type="image/png")
