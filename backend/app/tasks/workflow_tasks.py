from ..db.models.workflow import WorkflowRun, WorkflowStepRun, SessionStep
from ..db.database import SessionLocal
from ..celery_app import celery_app
from uuid import UUID
from time import time
from sqlalchemy.orm import Session
from typing import List, Dict, Any

import logging
from datetime import datetime, timedelta
import asyncio
import os
try:
    from celery.exceptions import SoftTimeLimitExceeded  # type: ignore
except Exception:  # Celery not installed or unavailable
    class SoftTimeLimitExceeded(Exception):
        pass

# Domain models and schemas
from app.db.models import Dataset  # for loading dataset path synchronously
from app.schemas.imputation import ImputationRunRequest
from app.services.imputation.service import run_imputation
from app.services.outlier_detection import OutlierDetectionOrchestrator

logger = logging.getLogger(__name__)

# Environment-configurable time limits (seconds)
STEP_SOFT_TIME_LIMIT = int(os.getenv("WORKFLOW_STEP_SOFT_TIME_LIMIT", "1800"))  # 30 minutes
STEP_HARD_TIME_LIMIT = int(os.getenv("WORKFLOW_STEP_HARD_TIME_LIMIT", str(STEP_SOFT_TIME_LIMIT + 60)))


def _normalize_algo_for_imputation(algo: str | None) -> str:
    if not algo:
        return "auto"
    a = algo.strip().lower()
    mapping = {
        "knnimputer": "knn",
        "knn": "knn",
        "simple": "simple",
        "mice": "mice",
        "missforest": "missforest",
        "lightgbm": "lightgbm",
        "dask_group": "dask_group",
        "auto": "auto",
    }
    return mapping.get(a, "auto")


def run_imputation_tool(db: Session, run: WorkflowRun, step_row: WorkflowStepRun) -> Dict[str, Any]:
    ds = db.query(Dataset).filter(Dataset.id == run.dataset_id).first()
    if ds is None:
        raise RuntimeError(f"Dataset not found for id={run.dataset_id}")

    params = step_row.params or {}
    output_name = params.get("output_name")
    if not output_name:
        # Default deterministic output name
        output_name = f"imputed_{run.dataset_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv"

    req = ImputationRunRequest(
        dataset_id=run.dataset_id,
        strategy=_normalize_algo_for_imputation(step_row.algorithm),
        params=params,
        output_name=output_name,
        async_job=False,
    )

    result = run_imputation(ds, req, user_id=run.owner_id)

    metrics = result.get("summary", {})
    visuals: List[Dict[str, Any]] = []
    if result.get("imputed_dataset_path"):
        visuals.append({
            "type": "file",
            "label": "imputed_dataset",
            "path": result["imputed_dataset_path"],
        })
    if result.get("validation_report_path"):
        visuals.append({
            "type": "report",
            "label": "validation_report",
            "path": result["validation_report_path"],
        })
    return {"metrics": metrics, "visuals": visuals}


def run_outlier_detection_tool(db: Session, run: WorkflowRun, step_row: WorkflowStepRun) -> Dict[str, Any]:
    ds = db.query(Dataset).filter(Dataset.id == run.dataset_id).first()
    if ds is None:
        raise RuntimeError(f"Dataset not found for id={run.dataset_id}")

    params = step_row.params or {}
    orchestrator = OutlierDetectionOrchestrator(
        dataset_id=run.dataset_id,
        user_id=run.owner_id,
        file_path=ds.file_path,
        run_parameters=params,
    )
    # Run async pipeline synchronously in Celery
    results_summary, artifact_paths = asyncio.run(orchestrator.run_pipeline())

    metrics = results_summary or {}
    visuals: List[Dict[str, Any]] = []
    for key, path in (artifact_paths or {}).items():
        if path:
            visuals.append({"type": "artifact", "label": key, "path": path})
    return {"metrics": metrics, "visuals": visuals}


def run_transformation_tool(db: Session, run: WorkflowRun, step_row: WorkflowStepRun) -> Dict[str, Any]:
    # Placeholder adapter. Implement full sync adapter if needed.
    logger.info(
        "Transformation step executed as placeholder. step=%s algo=%s",
        step_row.step,
        step_row.algorithm,
    )
    metrics = {"status": "skipped", "reason": "not_implemented", "params": step_row.params or {}}
    visuals: List[Dict[str, Any]] = []
    return {"metrics": metrics, "visuals": visuals}


def dispatch_tool(db: Session, run: WorkflowRun, step_row: WorkflowStepRun) -> Dict[str, Any]:
    step_type = (step_row.step or "").lower()
    if step_type == "imputation":
        return run_imputation_tool(db, run, step_row)
    if step_type in ("outlier_detection", "outliers"):
        return run_outlier_detection_tool(db, run, step_row)
    if step_type == "transformation":
        return run_transformation_tool(db, run, step_row)
    raise RuntimeError(f"Unsupported step type: {step_row.step}")

def build_celery_chain(run_id: str, steps: List[Dict[str, Any]]):
    from celery import chain
    tasks = []
    total = len(steps)
    for idx, step in enumerate(steps):
        # Use immutable signatures so previous step's return value is not passed to the next step
        tasks.append(execute_step.si(run_id, idx, step, total))
    return chain(*tasks) if tasks else chain()

@celery_app.task(
    bind=True,
    acks_late=True,
    soft_time_limit=STEP_SOFT_TIME_LIMIT,
    time_limit=STEP_HARD_TIME_LIMIT,
    ignore_result=True,
)
def execute_step(self, run_id: str, order: int, step_def: Dict[str, Any], total_steps: int):
    db = SessionLocal()
    step_row = None
    try:
        run = db.query(WorkflowRun).filter(WorkflowRun.id == UUID(run_id)).first()
        if run is None:
            raise Exception(f"WorkflowRun not found: {run_id}")
        # Mark run as running on first step
        if order == 0 and run.status != "running":
            run.status = "running"
            db.commit()
        logger.info("Executing workflow step: run_id=%s order=%s step=%s algo=%s", run_id, order, (step_def or {}).get("step"), (step_def or {}).get("algorithm"))
        step_row = WorkflowStepRun(
            run_id=run.id,
            order=order,
            step=(step_def or {}).get("step"),
            substep=(step_def or {}).get("substep"),
            algorithm=(step_def or {}).get("algorithm"),
            params=(step_def or {}).get("params"),
            status="running"
        )
        db.add(step_row)
        db.commit()
        db.refresh(step_row)
        t0 = time()
        # Dispatch to actual tool service via adapters
        result = dispatch_tool(db, run, step_row)
        step_row.metrics = result.get("metrics")
        step_row.visuals = result.get("visuals")
        step_row.status = "success"
        step_row.elapsed_ms = int((time() - t0) * 1000)
        db.commit()
        # If last step, close the run
        if order == max(0, int(total_steps) - 1):
            run.status = "success"
            run.finished_at = datetime.utcnow()
            db.commit()
        # Optionally: publish_ws_update(run_id) here
    except SoftTimeLimitExceeded as exc:
        # Handle Celery soft timeouts explicitly to avoid steps stuck in 'running'
        try:
            db.rollback()
        except Exception:
            pass
        if step_row is not None:
            step_row.status = "failed"
            step_row.error = f"Step timed out after {STEP_SOFT_TIME_LIMIT} seconds"
            db.commit()
        try:
            run = run if 'run' in locals() else db.query(WorkflowRun).filter(WorkflowRun.id == UUID(run_id)).first()
            if run is not None:
                run.status = "failed"
                run.finished_at = datetime.utcnow()
                db.commit()
        except Exception:
            pass
        logger.error("Workflow step timed out: run_id=%s order=%s", run_id, order)
        raise
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        if step_row is not None:
            step_row.status = "failed"
            step_row.error = str(exc)
            db.commit()
        # Mark run failed and end time
        try:
            run = run if 'run' in locals() else db.query(WorkflowRun).filter(WorkflowRun.id == UUID(run_id)).first()
            if run is not None:
                run.status = "failed"
                run.finished_at = datetime.utcnow()
                db.commit()
        except Exception as inner_exc:  # noqa: F841
            # Best-effort; do not mask original error
            pass
        raise
    finally:
        db.close()


# -----------------------------------------------------------------------------
# Watchdog: mark stale session steps as failed to prevent indefinite 'running'
# -----------------------------------------------------------------------------
@celery_app.task(ignore_result=True, name="app.tasks.workflow_tasks.mark_stale_session_steps")
def mark_stale_session_steps():
    db = SessionLocal()
    try:
        running_timeout = int(os.getenv("SESSION_STEP_RUNNING_TIMEOUT", "3600"))  # 1h default
        batch_limit = int(os.getenv("WATCHDOG_BATCH_LIMIT", "200"))
        cutoff = datetime.utcnow() - timedelta(seconds=running_timeout)

        stale_steps = (
            db.query(SessionStep)
            .filter(SessionStep.status == "running")
            .filter(SessionStep.finished_at.is_(None))
            .filter(SessionStep.started_at < cutoff)
            .order_by(SessionStep.started_at)
            .limit(batch_limit)
            .all()
        )

        updated = 0
        for step in stale_steps:
            step.status = "failed"
            step.error = "Watchdog: timeout"
            step.finished_at = datetime.utcnow()
            updated += 1
        if updated:
            db.commit()
        logger.info("Watchdog marked %s stale session steps as failed", updated)
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        logger.error("Watchdog error: %s", e)
    finally:
        db.close()
