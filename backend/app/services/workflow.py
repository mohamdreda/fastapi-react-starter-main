from typing import Any, Dict, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from uuid import UUID
from ..db.models.workflow import WorkflowTemplate, WorkflowRun, WorkflowStepRun
from ..db.models.workflow import Session as WorkflowSession, SessionStep as WorkflowSessionStep
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..db import models as core_models  # type: ignore
    User = core_models.User  # noqa: F401
from ..schemas.workflow import WorkflowTemplateCreate, WorkflowTemplateUpdate, WorkflowRunCreate
from sqlalchemy.exc import IntegrityError
from ..schemas.session import SaveWorkflowFromSessionRequest
from datetime import datetime
import os
import logging
logger = logging.getLogger(__name__)
from ..config import get_settings

# --- Template CRUD ---
async def _next_template_version(db: AsyncSession, owner_id: int, name: str) -> int:
    """Return next version number for a template name within the user's namespace."""
    result = await db.execute(
        select(WorkflowTemplate.version).where(
            WorkflowTemplate.owner_id == owner_id,
            WorkflowTemplate.name == name,
        )
    )
    versions = [v for v in result.scalars().all() if isinstance(v, int)]
    return (max(versions) + 1) if versions else 1

async def create_template(db: AsyncSession, user: "User", payload: WorkflowTemplateCreate) -> WorkflowTemplate:
    version = await _next_template_version(db, int(user.id), payload.name)
    tmpl = WorkflowTemplate(
        owner_id=user.id,
        name=payload.name,
        description=payload.description,
        steps=[step.dict() for step in payload.steps],
        version=version,
    )
    db.add(tmpl)
    await db.commit()
    await db.refresh(tmpl)
    logger.info("Created template name=%s version=%s owner=%s steps=%d", tmpl.name, tmpl.version, user.id, len(tmpl.steps or []))
    return tmpl

async def list_templates(db: AsyncSession, user: "User"):
    result = await db.execute(
        select(WorkflowTemplate).where(WorkflowTemplate.owner_id == user.id)
    )
    return result.scalars().all()

async def update_template(
    db: AsyncSession,
    user: "User",
    template_id: UUID,
    payload: WorkflowTemplateUpdate,
) -> WorkflowTemplate:
    result = await db.execute(
        select(WorkflowTemplate).where(WorkflowTemplate.id == template_id)
    )
    tmpl = result.scalars().first()
    if not tmpl or int(tmpl.owner_id) != int(user.id):
        raise PermissionError("Template not found or access denied")

    # Apply changes if provided
    if payload.name is not None and payload.name != tmpl.name:
        # Ensure uniqueness on (name, version, owner_id)
        conflict_q = await db.execute(
            select(WorkflowTemplate.id).where(
                WorkflowTemplate.owner_id == user.id,
                WorkflowTemplate.name == payload.name,
                WorkflowTemplate.version == tmpl.version,
            )
        )
        if conflict_q.scalars().first():
            raise ValueError("A template with the same name and version already exists. Save as new template instead.")
        tmpl.name = payload.name

    if payload.description is not None:
        tmpl.description = payload.description

    if payload.steps is not None:
        tmpl.steps = [s.dict() for s in payload.steps]

    try:
        await db.commit()
    except IntegrityError as ie:
        await db.rollback()
        raise ValueError(str(ie))
    await db.refresh(tmpl)
    logger.info("Updated template id=%s name=%s version=%s", tmpl.id, tmpl.name, tmpl.version)
    return tmpl

# --- Runs listing ---
async def list_runs(db: AsyncSession, user: "User", dataset_id: Optional[int] = None):
    stmt = (
        select(WorkflowRun)
        .options(selectinload(WorkflowRun.steps))
        .where(WorkflowRun.owner_id == user.id)
    )
    if dataset_id is not None:
        stmt = stmt.where(WorkflowRun.dataset_id == dataset_id)
    stmt = stmt.order_by(WorkflowRun.started_at.desc())
    result = await db.execute(stmt)
    return result.scalars().all()

# --- Save template from a captured session ---
async def save_template_from_session(
    db: AsyncSession,
    user: "User",
    session_id: UUID,
    payload: SaveWorkflowFromSessionRequest,
) -> WorkflowTemplate:
    # Load session with steps and ownership check
    session_res = await db.execute(
        select(WorkflowSession).options(selectinload(WorkflowSession.steps)).where(
            WorkflowSession.id == session_id
        )
    )
    session_obj = session_res.scalars().first()
    if not session_obj or int(session_obj.owner_id) != int(user.id):
        raise PermissionError("Session not found or access denied")

    # Select and order steps
    steps = session_obj.steps or []
    selected_ids = set((payload.selected_step_ids or []))
    filtered = [s for s in steps if not selected_ids or s.id in selected_ids]
    filtered.sort(key=lambda s: (s.order or 0))

    # Transform to template step dicts
    template_steps: List[Dict[str, Any]] = []
    for s in filtered:
        template_steps.append({
            "step": s.step,
            "substep": s.substep,
            "algorithm": s.algorithm or "",
            "params": s.params or {},
        })

    if not template_steps:
        raise ValueError("No steps selected from session to save as template")

    version = await _next_template_version(db, int(user.id), payload.name)
    tmpl = WorkflowTemplate(
        owner_id=user.id,
        name=payload.name,
        description=payload.description,
        steps=template_steps,
        version=version,
    )
    db.add(tmpl)
    await db.commit()
    await db.refresh(tmpl)
    logger.info(
        "Saved template from session id=%s name=%s version=%s steps=%d",
        session_id,
        tmpl.name,
        tmpl.version,
        len(template_steps),
    )
    return tmpl

# --- Run orchestration ---
async def queue_run(db: AsyncSession, user: "User", payload: WorkflowRunCreate) -> WorkflowRun:
    run = WorkflowRun(
        dataset_id=payload.dataset_id,
        template_id=payload.template_id,
        owner_id=user.id,
        status="queued",
        started_at=datetime.utcnow(),
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    # build Celery chain and dispatch using explicit template steps
    steps: List[Dict[str, Any]] = []
    if run.template_id:
        res = await db.execute(
            select(WorkflowTemplate.steps).where(WorkflowTemplate.id == run.template_id)
        )
        steps = res.scalar_one_or_none() or []
    settings = get_settings()
    sync_mode = getattr(settings, "WORKFLOWS_SYNC", False)
    logger.info(f"Workflow {run.id}: sync_mode={sync_mode}, steps={len(steps)}")
    if steps:
        if sync_mode:
            # Lazy import to avoid circulars and heavy deps at import time
            from ..tasks.workflow_tasks import dispatch_tool
            from ..db.database import SessionLocal
            from time import time as _time

            db_sync = SessionLocal()
            logger.debug("Workflow %s: opened sync DB session", run.id)
            try:
                run_row = db_sync.query(WorkflowRun).filter(WorkflowRun.id == run.id).first()
                if run_row is None:
                    raise RuntimeError(f"WorkflowRun not found for id={run.id}")
                # Mark running
                run_row.status = "running"
                db_sync.commit()
                logger.info("Workflow %s: marked running", run.id)

                total = len(steps)
                logger.info("Workflow %s: executing %d step(s) in sync mode", run.id, total)
                for idx, step_def in enumerate(steps):
                    step_row = WorkflowStepRun(
                        run_id=run.id,
                        order=idx,
                        step=(step_def or {}).get("step"),
                        substep=(step_def or {}).get("substep"),
                        algorithm=(step_def or {}).get("algorithm"),
                        params=(step_def or {}).get("params"),
                        status="running",
                    )
                    db_sync.add(step_row)
                    db_sync.commit()
                    db_sync.refresh(step_row)
                    logger.debug(
                        "Workflow %s: step %d/%d persisted id=%s step=%s algorithm=%s",
                        run.id,
                        idx + 1,
                        total,
                        step_row.id,
                        (step_def or {}).get("step"),
                        (step_def or {}).get("algorithm"),
                    )

                    t0 = _time()
                    try:
                        logger.debug("Workflow %s: dispatching tool for step %s", run.id, step_row.id)
                        result = dispatch_tool(db_sync, run_row, step_row)
                        step_row.metrics = result.get("metrics")
                        step_row.visuals = result.get("visuals")
                        step_row.status = "success"
                        step_row.elapsed_ms = int((_time() - t0) * 1000)
                        db_sync.commit()
                        logger.info(
                            "Workflow %s: step %s success in %d ms",
                            run.id,
                            step_row.id,
                            step_row.elapsed_ms or -1,
                        )
                    except Exception as exc:
                        logger.exception("Workflow %s: step %s failed: %s", run.id, step_row.id, exc)
                        try:
                            db_sync.rollback()
                        except Exception:
                            pass
                        step_row.status = "failed"
                        step_row.error = str(exc)
                        db_sync.commit()
                        run_row.status = "failed"
                        run_row.finished_at = datetime.utcnow()
                        db_sync.commit()
                        raise

                # All steps succeeded
                run_row.status = "success"
                run_row.finished_at = datetime.utcnow()
                db_sync.commit()
                logger.info("Workflow %s: all steps succeeded; finished_at=%s", run.id, run_row.finished_at)
            finally:
                db_sync.close()
            # Return the fresh run state with steps
            return await get_run(db, run.id)
        else:
            from ..tasks.workflow_tasks import build_celery_chain
            task_chain = build_celery_chain(str(run.id), steps)
            logger.info("Workflow %s: enqueued Celery chain with %d step(s)", run.id, len(steps))
            task_chain.apply_async()
    if not steps:
        logger.warning("Workflow %s: template has no steps", run.id)
    return run

async def get_run(db: AsyncSession, run_id: UUID) -> WorkflowRun | None:
    result = await db.execute(
        select(WorkflowRun)
        .options(selectinload(WorkflowRun.steps))
        .where(WorkflowRun.id == run_id)
    )
    return result.scalars().first()

async def get_run_steps(db: AsyncSession, run_id: UUID):
    result = await db.execute(
        select(WorkflowStepRun).where(WorkflowStepRun.run_id == run_id).order_by(WorkflowStepRun.order)
    )
    return result.scalars().all()
