from typing import Optional, List, Any, Dict
from uuid import UUID
from datetime import datetime
import logging

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..db.models.workflow import Session as WorkflowSession, SessionStep as WorkflowSessionStep

if False:  # type checking only
    from ..db import models as core_models  # noqa: F401
    User = core_models.User  # noqa: F401

logger = logging.getLogger(__name__)


async def _get_session_owned(db: AsyncSession, user_id: int, session_id: UUID) -> WorkflowSession | None:
    res = await db.execute(
        select(WorkflowSession)
        .options(selectinload(WorkflowSession.steps))
        .where(WorkflowSession.id == session_id)
    )
    session = res.scalars().first()
    if not session or int(session.owner_id) != int(user_id):
        return None
    return session


async def create_session(db: AsyncSession, user: "User", title: Optional[str] = None, description: Optional[str] = None) -> WorkflowSession:
    sess = WorkflowSession(owner_id=user.id, title=title, description=description)
    db.add(sess)
    await db.commit()
    await db.refresh(sess)
    logger.info("Session created id=%s owner=%s title=%s", sess.id, user.id, title)
    return sess


async def list_sessions(db: AsyncSession, user: "User") -> List[WorkflowSession]:
    res = await db.execute(
        select(WorkflowSession)
        .options(selectinload(WorkflowSession.steps))
        .where(WorkflowSession.owner_id == user.id)
        .order_by(WorkflowSession.created_at.desc())
    )
    return res.scalars().all()


async def get_session_with_steps(db: AsyncSession, user: "User", session_id: UUID) -> WorkflowSession | None:
    return await _get_session_owned(db, int(user.id), session_id)


async def close_session(db: AsyncSession, user: "User", session_id: UUID) -> WorkflowSession:
    sess = await _get_session_owned(db, int(user.id), session_id)
    if not sess:
        raise PermissionError("Session not found or access denied")
    sess.closed_at = datetime.utcnow()
    await db.commit()
    await db.refresh(sess)
    logger.info("Session closed id=%s owner=%s", sess.id, user.id)
    return sess


async def _next_order(db: AsyncSession, session_id: UUID) -> int:
    res = await db.execute(
        select(func.max(WorkflowSessionStep.order)).where(WorkflowSessionStep.session_id == session_id)
    )
    max_ord = res.scalar_one_or_none()
    if max_ord is None:
        return 0
    try:
        return int(max_ord) + 1
    except Exception:
        return 0


async def add_step(
    db: AsyncSession,
    user: "User",
    session_id: UUID,
    tool: str,
    step: str,
    substep: Optional[str] = None,
    algorithm: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> WorkflowSessionStep:
    sess = await _get_session_owned(db, int(user.id), session_id)
    if not sess:
        raise PermissionError("Session not found or access denied")

    order = await _next_order(db, session_id)
    sstep = WorkflowSessionStep(
        session_id=session_id,
        order=order,
        tool=tool,
        step=step,
        substep=substep,
        algorithm=algorithm,
        params=params or {},
        status="queued",
        started_at=datetime.utcnow(),
    )
    db.add(sstep)
    await db.commit()
    await db.refresh(sstep)
    logger.info("Session step added session=%s step_id=%s order=%s tool=%s step=%s", session_id, sstep.id, order, tool, step)
    return sstep


async def update_step(
    db: AsyncSession,
    user: Optional["User"],
    step_id: UUID,
    *,
    status: Optional[str] = None,
    error: Optional[str] = None,
    finished_at: Optional[datetime] = None,
    run_ref_type: Optional[str] = None,
    run_ref_id: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> WorkflowSessionStep | None:
    # Load step and session for ownership check (if user provided)
    res = await db.execute(
        select(WorkflowSessionStep)
        .options(selectinload(WorkflowSessionStep.session))
        .where(WorkflowSessionStep.id == step_id)
    )
    step_row = res.scalars().first()
    if not step_row:
        return None
    if user is not None and int(step_row.session.owner_id) != int(user.id):
        raise PermissionError("Access denied for updating session step")

    if status is not None:
        step_row.status = status
    if error is not None:
        step_row.error = error
    if finished_at is not None:
        step_row.finished_at = finished_at
    if run_ref_type is not None:
        step_row.run_ref_type = run_ref_type
    if run_ref_id is not None:
        step_row.run_ref_id = run_ref_id
    if params is not None:
        # Merge params shallowly
        base = step_row.params or {}
        try:
            base.update(params)
        except Exception:
            base = params
        step_row.params = base

    await db.commit()
    await db.refresh(step_row)
    logger.debug("Session step updated id=%s status=%s", step_row.id, step_row.status)
    return step_row


async def get_step_by_id(
    db: AsyncSession,
    user: "User",
    step_id: UUID,
) -> WorkflowSessionStep | None:
    res = await db.execute(
        select(WorkflowSessionStep)
        .options(selectinload(WorkflowSessionStep.session))
        .where(WorkflowSessionStep.id == step_id)
    )
    step_row = res.scalars().first()
    if not step_row:
        return None
    if int(step_row.session.owner_id) != int(user.id):
        raise PermissionError("Access denied for fetching session step")
    return step_row


async def get_step_by_run_ref(
    db: AsyncSession,
    user: "User",
    run_ref_type: str,
    run_ref_id: str,
) -> WorkflowSessionStep | None:
    res = await db.execute(
        select(WorkflowSessionStep)
        .options(selectinload(WorkflowSessionStep.session))
        .where(
            WorkflowSessionStep.run_ref_type == run_ref_type,
            WorkflowSessionStep.run_ref_id == run_ref_id,
        )
        .order_by(WorkflowSessionStep.started_at.desc())
    )
    step_row = res.scalars().first()
    if not step_row:
        return None
    if int(step_row.session.owner_id) != int(user.id):
        raise PermissionError("Access denied for fetching session step by run_ref")
    return step_row
