from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from typing import List

from ..dependencies import get_db, get_current_user
from ..db.models import User
from ..schemas.session import (
    SessionCreate,
    SessionRead,
    SessionStepCreate,
    SessionStepRead,
)
from ..services import sessions as session_service

router = APIRouter(prefix="/api/v1/sessions", tags=["Sessions"])


@router.post("/", response_model=SessionRead)
async def create_session(
    payload: SessionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sess = await session_service.create_session(db, current_user, payload.title, payload.description)
    # Reload with steps eagerly loaded to avoid async lazy load during serialization
    loaded = await session_service.get_session_with_steps(db, current_user, sess.id)
    return loaded


@router.post("/open", response_model=SessionRead)
async def open_session(
    payload: SessionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Alias for creating a session. Useful for explicit 'open' semantics from the frontend."""
    sess = await session_service.create_session(db, current_user, payload.title, payload.description)
    loaded = await session_service.get_session_with_steps(db, current_user, sess.id)
    return loaded


@router.get("/", response_model=List[SessionRead])
async def list_sessions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await session_service.list_sessions(db, current_user)


@router.get("/{session_id}", response_model=SessionRead)
async def get_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sess = await session_service.get_session_with_steps(db, current_user, session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found or access denied")
    return sess


@router.get("/{session_id}/steps", response_model=List[SessionStepRead])
async def get_session_steps(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return only the steps of a session for efficient timeline polling."""
    sess = await session_service.get_session_with_steps(db, current_user, session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found or access denied")
    return list(sess.steps or [])


@router.post("/{session_id}/close", response_model=SessionRead)
async def close_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        _ = await session_service.close_session(db, current_user, session_id)
        loaded = await session_service.get_session_with_steps(db, current_user, session_id)
        return loaded
    except PermissionError:
        raise HTTPException(status_code=403, detail="Session not found or access denied")


@router.post("/{session_id}/steps", response_model=SessionStepRead)
async def add_step(
    session_id: UUID,
    payload: SessionStepCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        step = await session_service.add_step(
            db,
            current_user,
            session_id,
            tool=payload.tool,
            step=payload.step,
            substep=payload.substep,
            algorithm=payload.algorithm,
            params=payload.params,
        )
        return step
    except PermissionError:
        raise HTTPException(status_code=403, detail="Session not found or access denied")


# -----------------------------------------------------------------------------
# Session Step status endpoints for polling
# -----------------------------------------------------------------------------
@router.get("/steps/{step_id}", response_model=SessionStepRead)
async def get_session_step(
    step_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        step = await session_service.get_step_by_id(db, current_user, step_id)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied for fetching session step")
    if not step:
        raise HTTPException(status_code=404, detail="Session step not found")
    return step


@router.get("/steps/by-run-ref", response_model=SessionStepRead)
async def get_session_step_by_run_ref(
    run_ref_type: str = Query(..., description="Reference type, e.g. 'imputation'"),
    run_ref_id: str = Query(..., description="Reference id, e.g. Celery task id"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        step = await session_service.get_step_by_run_ref(db, current_user, run_ref_type, run_ref_id)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied for fetching session step by run_ref")
    if not step:
        raise HTTPException(status_code=404, detail="Session step not found for given run reference")
    return step
