from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from ..schemas.workflow import WorkflowTemplateCreate, WorkflowTemplateUpdate, WorkflowTemplateRead, WorkflowRunCreate, WorkflowRunRead, WorkflowStepRunRead
from ..schemas.session import SaveWorkflowFromSessionRequest
from ..services import workflow as workflow_service
from ..dependencies import get_db, get_current_user
from ..db.models import User
from ..utils.event_bus import workflow_ws_listener

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])

@router.post("/templates", response_model=WorkflowTemplateRead)
async def create_template(payload: WorkflowTemplateCreate,
                    db: AsyncSession = Depends(get_db),
                    current_user: User = Depends(get_current_user)):
    return await workflow_service.create_template(db, current_user, payload)

@router.get("/templates", response_model=list[WorkflowTemplateRead])
async def list_templates(db: AsyncSession = Depends(get_db),
                  current_user: User = Depends(get_current_user)):
    return await workflow_service.list_templates(db, current_user)

@router.put("/templates/{template_id}", response_model=WorkflowTemplateRead)
async def update_template(template_id: UUID,
                    payload: WorkflowTemplateUpdate,
                    db: AsyncSession = Depends(get_db),
                    current_user: User = Depends(get_current_user)):
    try:
        return await workflow_service.update_template(db, current_user, template_id, payload)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")
    except ValueError as e:
        msg = str(e)
        if msg == "Template not found":
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)

@router.post("/runs", response_model=WorkflowRunRead)
async def start_run(payload: WorkflowRunCreate,
              db: AsyncSession = Depends(get_db),
              current_user: User = Depends(get_current_user)):
    return await workflow_service.queue_run(db, current_user, payload)

@router.get("/runs", response_model=list[WorkflowRunRead])
async def list_runs(dataset_id: int | None = None,
              db: AsyncSession = Depends(get_db),
              current_user: User = Depends(get_current_user)):
    return await workflow_service.list_runs(db, current_user, dataset_id)

@router.get("/runs/{run_id}", response_model=WorkflowRunRead)
async def get_run(run_id: UUID,
            db: AsyncSession = Depends(get_db),
            current_user: User = Depends(get_current_user)):
    run = await workflow_service.get_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run

@router.get("/runs/{run_id}/steps", response_model=list[WorkflowStepRunRead])
async def get_run_steps(run_id: UUID,
                  db: AsyncSession = Depends(get_db),
                  current_user: User = Depends(get_current_user)):
    return await workflow_service.get_run_steps(db, run_id)

@router.post("/templates/from-session/{session_id}", response_model=WorkflowTemplateRead)
async def save_template_from_session(
    session_id: UUID,
    payload: SaveWorkflowFromSessionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return await workflow_service.save_template_from_session(db, current_user, session_id, payload)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Session not found or access denied")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.websocket("/ws/workflow/{run_id}")
async def workflow_ws(websocket: WebSocket, run_id: str):
    await websocket.accept()
    try:
        await workflow_ws_listener(websocket, run_id)
    except WebSocketDisconnect:
        pass
