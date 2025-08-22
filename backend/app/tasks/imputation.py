from celery import shared_task
from datetime import datetime
from typing import Optional
from uuid import UUID
import logging

from app.schemas.imputation import ImputationRunRequest
from app.services.imputation.service import run_imputation
from app.db.database import SessionLocal
from app.db.models import Dataset
from app.db.models.workflow import SessionStep as WorkflowSessionStep

logger = logging.getLogger(__name__)


@shared_task(name="tasks.imputation.run")
def run(req_dict: dict, dataset_id: int, user_id: int, session_step_id: Optional[str] = None):
    db = SessionLocal()
    try:
        # Optionally mark step running (route usually did this already)
        if session_step_id:
            try:
                step = db.query(WorkflowSessionStep).filter(WorkflowSessionStep.id == UUID(session_step_id)).first()
                if step is not None:
                    step.status = "running"
                    db.commit()
                else:
                    logger.warning("Session step not found when marking running: %s", session_step_id)
            except Exception as e:
                logger.warning("Failed to mark session step running in worker: %s", e)
                try:
                    db.rollback()
                except Exception:
                    pass

        dataset = db.query(Dataset).filter(Dataset.id == int(dataset_id)).first()
        if dataset is None:
            raise RuntimeError(f"Dataset not found: id={dataset_id}")

        req = ImputationRunRequest(**req_dict)
        result = run_imputation(dataset, req, user_id)

        if session_step_id:
            try:
                step = db.query(WorkflowSessionStep).filter(WorkflowSessionStep.id == UUID(session_step_id)).first()
                if step is not None:
                    step.status = "success"
                    step.finished_at = datetime.utcnow()
                    db.commit()
                else:
                    logger.warning("Session step not found when marking success: %s", session_step_id)
            except Exception as e:
                logger.warning("Failed to mark session step success in worker: %s", e)
                try:
                    db.rollback()
                except Exception:
                    pass

        return result
    except Exception as e:
        if session_step_id:
            try:
                step = db.query(WorkflowSessionStep).filter(WorkflowSessionStep.id == UUID(session_step_id)).first()
                if step is not None:
                    step.status = "failed"
                    step.error = str(e)
                    step.finished_at = datetime.utcnow()
                    db.commit()
                else:
                    logger.warning("Session step not found when marking failed: %s", session_step_id)
            except Exception as inner:
                logger.warning("Failed to mark session step failed in worker: %s", inner)
                try:
                    db.rollback()
                except Exception:
                    pass
        raise
    finally:
        db.close()
