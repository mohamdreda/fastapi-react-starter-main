from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path
from datetime import datetime
import shutil

from ..db.database import get_db
from ..db.models import User
from ..services.auth import get_current_user
from ..utils.logger import setup_logger
from ..config import get_settings

settings = get_settings()
logger = setup_logger(__name__)
router = APIRouter(prefix="/api/v1/artifacts", tags=["Artifacts"])

ARTIFACT_DIR = Path(settings.UPLOAD_DIR) / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def upload_artifact(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),  # placeholder, not used yet
    current_user: User = Depends(get_current_user)
):
    """Upload an arbitrary file (e.g., candidate pairs CSV/JSON) and return its storage path.

    File is placed under <UPLOAD_DIR>/artifacts/<user_id>/ with a timestamped filename.
    """
    try:
        # Ensure user subdir
        user_dir = ARTIFACT_DIR / str(current_user.id)
        user_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        dest_path = user_dir / safe_filename

        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        web_path = f"/{dest_path.as_posix()}"  # leading slash so frontend can call directly

        return {
            "status": "success",
            "artifact_path": web_path,
            "filename": file.filename,
        }
    except Exception as e:
        logger.error(f"Artifact upload error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Upload failed")
    finally:
        await file.close()
