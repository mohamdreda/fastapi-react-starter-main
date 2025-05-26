from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path
import os
import shutil
from datetime import datetime
from ..db.database import get_db
from ..db.models import User
from ..services.auth import get_current_user
from ..services.upload import process_uploaded_file
from ..utils.logger import setup_logger
from ..config import get_settings

logger = setup_logger(__name__)
settings = get_settings()
router = APIRouter()

# Use absolute path for uploads
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/")
async def upload_file(
    file: UploadFile = File(...),
    file_type: str = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload and process a new dataset file"""
    try:
        # Validate file type
        if not file_type:
            file_type = file.filename.split('.')[-1].lower()
            
        if file_type not in settings.ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format. Allowed formats: {', '.join(settings.ALLOWED_FILE_TYPES)}"
            )

        # Check file size
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Reset position
        
        if size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:  # Convert MB to bytes
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size allowed is {settings.MAX_FILE_SIZE_MB}MB"
            )

        # Create user-specific directory
        user_dir = UPLOAD_DIR / str(current_user.id)
        user_dir.mkdir(exist_ok=True)
        
        # Save file with timestamp to avoid duplicates
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = user_dir / safe_filename
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process file and create dataset entry
        dataset_id = await process_uploaded_file(db, str(file_path), current_user.id, file_type)
        
        return {
            "status": "success",
            "message": "File uploaded successfully",
            "dataset_id": dataset_id,
            "filename": file.filename,
            "file_type": file_type,
            "file_path": str(file_path)
        }
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    finally:
        await file.close()
