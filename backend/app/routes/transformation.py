"""
Data Transformation Routes

This module contains the API routes for data transformation operations.
"""
import os
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import select as sync_select
import pandas as pd

# Local imports
from app.db.database import get_db
from app.db.models import User, Dataset, TransformationRun
from app.services.data_transformation.transformation_service import DataTransformationService
from app.services.auth import get_current_user
from app.schemas.transformation import (
    TransformationConfig,
    TransformationRequest,
    TransformationResponse,
    CategoricalEncodingMethod,
    FeatureScalingMethod
)
from app.utils.file_utils import save_dataframe, get_file_extension
from app.config.config import get_settings
settings = get_settings()

# Sessions service for optional workflow capture
from app.services import sessions as session_service

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/dataset-columns/{dataset_id}", response_model=Dict[str, List[str]])
async def get_dataset_columns(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get columns from a dataset file.
    
    Args:
        dataset_id: ID of the dataset
        current_user: Current authenticated user
        db: Async database session
        
    Returns:
        Dictionary with columns list
    """
    try:
        # Get the dataset
        query = select(Dataset).where(
            Dataset.id == dataset_id,
            Dataset.user_id == current_user.id
        )
        result = await db.execute(query)
        dataset = result.scalar_one_or_none()
        
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found"
            )
        
        # Read the dataset file
        file_path = dataset.file_path
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset file not found"
            )
        
        # Determine file type and read accordingly
        file_ext = get_file_extension(file_path).lower()
        
        if file_ext == 'csv':
            df = pd.read_csv(file_path, nrows=5)  # Read just a few rows for efficiency
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(file_path, nrows=5)
        elif file_ext == 'json':
            df = pd.read_json(file_path)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format: {file_ext}"
            )
        
        # Get column names and infer numeric columns
        all_columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        return {
            "columns": all_columns,
            "numeric_columns": numeric_columns
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset columns: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/transform", response_model=TransformationResponse)
async def transform_dataset(
    request: TransformationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    session_id: Optional[uuid.UUID] = Query(None, description="Optional session to record this run as a step"),
):
    """
    Apply transformations to a dataset.
    
    Args:
        request: Transformation request containing dataset_id and config
        current_user: Current authenticated user
        db: Async database session
        
    Returns:
        Transformation result with download URL
    """
    if not request.dataset_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset ID is required"
        )
    
    try:
        # Get the dataset using async session
        result = await db.execute(
            select(Dataset).filter(
                Dataset.id == request.dataset_id,
                Dataset.user_id == current_user.id
            )
        )
        dataset = result.scalars().first()

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found or access denied"
            )
        # Optionally create and mark a session step as running
        session_step_id = None
        if session_id is not None:
            try:
                step_row = await session_service.add_step(
                    db,
                    current_user,
                    session_id,
                    tool="transformation",
                    step="transform",
                    algorithm="pipeline",
                    params=request.dict(),
                )
                session_step_id = step_row.id
                await session_service.update_step(
                    db,
                    current_user,
                    step_id=session_step_id,
                    status="running",
                    run_ref_type="transformation",
                )
            except PermissionError:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Session not found or access denied")
            except Exception as e:
                # Non-fatal for the transformation itself
                logger.warning(f"Unable to create/update session step: {e}")

        # Initialize the transformation service
        transformation_service = DataTransformationService(
            dataset_id=request.dataset_id,
            user_id=current_user.id,
            config=request.config.dict(),
            db=db
        )

        # Apply transformations
        transformed_df = await transformation_service.transform()

        # Generate a unique filename for the transformed data
        original_filename = os.path.basename(dataset.file_path)
        filename, ext = os.path.splitext(original_filename)
        new_filename = f"{filename}_transformed_{int(datetime.now().timestamp())}{ext}"

        # Save the transformed data
        output_dir = os.path.join(settings.UPLOAD_DIR, str(current_user.id), "transformed")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the dataframe with the correct parameters
        output_path, saved_filename = save_dataframe(
            df=transformed_df, 
            output_dir=output_dir,
            filename=new_filename,
            file_type=get_file_extension(new_filename)
        )

        # Create a new dataset record for the transformed data
        transformed_dataset = Dataset(
            user_id=current_user.id,
            filename=saved_filename,
            file_path=output_path,
            file_type=get_file_extension(saved_filename)
        )
        db.add(transformed_dataset)
        await db.commit()
        await db.refresh(transformed_dataset)

        # Get the original dataset to get its filename and path
        original_dataset = await db.get(Dataset, request.dataset_id)
        
        # Record the transformation run
        transformation_run = TransformationRun(
            user_id=current_user.id,
            dataset_id=request.dataset_id,  # This is the original dataset ID
            original_filename=original_dataset.filename,
            original_file_path=original_dataset.file_path,
            transformed_filename=transformed_dataset.filename,
            transformed_file_path=transformed_dataset.file_path,
            transformation_config=request.config.dict(),
            status="completed",
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )
        db.add(transformation_run)
        await db.commit()
        await db.refresh(transformation_run)

        # Update session step success if created
        if session_id is not None and session_step_id is not None:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    step_id=session_step_id,
                    status="success",
                    finished_at=datetime.utcnow(),
                    run_ref_id=str(transformation_run.id),
                )
            except Exception as e:
                logger.warning(f"Failed to update session step to success: {e}")

        return TransformationResponse(
            status="success",
            message="Dataset transformed successfully",
            original_dataset_id=request.dataset_id,
            transformed_dataset_id=transformed_dataset.id,
            download_url=f"/api/v1/transformation/download/{transformation_run.id}",
            transformation_id=transformation_run.id
        )

    except HTTPException as http_exc:
        try:
            # Mark session step as failed if any
            if 'session_step_id' in locals() and session_step_id is not None:
                await session_service.update_step(
                    db,
                    current_user,
                    step_id=session_step_id,
                    status="failed",
                    error=str(http_exc.detail) if hasattr(http_exc, 'detail') else str(http_exc),
                    finished_at=datetime.utcnow(),
                )
        except Exception:
            pass
        await db.rollback()
        raise
    except Exception as e:
        try:
            if 'session_step_id' in locals() and session_step_id is not None:
                await session_service.update_step(
                    db,
                    current_user,
                    step_id=session_step_id,
                    status="failed",
                    error=str(e),
                    finished_at=datetime.utcnow(),
                )
        except Exception:
            pass
        await db.rollback()
        logger.error(f"Error in transform_dataset: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error transforming dataset: {str(e)}"
        )

@router.post("/transform/upload", response_model=TransformationResponse)
async def transform_uploaded_file(
    file: UploadFile = File(...),
    config: str = Form(...),  # JSON string of the config
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    session_id: Optional[uuid.UUID] = Query(None, description="Optional session to record this run as a step"),
):
    """
    Upload a file and apply transformations.
    
    Args:
        file: The file to transform
        config: JSON string containing transformation configuration
        current_user: Current authenticated user
        db: Async database session
        
    Returns:
        Transformation result with download URL
    """
    try:
        # Parse the config
        try:
            config_data = json.loads(config)
            transformation_config = TransformationConfig(**config_data)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid transformation config: {str(e)}"
            )
        
        # Validate file type
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ['.csv', '.parquet', '.feather', '.json']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file type. Supported types: .csv, .parquet, .feather, .json"
            )
        
        # Create upload directory if it doesn't exist
        upload_dir = os.path.join(settings.UPLOAD_DIR, str(current_user.id))
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the uploaded file
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(upload_dir, unique_filename)
        
        try:
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Create a dataset record
            dataset = Dataset(
                user_id=current_user.id,
                filename=file.filename,
                file_path=file_path,
                file_size=os.path.getsize(file_path),
                file_type=file.content_type or f"application/{file_extension[1:]}",
                is_processed=False
            )
            db.add(dataset)
            await db.commit()
            await db.refresh(dataset)

            # Optionally create and mark a session step as running
            session_step_id = None
            if session_id is not None:
                try:
                    step_row = await session_service.add_step(
                        db,
                        current_user,
                        session_id,
                        tool="transformation",
                        step="transform_upload",
                        algorithm="pipeline",
                        params={"config": transformation_config.dict(), "filename": file.filename},
                    )
                    session_step_id = step_row.id
                    await session_service.update_step(
                        db,
                        current_user,
                        step_id=session_step_id,
                        status="running",
                        run_ref_type="transformation",
                    )
                except PermissionError:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Session not found or access denied")
                except Exception as e:
                    logger.warning(f"Unable to create/update session step: {e}")

            # Initialize the transformation service
            transformation_service = DataTransformationService(
                dataset_id=dataset.id,
                user_id=current_user.id,
                config=transformation_config.dict(),
                db=db
            )
            
            # Apply transformations
            transformed_df = transformation_service.transform()
            
            # Generate a unique filename for the transformed data
            transformed_filename = f"{os.path.splitext(file.filename)[0]}_transformed_{int(datetime.now().timestamp())}{file_extension}"
            transformed_dir = os.path.join(settings.UPLOAD_DIR, str(current_user.id), "transformed")
            os.makedirs(transformed_dir, exist_ok=True)
            transformed_path = os.path.join(transformed_dir, transformed_filename)
            
            # Save the transformed data
            save_dataframe(transformed_df, transformed_path)
            
            # Create a dataset record for the transformed data
            transformed_dataset = Dataset(
                user_id=current_user.id,
                filename=transformed_filename,
                file_path=transformed_path,
                file_size=os.path.getsize(transformed_path),
                file_type=f"application/{file_extension[1:]}" if file_extension else "application/octet-stream",
                is_processed=True
            )
            db.add(transformed_dataset)
            await db.commit()
            
            # Create a transformation run record
            transformation_run = TransformationRun(
                user_id=current_user.id,
                dataset_id=dataset.id,
                transformed_dataset_id=transformed_dataset.id,
                config=transformation_config.dict(),
                status="completed"
            )
            db.add(transformation_run)
            await db.commit()
            # Update session step success if created
            if session_id is not None and session_step_id is not None:
                try:
                    await session_service.update_step(
                        db,
                        current_user,
                        step_id=session_step_id,
                        status="success",
                        finished_at=datetime.utcnow(),
                        run_ref_id=str(transformation_run.id),
                    )
                except Exception as e:
                    logger.warning(f"Failed to update session step to success: {e}")
            
            return TransformationResponse(
                success=True,
                message="File uploaded and transformed successfully",
                dataset_id=transformed_dataset.id,
                download_url=f"/api/transform/download/{transformation_run.id}",
                transformation_id=transformation_run.id
            )
            
        except Exception as e:
            await db.rollback()
            # Clean up the uploaded file if it exists
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up file {file_path}: {str(cleanup_error)}")
            
            logger.error(f"Error processing uploaded file: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing uploaded file: {str(e)}"
            )
        
        await db.add(transformed_dataset)
        await db.commit()
        await db.refresh(transformed_dataset)
        
        # Record the transformation run
        transformation_run = TransformationRun(
            user_id=current_user.id,
            original_dataset_id=dataset.id,
            transformed_dataset_id=transformed_dataset.id,
            config=transformation_config.model_dump(),
            status="completed"
        )
        db.add(transformation_run)
        await db.commit()
        await db.refresh(transformation_run)
        
        return TransformationResponse(
            status="success",
            message="File uploaded and transformed successfully",
            original_dataset_id=dataset.id,
            transformed_dataset_id=transformed_dataset.id,
            download_url=f"/api/v1/transformation/download/{transformation_run.id}",
            transformation_id=transformation_run.id
        )
        
    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        # Best-effort mark failed if step exists
        try:
            if 'session_step_id' in locals() and session_step_id is not None:
                await session_service.update_step(
                    db,
                    current_user,
                    step_id=session_step_id,
                    status="failed",
                    error=str(e),
                    finished_at=datetime.utcnow(),
                )
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )


@router.get("/download/{transformation_id}", response_class=FileResponse)
async def download_transformed_data(
    transformation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Download the transformed data file.
    
    Args:
        transformation_id: ID of the transformation run
        current_user: Current authenticated user
        db: Async database session
        
    Returns:
        The transformed file for download
        
    Raises:
        HTTPException: If transformation or file not found or access denied
    """
    try:
        # Get the transformation run
        result = await db.execute(
            select(TransformationRun).filter(
                TransformationRun.id == transformation_id,
                (TransformationRun.user_id == current_user.id) | (current_user.role == "admin")
            )
        )
        transformation = result.scalars().first()

        if not transformation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Transformation not found or access denied"
            )

        # Get the transformed dataset using the transformed file path from the transformation record
        result = await db.execute(
            select(Dataset).filter(
                Dataset.file_path == transformation.transformed_file_path,
                (Dataset.user_id == current_user.id) | (current_user.role == "admin")
            )
        )
        dataset = result.scalars().first()

        if not dataset:
            logger.error(f"Transformed dataset not found for transformation ID: {transformation_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Transformed dataset record not found"
            )

        if not os.path.exists(dataset.file_path):
            logger.error(f"Transformed file not found on disk: {dataset.file_path}")

            # Update dataset status to indicate the file is missing
            try:
                dataset.is_processed = False
                await db.commit()
            except Exception as update_error:
                logger.error(f"Failed to update dataset status: {str(update_error)}")

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The transformed file could not be found on the server"
            )

        # Determine media type based on file extension
        media_types = {
            '.csv': 'text/csv',
            '.parquet': 'application/octet-stream',
            '.feather': 'application/octet-stream',
            '.json': 'application/json',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }

        file_ext = os.path.splitext(dataset.filename)[1].lower()
        media_type = media_types.get(file_ext, 'application/octet-stream')

        # Return the file for download
        return FileResponse(
            path=dataset.file_path,
            filename=dataset.filename,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={dataset.filename}"}
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in download endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request"
        )
    
    finally:
        # Ensure database connection is closed
        try:
            db.close()
        except Exception as db_error:
            logger.error(f"Error closing database connection: {str(db_error)}")
