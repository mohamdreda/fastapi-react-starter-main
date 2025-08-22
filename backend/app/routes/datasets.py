from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from ..db.database import get_db
from ..db.models import Dataset, User
from ..services.auth import get_current_user
from ..utils.logger import setup_logger
import json
import os
from pathlib import Path
import pandas as pd

logger = setup_logger(__name__)
router = APIRouter()

@router.get("", response_model=List[dict])
@router.get("/", response_model=List[dict])
async def get_datasets(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all datasets for the current user"""
    try:
        query = select(Dataset).where(Dataset.user_id == current_user.id)
        result = await db.execute(query)
        datasets = result.scalars().all()

        # Convert to dict - JSON fields are already parsed by SQLAlchemy
        datasets_list = []
        for dataset in datasets:
            dataset_dict = {
                "id": dataset.id,
                "filename": dataset.filename,
                "file_type": dataset.file_type,
                "file_path": dataset.file_path,
                "format": dataset.format,
                "missing_values": dataset.missing_values,
                "duplicates": dataset.duplicates,
                "data_types": dataset.data_types,
                "categorical_issues": dataset.categorical_issues,
                "summary_stats": dataset.summary_stats,
                "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
                "updated_at": dataset.updated_at.isoformat() if dataset.updated_at else None
            }
            datasets_list.append(dataset_dict)

        return datasets_list

    except Exception as e:
        logger.error(f"Error fetching datasets: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{dataset_id}", response_model=dict)
async def get_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific dataset by ID"""
    try:
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

        return {
            "id": dataset.id,
            "filename": dataset.filename,
            "file_type": dataset.file_type,
            "file_path": dataset.file_path,
            "format": dataset.format,
            "missing_values": dataset.missing_values,
            "duplicates": dataset.duplicates,
            "data_types": dataset.data_types,
            "categorical_issues": dataset.categorical_issues,
            "summary_stats": dataset.summary_stats,
            "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
            "updated_at": dataset.updated_at.isoformat() if dataset.updated_at else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/file/download")
async def download_file(
    path: str,
    current_user: User = Depends(get_current_user)
):
    """Download a file by its path"""
    try:
        # Log the received path for debugging
        logger.info(f"Download request received for path: {path}")
        
        # Handle both forward and backslashes in paths
        # First convert all to forward slashes for consistency
        path = path.replace('\\', '/')
        
        # If the path doesn't start with the uploads directory, prepend it
        # This helps with relative paths from the frontend
        if not path.startswith('uploads/'):
            base_dir = os.path.abspath(os.path.join(os.getcwd(), 'uploads'))
            full_path = os.path.join(base_dir, path.lstrip('/'))
        else:
            full_path = os.path.join(os.getcwd(), path)
        
        # Convert to OS-specific format
        normalized_path = os.path.normpath(full_path)
        
        logger.info(f"Normalized path: {normalized_path}")
        
        # Security check - ensure the path exists
        if not os.path.exists(normalized_path):
            logger.error(f"File not found: {normalized_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {normalized_path}"
            )
        
        # Get file name and extension
        file_name = os.path.basename(normalized_path)
        
        # Read file content
        with open(normalized_path, "rb") as file:
            file_content = file.read()
        
        # Determine content type based on extension
        content_type = "application/octet-stream"  # Default
        if normalized_path.lower().endswith(".csv"):
            content_type = "text/csv"
        elif normalized_path.lower().endswith(".xlsx") or normalized_path.lower().endswith(".xls"):
            content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif normalized_path.lower().endswith(".json"):
            content_type = "application/json"
        
        logger.info(f"Serving file: {normalized_path} as {content_type}")
        
        # Return file as response
        return Response(
            content=file_content,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={file_name}"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file {path}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

async def get_dataset_path(db: AsyncSession, dataset_id: int):
    query = select(Dataset).where(Dataset.id == dataset_id)
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    return dataset.file_path

async def check_dataset_access(db: AsyncSession, dataset_id: int, user_id: int):
    query = select(Dataset).where(
        Dataset.id == dataset_id,
        Dataset.user_id == user_id
    )
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to dataset"
        )

@router.get("/{dataset_id}/columns", response_model=List[str])
async def get_dataset_columns(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieve column names for a dataset by ID"""
    # Verify dataset access
    dataset_path = await get_dataset_path(db, dataset_id)
    await check_dataset_access(db, dataset_id, current_user.id)

    # Load dataset and return columns
    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path, nrows=0)
    elif dataset_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(dataset_path, nrows=0)
    elif dataset_path.endswith('.json'):
        df = pd.read_json(dataset_path, lines=True, nrows=0)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {os.path.splitext(dataset_path)[1]}"
        )

    return df.columns.tolist()

@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a dataset by ID"""
    try:
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

        await db.delete(dataset)
        await db.commit()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )