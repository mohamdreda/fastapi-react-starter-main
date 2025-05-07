from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from ..db.database import get_db
from ..db.models import Dataset, User
from ..services.auth import get_current_user
from ..utils.logger import setup_logger
import json

logger = setup_logger(__name__)
router = APIRouter()

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