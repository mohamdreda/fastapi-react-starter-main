"""
Dataset access utility functions.

This module provides functions for checking dataset access permissions
and retrieving dataset paths.
"""
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ..db.models import Dataset
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

async def get_dataset_path(db: AsyncSession, dataset_id: int) -> str:
    """
    Get the file path for a dataset.
    
    Args:
        db: Database session
        dataset_id: ID of the dataset
        
    Returns:
        Path to the dataset file
        
    Raises:
        HTTPException: If the dataset is not found
    """
    try:
        query = select(Dataset).where(Dataset.id == dataset_id)
        result = await db.execute(query)
        dataset = result.scalars().first()
        
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )
            
        return dataset.file_path
    except Exception as e:
        logger.error(f"Error retrieving dataset path: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving dataset path: {str(e)}"
        )

async def check_dataset_access(dataset_id: int, user_id: int, db: AsyncSession = None) -> str:
    """
    Check if a user has access to a dataset and return the dataset path.
    
    Args:
        dataset_id: ID of the dataset
        user_id: ID of the user
        db: Database session (optional, only needed if not using _load_dataset)
        
    Returns:
        Path to the dataset file
        
    Raises:
        HTTPException: If the dataset is not found or the user doesn't have access
    """
    try:
        query = select(Dataset).where(
            (Dataset.id == dataset_id) & 
            (Dataset.user_id == user_id)
        )
        result = await db.execute(query)
        dataset = result.scalars().first()
        
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found or you don't have access to it"
            )
            
        return dataset.file_path
    except Exception as e:
        logger.error(f"Error checking dataset access: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking dataset access: {str(e)}"
        )
