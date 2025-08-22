from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import Optional

from app.db.models import Dataset

async def get(db: AsyncSession, id: int) -> Optional[Dataset]:
    """
    Get a dataset by ID
    
    Args:
        db: Database session
        id: Dataset ID
        
    Returns:
        Dataset object if found, None otherwise
    """
    result = await db.execute(select(Dataset).filter(Dataset.id == id))
    return result.scalars().first()

async def get_by_user(db: AsyncSession, user_id: int):
    """
    Get all datasets for a user
    
    Args:
        db: Database session
        user_id: User ID
        
    Returns:
        List of Dataset objects
    """
    result = await db.execute(select(Dataset).filter(Dataset.user_id == user_id))
    return result.scalars().all()
