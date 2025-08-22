from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.database import get_db
from app.db.models import FeatureSet, FeatureTypeEnum, Dataset, User
from app.schemas.feature_set import FeatureSetCreate, FeatureSetUpdate, FeatureSetOut
from app.services.auth import get_current_user
from typing import List, Optional

router = APIRouter(
    prefix="/feature-sets",
    tags=["Feature Sets"],
    responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=FeatureSetOut, status_code=status.HTTP_201_CREATED)
async def create_feature_set(
    feature_set: FeatureSetCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Check for duplicate name for this user+dataset
    result = await db.execute(
        select(FeatureSet).where(
            FeatureSet.user_id == current_user.id,
            FeatureSet.dataset_id == feature_set.dataset_id,
            FeatureSet.name == feature_set.name
        )
    )
    existing = result.scalar()
    if existing:
        raise HTTPException(status_code=400, detail="Feature set with this name already exists for this dataset.")
    db_feature_set = FeatureSet(
        user_id=current_user.id,
        dataset_id=feature_set.dataset_id,
        name=feature_set.name,
        path=feature_set.path,
        feature_type=feature_set.feature_type,
        description=feature_set.description
    )
    db.add(db_feature_set)
    await db.commit()
    await db.refresh(db_feature_set)
    return db_feature_set

@router.get("/", response_model=List[FeatureSetOut])
async def list_feature_sets(
    dataset_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = select(FeatureSet).where(FeatureSet.user_id == current_user.id)
    if dataset_id:
        query = query.where(FeatureSet.dataset_id == dataset_id)
    result = await db.execute(query)
    return result.scalars().all()

@router.delete("/{feature_set_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_feature_set(
    feature_set_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(FeatureSet).where(
            FeatureSet.id == feature_set_id,
            FeatureSet.user_id == current_user.id
        )
    )
    obj = result.scalar()
    if not obj:
        raise HTTPException(status_code=404, detail="Feature set not found or not owned by user.")
    await db.delete(obj)
    await db.commit()
    return

@router.patch("/{feature_set_id}", response_model=FeatureSetOut)
async def update_feature_set(
    feature_set_id: int,
    update: FeatureSetUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(FeatureSet).where(
            FeatureSet.id == feature_set_id,
            FeatureSet.user_id == current_user.id
        )
    )
    obj = result.scalar()
    if not obj:
        raise HTTPException(status_code=404, detail="Feature set not found.")
    for k, v in update.dict(exclude_unset=True).items():
        setattr(obj, k, v)
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj
