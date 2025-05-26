from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db
from app.db.models import Dataset, User
from app.services.visualization import generate_visualizations
from app.dependencies import get_current_user
from typing import List, Dict, Any
import os
import pandas as pd

router = APIRouter()

def convert_missing_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle legacy missing values format"""
    if isinstance(data, dict) and 'missing_percentage' not in data:
        total = sum(int(v) for v in data.values())
        count = len(data)
        return {
            "total_missing": total,
            "missing_percentage": (total / count * 100) if count > 0 else 0.0,
            "per_column": {k: {"count": v, "percentage": (v/count*100) if count >0 else 0} 
                         for k, v in data.items()}
        }
    return data

@router.get("/{dataset_id}")
async def get_diagnosis_data(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get complete diagnosis report with visualization endpoints"""
    dataset = await db.get(Dataset, dataset_id)
    if not dataset or dataset.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {
        "id": dataset.id,
        "filename": dataset.filename,
        "analysis": {
            "missing_values": convert_missing_values(dataset.missing_values or {}),
            "duplicates": dataset.duplicates or 0,
            "categorical_issues": dataset.categorical_issues or {},
            "summary_stats": dataset.summary_stats or {}
        },
        "visualizations": [
            # Basic visualizations
            {"type": "missing", "endpoint": f"/api/v1/diagnosis/{dataset_id}/visualization/missing", "category": "basic"},
            {"type": "duplicates", "endpoint": f"/api/v1/diagnosis/{dataset_id}/visualization/duplicates", "category": "basic"},
            {"type": "categorical", "endpoint": f"/api/v1/diagnosis/{dataset_id}/visualization/categorical", "category": "basic"},
            {"type": "outliers", "endpoint": f"/api/v1/diagnosis/{dataset_id}/visualization/outliers", "category": "basic"},
            
            # Advanced visualizations
            {"type": "structure", "endpoint": f"/api/v1/diagnosis/{dataset_id}/visualization/structure", "category": "advanced"},
            {"type": "distribution", "endpoint": f"/api/v1/diagnosis/{dataset_id}/visualization/distribution", "category": "advanced"},
            {"type": "correlation", "endpoint": f"/api/v1/diagnosis/{dataset_id}/visualization/correlation", "category": "advanced"},
            {"type": "summary", "endpoint": f"/api/v1/diagnosis/{dataset_id}/visualization/summary", "category": "advanced"},
            {"type": "categorical_consistency", "endpoint": f"/api/v1/diagnosis/{dataset_id}/visualization/categorical_consistency", "category": "advanced"}
        ]
    }

@router.get("/{dataset_id}/visualization/{viz_type}")
async def get_visualization(
    dataset_id: int,
    viz_type: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Endpoint to retrieve specific visualization data"""
    valid_types = [
        # Basic visualizations
        "missing", "duplicates", "categorical", "outliers",
        # Advanced visualizations
        "structure", "distribution", "correlation", "summary", "categorical_consistency"
    ]
    if viz_type not in valid_types:
        raise HTTPException(status_code=400, detail="Invalid visualization type")

    dataset = await db.get(Dataset, dataset_id)
    if not dataset or dataset.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        if not os.path.exists(dataset.file_path):
            raise FileNotFoundError("Data file not found")
            
        viz_data = await generate_visualizations(
            file_path=dataset.file_path,
            file_type=dataset.file_type,
            viz_type=viz_type
        )
        return {
            "dataset_id": dataset_id,
            "visualization_type": viz_type,
            "data": viz_data
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except pd.errors.EmptyDataError as e:
        raise HTTPException(status_code=422, detail="Empty dataset")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")
