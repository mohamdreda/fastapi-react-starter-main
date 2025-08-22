from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db
from app.db.models import Dataset, User
from app.services.visualization import generate_visualizations
from app.services.data_quality import DataQualityAnalyzer
from app.services.upload import read_dataframe
from app.dependencies import get_current_user
from typing import List, Dict, Any, Optional
import os
import json
import pandas as pd
from pathlib import Path
from uuid import UUID
from datetime import datetime
from app.services import sessions as session_service

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
    refresh: bool = False,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    session_id: Optional[UUID] = Query(None, description="Optional session to record this run as a step"),
):
    """Get complete diagnosis report with visualization endpoints"""
    dataset = await db.get(Dataset, dataset_id)
    if not dataset or dataset.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Optionally create a session step for this diagnosis analysis
    session_step_id = None
    step_failed = False
    if session_id is not None:
        try:
            step_row = await session_service.add_step(
                db,
                current_user,
                session_id,
                tool="diagnosis",
                step="analysis",
                params={"dataset_id": dataset_id, "refresh": refresh},
            )
            session_step_id = step_row.id
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    step_id=step_row.id,
                    status="running",
                    run_ref_type="diagnosis",
                    run_ref_id=str(dataset_id),
                )
            except Exception as e:
                print(f"WARNING: Failed to mark diagnosis session step running: {e}")
        except PermissionError:
            raise HTTPException(status_code=403, detail="Session not found or access denied")
        except Exception as e:
            print(f"WARNING: Unable to create diagnosis session step: {e}")
    
    # If refresh is requested, reanalyze the dataset with the enhanced analyzer
    if refresh and os.path.exists(dataset.file_path):
        try:
            # Read the file
            df = read_dataframe(Path(dataset.file_path), dataset.file_type)
            
            # Use the enhanced data quality analyzer
            analyzer = DataQualityAnalyzer(df)
            analysis_results = analyzer.get_full_analysis()
            
            # Update the dataset with new analysis
            dataset.missing_values = json.dumps(analysis_results['missing_values'])
            dataset.duplicates = analysis_results['duplicates']['duplicate_count']
            dataset.data_types = json.dumps(analysis_results['data_types']['column_types'])
            dataset.categorical_issues = json.dumps(analysis_results['categorical_consistency'])
            dataset.analysis_metadata = json.dumps({
                'id_columns': analysis_results['id_columns'],
                'duplicate_details': analysis_results['duplicates'],
                'type_issues': analysis_results['data_types']['type_issues']
            })
            
            await db.commit()
        except Exception as e:
            # Log error and mark step failed, but continue with existing data
            print(f"Error refreshing analysis: {str(e)}")
            if session_step_id is not None:
                try:
                    await session_service.update_step(
                        db,
                        current_user,
                        step_id=session_step_id,
                        status="failed",
                        error=str(e),
                        finished_at=datetime.utcnow(),
                    )
                except Exception as upd_e:
                    print(f"WARNING: Failed to update diagnosis session step on error: {upd_e}")
            step_failed = True
    
    # Parse JSON fields
    missing_values = json.loads(dataset.missing_values) if dataset.missing_values else {}
    categorical_issues = json.loads(dataset.categorical_issues) if dataset.categorical_issues else {}
    summary_stats = json.loads(dataset.summary_stats) if dataset.summary_stats else {}
    data_types = json.loads(dataset.data_types) if dataset.data_types else {}
    metadata = json.loads(dataset.analysis_metadata) if dataset.analysis_metadata else {}
    
    # Enhanced response with detailed information
    resp = {
        "id": dataset.id,
        "filename": dataset.filename,
        "file_type": dataset.file_type,
        "analysis": {
            "missing_values": convert_missing_values(missing_values),
            "duplicates": {
                "count": dataset.duplicates or 0,
                "details": metadata.get('duplicate_details', {})
            },
            "categorical_issues": categorical_issues,
            "summary_stats": summary_stats,
            "data_types": data_types,
            "id_columns": metadata.get('id_columns', []),
            "type_issues": metadata.get('type_issues', {})
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

    # Finalize session step on success
    if session_step_id is not None and not step_failed:
        try:
            await session_service.update_step(
                db,
                current_user,
                step_id=session_step_id,
                status="success",
                finished_at=datetime.utcnow(),
            )
        except Exception as upd_e:
            print(f"WARNING: Failed to update diagnosis session step on success: {upd_e}")

    # Include session_step_id in response when available
    resp["session_step_id"] = str(session_step_id) if session_step_id else None
    return resp

@router.get("/{dataset_id}/visualization/{viz_type}")
async def get_visualization(
    dataset_id: int,
    viz_type: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    session_id: Optional[UUID] = Query(None, description="Optional session to record this run as a step"),
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

    # Optionally create a session step for this specific visualization
    session_step_id = None
    if session_id is not None:
        try:
            step_row = await session_service.add_step(
                db,
                current_user,
                session_id,
                tool="diagnosis",
                step="visualization",
                substep=viz_type,
                params={"dataset_id": dataset_id, "viz_type": viz_type},
            )
            session_step_id = step_row.id
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    step_id=step_row.id,
                    status="running",
                    run_ref_type="diagnosis",
                    run_ref_id=f"{dataset_id}:{viz_type}",
                )
            except Exception as e:
                print(f"WARNING: Failed to mark diagnosis viz session step running: {e}")
        except PermissionError:
            raise HTTPException(status_code=403, detail="Session not found or access denied")
        except Exception as e:
            print(f"WARNING: Unable to create diagnosis viz session step: {e}")

    try:
        if not os.path.exists(dataset.file_path):
            raise FileNotFoundError("Data file not found")
            
        viz_data = await generate_visualizations(
            file_path=dataset.file_path,
            file_type=dataset.file_type,
            viz_type=viz_type
        )

        # On success, update session step
        if session_step_id is not None:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    step_id=session_step_id,
                    status="success",
                    finished_at=datetime.utcnow(),
                )
            except Exception as upd_e:
                print(f"WARNING: Failed to update diagnosis viz step on success: {upd_e}")

        return {
            "dataset_id": dataset_id,
            "visualization_type": viz_type,
            "data": viz_data,
            "session_step_id": str(session_step_id) if session_step_id else None,
        }
    except FileNotFoundError as e:
        if session_step_id is not None:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    step_id=session_step_id,
                    status="failed",
                    error=str(e),
                    finished_at=datetime.utcnow(),
                )
            except Exception as upd_e:
                print(f"WARNING: Failed to update diagnosis viz step on FileNotFound: {upd_e}")
        raise HTTPException(status_code=404, detail=str(e))
    except pd.errors.EmptyDataError as e:
        if session_step_id is not None:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    step_id=session_step_id,
                    status="failed",
                    error=str(e),
                    finished_at=datetime.utcnow(),
                )
            except Exception as upd_e:
                print(f"WARNING: Failed to update diagnosis viz step on EmptyDataError: {upd_e}")
        raise HTTPException(status_code=422, detail="Empty dataset")
    except ValueError as e:
        if session_step_id is not None:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    step_id=session_step_id,
                    status="failed",
                    error=str(e),
                    finished_at=datetime.utcnow(),
                )
            except Exception as upd_e:
                print(f"WARNING: Failed to update diagnosis viz step on ValueError: {upd_e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if session_step_id is not None:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    step_id=session_step_id,
                    status="failed",
                    error=str(e),
                    finished_at=datetime.utcnow(),
                )
            except Exception as upd_e:
                print(f"WARNING: Failed to update diagnosis viz step on generic error: {upd_e}")
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")