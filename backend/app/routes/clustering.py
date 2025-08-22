"""
Clustering Routes

This module contains the API routes for clustering operations (DBSCAN, OPTICS, DENCLUE)
"""
import os
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, Body, status, Query, Response, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Dict, Any, Optional, List
import os
import pandas as pd
import json
from pathlib import Path
import uuid
import shutil
from uuid import UUID
from datetime import datetime
from app.services import sessions as session_service

from app.db.database import get_db
from app.db.models import User, Dataset, ClusteringResult
from app.services.auth import get_current_user
from app.services.outlier_detection.clustering.dbscan_service import run_dbscan_clustering
from app.services.outlier_detection.clustering.optics_service import run_optics_clustering
from app.services.outlier_detection.clustering.denclue_service import run_denclue_clustering
from app.utils.file_utils import validate_file_extension
from ..config.config import get_settings

settings = get_settings()
router = APIRouter()

@router.post("/density/upload", response_model=Dict[str, Any])
async def run_density_clustering_upload(
    algorithm: str = Form(...),
    parameters: str = Form(...),
    file: UploadFile = File(...),
    true_labels_file: Optional[UploadFile] = File(None),
    session_id: Optional[UUID] = Query(None, description="Optional workflow session id"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Run density-based clustering algorithm on an uploaded file"""
    # Validate file extension
    valid_extensions = ['.csv', '.xlsx', '.xls']
    if not validate_file_extension(file.filename, valid_extensions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format. Supported formats: {', '.join(valid_extensions)}"
        )
    
    # Create a temporary directory for the uploaded file
    temp_dir = Path(f"app/uploads/temp/{current_user.id}_{uuid.uuid4()}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the uploaded file
    file_path = temp_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Session steps (optional)
    _cluster_step = None
    if session_id:
        try:
            _cluster_step = await session_service.add_step(
                db,
                current_user,
                session_id,
                tool="clustering",
                step="run",
                substep="density_upload",
                algorithm=algorithm,
                params={"source": "upload", "original_filename": file.filename},
            )
        except Exception as e:
            print(f"CLUSTERING UPLOAD: session step create error: {e}")
    try:
        # Parse parameters from JSON string
        parameters = json.loads(parameters)
        
        # Load the dataset based on file extension
        if file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:  # Default to CSV
            df = pd.read_csv(file_path)
    except Exception as e:
        # Update session step -> failed
        try:
            if _cluster_step:
                await session_service.update_step(
                    db,
                    current_user,
                    _cluster_step.id,
                    status="failed",
                    error=str(e),
                    finished_at=datetime.utcnow(),
                )
        except Exception as upd_e:
            print(f"CLUSTERING UPLOAD: session step fail update error: {upd_e}")
        # Clean up the temporary file
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error processing file: {str(e)}"
        )
        
    # Process true labels if provided
    true_labels = None
    if true_labels_file:
        try:
            labels_file_path = temp_dir / true_labels_file.filename
            with open(labels_file_path, "wb") as buffer:
                shutil.copyfileobj(true_labels_file.file, buffer)
                
            # Read true labels based on file extension
            if labels_file_path.suffix.lower() in ['.xlsx', '.xls']:
                labels_df = pd.read_excel(labels_file_path)
            else:  # Default to CSV
                labels_df = pd.read_csv(labels_file_path)
                
            # Extract labels column (assuming it's the first column)
            if not labels_df.empty:
                true_labels = labels_df.iloc[:, 0].values
                
                # Validate that number of labels matches number of data points
                if len(true_labels) != len(df):
                    raise ValueError(f"Number of true labels ({len(true_labels)}) does not match number of data points ({len(df)})")
        except Exception as e:
            # Log the error but continue without true labels
            print(f"Error processing true labels file: {str(e)}")
            true_labels = None
    
    # Run clustering algorithm
    try:
        # Step -> running
        try:
            if _cluster_step:
                await session_service.update_step(db, current_user, _cluster_step.id, status="running")
        except Exception as upd_e:
            print(f"CLUSTERING UPLOAD: session step running update error: {upd_e}")

        if algorithm == "dbscan":
            results = run_dbscan_clustering(df, parameters, true_labels)
        elif algorithm == "optics":
            results = run_optics_clustering(df, parameters, true_labels)
        elif algorithm == "denclue":
            results = run_denclue_clustering(df, parameters, true_labels)
        else:
            # Clean up the temporary file
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported algorithm: {algorithm}"
            )
    except Exception as e:
        # Update session step -> failed
        try:
            if _cluster_step:
                await session_service.update_step(
                    db,
                    current_user,
                    _cluster_step.id,
                    status="failed",
                    error=str(e),
                    finished_at=datetime.utcnow(),
                )
        except Exception as upd_e:
            print(f"CLUSTERING UPLOAD: session step fail update error: {upd_e}")
        # Clean up the temporary file
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running clustering: {str(e)}"
        )
    
    # Save results to database
    output_dir = Path(f"app/outputs/clustering/{current_user.id}/uploaded")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save clustered data to CSV
    result_filename = f"{algorithm}_clustering_results_{uuid.uuid4()}.csv"
    result_path = output_dir / result_filename
    results["clustered_data"].to_csv(result_path, index=False)
    
    # Create database entry with enhanced metadata
    clustering_result = ClusteringResult(
        user_id=current_user.id,
        dataset_id=None,  # No associated dataset since this is a direct upload
        algorithm=algorithm,
        parameters=parameters,
        result_path=str(result_path),
        result_metadata={
            "n_clusters": results.get("n_clusters"),
            "analysis_summary": results.get("analysis_summary"),
            "visualizations": results.get("visualizations"),
            "original_filename": file.filename
        }
    )
    
    db.add(clustering_result)
    await db.commit()
    await db.refresh(clustering_result)
    
    # Clean up the temporary file
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Session step -> success
    try:
        if _cluster_step:
            await session_service.update_step(
                db,
                current_user,
                _cluster_step.id,
                status="success",
                finished_at=datetime.utcnow(),
                run_ref_type="clustering",
                run_ref_id=f"{algorithm}_upload_run:{clustering_result.id}",
            )
    except Exception as upd_e:
        print(f"CLUSTERING UPLOAD: session step update error: {upd_e}")
    
    # Return enhanced results
    return {
        "id": clustering_result.id,
        "algorithm": algorithm,
        "n_clusters": results.get("n_clusters"),
        "cluster_labels": results.get("cluster_labels"),
        "analysis_summary": results.get("analysis_summary"),
        "visualizations": results.get("visualizations"),
        "original_filename": file.filename
    }

@router.post("/density", response_model=Dict[str, Any])
async def run_density_clustering(
    payload: Dict[str, Any] = Body(...),
    session_id: Optional[UUID] = Query(None, description="Optional workflow session id"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Run density-based clustering algorithm on a dataset"""
    dataset_id = payload.get("dataset_id")
    algorithm = payload.get("algorithm", "dbscan")
    parameters = payload.get("parameters", {})
    
    # Session steps (optional)
    _cluster_step = None
    if session_id:
        try:
            _cluster_step = await session_service.add_step(
                db,
                current_user,
                session_id,
                tool="clustering",
                step="run",
                substep="density",
                algorithm=algorithm,
                params={
                    "dataset_id": dataset_id,
                    "algorithm": algorithm,
                    "parameters": parameters,
                    "true_labels_path": payload.get("true_labels_path"),
                },
            )
        except Exception as e:
            print(f"CLUSTERING RUN: session step create error: {e}")
    
    # Validate dataset ownership
    dataset = await db.get(Dataset, dataset_id)
    if not dataset or dataset.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found or access denied"
        )
    
    # Load dataset
    dataset_path = dataset.file_path
    if not os.path.exists(dataset_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found"
        )
    
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        # Update session step -> failed
        try:
            if _cluster_step:
                await session_service.update_step(
                    db,
                    current_user,
                    _cluster_step.id,
                    status="failed",
                    error=str(e),
                    finished_at=datetime.utcnow(),
                )
        except Exception as upd_e:
            print(f"CLUSTERING RUN: session step fail update error: {upd_e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error loading dataset: {str(e)}"
        )
        
    # Process true labels if provided in the payload
    true_labels = None
    true_labels_path = payload.get("true_labels_path")
    
    if true_labels_path:
        try:
            # Validate true labels file exists and is accessible
            if not os.path.exists(true_labels_path):
                print(f"True labels file not found: {true_labels_path}")
            else:
                # Load true labels based on file extension
                if true_labels_path.lower().endswith(('.xlsx', '.xls')):
                    labels_df = pd.read_excel(true_labels_path)
                else:  # Default to CSV
                    labels_df = pd.read_csv(true_labels_path)
                    
                # Extract labels column (assuming it's the first column)
                if not labels_df.empty:
                    true_labels = labels_df.iloc[:, 0].values
                    
                    # Validate that number of labels matches number of data points
                    if len(true_labels) != len(df):
                        print(f"Warning: Number of true labels ({len(true_labels)}) does not match number of data points ({len(df)})")
                        true_labels = None
        except Exception as e:
            # Log the error but continue without true labels
            print(f"Error processing true labels file: {str(e)}")
            true_labels = None
    
    # Update session step -> running
    try:
        if _cluster_step:
            await session_service.update_step(db, current_user, _cluster_step.id, status="running")
    except Exception as upd_e:
        print(f"CLUSTERING RUN: session step running update error: {upd_e}")
    
    # Run clustering algorithm
    try:
        if algorithm == "dbscan":
            results = run_dbscan_clustering(df, parameters, true_labels)
        elif algorithm == "optics":
            results = run_optics_clustering(df, parameters, true_labels)
        elif algorithm == "denclue":
            results = run_denclue_clustering(df, parameters, true_labels)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported algorithm: {algorithm}"
            )
    except Exception as e:
        # Update session step -> failed
        try:
            if _cluster_step:
                await session_service.update_step(
                    db,
                    current_user,
                    _cluster_step.id,
                    status="failed",
                    error=str(e),
                    finished_at=datetime.utcnow(),
                )
        except Exception as upd_e:
            print(f"CLUSTERING RUN: session step fail update error: {upd_e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running clustering: {str(e)}"
        )
    
    # Save results to database
    output_dir = Path(f"app/outputs/clustering/{current_user.id}/{dataset_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save clustered data to CSV
    result_filename = f"{algorithm}_clustering_results_{dataset_id}.csv"
    result_path = output_dir / result_filename
    results["clustered_data"].to_csv(result_path, index=False)
    
    # Create database entry with enhanced metadata
    clustering_result = ClusteringResult(
        user_id=current_user.id,
        dataset_id=dataset_id,
        algorithm=algorithm,
        parameters=parameters,
        result_path=str(result_path),
        result_metadata={
            "n_clusters": results.get("n_clusters"),
            "analysis_summary": results.get("analysis_summary"),
            "visualizations": results.get("visualizations"),
            "dataset_name": dataset.filename
        }
    )
    
    db.add(clustering_result)
    await db.commit()
    await db.refresh(clustering_result)
    
    # Session step -> success
    try:
        if _cluster_step:
            await session_service.update_step(
                db,
                current_user,
                _cluster_step.id,
                status="success",
                finished_at=datetime.utcnow(),
                run_ref_type="clustering",
                run_ref_id=f"{algorithm}_run:{clustering_result.id}",
            )
    except Exception as upd_e:
        print(f"CLUSTERING RUN: session step update error: {upd_e}")
    
    # Return enhanced results
    return {
        "id": clustering_result.id,
        "algorithm": algorithm,
        "n_clusters": results.get("n_clusters"),
        "cluster_labels": results.get("cluster_labels"),
        "analysis_summary": results.get("analysis_summary"),
        "visualizations": results.get("visualizations"),
        "dataset_name": dataset.filename
    }

@router.get("/download/{clustering_id}")
async def download_clustering_result(
    clustering_id: int,
    filename: Optional[str] = Query(None),
    session_id: Optional[UUID] = Query(None, description="Optional workflow session id"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Download clustering result file with optional custom filename"""
    # Get clustering result
    clustering_result = await db.get(ClusteringResult, clustering_id)
    if not clustering_result or clustering_result.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Clustering result not found or access denied"
        )
    
    # Check if file exists
    result_path = clustering_result.result_path
    if not os.path.exists(result_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Clustering result file not found"
        )
    
    # Use custom filename if provided, otherwise use default
    if filename:
        # Validate filename ends with .csv
        if not filename.lower().endswith('.csv'):
            filename += '.csv'
        # Sanitize filename
        filename = os.path.basename(filename)
    else:
        # Use default filename based on algorithm and original filename if available
        original_filename = clustering_result.result_metadata.get("original_filename", "results") if clustering_result.result_metadata else "results"
        base_name = os.path.splitext(original_filename)[0]
        filename = f"{clustering_result.algorithm}_{base_name}_results.csv"
    
    # Session step (optional): download
    try:
        if session_id:
            step = await session_service.add_step(
                db,
                current_user,
                session_id,
                tool="clustering",
                step="download",
                substep=clustering_result.algorithm,
                algorithm=clustering_result.algorithm,
                params={"clustering_id": clustering_id, "filename": filename},
            )
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    step.id,
                    status="success",
                    finished_at=datetime.utcnow(),
                    run_ref_type="clustering",
                    run_ref_id=f"download:{clustering_id}",
                )
            except Exception as upd_e:
                print(f"CLUSTERING DOWNLOAD: session step update error: {upd_e}")
    except Exception as step_e:
        print(f"CLUSTERING DOWNLOAD: session step create error: {step_e}")
    
    return FileResponse(
        path=result_path,
        filename=filename,
        media_type="text/csv"
    )

@router.get("/results", response_model=List[Dict[str, Any]])
async def list_clustering_results(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all clustering results for the current user"""
    # Query for clustering results
    query = select(ClusteringResult).where(ClusteringResult.user_id == current_user.id)
    result = await db.execute(query)
    clustering_results = result.scalars().all()
    
    # Format results
    formatted_results = []
    for cr in clustering_results:
        dataset_name = "Unknown"
        if cr.dataset_id:
            dataset = await db.get(Dataset, cr.dataset_id)
            dataset_name = dataset.filename if dataset else "Unknown"
        else:
            # For uploaded files without dataset association
            dataset_name = cr.result_metadata.get("original_filename", "Uploaded file") if cr.result_metadata else "Uploaded file"
            
        formatted_results.append({
            "id": cr.id,
            "algorithm": cr.algorithm,
            "dataset_name": dataset_name,
            "created_at": cr.created_at.isoformat(),
            "n_clusters": cr.result_metadata.get("n_clusters") if cr.result_metadata else None,
        })
    
    return formatted_results
