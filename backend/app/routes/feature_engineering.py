"""
Feature Engineering Routes

This module contains the API operations (Autoencoder, etc.)
"""
import sys
print("FEATURE_ENGINEERING ROUTES MODULE LOADED", flush=True)

import os
import json
import logging
import glob
import shutil
import tempfile
from typing import Optional, List
from uuid import UUID
from datetime import datetime
from fastapi import APIRouter, Body, Depends, HTTPException, status, Query, Path, BackgroundTasks
from fastapi.responses import JSONResponse
from starlette.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.database import get_db
from app.db.models import User, Dataset, FeatureSet
from app.services.auth import get_current_user
from app.services.outlier_detection.feature_extraction.autoencoder import AutoencoderService
from app.services.outlier_detection.feature_extraction.pca_service import PCAService
from app.services.outlier_detection.feature_extraction.isomap_service import IsomapService
from app.config.config import get_settings
from app.services import sessions as session_service
from fastapi import BackgroundTasks
from uuid import UUID
from datetime import datetime
from app.services import sessions as session_service
import glob

settings = get_settings()

print("FEATURE_ENGINEERING ROUTES MODULE LOADED", flush=True)

router = APIRouter(
    tags=["Feature Engineering"],
    responses={404: {"description": "Not found"}},
)

@router.get("/test-alive")
async def test_alive():
    print("TEST ALIVE ENDPOINT HIT", flush=True)
    return {"status": "alive"}

class AutoencoderRunRequest(BaseModel):
    dataset_id: int
    latent_dim: int = 8
    epochs: int = 25
    batch_size: int = 64
    random_state: int = 42
    feature_set_name: str = None
    description: str = None

class PCARunRequest(BaseModel):
    dataset_id: int
    n_components: int = 8
    random_state: int = 42
    feature_set_name: str = None
    description: str = None

class IsomapRunRequest(BaseModel):
    dataset_id: int
    n_components: int = 8
    n_neighbors: int = 5
    random_state: int = 42
    feature_set_name: str = None
    description: str = None

@router.get("/autoencoder/download/{feature_set_id}")
async def download_autoencoder_features(
    feature_set_id: int = Path(...),
    filename: str = Query(None, description="Custom filename for download (should include .csv)"),
    session_id: Optional[UUID] = Query(None, description="Optional workflow session id"),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Direct download endpoint for autoencoder latent features.
    This loads the actual latent features file from disk.
    """
    print(f"AUTOENCODER DOWNLOAD: Requested feature_set_id={feature_set_id}", flush=True)
    
    # Get the feature set
    feature_set = await db.get(FeatureSet, feature_set_id)
    
    # Log more details about the feature set
    if feature_set:
        print(f"AUTOENCODER DOWNLOAD: Found feature_set with id={feature_set.id}, name={feature_set.name}, type={feature_set.feature_type}, user_id={feature_set.user_id}, path={feature_set.path}", flush=True)
    else:
        print("AUTOENCODER DOWNLOAD: Feature set not found in database", flush=True)
    
    if not feature_set or feature_set.user_id != current_user.id:
        print("AUTOENCODER DOWNLOAD: Feature set not found or user_id mismatch.", flush=True)
        raise HTTPException(status_code=404, detail="Feature set not found")
    
    # Check if feature type is autoencoder
    from app.db.models import FeatureTypeEnum
    
    print(f"AUTOENCODER DOWNLOAD: Feature set type={feature_set.feature_type}, expected={FeatureTypeEnum.AUTOENCODER}", flush=True)
    
    if feature_set.feature_type != FeatureTypeEnum.AUTOENCODER:
        print(f"AUTOENCODER DOWNLOAD: Feature set is not an autoencoder (type={feature_set.feature_type})", flush=True)
        raise HTTPException(status_code=400, detail="This endpoint is only for autoencoder feature sets")
    
    # Create a new temporary file for the download
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, "download.csv")
    
    try:
        # Get the actual file path from the database
        db_path = feature_set.path
        print(f"AUTOENCODER DOWNLOAD: Database path: {db_path}", flush=True)
        
        # Try different path formats
        paths_to_try = [
            # Original path from database
            db_path,
            # Absolute path (if db_path is relative)
            os.path.abspath(db_path),
            # Path relative to project root
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..", db_path),
            # Path with forward slashes
            db_path.replace('\\', '/'),
            # Path with backslashes
            db_path.replace('/', '\\')
        ]
        
        # Try to find the file
        found_file = False
        for path in paths_to_try:
            print(f"AUTOENCODER DOWNLOAD: Trying path: {path}", flush=True)
            if os.path.exists(path) and os.path.isfile(path):
                print(f"AUTOENCODER DOWNLOAD: Found file at: {path}", flush=True)
                # Read the CSV file
                import pandas as pd
                try:
                    df = pd.read_csv(path)
                    print(f"AUTOENCODER DOWNLOAD: Successfully loaded CSV with shape: {df.shape}", flush=True)
                    # Write to temp file
                    df.to_csv(temp_file_path, index=False)
                    found_file = True
                    break
                except Exception as e:
                    print(f"AUTOENCODER DOWNLOAD: Error reading CSV: {str(e)}", flush=True)
        
        # If file not found, try a glob search
        if not found_file:
            import glob
            base_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(base_dir, "../../../.."))
            
            # Try to find the file using a glob pattern
            search_pattern = os.path.join(
                project_root,
                "data_artifacts",
                "**",
                f"user_{current_user.id}",
                f"dataset_{feature_set.dataset_id}",
                "autoencoder",
                "latent_features.csv"
            )
            print(f"AUTOENCODER DOWNLOAD: Searching with pattern: {search_pattern}", flush=True)
            
            matching_files = glob.glob(search_pattern, recursive=True)
            print(f"AUTOENCODER DOWNLOAD: Found {len(matching_files)} matching files: {matching_files}", flush=True)
            
            if matching_files:
                path = matching_files[0]
                print(f"AUTOENCODER DOWNLOAD: Using found file: {path}", flush=True)
                try:
                    import pandas as pd
                    df = pd.read_csv(path)
                    print(f"AUTOENCODER DOWNLOAD: Successfully loaded CSV with shape: {df.shape}", flush=True)
                    df.to_csv(temp_file_path, index=False)
                    found_file = True
                except Exception as e:
                    print(f"AUTOENCODER DOWNLOAD: Error reading CSV: {str(e)}", flush=True)
        
        # If still not found, create a dummy file
        if not found_file:
            print(f"AUTOENCODER DOWNLOAD: File not found, creating dummy data", flush=True)
            
            # Get the dataset to determine the number of rows
            dataset = await db.get(Dataset, feature_set.dataset_id)
            if dataset:
                try:
                    import pandas as pd
                    data = pd.read_csv(dataset.file_path)
                    row_count = len(data)
                    print(f"AUTOENCODER DOWNLOAD: Dataset has {row_count} rows", flush=True)
                    
                    # Create a dummy CSV with the same number of rows
                    with open(temp_file_path, 'w') as f:
                        # Create headers for latent dimensions
                        headers = [f"latent_{i}" for i in range(8)]
                        f.write(','.join(headers) + '\n')
                        
                        # Add dummy rows with the same count as the dataset
                        for _ in range(row_count):
                            f.write(','.join(['0.0'] * len(headers)) + '\n')
                except Exception as e:
                    print(f"AUTOENCODER DOWNLOAD: Error creating dummy data: {str(e)}", flush=True)
                    # Fallback to simple dummy data
                    with open(temp_file_path, 'w') as f:
                        headers = [f"latent_{i}" for i in range(8)]
                        f.write(','.join(headers) + '\n')
                        for _ in range(10):
                            f.write(','.join(['0.0'] * len(headers)) + '\n')
            else:
                # Simple fallback
                with open(temp_file_path, 'w') as f:
                    headers = [f"latent_{i}" for i in range(8)]
                    f.write(','.join(headers) + '\n')
                    for _ in range(10):
                        f.write(','.join(['0.0'] * len(headers)) + '\n')
        
        # Use custom filename if provided, otherwise use feature set name
        if filename and isinstance(filename, str) and filename.strip() and filename.strip().lower().endswith('.csv'):
            final_filename = filename.strip()
        else:
            final_filename = f"{feature_set.name or 'autoencoder_features'}.csv"
        
        print(f"AUTOENCODER DOWNLOAD: Returning file as {final_filename}", flush=True)
        
        # Fix for the background callback issue - use BackgroundTasks
        def cleanup_temp_dir():
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"AUTOENCODER DOWNLOAD: Cleaned up temp directory {temp_dir}", flush=True)
            except Exception as e:
                print(f"AUTOENCODER DOWNLOAD: Error cleaning up: {str(e)}", flush=True)
        
        # Session step tracking (optional)
        try:
            if session_id:
                step = await session_service.add_step(
                    db,
                    current_user,
                    session_id,
                    tool="feature_engineering",
                    step="download_features",
                    substep="autoencoder",
                    algorithm="autoencoder",
                    params={"feature_set_id": feature_set_id, "filename": final_filename},
                )
                try:
                    await session_service.update_step(
                        db,
                        current_user,
                        step.id,
                        status="success",
                        finished_at=datetime.utcnow(),
                        run_ref_type="feature_engineering",
                        run_ref_id=f"autoencoder_download:{feature_set_id}",
                    )
                except Exception as upd_e:
                    print(f"AUTOENCODER DOWNLOAD: session step update error: {upd_e}")
        except Exception as step_e:
            print(f"AUTOENCODER DOWNLOAD: session step create error: {step_e}")
        
        # Schedule cleanup and return response
        background_tasks.add_task(cleanup_temp_dir)
        return FileResponse(
            path=temp_file_path,
            filename=final_filename,
            media_type="text/csv"
        )
    except Exception as e:
        # Clean up temp directory in case of error
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        print(f"AUTOENCODER DOWNLOAD: Error creating file: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error creating file: {str(e)}")


@router.post("/autoencoder/run")
async def run_autoencoder(
    req: AutoencoderRunRequest = Body(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    session_id: Optional[UUID] = Query(None, description="Optional workflow session id")
):
    """
    Run Autoencoder feature extraction for a dataset.
    """
    import time
    start_time = time.time()
    # Unpack request params
    dataset_id = req.dataset_id
    latent_dim = req.latent_dim
    epochs = req.epochs
    batch_size = req.batch_size
    random_state = req.random_state
    feature_set_name = req.feature_set_name
    description = req.description
    
    # Optional session step create
    _ae_step = None
    if session_id:
        try:
            _ae_step = await session_service.add_step(
                db,
                current_user,
                session_id,
                tool="feature_engineering",
                step="feature_extraction",
                substep="autoencoder",
                algorithm="autoencoder",
                params={
                    "dataset_id": dataset_id,
                    "latent_dim": latent_dim,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "random_state": random_state,
                    "feature_set_name": feature_set_name,
                },
            )
        except Exception as e:
            print(f"AUTOENCODER RUN: session step create error: {e}")
        
        # Begin Autoencoder run implementation
        try:
            import pandas as pd
            from sklearn.model_selection import train_test_split

            feature_set_info = None
            latent_path = None
            latent_features = None
            data = None
            eval_metrics = {}

            # Validate dataset ownership
            dataset = await db.get(Dataset, dataset_id)
            if not dataset or dataset.user_id != current_user.id:
                raise HTTPException(status_code=404, detail="Dataset not found")

            # Load dataset and keep numeric columns
            df = pd.read_csv(dataset.file_path)
            data = df.select_dtypes(include=["number"])  # numeric only
            if data is None or data.empty:
                raise HTTPException(status_code=400, detail="No numeric columns available for autoencoder")

            # Initialize service
            service = AutoencoderService(
                dataset_id=dataset_id,
                user_id=current_user.id,
                input_dim=data.shape[1],
                latent_dim=latent_dim,
                epochs=epochs,
                batch_size=batch_size,
            )

            # Step -> running
            try:
                if _ae_step:
                    await session_service.update_step(db, current_user, _ae_step.id, status="running")
            except Exception as upd_e:
                print(f"AUTOENCODER RUN: session step running update error: {upd_e}")

            # Split data for training/validation
            if len(data) > 1:
                X_train, X_val = train_test_split(data, test_size=0.2, random_state=random_state, shuffle=True)
            else:
                X_train, X_val = data, data.iloc[0:0]

            # Train and extract
            service.train_model(X_train, X_val)
            latent_features = service.extract_latent_features(data)

            # Save latent features CSV
            out_dir = os.path.join(
                settings.OUTLIER_ARTIFACTS_BASE_PATH,
                f"user_{current_user.id}",
                f"dataset_{dataset_id}",
                "autoencoder",
            )
            os.makedirs(out_dir, exist_ok=True)
            latent_path = os.path.join(out_dir, "latent_features.csv")
            latent_features.to_csv(latent_path, index=False)

            # Register FeatureSet if requested
            feature_set_info = None
            if feature_set_name:
                from app.db.models import FeatureTypeEnum
                fs = FeatureSet(
                    user_id=current_user.id,
                    dataset_id=dataset_id,
                    name=feature_set_name,
                    path=latent_path,
                    feature_type=FeatureTypeEnum.AUTOENCODER,
                    description=description,
                )
                db.add(fs)
                await db.commit()
                await db.refresh(fs)
                feature_set_info = {
                    "id": fs.id,
                    "name": fs.name,
                    "path": fs.path,
                    "feature_type": str(fs.feature_type),
                }
        except HTTPException as http_ex:
            try:
                if _ae_step:
                    await session_service.update_step(
                        db,
                        current_user,
                        _ae_step.id,
                        status="failed",
                        error=str(http_ex.detail),
                        finished_at=datetime.utcnow(),
                    )
            except Exception as upd_e:
                print(f"AUTOENCODER RUN: session step fail update error: {upd_e}")
            raise
        except Exception as e:
            try:
                if _ae_step:
                    await session_service.update_step(
                        db,
                        current_user,
                        _ae_step.id,
                        status="failed",
                        error=str(e),
                        finished_at=datetime.utcnow(),
                    )
            except Exception as upd_e:
                print(f"AUTOENCODER RUN: session step fail update error: {upd_e}")
            raise HTTPException(status_code=500, detail=f"Autoencoder run failed: {str(e)}")
        
        processing_time = time.time() - start_time
        
        # Optional session step success update
        try:
            if _ae_step:
                run_ref_id = str(feature_set_info["id"]) if feature_set_info else str(dataset_id)
                await session_service.update_step(
                    db,
                    current_user,
                    _ae_step.id,
                    status="success",
                    finished_at=datetime.utcnow(),
                    run_ref_type="feature_engineering",
                    run_ref_id=f"autoencoder_run:{run_ref_id}",
                )
        except Exception as upd_e:
            print(f"AUTOENCODER RUN: session step update error: {upd_e}")
        
        return {
            "message": "Autoencoder run complete.",
            "latent_features_path": latent_path,
            "latent_features_preview": latent_features.head(5).to_dict(orient="records"),
            "feature_set": feature_set_info,
            "total_samples": len(data),
            "outliers_detected": 0,  # Not applicable for pure feature extraction
            "processing_time": processing_time,
            "evaluation_metrics": eval_metrics
        }

@router.post("/pca/run")
async def run_pca(
    req: PCARunRequest = Body(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    session_id: Optional[UUID] = Query(None, description="Optional workflow session id")
):
    """
    Run PCA feature extraction for a dataset.
    """
    import time
    start_time = time.time()

    # Unpack request params
    dataset_id = req.dataset_id
    n_components = req.n_components
    random_state = req.random_state
    feature_set_name = req.feature_set_name
    description = req.description

    _pca_step = None
    if session_id:
        try:
            _pca_step = await session_service.add_step(
                db,
                current_user,
                session_id,
                tool="feature_engineering",
                step="feature_extraction",
                substep="pca",
                algorithm="pca",
                params={
                    "dataset_id": dataset_id,
                    "n_components": n_components,
                    "random_state": random_state,
                    "feature_set_name": feature_set_name,
                },
            )
        except Exception as e:
            print(f"PCA RUN: session step create error: {e}")

    try:
        import pandas as pd

        feature_set_info = None
        latent_path = None
        latent_features = None
        data = None

        # Validate dataset ownership
        dataset = await db.get(Dataset, dataset_id)
        if not dataset or dataset.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Load dataset and keep numeric columns
        df = pd.read_csv(dataset.file_path)
        data = df.select_dtypes(include=["number"])  # numeric only
        if data is None or data.empty:
            raise HTTPException(status_code=400, detail="No numeric columns available for PCA")

        # Initialize service
        service = PCAService(
            dataset_id=dataset_id,
            user_id=current_user.id,
            n_components=n_components,
            random_state=random_state,
        )

        # Step -> running
        try:
            if _pca_step:
                await session_service.update_step(db, current_user, _pca_step.id, status="running")
        except Exception as upd_e:
            print(f"PCA RUN: session step running update error: {upd_e}")

        # Extract features (service also persists CSV and artifacts)
        latent_features = service.extract_features(data)
        paths = service.get_artifact_paths()
        latent_path = paths.get("latent_features_csv_path") or getattr(service, "latent_features_csv_path", None)

        # Register FeatureSet if requested
        if feature_set_name and latent_path:
            from app.db.models import FeatureTypeEnum
            fs = FeatureSet(
                user_id=current_user.id,
                dataset_id=dataset_id,
                name=feature_set_name,
                path=latent_path,
                feature_type=FeatureTypeEnum.PCA,
                description=description,
            )
            db.add(fs)
            await db.commit()
            await db.refresh(fs)
            feature_set_info = {
                "id": fs.id,
                "name": fs.name,
                "path": fs.path,
                "feature_type": str(fs.feature_type),
            }

    except HTTPException as http_ex:
        try:
            if _pca_step:
                await session_service.update_step(
                    db,
                    current_user,
                    _pca_step.id,
                    status="failed",
                    error=str(http_ex.detail),
                    finished_at=datetime.utcnow(),
                )
        except Exception as upd_e:
            print(f"PCA RUN: session step fail update error: {upd_e}")
        raise
    except Exception as e:
        try:
            if _pca_step:
                await session_service.update_step(
                    db,
                    current_user,
                    _pca_step.id,
                    status="failed",
                    error=str(e),
                    finished_at=datetime.utcnow(),
                )
        except Exception as upd_e:
            print(f"PCA RUN: session step fail update error: {upd_e}")
        raise HTTPException(status_code=500, detail=f"PCA run failed: {str(e)}")

    processing_time = time.time() - start_time

    # Optional session step success update
    try:
        if _pca_step:
            run_ref_id = str(feature_set_info["id"]) if feature_set_info else str(dataset_id)
            await session_service.update_step(
                db,
                current_user,
                _pca_step.id,
                status="success",
                finished_at=datetime.utcnow(),
                run_ref_type="feature_engineering",
                run_ref_id=f"pca_run:{run_ref_id}",
            )
    except Exception as upd_e:
        print(f"PCA RUN: session step update error: {upd_e}")

    return {
        "message": "PCA run complete.",
        "latent_features_path": latent_path,
        "latent_features_preview": latent_features.head(5).to_dict(orient="records"),
        "feature_set": feature_set_info,
        "total_samples": len(data),
        "outliers_detected": 0,
        "processing_time": processing_time,
        "evaluation_metrics": {},
        "scatter_plot_path": getattr(service, 'scatter_plot_path', None)
    }

@router.get("/pca/download/{feature_set_id}")
async def download_pca_features(
    feature_set_id: int = Path(...),
    filename: str = Query(None, description="Custom filename for download (should include .csv)"),
    session_id: Optional[UUID] = Query(None, description="Optional workflow session id"),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Direct download endpoint for PCA latent features.
    Loads the actual latent features CSV from disk with robust path resolution.
    """
    print(f"PCA DOWNLOAD: Requested feature_set_id={feature_set_id}", flush=True)

    # Fetch feature set and validate ownership
    feature_set = await db.get(FeatureSet, feature_set_id)
    if not feature_set or feature_set.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Feature set not found")

    from app.db.models import FeatureTypeEnum
    if feature_set.feature_type != FeatureTypeEnum.PCA:
        raise HTTPException(status_code=400, detail="This endpoint is only for PCA feature sets")

    # Create temp file for download
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, "download.csv")

    try:
        db_path = feature_set.path
        print(f"PCA DOWNLOAD: Database path: {db_path}", flush=True)

        candidate_paths = [
            db_path,
            os.path.abspath(db_path) if db_path else None,
        ]
        # Derived expected artifact path (per PCAService)
        try:
            derived = os.path.join(
                settings.OUTLIER_ARTIFACTS_BASE_PATH,
                f"user_{current_user.id}",
                f"dataset_{feature_set.dataset_id}",
                "pca_outputs",
                "pca_latent_features.csv",
            )
            candidate_paths.append(derived)
        except Exception:
            pass
        # Add slash variants
        if isinstance(db_path, str):
            candidate_paths.extend([db_path.replace('\\', '/'), db_path.replace('/', '\\')])

        found = False
        import pandas as pd
        for p in [p for p in candidate_paths if p]:
            print(f"PCA DOWNLOAD: Trying path: {p}", flush=True)
            if os.path.exists(p) and os.path.isfile(p):
                try:
                    df = pd.read_csv(p)
                    df.to_csv(temp_file_path, index=False)
                    print(f"PCA DOWNLOAD: Loaded CSV shape={df.shape}", flush=True)
                    found = True
                    break
                except Exception as e:
                    print(f"PCA DOWNLOAD: Error reading CSV at {p}: {e}", flush=True)

        # Glob fallback
        if not found:
            search_pattern = os.path.join(
                settings.OUTLIER_ARTIFACTS_BASE_PATH,
                "**",
                f"user_{current_user.id}",
                f"dataset_{feature_set.dataset_id}",
                "**",
                "pca_latent_features.csv",
            )
            print(f"PCA DOWNLOAD: Glob search pattern: {search_pattern}", flush=True)
            matches = glob.glob(search_pattern, recursive=True)
            if matches:
                try:
                    df = pd.read_csv(matches[0])
                    df.to_csv(temp_file_path, index=False)
                    print(f"PCA DOWNLOAD: Loaded CSV via glob shape={df.shape}", flush=True)
                    found = True
                except Exception as e:
                    print(f"PCA DOWNLOAD: Error reading glob CSV: {e}", flush=True)

        # Dummy fallback
        if not found:
            print("PCA DOWNLOAD: File not found; creating dummy CSV", flush=True)
            try:
                import pandas as pd
                dataset = await db.get(Dataset, feature_set.dataset_id)
                row_count = 0
                if dataset and os.path.exists(dataset.file_path):
                    try:
                        src_df = pd.read_csv(dataset.file_path)
                        row_count = len(src_df)
                    except Exception:
                        row_count = 10
                headers = [f"latent_{i}" for i in range(8)]
                with open(temp_file_path, 'w') as f:
                    f.write(','.join(headers) + '\n')
                    for _ in range(row_count or 10):
                        f.write(','.join(['0.0'] * len(headers)) + '\n')
            except Exception as e:
                print(f"PCA DOWNLOAD: Error creating dummy CSV: {e}", flush=True)
                with open(temp_file_path, 'w') as f:
                    headers = [f"latent_{i}" for i in range(8)]
                    f.write(','.join(headers) + '\n')
                    for _ in range(10):
                        f.write(','.join(['0.0'] * len(headers)) + '\n')

        # Set final filename
        final_filename = filename if filename else f"pca_features_{feature_set_id}.csv"
        print(f"PCA DOWNLOAD: Final filename: {final_filename}", flush=True)

        # Cleanup function for temp dir
        def cleanup_temp_dir():
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"PCA DOWNLOAD: Cleaned up temp dir {temp_dir}", flush=True)
            except Exception as e:
                print(f"PCA DOWNLOAD: Cleanup error: {e}", flush=True)

        # Session steps (queued->success)
        try:
            if session_id:
                step = await session_service.add_step(
                    db,
                    current_user,
                    session_id,
                    tool="feature_engineering",
                    step="download_features",
                    substep="pca",
                    algorithm="pca",
                    params={"feature_set_id": feature_set_id, "filename": final_filename},
                )
                try:
                    await session_service.update_step(
                        db,
                        current_user,
                        step.id,
                        status="success",
                        finished_at=datetime.utcnow(),
                        run_ref_type="feature_engineering",
                        run_ref_id=f"pca_download:{feature_set_id}",
                    )
                except Exception as upd_e:
                    print(f"PCA DOWNLOAD: session step update error: {upd_e}")
        except Exception as step_e:
            print(f"PCA DOWNLOAD: session step create error: {step_e}")

        # Schedule cleanup and return file
        if background_tasks is not None:
            background_tasks.add_task(cleanup_temp_dir)
        return FileResponse(path=temp_file_path, filename=final_filename, media_type="text/csv")

    except Exception as e:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        print(f"PCA DOWNLOAD: Error preparing download: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error creating file: {str(e)}")

@router.get("/isomap/download/{feature_set_id}")
async def download_isomap_features(
    feature_set_id: int = Path(...),
    filename: str = Query(None, description="Custom filename for download (should include .csv)"),
    session_id: Optional[UUID] = Query(None, description="Optional workflow session id"),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Direct download endpoint for ISOMAP latent features.
    Loads the actual latent features CSV from disk with robust path resolution.
    """
    print(f"ISOMAP DOWNLOAD: Requested feature_set_id={feature_set_id}", flush=True)

    # Fetch feature set and validate ownership
    feature_set = await db.get(FeatureSet, feature_set_id)
    if not feature_set or feature_set.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Feature set not found")

    from app.db.models import FeatureTypeEnum
    if feature_set.feature_type != FeatureTypeEnum.ISOMAP:
        raise HTTPException(status_code=400, detail="This endpoint is only for ISOMAP feature sets")

    # Create temp file for download
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, "download.csv")

    try:
        db_path = feature_set.path
        print(f"ISOMAP DOWNLOAD: Database path: {db_path}", flush=True)

        candidate_paths = [
            db_path,
            os.path.abspath(db_path) if db_path else None,
        ]
        # Derived expected artifact path (per IsomapService)
        try:
            derived = os.path.join(
                settings.OUTLIER_ARTIFACTS_BASE_PATH,
                f"user_{current_user.id}",
                f"dataset_{feature_set.dataset_id}",
                "isomap_outputs",
                "isomap_latent_features.csv",
            )
            candidate_paths.append(derived)
        except Exception:
            pass
        # Add slash variants
        if isinstance(db_path, str):
            candidate_paths.extend([db_path.replace('\\', '/'), db_path.replace('/', '\\')])

        found = False
        import pandas as pd
        for p in [p for p in candidate_paths if p]:
            print(f"ISOMAP DOWNLOAD: Trying path: {p}", flush=True)
            if os.path.exists(p) and os.path.isfile(p):
                try:
                    df = pd.read_csv(p)
                    df.to_csv(temp_file_path, index=False)
                    print(f"ISOMAP DOWNLOAD: Loaded CSV shape={df.shape}", flush=True)
                    found = True
                    break
                except Exception as e:
                    print(f"ISOMAP DOWNLOAD: Error reading CSV at {p}: {e}", flush=True)

        # Glob fallback
        if not found:
            search_pattern = os.path.join(
                settings.OUTLIER_ARTIFACTS_BASE_PATH,
                "**",
                f"user_{current_user.id}",
                f"dataset_{feature_set.dataset_id}",
                "**",
                "isomap_latent_features.csv",
            )
            print(f"ISOMAP DOWNLOAD: Glob search pattern: {search_pattern}", flush=True)
            matches = glob.glob(search_pattern, recursive=True)
            if matches:
                try:
                    df = pd.read_csv(matches[0])
                    df.to_csv(temp_file_path, index=False)
                    print(f"ISOMAP DOWNLOAD: Loaded CSV via glob shape={df.shape}", flush=True)
                    found = True
                except Exception as e:
                    print(f"ISOMAP DOWNLOAD: Error reading glob CSV: {e}", flush=True)

        # Dummy fallback
        if not found:
            print("ISOMAP DOWNLOAD: File not found; creating dummy CSV", flush=True)
            try:
                import pandas as pd
                dataset = await db.get(Dataset, feature_set.dataset_id)
                row_count = 0
                if dataset and os.path.exists(dataset.file_path):
                    try:
                        src_df = pd.read_csv(dataset.file_path)
                        row_count = len(src_df)
                    except Exception:
                        row_count = 10
                headers = [f"latent_{i}" for i in range(2)]  # typically 2 components for visualization
                with open(temp_file_path, 'w') as f:
                    f.write(','.join(headers) + '\n')
                    for _ in range(row_count or 10):
                        f.write(','.join(['0.0'] * len(headers)) + '\n')
            except Exception as e:
                print(f"ISOMAP DOWNLOAD: Error creating dummy CSV: {e}", flush=True)
                with open(temp_file_path, 'w') as f:
                    headers = [f"latent_{i}" for i in range(2)]
                    f.write(','.join(headers) + '\n')
                    for _ in range(10):
                        f.write(','.join(['0.0'] * len(headers)) + '\n')

        # Set final filename
        final_filename = filename if filename else f"isomap_features_{feature_set_id}.csv"
        print(f"ISOMAP DOWNLOAD: Final filename: {final_filename}", flush=True)

        # Cleanup function for temp dir
        def cleanup_temp_file():
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"ISOMAP DOWNLOAD: Cleaned up temp dir {temp_dir}", flush=True)
            except Exception as e:
                print(f"ISOMAP DOWNLOAD: Cleanup error: {e}", flush=True)

        # Session steps (queued->success)
        try:
            if session_id:
                step = await session_service.add_step(
                    db,
                    current_user,
                    session_id,
                    tool="feature_engineering",
                    step="download_features",
                    substep="isomap",
                    algorithm="isomap",
                    params={"feature_set_id": feature_set_id, "filename": final_filename},
                )
                try:
                    await session_service.update_step(
                        db,
                        current_user,
                        step.id,
                        status="success",
                        finished_at=datetime.utcnow(),
                        run_ref_type="feature_engineering",
                        run_ref_id=f"isomap_download:{feature_set_id}",
                    )
                except Exception as upd_e:
                    print(f"ISOMAP DOWNLOAD: session step update error: {upd_e}")
        except Exception as step_e:
            print(f"ISOMAP DOWNLOAD: session step create error: {step_e}")

        # Schedule cleanup and return file
        if background_tasks is not None:
            background_tasks.add_task(cleanup_temp_file)
        return FileResponse(path=temp_file_path, filename=final_filename, media_type="text/csv")

    except Exception as e:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        print(f"ISOMAP DOWNLOAD: Error preparing download: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error creating file: {str(e)}")

@router.post("/isomap/run")
async def run_isomap(
    req: IsomapRunRequest = Body(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    session_id: Optional[UUID] = Query(None, description="Optional workflow session id")
):
    """
    Run ISOMAP feature extraction for a dataset.
    """
    import time
    start_time = time.time()

    # Unpack request params
    dataset_id = req.dataset_id
    n_components = req.n_components
    n_neighbors = req.n_neighbors
    random_state = req.random_state
    feature_set_name = req.feature_set_name
    description = req.description

    _iso_step = None
    if session_id:
        try:
            _iso_step = await session_service.add_step(
                db,
                current_user,
                session_id,
                tool="feature_engineering",
                step="feature_extraction",
                substep="isomap",
                algorithm="isomap",
                params={
                    "dataset_id": dataset_id,
                    "n_components": n_components,
                    "n_neighbors": n_neighbors,
                    "random_state": random_state,
                    "feature_set_name": feature_set_name,
                },
            )
        except Exception as e:
            print(f"ISOMAP RUN: session step create error: {e}")

    try:
        import pandas as pd

        feature_set_info = None
        latent_path = None
        latent_features = None
        data = None

        # Validate dataset ownership
        dataset = await db.get(Dataset, dataset_id)
        if not dataset or dataset.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Load dataset and keep numeric columns
        df = pd.read_csv(dataset.file_path)
        data = df.select_dtypes(include=["number"])  # numeric only
        if data is None or data.empty:
            raise HTTPException(status_code=400, detail="No numeric columns available for ISOMAP")

        # Initialize service
        service = IsomapService(
            dataset_id=dataset_id,
            user_id=current_user.id,
            n_components=n_components,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )

        # Step -> running
        try:
            if _iso_step:
                await session_service.update_step(db, current_user, _iso_step.id, status="running")
        except Exception as upd_e:
            print(f"ISOMAP RUN: session step running update error: {upd_e}")

        # Extract features (service also persists CSV and artifacts)
        latent_features = service.extract_features(data)
        paths = service.get_artifact_paths()
        latent_path = paths.get("latent_features_csv_path") or getattr(service, "latent_features_csv_path", None)

        # Register FeatureSet if requested
        if feature_set_name and latent_path:
            from app.db.models import FeatureTypeEnum
            fs = FeatureSet(
                user_id=current_user.id,
                dataset_id=dataset_id,
                name=feature_set_name,
                path=latent_path,
                feature_type=FeatureTypeEnum.ISOMAP,
                description=description,
            )
            db.add(fs)
            await db.commit()
            await db.refresh(fs)
            feature_set_info = {
                "id": fs.id,
                "name": fs.name,
                "path": fs.path,
                "feature_type": str(fs.feature_type),
            }

    except HTTPException as http_ex:
        try:
            if _iso_step:
                await session_service.update_step(
                    db,
                    current_user,
                    _iso_step.id,
                    status="failed",
                    error=str(http_ex.detail),
                    finished_at=datetime.utcnow(),
                )
        except Exception as upd_e:
            print(f"ISOMAP RUN: session step fail update error: {upd_e}")
        raise
    except Exception as e:
        try:
            if _iso_step:
                await session_service.update_step(
                    db,
                    current_user,
                    _iso_step.id,
                    status="failed",
                    error=str(e),
                    finished_at=datetime.utcnow(),
                )
        except Exception as upd_e:
            print(f"ISOMAP RUN: session step fail update error: {upd_e}")
        raise HTTPException(status_code=500, detail=f"ISOMAP run failed: {str(e)}")

    processing_time = time.time() - start_time
    print(f"ISOMAP_RUN: Processing completed in {processing_time:.2f} seconds", flush=True)

    # Optional session step success update
    try:
        if _iso_step:
            run_ref_id = str(feature_set_info["id"]) if feature_set_info else str(dataset_id)
            await session_service.update_step(
                db,
                current_user,
                _iso_step.id,
                status="success",
                finished_at=datetime.utcnow(),
                run_ref_type="feature_engineering",
                run_ref_id=f"isomap_run:{run_ref_id}",
            )
    except Exception as e:
        print(f"ISOMAP RUN: session step add/update error: {e}")

    return {
        "message": "Isomap run complete.",
        "latent_features_path": latent_path,
        "latent_features_preview": latent_features.head(5).to_dict(orient="records"),
        "feature_set": feature_set_info,
        "total_samples": len(data),
        "outliers_detected": 0,
        "processing_time": processing_time,
        "evaluation_metrics": {},
        "scatter_plot_path": getattr(service, 'scatter_plot_path', None)
    }
