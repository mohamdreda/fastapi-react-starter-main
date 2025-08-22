"""
API routes for the modular deduplication pipeline.

This module defines the endpoints for each step of the pipeline:
1. Preprocessing
2. Blocking
3. Similarity Calculation
4. Classification
5. Clustering
6. Resolution
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
import os
import json
import logging
from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime
from app.services import sessions as session_service

from app.db.database import get_db
from app.services.auth import get_current_user
from app.db.models import User
from app.schemas.deduplication_pipeline import (
    PreprocessingRequest, PreprocessingResponse,
    BlockingRequest, BlockingResponse,
    SimilarityRequest, SimilarityResponse,
    ClassificationRequest, ClassificationResponse,
    ClusteringRequest, ClusteringResponse,
    ResolutionRequest, ResolutionResponse,
    ManualResolutionRequest, ManualResolutionResponse,
    LegacyDeduplicationRequest, LegacyDeduplicationResponse
)
from app.services.datasets import get_dataset_path, check_dataset_access
from app.services.deduplication.preprocessing.service import run_preprocessing
from app.services.deduplication.blocking.service import run_blocking
from app.services.deduplication.similarity.service import run_similarity_calculation
from app.services.deduplication.classification.service import run_classification
from app.services.deduplication.clustering.service import run_clustering
from app.services.deduplication.resolution.service import run_resolution, apply_manual_resolution
from app.services.deduplication.legacy.service import run_legacy_deduplication, get_legacy_algorithms

router = APIRouter(
    prefix="/deduplication/pipeline",
    tags=["deduplication_pipeline"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

async def _load_dataset(dataset_id: int, user_id: int, db: AsyncSession) -> pd.DataFrame:
    """
    Load a dataset from file.
    
    Args:
        dataset_id: ID of the dataset
        user_id: ID of the user
        db: Database session
        
    Returns:
        DataFrame with the dataset
    """
    # Check dataset access
    dataset_path = await get_dataset_path(db, dataset_id)
    await check_dataset_access(dataset_id, user_id, db)
    
    # Load dataset based on file extension
    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
    elif dataset_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(dataset_path)
    elif dataset_path.endswith('.json'):
        df = pd.read_json(dataset_path)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {os.path.splitext(dataset_path)[1]}"
        )
    
    return df

async def _load_json_artifact(artifact_path: str) -> Any:
    """
    Load a candidate-pairs artifact (JSON or CSV).

    The Blocking step can save candidate pairs in either format; we support
    both so that downstream steps (Similarity, Classification, etc.) can be
    fed the CSV that users often download for inspection.
    
    Returns whatever python structure downstream expects:
    • JSON -> arbitrary object (usually list[tuple[int,int]])
    • CSV  -> list[Tuple[int,int]] derived from the two id columns
    """
    try:
        # Convert web paths like '/uploads/...' to local filesystem paths
        if artifact_path.startswith('/') and not os.path.exists(artifact_path):
            candidate = os.path.join(os.getcwd(), artifact_path.lstrip('/'))
            if os.path.exists(candidate):
                artifact_path = candidate
        
        if artifact_path.endswith('.csv'):
            import csv
            pairs: list[tuple[int, int]] = []
            with open(artifact_path, newline='') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                # CSV cluster assignments (cluster_id, record_id)
                if 'cluster_id' in headers and 'record_id' in headers:
                    clusters_dict: Dict[int, list[int]] = {}
                    for row in reader:
                        try:
                            cid = int(row.get('cluster_id'))
                            rid = int(row.get('record_id'))
                        except (TypeError, ValueError):
                            continue
                        clusters_dict.setdefault(cid, []).append(rid)
                    # Keep only clusters with more than 1 record (true duplicates)
                    clusters = [members for members in clusters_dict.values() if len(members) > 1]
                    return clusters

                # CSV similarity or classification results
                if 'similarity' in headers or 'confidence' in headers:
                    results: list[dict] = []
                    for row in reader:
                        try:
                            r1 = int(row.get('record1_id') or row.get('id1') or row.get('record_id_1'))
                            r2 = int(row.get('record2_id') or row.get('id2') or row.get('record_id_2'))
                            score_field = 'similarity' if 'similarity' in headers else 'confidence'
                            score = float(row.get(score_field))
                        except (TypeError, ValueError):
                            continue  # malformed row
                        results.append({
                            "record1_id": r1,
                            "record2_id": r2,
                            score_field: score,
                            # Ensure both keys exist for downstream convenience
                            "similarity": score if score_field == 'similarity' else None,
                            "confidence": score if score_field == 'confidence' else None,
                            "field_similarities": {},
                            "record1_data": {},
                            "record2_data": {}
                        })
                    return results
                # Otherwise treat as candidate-pairs CSV (blocking output)
                pairs: list[tuple[int, int]] = []
                # We already consumed header row; DictReader keeps iterator at first row
                for row in reader:
                    try:
                        r1 = int(row.get('record1_id'))
                        r2 = int(row.get('record2_id'))
                        pairs.append((r1, r2))
                    except (TypeError, ValueError):
                        continue
                return pairs

        else:
            with open(artifact_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error loading artifact: {str(e)}"
        )

@router.post("/preprocessing", response_model=PreprocessingResponse)
async def preprocess_data(
    request: PreprocessingRequest,
    session_id: Optional[UUID] = Query(None, description="Optional workflow session id"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Preprocess a dataset for deduplication."""
    _step = None
    try:
        # Optionally create a session step
        if session_id:
            try:
                _step = await session_service.add_step(
                    db,
                    current_user,
                    session_id,
                    tool="deduplication_pipeline",
                    step="preprocessing",
                    substep="basic",
                    algorithm="basic",
                    params={
                        "dataset_id": request.dataset_id,
                        "text_columns": request.text_columns,
                        "numeric_columns": request.numeric_columns,
                        "categorical_columns": request.categorical_columns,
                        "output_name": request.output_name,
                    },
                )
            except Exception as e:
                logger.warning(f"PREPROCESSING: session step create error: {e}")

        # Load dataset from storage (ensures user ownership)
        df = await _load_dataset(request.dataset_id, current_user.id, db)

        # Mark step running
        if _step:
            try:
                await session_service.update_step(db, current_user, _step.id, status="running")
            except Exception as upd_e:
                logger.warning(f"PREPROCESSING: session step running update error: {upd_e}")

        # Run preprocessing service
        result = await run_preprocessing(
            df,
            request.text_columns,
            request.numeric_columns,
            request.categorical_columns,
            request.dataset_id,
            current_user.id,
            request.output_name,
        )

        # Finalize success and return session_step_id
        if _step:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    _step.id,
                    status="success",
                    finished_at=datetime.utcnow(),
                    run_ref_type="deduplication_pipeline",
                    run_ref_id=f"preprocessing:{request.dataset_id}",
                )
            except Exception as upd_e:
                logger.warning(f"PREPROCESSING: session step success update error: {upd_e}")
            result["session_step_id"] = str(_step.id)

        # Validate and return response according to schema
        return PreprocessingResponse(**result)

    except Exception as e:
        # Update session step -> failed
        if _step:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    _step.id,
                    status="failed",
                    error=str(e),
                    finished_at=datetime.utcnow(),
                )
            except Exception as upd_e2:
                logger.warning(f"PREPROCESSING: session step failed update error: {upd_e2}")
        logger.error(f"Error in preprocessing: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in preprocessing: {str(e)}",
        )

@router.post("/blocking", response_model=BlockingResponse)
async def block_dataset(
    request: BlockingRequest,
    session_id: Optional[UUID] = Query(None, description="Optional workflow session id"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generate candidate pairs using blocking techniques.
    
    This endpoint handles the second step of the deduplication pipeline:
    - MinHash LSH blocking
    - SimHash blocking
    """
    _step = None
    try:
        # Create session step (optional)
        if session_id:
            try:
                _step = await session_service.add_step(
                    db,
                    current_user,
                    session_id,
                    tool="deduplication_pipeline",
                    step="blocking",
                    substep=request.method,
                    algorithm=request.method,
                    params={
                        "dataset_id": request.dataset_id,
                        "key_fields": request.key_fields,
                        "params": request.params,
                        "output_name": request.output_name,
                    },
                )
            except Exception as e:
                logger.warning(f"BLOCKING: session step create error: {e}")
        # Load dataset
        df = await _load_dataset(request.dataset_id, current_user.id, db)
        
        # Mark step running
        if _step:
            try:
                await session_service.update_step(db, current_user, _step.id, status="running")
            except Exception as upd_e:
                logger.warning(f"BLOCKING: session step running update error: {upd_e}")
        
        # Run blocking
        result = await run_blocking(
            df,
            request.method,
            request.key_fields,
            request.params,
            request.dataset_id,
            current_user.id,
            request.output_name
        )
        
        # Finish step success and attach session_step_id
        if _step:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    _step.id,
                    status="success",
                    finished_at=datetime.utcnow(),
                    run_ref_type="deduplication_pipeline",
                    run_ref_id=f"blocking:{request.dataset_id}",
                )
            except Exception as upd_e:
                logger.warning(f"BLOCKING: session step success update error: {upd_e}")
            result["session_step_id"] = str(_step.id)
        
        return BlockingResponse(**result)
        
    except Exception as e:
        # Update session step -> failed
        if _step:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    _step.id,
                    status="failed",
                    error=str(e),
                    finished_at=datetime.utcnow(),
                )
            except Exception as upd_e2:
                logger.warning(f"BLOCKING: session step failed update error: {upd_e2}")
        logger.error(f"Error in blocking: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in blocking: {str(e)}"
        )

@router.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity(
    request: SimilarityRequest,
    session_id: Optional[UUID] = Query(None, description="Optional workflow session id"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Calculate similarity between candidate pairs.
    
    This endpoint handles the third step of the deduplication pipeline:
    - Field-specific similarity metrics
    - Composite weighted similarity
    """
    _step = None
    try:
        # Create session step (optional)
        if session_id:
            try:
                _step = await session_service.add_step(
                    db,
                    current_user,
                    session_id,
                    tool="deduplication_pipeline",
                    step="similarity",
                    substep="composite",
                    algorithm="composite",
                    params={
                        "dataset_id": request.dataset_id,
                        "candidate_pairs_path": request.candidate_pairs_path,
                        "field_configs": request.field_configs,
                        "threshold": request.threshold,
                    },
                )
            except Exception as e:
                logger.warning(f"SIMILARITY: session step create error: {e}")
        
        # Load dataset
        df = await _load_dataset(request.dataset_id, current_user.id, db)
        
        # Load candidate pairs
        candidate_pairs = await _load_json_artifact(request.candidate_pairs_path)
        
        # Mark step running
        if _step:
            try:
                await session_service.update_step(db, current_user, _step.id, status="running")
            except Exception as upd_e:
                logger.warning(f"SIMILARITY: session step running update error: {upd_e}")
        
        # Run similarity calculation
        result = await run_similarity_calculation(
            df,
            candidate_pairs,
            request.field_configs,
            request.dataset_id,
            current_user.id,
            request.threshold
        )
        
        # Finish step success and attach session_step_id
        if _step:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    _step.id,
                    status="success",
                    finished_at=datetime.utcnow(),
                    run_ref_type="deduplication_pipeline",
                    run_ref_id=f"similarity:{request.dataset_id}",
                )
            except Exception as upd_e:
                logger.warning(f"SIMILARITY: session step success update error: {upd_e}")
            result["session_step_id"] = str(_step.id)
        
        return SimilarityResponse(**result)
        
    except Exception as e:
        # Update session step -> failed
        if _step:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    _step.id,
                    status="failed",
                    error=str(e),
                    finished_at=datetime.utcnow(),
                )
            except Exception as upd_e2:
                logger.warning(f"SIMILARITY: session step failed update error: {upd_e2}")
        logger.error(f"Error in similarity calculation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in similarity calculation: {str(e)}"
        )

@router.post("/classification", response_model=ClassificationResponse)
async def classify_pairs(
    request: ClassificationRequest,
    session_id: Optional[UUID] = Query(None, description="Optional workflow session id"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Classify candidate pairs as duplicates or non-duplicates.
    
    This endpoint handles the fourth step of the deduplication pipeline:
    - Random Forest classification
    - XGBoost classification
    - Siamese Network classification
    """
    _step = None
    try:
        # Create session step (optional)
        if session_id:
            try:
                _step = await session_service.add_step(
                    db,
                    current_user,
                    session_id,
                    tool="deduplication_pipeline",
                    step="classification",
                    substep=request.method,
                    algorithm=request.method,
                    params={
                        "dataset_id": request.dataset_id,
                        "similarity_results_path": request.similarity_results_path,
                        "method": request.method,
                        "params": request.params,
                    },
                )
            except Exception as e:
                logger.warning(f"CLASSIFICATION: session step create error: {e}")
        
        # Load similarity results
        similarity_results = await _load_json_artifact(request.similarity_results_path)
        
        # Mark step running
        if _step:
            try:
                await session_service.update_step(db, current_user, _step.id, status="running")
            except Exception as upd_e:
                logger.warning(f"CLASSIFICATION: session step running update error: {upd_e}")
        
        # Run classification
        result = await run_classification(
            similarity_results,
            request.method,
            request.params,
            request.dataset_id,
            current_user.id
        )
        
        # Finish step success and attach session_step_id
        if _step:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    _step.id,
                    status="success",
                    finished_at=datetime.utcnow(),
                    run_ref_type="deduplication_pipeline",
                    run_ref_id=f"classification:{request.dataset_id}",
                )
            except Exception as upd_e:
                logger.warning(f"CLASSIFICATION: session step success update error: {upd_e}")
            result["session_step_id"] = str(_step.id)
        
        return ClassificationResponse(**result)
        
    except Exception as e:
        # Update session step -> failed
        if _step:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    _step.id,
                    status="failed",
                    error=str(e),
                    finished_at=datetime.utcnow(),
                )
            except Exception as upd_e2:
                logger.warning(f"CLASSIFICATION: session step failed update error: {upd_e2}")
        logger.error(f"Error in classification: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in classification: {str(e)}"
        )

@router.post("/clustering", response_model=ClusteringResponse)
async def cluster_duplicates(
    request: ClusteringRequest,
    session_id: Optional[UUID] = Query(None, description="Optional workflow session id"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Cluster duplicate records.
    
    This endpoint handles the fifth step of the deduplication pipeline:
    - Graph-based clustering
    - Density-based clustering
    """
    _step = None
    try:
        # Create session step (optional)
        if session_id:
            try:
                _step = await session_service.add_step(
                    db,
                    current_user,
                    session_id,
                    tool="deduplication_pipeline",
                    step="clustering",
                    substep=request.method,
                    algorithm=request.method,
                    params={
                        "dataset_id": request.dataset_id,
                        "classification_results_path": request.classification_results_path,
                        "method": request.method,
                        "params": request.params,
                    },
                )
            except Exception as e:
                logger.warning(f"CLUSTERING: session step create error: {e}")
        
        # Load dataset
        df = await _load_dataset(request.dataset_id, current_user.id, db)
        
        # Load classification results
        classification_results = await _load_json_artifact(request.classification_results_path)
        
        # Mark step running
        if _step:
            try:
                await session_service.update_step(db, current_user, _step.id, status="running")
            except Exception as upd_e:
                logger.warning(f"CLUSTERING: session step running update error: {upd_e}")
        
        # Run clustering
        result = await run_clustering(
            df,
            classification_results,
            request.method,
            request.params,
            request.dataset_id,
            current_user.id
        )
        
        # Finalize success and attach session_step_id
        if _step:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    _step.id,
                    status="success",
                    finished_at=datetime.utcnow(),
                    run_ref_type="deduplication_pipeline",
                    run_ref_id=f"clustering:{request.dataset_id}",
                )
            except Exception as upd_e:
                logger.warning(f"CLUSTERING: session step success update error: {upd_e}")
            result["session_step_id"] = str(_step.id)
        
        return ClusteringResponse(**result)
        
    except Exception as e:
        # Update session step -> failed
        if _step:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    _step.id,
                    status="failed",
                    error=str(e),
                    finished_at=datetime.utcnow(),
                )
            except Exception as upd_e2:
                logger.warning(f"CLUSTERING: session step failed update error: {upd_e2}")
        logger.error(f"Error in clustering: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in clustering: {str(e)}"
        )

@router.post("/resolution", response_model=ResolutionResponse)
async def resolve_duplicates(
    request: ResolutionRequest,
    session_id: Optional[UUID] = Query(None, description="Optional workflow session id"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Resolve duplicate clusters.
    
    This endpoint handles the final step of the deduplication pipeline:
    - Keep First Record
    - Keep Most Complete Record
    - Merge Records
    - Manual Review
    """
    _step = None
    try:
        # Optionally create a session step
        if session_id:
            try:
                _step = await session_service.add_step(
                    db,
                    current_user,
                    session_id,
                    tool="deduplication_pipeline",
                    step="resolution",
                    substep=request.method,
                    algorithm=request.method,
                    params={
                        "dataset_id": request.dataset_id,
                        "clustering_results_path": request.clustering_results_path,
                        "method": request.method,
                        "params": request.params,
                    },
                )
            except Exception as e:
                logger.warning(f"RESOLUTION: session step create error: {e}")
        
        # Load dataset
        df = await _load_dataset(request.dataset_id, current_user.id, db)
        
        # Load clustering results
        clusters = await _load_json_artifact(request.clustering_results_path)
        
        # Mark step running
        if _step:
            try:
                await session_service.update_step(db, current_user, _step.id, status="running")
            except Exception as upd_e:
                logger.warning(f"RESOLUTION: session step running update error: {upd_e}")
        
        # Run resolution
        result = await run_resolution(
            df,
            clusters,
            request.method,
            request.params,
            request.dataset_id,
            current_user.id
        )
        
        # Finalize success and attach session_step_id
        if _step:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    _step.id,
                    status="success",
                    finished_at=datetime.utcnow(),
                    run_ref_type="deduplication_pipeline",
                    run_ref_id=f"resolution:{request.dataset_id}",
                )
            except Exception as upd_e:
                logger.warning(f"RESOLUTION: session step success update error: {upd_e}")
            result["session_step_id"] = str(_step.id)
        
        return ResolutionResponse(**result)
        
    except Exception as e:
        # Update session step -> failed
        if _step:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    _step.id,
                    status="failed",
                    error=str(e),
                    finished_at=datetime.utcnow(),
                )
            except Exception as upd_e2:
                logger.warning(f"RESOLUTION: session step failed update error: {upd_e2}")
        logger.error(f"Error in resolution: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in resolution: {str(e)}"
        )

@router.post("/manual-resolution", response_model=ManualResolutionResponse)
async def apply_manual_resolution_decisions(
    request: ManualResolutionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Apply manual resolution decisions.
    
    This endpoint handles the application of manual resolution decisions:
    - Apply user-selected records to keep for each cluster
    """
    try:
        # Load dataset
        df = await _load_dataset(request.dataset_id, current_user.id, db)
        
        # Load manual review data
        manual_review_data = await _load_json_artifact(request.manual_review_path)
        
        # Apply manual resolution
        resolved_df = apply_manual_resolution(df, request.resolution_decisions)
        
        # Save the resolved DataFrame
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"manually_deduplicated_dataset_{timestamp}.csv"
        
        from app.config.config import get_settings
        settings = get_settings()
        
        from app.services.deduplication.resolution.service import _get_resolution_artifact_path
        output_path = _get_resolution_artifact_path(
            settings.DATASET_UPLOAD_DIR,
            request.dataset_id,
            current_user.id,
            output_filename
        )
        
        resolved_df.to_csv(output_path, index=False)
        
        # Create response
        result = {
            "status": "success",
            "message": "Manual resolution applied successfully",
            "summary": {
                "method": "manual",
                "records_kept": len(resolved_df),
                "records_removed": len(df) - len(resolved_df),
                "output_path": output_path
            },
            "resolved_dataset_path": output_path,
            "records_kept": len(resolved_df),
            "records_removed": len(df) - len(resolved_df)
        }
        
        return ManualResolutionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in manual resolution: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in manual resolution: {str(e)}"
        )

@router.get("/algorithms", response_model=Dict[str, Any])
async def list_pipeline_algorithms():
    """
    List available algorithms for each step of the deduplication pipeline.
    """
    return {
        "preprocessing": {
            "text": ["lowercase_strip"],
            "numeric": ["min_max_scaling"],
            "categorical": ["label_encoding"]
        },
        "blocking": {
            "methods": ["minhash_lsh", "simhash"],
            "description": "Techniques to efficiently generate candidate pairs"
        },
        "similarity": {
            "text": ["jaro_winkler", "levenshtein", "token_sort", "token_set"],
            "numeric": ["normalized_distance", "exact_match"],
            "categorical": ["exact_match", "jaccard"],
            "description": "Field-specific similarity metrics"
        },
        "classification": {
            "methods": ["random_forest", "xgboost", "siamese_network"],
            "description": "ML-based duplicate classification"
        },
        "clustering": {
            "methods": ["graph_connected_components", "graph_community_detection", "dbscan", "optics"],
            "description": "Techniques to group duplicate records"
        },
        "resolution": {
            "methods": ["keep_first", "keep_most_complete", "merge", "manual"],
            "description": "Strategies for handling identified duplicates"
        },
        "legacy": {
            "algorithms": get_legacy_algorithms(),
            "description": "Legacy algorithms retained for comparison purposes"
        }
    }

@router.post("/legacy", response_model=LegacyDeduplicationResponse)
async def run_legacy_algorithm(
    request: LegacyDeduplicationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Run a legacy deduplication algorithm (fuzzy matching or deep ER) in the pipeline structure.
    """
    try:
        # Check dataset access
        dataset_path = await check_dataset_access(request.dataset_id, current_user.id, db)
        
        # Load dataset
        df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else \
             pd.read_excel(dataset_path) if dataset_path.endswith(('.xls', '.xlsx')) else \
             pd.read_json(dataset_path) if dataset_path.endswith('.json') else None
        
        if df is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file format"
            )
        
        # Run legacy algorithm
        result = await run_legacy_deduplication(
            df=df,
            algorithm=request.algorithm,
            params=request.params,
            dataset_id=request.dataset_id,
            user_id=current_user.id
        )
        
        if result.get("status") == "error":
            return LegacyDeduplicationResponse(
                status="error",
                message=result.get("message", "Error running legacy algorithm"),
                error=result.get("error")
            )
        
        return LegacyDeduplicationResponse(
            status="success",
            message=result.get("message", "Legacy algorithm completed successfully"),
            results_path=result.get("results_path"),
            num_duplicates=result.get("num_duplicates"),
            num_clusters=result.get("num_clusters"),
            preview=result.get("preview"),
            summary={
                "algorithm": request.algorithm,
                "params": request.params,
                **{k: v for k, v in result.items() if k not in ["preview", "status", "message", "error"]}
            }
        )
        
    except Exception as e:
        logger.error(f"Error running legacy algorithm: {str(e)}", exc_info=True)
        return LegacyDeduplicationResponse(
            status="error",
            message="Error running legacy algorithm",
            error=str(e)
        )
