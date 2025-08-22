import pandas as pd
import os
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException
import json
from datetime import datetime

from app.schemas.deduplication_pipeline import LegacyDeduplicationRequest as DeduplicationRequest, LegacyDeduplicationResponse as DeduplicationResponse
from app.crud.crud_dataset import get as get_dataset
from app.services.deduplication.legacy.fuzzy_matching import fuzzy_matching_deduplication
from app.services.deduplication.legacy.deep_er import deep_er_deduplication
from app.services.deduplication.legacy.service import get_legacy_algorithms
from app.config.config import Settings

settings = Settings()

logger = logging.getLogger(__name__)

async def run_deduplication(db: AsyncSession, request: DeduplicationRequest, user_id: int) -> DeduplicationResponse:
    """
    Run deduplication on a dataset using the specified algorithm and parameters.
    Can optionally remove duplicates and save a cleaned dataset.
    
    Args:
        db: Database session
        request: Deduplication request with dataset_id, algorithm and parameters
        user_id: ID of the current user
        
    Returns:
        DeduplicationResponse with status, message, number of duplicates and result preview
        If remove_duplicates is True, also includes cleaned_dataset_path
    """
    # Get dataset
    dataset = await get_dataset(db, id=request.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Check if user has access to dataset
    if dataset.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this dataset")
    
    # Get dataset path
    dataset_path = dataset.file_path
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found")
    
    try:
        # Load dataset
        file_extension = os.path.splitext(dataset_path)[1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(dataset_path)
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(dataset_path)
        elif file_extension == '.json':
            df = pd.read_json(dataset_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")
        
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Run appropriate deduplication algorithm
        if request.algorithm.lower() in ['fuzzy', 'fuzzy_matching']:
            result = fuzzy_matching_deduplication(df, request.params, dataset.id, user_id)
        elif request.algorithm.lower() in ['deep', 'deep_er']:
            result = deep_er_deduplication(df, request.params, dataset.id, user_id)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown algorithm: {request.algorithm}")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Make sure the base directory exists
        os.makedirs(settings.DATASET_UPLOAD_DIR, exist_ok=True)
        # Create the results directory for this dataset
        results_dir = os.path.join(settings.DATASET_UPLOAD_DIR, f"deduplication_results/{dataset.id}")
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, f"dedup_{request.algorithm}_{timestamp}.json")
        cleaned_dataset_path = None
        
        # Handle duplicate removal if requested
        if request.remove_duplicates and result.get("result_preview") and len(result.get("result_preview", [])) > 0:
            logger.info(f"Removing {result.get('num_duplicates', 0)} duplicates from dataset")
            
            # Get the IDs of records to keep (remove the second record in each duplicate pair)
            duplicate_ids_to_remove = set()
            for duplicate_pair in result.get("result_preview", []):
                # Add the second record's ID to the set of IDs to remove
                if "record2" in duplicate_pair and "id" in duplicate_pair["record2"]:
                    duplicate_ids_to_remove.add(duplicate_pair["record2"]["id"])
            
            # Filter the dataframe to keep only non-duplicate records
            if "id" in df.columns:
                cleaned_df = df[~df["id"].isin(duplicate_ids_to_remove)]
                
                # Save the cleaned dataset with custom filename if provided
                if request.output_filename:
                    # Ensure the filename has the correct extension
                    base_name = request.output_filename
                    file_extension = os.path.splitext(dataset_path)[1].lower()
                    if not base_name.endswith(file_extension):
                        base_name = f"{base_name}{file_extension}"
                    cleaned_dataset_filename = base_name
                else:
                    cleaned_dataset_filename = f"cleaned_{os.path.basename(dataset_path)}"
                cleaned_dataset_path = os.path.join(results_dir, cleaned_dataset_filename)
                
                # Save in the same format as the original
                file_extension = os.path.splitext(dataset_path)[1].lower()
                if file_extension == '.csv':
                    cleaned_df.to_csv(cleaned_dataset_path, index=False)
                elif file_extension in ['.xls', '.xlsx']:
                    cleaned_df.to_excel(cleaned_dataset_path, index=False)
                elif file_extension == '.json':
                    cleaned_df.to_json(cleaned_dataset_path, orient='records')
                
                logger.info(f"Cleaned dataset saved to {cleaned_dataset_path}")
                
                # Update the summary with information about removed duplicates
                summary = {
                    "dataset_id": dataset.id,
                    "algorithm": request.algorithm,
                    "params": request.params,
                    "num_duplicates": result.get("num_duplicates", 0),
                    "duplicates_removed": True,
                    "num_records_before": len(df),
                    "num_records_after": len(cleaned_df),
                    "cleaned_dataset_path": cleaned_dataset_path,
                    "timestamp": timestamp,
                    "message": result.get("message", "")
                }
            else:
                logger.warning("Cannot remove duplicates: 'id' column not found in dataset")
                summary = {
                    "dataset_id": dataset.id,
                    "algorithm": request.algorithm,
                    "params": request.params,
                    "num_duplicates": result.get("num_duplicates", 0),
                    "duplicates_removed": False,
                    "error": "'id' column not found in dataset",
                    "timestamp": timestamp,
                    "message": result.get("message", "")
                }
        else:
            # Standard summary without duplicate removal
            summary = {
                "dataset_id": dataset.id,
                "algorithm": request.algorithm,
                "params": request.params,
                "num_duplicates": result.get("num_duplicates", 0),
                "duplicates_removed": False,
                "timestamp": timestamp,
                "message": result.get("message", "")
            }
        
        # Save summary results
        with open(results_path, 'w') as f:
            json.dump(summary, f)
        
        logger.info(f"Deduplication results saved to {results_path}")
        
        return DeduplicationResponse(
            status="success",
            message=result.get("message", "Deduplication process completed."),
            num_duplicates=result.get("num_duplicates"),
            result_preview=result.get("result_preview"),
            duplicates_removed=summary.get("duplicates_removed", False),
            cleaned_dataset_path=cleaned_dataset_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in deduplication: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Deduplication failed: {str(e)}")
