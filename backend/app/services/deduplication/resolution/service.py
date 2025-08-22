"""
Resolution service for data deduplication.

This module implements strategies for handling identified duplicate clusters:
- Keep First Record: Keep only the first record in each cluster
- Keep Most Complete Record: Keep the record with fewest missing values
- Merge Records: Create a new record by combining values from duplicates
- Manual Review: Tools for user-guided resolution
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Set, Optional
import os
import json
import logging
from datetime import datetime

from app.config.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

def _get_resolution_artifact_path(
    base_path: str,
    dataset_id: int,
    user_id: int,
    artifact_name: str
) -> str:
    """Create and return the path for resolution artifacts."""
    dir_path = os.path.join(base_path, f"user_{user_id}", f"dataset_{dataset_id}", "deduplication", "resolution")
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, artifact_name)

def keep_first_record(
    df: pd.DataFrame,
    clusters: List[List[int]]
) -> pd.DataFrame:
    """
    Keep only the first record in each cluster.
    
    Args:
        df: Input DataFrame
        clusters: List of clusters, where each cluster is a list of record IDs
        
    Returns:
        DataFrame with duplicates removed
    """
    # Create a set of records to remove
    records_to_remove = set()
    
    for cluster in clusters:
        # Sort record IDs to ensure consistent results
        sorted_cluster = sorted(cluster)
        
        # Keep the first record, remove the rest
        records_to_remove.update(sorted_cluster[1:])
    
    # Create a mask for records to keep
    mask = ~df.index.isin(records_to_remove)
    
    # Return filtered DataFrame
    return df[mask].copy()

def keep_most_complete_record(
    df: pd.DataFrame,
    clusters: List[List[int]]
) -> pd.DataFrame:
    """
    Keep the record with the fewest missing values in each cluster.
    
    Args:
        df: Input DataFrame
        clusters: List of clusters, where each cluster is a list of record IDs
        
    Returns:
        DataFrame with duplicates removed
    """
    # Create a set of records to remove
    records_to_remove = set()
    
    for cluster in clusters:
        # Calculate number of missing values for each record
        missing_counts = {}
        for record_id in cluster:
            if record_id < len(df):
                missing_counts[record_id] = df.iloc[record_id].isna().sum()
        
        if not missing_counts:
            continue
        
        # Find the record with the fewest missing values
        best_record = min(missing_counts.items(), key=lambda x: x[1])[0]
        
        # Remove all other records in the cluster
        records_to_remove.update([r for r in cluster if r != best_record])
    
    # Create a mask for records to keep
    mask = ~df.index.isin(records_to_remove)
    
    # Return filtered DataFrame
    return df[mask].copy()

def merge_records(
    df: pd.DataFrame,
    clusters: List[List[int]]
) -> pd.DataFrame:
    """
    Create a new record by combining values from duplicates.
    
    Args:
        df: Input DataFrame
        clusters: List of clusters, where each cluster is a list of record IDs
        
    Returns:
        DataFrame with merged records
    """
    # Create a copy of the DataFrame
    result_df = df.copy()
    
    # Create a set of records to remove
    records_to_remove = set()
    
    # Process each cluster
    for cluster in clusters:
        # Skip clusters with only one record
        if len(cluster) <= 1:
            continue
        
        # Get records in the cluster
        cluster_records = [df.iloc[record_id] for record_id in cluster if record_id < len(df)]
        
        if not cluster_records:
            continue
        
        # Create a merged record
        merged_record = {}
        
        # Process each column
        for column in df.columns:
            # Get non-null values for this column
            values = [record[column] for record in cluster_records if pd.notna(record[column])]
            
            if values:
                # Use the most common value
                if column == 'id' or column.endswith('_id'):
                    # For ID columns, use the first ID
                    merged_record[column] = values[0]
                else:
                    # For other columns, use the most common value
                    value_counts = pd.Series(values).value_counts()
                    merged_record[column] = value_counts.index[0]
            else:
                # If all values are null, use null
                merged_record[column] = np.nan
        
        # Add the merged record to the result DataFrame
        # Use the first record's index as the location for the merged record
        first_record_id = cluster[0]
        if first_record_id < len(result_df):
            result_df.iloc[first_record_id] = pd.Series(merged_record)
            
            # Mark other records for removal
            records_to_remove.update(cluster[1:])
    
    # Create a mask for records to keep
    mask = ~result_df.index.isin(records_to_remove)
    
    # Return filtered DataFrame
    return result_df[mask].copy()

def prepare_manual_review(
    df: pd.DataFrame,
    clusters: List[List[int]]
) -> Dict[str, Any]:
    """
    Prepare data for manual review of duplicates.
    
    Args:
        df: Input DataFrame
        clusters: List of clusters, where each cluster is a list of record IDs
        
    Returns:
        Dictionary with data for manual review
    """
    # Create a list of clusters with record data
    clusters_with_data = []
    
    for i, cluster in enumerate(clusters):
        cluster_info = {
            "cluster_id": i,
            "records": []
        }
        
        # Add records in the cluster
        for record_id in cluster:
            if record_id < len(df):
                record_data = df.iloc[record_id].to_dict()
                cluster_info["records"].append({
                    "record_id": int(record_id),
                    "data": record_data
                })
        
        clusters_with_data.append(cluster_info)
    
    return {
        "total_clusters": len(clusters_with_data),
        "total_records": sum(len(cluster["records"]) for cluster in clusters_with_data),
        "clusters": clusters_with_data
    }

def apply_manual_resolution(
    df: pd.DataFrame,
    resolution_decisions: Dict[int, List[int]]
) -> pd.DataFrame:
    """
    Apply manual resolution decisions.
    
    Args:
        df: Input DataFrame
        resolution_decisions: Dictionary mapping cluster IDs to lists of record IDs to keep
        
    Returns:
        DataFrame with manually resolved duplicates
    """
    # Create a set of all records
    all_records = set(range(len(df)))
    
    # Create a set of records to keep
    records_to_keep = set()
    
    # Add records to keep based on resolution decisions
    for cluster_id, record_ids in resolution_decisions.items():
        records_to_keep.update(record_ids)
    
    # Calculate records to remove
    records_to_remove = all_records - records_to_keep
    
    # Create a mask for records to keep
    mask = ~df.index.isin(records_to_remove)
    
    # Return filtered DataFrame
    return df[mask].copy()

async def run_resolution(
    df: pd.DataFrame,
    clusters: List[List[int]],
    method: str,
    params: Dict[str, Any],
    dataset_id: int,
    user_id: int
) -> Dict[str, Any]:
    """
    Run resolution on duplicate clusters.
    
    Args:
        df: Input DataFrame
        clusters: List of clusters, where each cluster is a list of record IDs
        method: Resolution method ('keep_first', 'keep_most_complete', 'merge', 'manual')
        params: Parameters for the resolution method
        dataset_id: ID of the dataset
        user_id: ID of the user
        
    Returns:
        Dictionary with resolution results and metadata
    """
    try:
        # Run the appropriate resolution method
        if method == 'keep_first':
            # Keep the first record in each cluster
            resolved_df = keep_first_record(df, clusters)
            
            # Create a summary
            summary = {
                "method": "keep_first",
                "total_clusters": len(clusters),
                "total_records": len(df),
                "records_kept": len(resolved_df),
                "records_removed": len(df) - len(resolved_df)
            }
            
        elif method == 'keep_most_complete':
            # Keep the record with the fewest missing values
            resolved_df = keep_most_complete_record(df, clusters)
            
            # Create a summary
            summary = {
                "method": "keep_most_complete",
                "total_clusters": len(clusters),
                "total_records": len(df),
                "records_kept": len(resolved_df),
                "records_removed": len(df) - len(resolved_df)
            }
            
        elif method == 'merge':
            # Merge records in each cluster
            resolved_df = merge_records(df, clusters)
            
            # Create a summary
            summary = {
                "method": "merge",
                "total_clusters": len(clusters),
                "total_records": len(df),
                "records_kept": len(resolved_df),
                "records_removed": len(df) - len(resolved_df)
            }
            
        elif method == 'manual':
            # For manual resolution, we just prepare the data
            # The actual resolution will be done in a separate step
            manual_review_data = prepare_manual_review(df, clusters)
            
            # Save the manual review data
            manual_review_path = _get_resolution_artifact_path(
                settings.DATASET_UPLOAD_DIR,
                dataset_id,
                user_id,
                "manual_review_data.json"
            )
            
            with open(manual_review_path, 'w') as f:
                json.dump(manual_review_data, f)
            
            # Create a summary
            summary = {
                "method": "manual",
                "total_clusters": len(clusters),
                "total_records": len(df),
                "manual_review_path": manual_review_path
            }
            
            # Return early for manual resolution
            return {
                "status": "success",
                "message": "Manual review data prepared successfully",
                "summary": summary,
                "manual_review_path": manual_review_path,
                "manual_review_data": manual_review_data
            }
            
        else:
            return {
                "status": "error",
                "message": f"Unknown resolution method: {method}"
            }
        
        # Save the resolved DataFrame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"deduplicated_dataset_{timestamp}.csv"
        
        output_path = _get_resolution_artifact_path(
            settings.DATASET_UPLOAD_DIR,
            dataset_id,
            user_id,
            output_filename
        )
        
        resolved_df.to_csv(output_path, index=False)
        
        # Add output path to summary
        summary["output_path"] = output_path
        
        return {
            "status": "success",
            "message": "Resolution completed successfully",
            "summary": summary,
            "resolved_dataset_path": output_path,
            "records_kept": len(resolved_df),
            "records_removed": len(df) - len(resolved_df)
        }
        
    except Exception as e:
        logger.error(f"Error in resolution: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error in resolution: {str(e)}",
            "error": str(e)
        }
