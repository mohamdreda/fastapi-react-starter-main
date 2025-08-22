"""
Deep Entity Resolution deduplication algorithm.

This module contains the legacy deep entity resolution algorithm that has been retained
for comparison purposes with the new modular deduplication pipeline.
"""
import pandas as pd
from typing import Dict, Any, List, Tuple, Set, Optional
import numpy as np
import logging
import os
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

def _get_deep_er_artifact_path(
    base_path: str,
    dataset_id: int,
    user_id: int,
    artifact_name: str
) -> str:
    """Create and return the path for deep ER artifacts."""
    dir_path = os.path.join(base_path, f"user_{user_id}", f"dataset_{dataset_id}", "deduplication", "legacy")
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, artifact_name)

def deep_er_deduplication(
    df: pd.DataFrame,
    params: Dict[str, Any],
    dataset_id: int,
    user_id: int
) -> Dict[str, Any]:
    """
    Simulates a Deep Entity Resolution approach using text embeddings and similarity.
    This is a simplified version that:
    1. Converts text data to TF-IDF vectors
    2. Computes cosine similarity between records
    3. Identifies pairs above a threshold as duplicates
    
    Note: A true Deep ER would use neural networks like BERT/transformers
    
    Args:
        df: Input DataFrame
        params: Algorithm parameters including:
            - threshold: Similarity threshold (default: 0.8)
            - text_columns: List of text columns to use (default: all string columns)
            - max_features: Maximum number of features for TF-IDF (default: 5000)
            - ngram_range: N-gram range for TF-IDF (default: (1, 2))
            - max_pairs: Maximum number of pairs to compare (default: 10000)
        dataset_id: ID of the dataset
        user_id: ID of the user
        
    Returns:
        Dictionary with results including potential duplicates
    """
    try:
        # Extract parameters with defaults
        threshold = params.get('threshold', 0.8)
        text_columns = params.get('text_columns', None)
        max_features = params.get('max_features', 5000)
        ngram_range = params.get('ngram_range', (1, 2))
        max_pairs = params.get('max_pairs', 10000)
        
        # If no text columns specified, use all string columns
        if not text_columns:
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Ensure we have at least one column to compare
        if not text_columns:
            return {
                "status": "error",
                "message": "No text columns found for comparison"
            }
        
        # Combine text columns into a single text field
        df['combined_text'] = df[text_columns].fillna('').astype(str).apply(lambda x: ' '.join(x), axis=1)
        
        # Limit the number of comparisons to avoid excessive computation
        n_rows = len(df)
        n_comparisons = (n_rows * (n_rows - 1)) // 2
        
        if n_comparisons > max_pairs:
            logger.warning(f"Large dataset detected ({n_rows} rows, {n_comparisons} potential comparisons). "
                          f"Limiting to {max_pairs} comparisons.")
            # Sample a subset of rows for comparison
            sample_size = int(np.sqrt(2 * max_pairs))
            sample_indices = np.random.choice(n_rows, min(sample_size, n_rows), replace=False)
            df_sample = df.iloc[sample_indices].reset_index(drop=True)
        else:
            df_sample = df.copy()
            sample_indices = list(range(n_rows))
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        
        # Fit and transform the combined text
        tfidf_matrix = vectorizer.fit_transform(df_sample['combined_text'])
        
        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find duplicate pairs
        duplicate_pairs = []
        
        for i in range(len(df_sample)):
            for j in range(i + 1, len(df_sample)):
                similarity = similarity_matrix[i, j]
                
                if similarity >= threshold:
                    # Get original indices if we sampled
                    orig_i = sample_indices[i]
                    orig_j = sample_indices[j]
                    
                    duplicate_pairs.append({
                        "index1": int(orig_i),
                        "index2": int(orig_j),
                        "similarity": float(similarity),
                        "record1": df.iloc[orig_i].to_dict(),
                        "record2": df.iloc[orig_j].to_dict()
                    })
        
        # Create clusters from duplicate pairs
        clusters = []
        index_to_cluster = {}
        
        for pair in duplicate_pairs:
            idx1, idx2 = pair["index1"], pair["index2"]
            
            # Check if either index is already in a cluster
            cluster1 = index_to_cluster.get(idx1)
            cluster2 = index_to_cluster.get(idx2)
            
            if cluster1 is not None and cluster2 is not None:
                # Both are in clusters, merge if different
                if cluster1 != cluster2:
                    clusters[cluster1].extend(clusters[cluster2])
                    # Update index_to_cluster for all indices in cluster2
                    for idx in clusters[cluster2]:
                        index_to_cluster[idx] = cluster1
                    # Mark cluster2 as empty
                    clusters[cluster2] = []
            elif cluster1 is not None:
                # idx1 is in a cluster, add idx2
                clusters[cluster1].append(idx2)
                index_to_cluster[idx2] = cluster1
            elif cluster2 is not None:
                # idx2 is in a cluster, add idx1
                clusters[cluster2].append(idx1)
                index_to_cluster[idx1] = cluster2
            else:
                # Neither is in a cluster, create new cluster
                clusters.append([idx1, idx2])
                index_to_cluster[idx1] = len(clusters) - 1
                index_to_cluster[idx2] = len(clusters) - 1
        
        # Remove empty clusters
        clusters = [c for c in clusters if c]
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"deep_er_results_{timestamp}.json"
        
        results_path = _get_deep_er_artifact_path(
            settings.DATASET_UPLOAD_DIR,
            dataset_id,
            user_id,
            results_filename
        )
        
        # Prepare results for saving
        results_data = {
            "algorithm": "deep_er",
            "params": params,
            "duplicate_pairs": duplicate_pairs,
            "clusters": clusters,
            "timestamp": timestamp
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_data, f)
        
        # Create a preview of results
        preview = duplicate_pairs[:10] if duplicate_pairs else []
        
        return {
            "status": "success",
            "message": "Deep ER deduplication completed successfully",
            "num_duplicates": len(duplicate_pairs),
            "num_clusters": len(clusters),
            "preview": preview,
            "results_path": results_path
        }
        
    except Exception as e:
        logger.error(f"Error in Deep ER deduplication: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error in Deep ER deduplication: {str(e)}",
            "error": str(e)
        }
