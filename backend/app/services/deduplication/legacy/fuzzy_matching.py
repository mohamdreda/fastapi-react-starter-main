"""
Fuzzy matching deduplication algorithm.

This module contains the legacy fuzzy matching algorithm that has been retained
for comparison purposes with the new modular deduplication pipeline.
"""
import pandas as pd
from typing import Dict, Any, List, Tuple, Set, Optional
import numpy as np
import logging
import os
import json
from datetime import datetime

# Try to import thefuzz, but provide a fallback implementation if not available
try:
    from thefuzz import fuzz
    HAS_THEFUZZ = True
except ImportError:
    # Simple Levenshtein distance implementation as fallback
    HAS_THEFUZZ = False
    
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
    
    # Simple ratio calculation as fallback for fuzz.ratio
    def ratio(s1, s2):
        if not s1 and not s2:
            return 100.0
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 100.0
        distance = levenshtein_distance(s1, s2)
        similarity = 1.0 - (distance / max_len)
        return similarity * 100.0
    
    # Create a fuzz module with a ratio function
    class FuzzFallback:
        @staticmethod
        def ratio(s1, s2):
            return ratio(s1, s2)
        
        @staticmethod
        def partial_ratio(s1, s2):
            # Simplified implementation
            return ratio(s1, s2)
        
        @staticmethod
        def token_sort_ratio(s1, s2):
            # Sort tokens and compare
            s1_sorted = ' '.join(sorted(s1.lower().split()))
            s2_sorted = ' '.join(sorted(s2.lower().split()))
            return ratio(s1_sorted, s2_sorted)
        
        @staticmethod
        def token_set_ratio(s1, s2):
            # Compare token sets
            s1_tokens = set(s1.lower().split())
            s2_tokens = set(s2.lower().split())
            
            # Intersection and differences
            intersection = s1_tokens.intersection(s2_tokens)
            diff1 = s1_tokens - intersection
            diff2 = s2_tokens - intersection
            
            # Convert sets back to strings
            intersection_str = ' '.join(sorted(intersection))
            diff1_str = ' '.join(sorted(diff1))
            diff2_str = ' '.join(sorted(diff2))
            
            # Calculate ratios
            base_ratio = ratio(intersection_str, intersection_str)
            
            if not diff1_str and not diff2_str:
                return base_ratio
                
            combined1 = f"{intersection_str} {diff1_str}".strip()
            combined2 = f"{intersection_str} {diff2_str}".strip()
            
            r1 = ratio(combined1, combined2)
            
            return max(base_ratio, r1)
    
    fuzz = FuzzFallback()

from app.config.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

def _get_fuzzy_artifact_path(
    base_path: str,
    dataset_id: int,
    user_id: int,
    artifact_name: str
) -> str:
    """Create and return the path for fuzzy matching artifacts."""
    dir_path = os.path.join(base_path, f"user_{user_id}", f"dataset_{dataset_id}", "deduplication", "legacy")
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, artifact_name)

def fuzzy_matching_deduplication(
    df: pd.DataFrame,
    params: Dict[str, Any],
    dataset_id: int,
    user_id: int
) -> Dict[str, Any]:
    """
    Detects duplicates using fuzzy matching.
    NOTE: This implementation is basic and can be very slow (O(n^2)) on large datasets.
    It compares string columns based on the provided parameters.
    
    Args:
        df: Input DataFrame
        params: Algorithm parameters including:
            - threshold: Similarity threshold (default: 80)
            - columns: List of columns to compare (default: all string columns)
            - method: Fuzzy matching method ('ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio')
            - max_pairs: Maximum number of pairs to compare (default: 10000)
        dataset_id: ID of the dataset
        user_id: ID of the user
        
    Returns:
        Dictionary with results including potential duplicates
    """
    try:
        # Extract parameters with defaults
        threshold = params.get('threshold', 80)
        columns = params.get('columns', None)
        method = params.get('method', 'token_sort_ratio')
        max_pairs = params.get('max_pairs', 10000)
        
        # If no columns specified, use all string columns
        if not columns:
            columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Ensure we have at least one column to compare
        if not columns:
            return {
                "status": "error",
                "message": "No string columns found for comparison"
            }
        
        # Select the appropriate fuzzy matching method
        if method == 'ratio':
            fuzzy_func = fuzz.ratio
        elif method == 'partial_ratio':
            fuzzy_func = fuzz.partial_ratio
        elif method == 'token_sort_ratio':
            fuzzy_func = fuzz.token_sort_ratio
        elif method == 'token_set_ratio':
            fuzzy_func = fuzz.token_set_ratio
        else:
            fuzzy_func = fuzz.token_sort_ratio  # Default
        
        # Initialize results
        duplicate_pairs = []
        
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
        
        # Compare each pair of records
        for i in range(len(df_sample)):
            for j in range(i + 1, len(df_sample)):
                # Calculate similarity for each column
                similarities = []
                
                for col in columns:
                    # Skip if either value is missing
                    if pd.isna(df_sample.iloc[i][col]) or pd.isna(df_sample.iloc[j][col]):
                        continue
                    
                    # Convert to string if needed
                    val1 = str(df_sample.iloc[i][col])
                    val2 = str(df_sample.iloc[j][col])
                    
                    # Skip empty strings
                    if not val1 or not val2:
                        continue
                    
                    # Calculate similarity
                    similarity = fuzzy_func(val1, val2)
                    similarities.append(similarity)
                
                # Calculate average similarity if we have any valid comparisons
                if similarities:
                    avg_similarity = sum(similarities) / len(similarities)
                    
                    # If above threshold, add to duplicate pairs
                    if avg_similarity >= threshold:
                        # Get original indices if we sampled
                        if n_comparisons > max_pairs:
                            orig_i = sample_indices[i]
                            orig_j = sample_indices[j]
                        else:
                            orig_i = i
                            orig_j = j
                            
                        duplicate_pairs.append({
                            "index1": int(orig_i),
                            "index2": int(orig_j),
                            "similarity": float(avg_similarity),
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
        results_filename = f"fuzzy_matching_results_{timestamp}.json"
        
        results_path = _get_fuzzy_artifact_path(
            settings.DATASET_UPLOAD_DIR,
            dataset_id,
            user_id,
            results_filename
        )
        
        # Prepare results for saving
        results_data = {
            "algorithm": "fuzzy_matching",
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
            "message": "Fuzzy matching deduplication completed successfully",
            "num_duplicates": len(duplicate_pairs),
            "num_clusters": len(clusters),
            "preview": preview,
            "results_path": results_path
        }
        
    except Exception as e:
        logger.error(f"Error in fuzzy matching deduplication: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error in fuzzy matching deduplication: {str(e)}",
            "error": str(e)
        }
