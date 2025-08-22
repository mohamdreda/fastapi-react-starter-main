"""
Similarity calculation service for data deduplication.

This module implements field-specific similarity metrics:
- Text similarity: Jaro-Winkler, TF-IDF + Cosine
- Numeric similarity: Normalized distance
- Categorical similarity: Exact match, Jaccard similarity
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Set, Optional
import os
import json
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

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
            return ratio(s1, s2)  # Simplified fallback
        
        @staticmethod
        def token_sort_ratio(s1, s2):
            # Sort tokens and then compare
            s1_sorted = ' '.join(sorted(s1.split()))
            s2_sorted = ' '.join(sorted(s2.split()))
            return ratio(s1_sorted, s2_sorted)
        
        @staticmethod
        def token_set_ratio(s1, s2):
            # Compare sets of tokens
            set1 = set(s1.split())
            set2 = set(s2.split())
            
            # Calculate Jaccard similarity
            if not set1 and not set2:
                return 100.0
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            return (intersection / union) * 100.0
    
    fuzz = FuzzFallback()

from app.config.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

def _get_similarity_artifact_path(
    base_path: str,
    dataset_id: int,
    user_id: int,
    artifact_name: str
) -> str:
    """Create and return the path for similarity artifacts."""
    dir_path = os.path.join(base_path, f"user_{user_id}", f"dataset_{dataset_id}", "deduplication", "similarity")
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, artifact_name)

def calculate_text_similarity(
    value1: str,
    value2: str,
    method: str = 'jaro_winkler'
) -> float:
    """
    Calculate similarity between two text values.
    
    Args:
        value1: First text value
        value2: Second text value
        method: Similarity method ('jaro_winkler', 'levenshtein', 'token_sort', 'token_set')
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Handle missing values
    if pd.isna(value1) or pd.isna(value2):
        return 0.0
    
    # Convert to string
    str1 = str(value1).lower().strip()
    str2 = str(value2).lower().strip()
    
    # If both strings are empty, they're identical
    if not str1 and not str2:
        return 1.0
    
    # If one string is empty, they're completely different
    if not str1 or not str2:
        return 0.0
    
    # Calculate similarity based on method
    if method == 'jaro_winkler':
        # Use thefuzz's implementation of Jaro-Winkler
        return fuzz.ratio(str1, str2) / 100.0
    
    elif method == 'levenshtein':
        # Normalized Levenshtein distance
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0
        
        if HAS_THEFUZZ:
            distance = 100 - fuzz.ratio(str1, str2)
        else:
            distance = levenshtein_distance(str1, str2)
            
        return 1.0 - (distance / max_len)
    
    elif method == 'token_sort':
        # Sort tokens and then compare
        return fuzz.token_sort_ratio(str1, str2) / 100.0
    
    elif method == 'token_set':
        # Compare sets of tokens
        return fuzz.token_set_ratio(str1, str2) / 100.0
    
    else:
        # Default to Jaro-Winkler
        return fuzz.ratio(str1, str2) / 100.0

def calculate_numeric_similarity(
    value1: float,
    value2: float,
    method: str = 'normalized_distance'
) -> float:
    """
    Calculate similarity between two numeric values.
    
    Args:
        value1: First numeric value
        value2: Second numeric value
        method: Similarity method ('normalized_distance', 'exact_match')
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Handle missing values
    if pd.isna(value1) or pd.isna(value2):
        return 0.0
    
    # Convert to float
    try:
        num1 = float(value1)
        num2 = float(value2)
    except (ValueError, TypeError):
        return 0.0
    
    # Calculate similarity based on method
    if method == 'normalized_distance':
        # Calculate absolute difference
        diff = abs(num1 - num2)
        
        # Normalize by the maximum value
        max_val = max(abs(num1), abs(num2))
        
        if max_val == 0:
            return 1.0  # Both values are 0
        
        # Convert to similarity (1.0 - normalized_difference)
        return max(0.0, 1.0 - (diff / max_val))
    
    elif method == 'exact_match':
        # 1.0 if values are equal, 0.0 otherwise
        return 1.0 if num1 == num2 else 0.0
    
    else:
        # Default to normalized distance
        diff = abs(num1 - num2)
        max_val = max(abs(num1), abs(num2))
        
        if max_val == 0:
            return 1.0
        
        return max(0.0, 1.0 - (diff / max_val))

def calculate_categorical_similarity(
    value1: Any,
    value2: Any,
    method: str = 'exact_match'
) -> float:
    """
    Calculate similarity between two categorical values.
    
    Args:
        value1: First categorical value
        value2: Second categorical value
        method: Similarity method ('exact_match', 'jaccard')
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Handle missing values
    if pd.isna(value1) or pd.isna(value2):
        return 0.0
    
    # Convert to string
    str1 = str(value1).lower().strip()
    str2 = str(value2).lower().strip()
    
    # Calculate similarity based on method
    if method == 'exact_match':
        # 1.0 if values are equal, 0.0 otherwise
        return 1.0 if str1 == str2 else 0.0
    
    elif method == 'jaccard':
        # Split into tokens and calculate Jaccard similarity
        tokens1 = set(str1.split())
        tokens2 = set(str2.split())
        
        # If both sets are empty, they're identical
        if not tokens1 and not tokens2:
            return 1.0
        
        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union
    
    else:
        # Default to exact match
        return 1.0 if str1 == str2 else 0.0

def calculate_composite_similarity(
    record1: Dict[str, Any],
    record2: Dict[str, Any],
    field_configs: Dict[str, Dict[str, Any]]
) -> float:
    """
    Calculate composite similarity between two records based on field configurations.
    
    Args:
        record1: First record as dictionary
        record2: Second record as dictionary
        field_configs: Configuration for each field with type, method, and weight
        
    Returns:
        Composite similarity score (0.0 to 1.0)
    """
    total_weight = 0.0
    weighted_sum = 0.0
    
    for field, config in field_configs.items():
        # Skip if field is not in both records
        if field not in record1 or field not in record2:
            continue
        
        field_type = config.get('type', 'text')
        method = config.get('method', None)
        weight = config.get('weight', 1.0)
        
        # Calculate field similarity based on type
        if field_type == 'text':
            default_method = 'jaro_winkler'
            similarity = calculate_text_similarity(
                record1[field], 
                record2[field], 
                method=method if method else default_method
            )
        
        elif field_type == 'numeric':
            default_method = 'normalized_distance'
            similarity = calculate_numeric_similarity(
                record1[field], 
                record2[field], 
                method=method if method else default_method
            )
        
        elif field_type == 'categorical':
            default_method = 'exact_match'
            similarity = calculate_categorical_similarity(
                record1[field], 
                record2[field], 
                method=method if method else default_method
            )
        
        else:
            # Default to text similarity
            similarity = calculate_text_similarity(
                record1[field], 
                record2[field]
            )
        
        # Add to weighted sum
        weighted_sum += similarity * weight
        total_weight += weight
    
    # Return normalized similarity
    if total_weight > 0:
        return weighted_sum / total_weight
    else:
        return 0.0

async def run_similarity_calculation(
    df: pd.DataFrame,
    candidate_pairs: List[List[int]],
    field_configs: Dict[str, Dict[str, Any]],
    dataset_id: int,
    user_id: int,
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Calculate similarity for candidate pairs.
    
    Args:
        df: Input DataFrame
        candidate_pairs: List of candidate pairs as [index1, index2]
        field_configs: Configuration for each field with type, method, and weight
        threshold: Similarity threshold for filtering pairs
        dataset_id: ID of the dataset
        user_id: ID of the user
        
    Returns:
        Dictionary with similarity results and metadata
    """
    try:
        # Calculate similarity for each candidate pair
        similarity_results = []
        
        for idx1, idx2 in candidate_pairs:
            # Get records
            record1 = df.iloc[idx1].to_dict()
            record2 = df.iloc[idx2].to_dict()
            
            # Calculate composite similarity
            similarity = calculate_composite_similarity(record1, record2, field_configs)
            
            # Add to results if above threshold
            if similarity >= threshold:
                result = {
                    "record1_id": int(idx1),
                    "record2_id": int(idx2),
                    "similarity": float(similarity),
                    "record1_data": record1,
                    "record2_data": record2,
                    "field_similarities": {}
                }
                
                # Calculate individual field similarities
                for field, config in field_configs.items():
                    if field in record1 and field in record2:
                        field_type = config.get('type', 'text')
                        method = config.get('method', None)
                        
                        if field_type == 'text':
                            default_method = 'jaro_winkler'
                            field_sim = calculate_text_similarity(
                                record1[field], 
                                record2[field], 
                                method=method if method else default_method
                            )
                        
                        elif field_type == 'numeric':
                            default_method = 'normalized_distance'
                            field_sim = calculate_numeric_similarity(
                                record1[field], 
                                record2[field], 
                                method=method if method else default_method
                            )
                        
                        elif field_type == 'categorical':
                            default_method = 'exact_match'
                            field_sim = calculate_categorical_similarity(
                                record1[field], 
                                record2[field], 
                                method=method if method else default_method
                            )
                        
                        else:
                            field_sim = calculate_text_similarity(record1[field], record2[field])
                        
                        result["field_similarities"][field] = float(field_sim)
                
                similarity_results.append(result)
        
        # Sort results by similarity (descending)
        similarity_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Save the similarity results (JSON)
        json_path = _get_similarity_artifact_path(
            settings.DATASET_UPLOAD_DIR,
            dataset_id,
            user_id,
            "similarity_results.json"
        )
        with open(json_path, 'w') as f:
            json.dump(similarity_results, f, ensure_ascii=False)

        # Also save as CSV (record1_id, record2_id, similarity)
        import pandas as pd
        sim_df = pd.DataFrame([
            {"record1_id": r["record1_id"], "record2_id": r["record2_id"], "similarity": r["similarity"]}
            for r in similarity_results
        ])
        csv_path = _get_similarity_artifact_path(
            settings.DATASET_UPLOAD_DIR,
            dataset_id,
            user_id,
            "similarity_results.csv"
        )
        sim_df.to_csv(csv_path, index=False)

        # Convert paths to web paths for frontend
        def _to_web(p: str) -> str:
            p = p.replace(os.sep, '/')
            return p if p.startswith('/') else '/' + p
        web_json_path = _to_web(json_path)
        web_csv_path = _to_web(csv_path)

        # Create a summary of similarity calculation
        summary = {
            "candidate_pairs": len(candidate_pairs),
            "similar_pairs": len(similarity_results),
            "threshold": threshold,
            "field_configs": field_configs,
            "json_path": web_json_path,
            "csv_path": web_csv_path,
            "output_path": web_csv_path
        }
        
        return {
            "status": "success",
            "message": "Similarity calculation completed successfully",
            "summary": summary,
            "similarity_results_path": web_csv_path,
            "preview": similarity_results[:10]  # Show first 10 results
        }
        
    except Exception as e:
        logger.error(f"Error in similarity calculation: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error in similarity calculation: {str(e)}",
            "error": str(e)
        }
