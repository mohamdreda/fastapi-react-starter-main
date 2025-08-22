"""
Legacy deduplication algorithms service.

This module integrates the legacy fuzzy matching and deep entity resolution algorithms
into the new modular deduplication pipeline structure. These algorithms are retained
for comparison purposes with the new modular approach.
"""
import pandas as pd
from typing import Dict, Any, List
import logging
from datetime import datetime

from .fuzzy_matching import fuzzy_matching_deduplication
from .deep_er import deep_er_deduplication

logger = logging.getLogger(__name__)

async def run_legacy_deduplication(
    df: pd.DataFrame,
    algorithm: str,
    params: Dict[str, Any],
    dataset_id: int,
    user_id: int
) -> Dict[str, Any]:
    """
    Run a legacy deduplication algorithm.
    
    Args:
        df: Input DataFrame
        algorithm: Algorithm name ('fuzzy', 'deep_er')
        params: Algorithm parameters
        dataset_id: ID of the dataset
        user_id: ID of the user
        
    Returns:
        Dictionary with results including potential duplicates
    """
    try:
        if algorithm == 'fuzzy':
            return fuzzy_matching_deduplication(df, params, dataset_id, user_id)
        elif algorithm == 'deep_er':
            return deep_er_deduplication(df, params, dataset_id, user_id)
        else:
            return {
                "status": "error",
                "message": f"Unknown legacy algorithm: {algorithm}"
            }
    except Exception as e:
        logger.error(f"Error in legacy deduplication: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error in legacy deduplication: {str(e)}",
            "error": str(e)
        }

def get_legacy_algorithms() -> List[Dict[str, Any]]:
    """
    Get a list of available legacy deduplication algorithms.
    
    Returns:
        List of algorithm information dictionaries
    """
    return [
        {
            "name": "fuzzy",
            "display_name": "Fuzzy Matching",
            "description": "String similarity-based fuzzy matching for duplicate detection",
            "parameters": {
                "threshold": {
                    "type": "number",
                    "default": 80,
                    "description": "Similarity threshold (0-100)"
                },
                "columns": {
                    "type": "array",
                    "default": None,
                    "description": "List of columns to compare (default: all string columns)"
                },
                "method": {
                    "type": "string",
                    "default": "token_sort_ratio",
                    "options": ["ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio"],
                    "description": "Fuzzy matching method"
                },
                "max_pairs": {
                    "type": "number",
                    "default": 10000,
                    "description": "Maximum number of pairs to compare"
                }
            }
        },
        {
            "name": "deep_er",
            "display_name": "Deep Entity Resolution",
            "description": "Text embedding-based entity resolution using TF-IDF and cosine similarity",
            "parameters": {
                "threshold": {
                    "type": "number",
                    "default": 0.8,
                    "description": "Similarity threshold (0.0-1.0)"
                },
                "text_columns": {
                    "type": "array",
                    "default": None,
                    "description": "List of text columns to use (default: all string columns)"
                },
                "max_features": {
                    "type": "number",
                    "default": 5000,
                    "description": "Maximum number of features for TF-IDF"
                },
                "ngram_range": {
                    "type": "array",
                    "default": [1, 2],
                    "description": "N-gram range for TF-IDF"
                },
                "max_pairs": {
                    "type": "number",
                    "default": 10000,
                    "description": "Maximum number of pairs to compare"
                }
            }
        }
    ]
