"""
Blocking service for data deduplication.

This module implements efficient blocking techniques to reduce the number of comparisons:
- MinHash LSH: Locality-Sensitive Hashing with MinHash signatures
- SimHash: Hash-based similarity estimation for text data
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Set, Optional
import os
import json
import logging
import hashlib
from itertools import combinations

# Try to import datasketch for MinHash LSH
try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False

from app.config.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

def _get_blocking_artifact_path(
    base_path: str,
    dataset_id: int,
    user_id: int,
    artifact_name: str
) -> str:
    """Create and return the path for blocking artifacts."""
    dir_path = os.path.join(base_path, f"user_{user_id}", f"dataset_{dataset_id}", "deduplication", "blocking")
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, artifact_name)

def _create_minhash_signature(
    row: pd.Series,
    key_fields: List[str],
    num_perm: int = 128
):  # Return type annotation removed to avoid NameError
    """
    Create a MinHash signature for a row using specified key fields.
    
    Args:
        row: Row from DataFrame
        key_fields: List of column names to use for signature
        num_perm: Number of permutations for MinHash
        
    Returns:
        MinHash object
    """
    if not HAS_DATASKETCH:
        raise ImportError("datasketch package is required for MinHash LSH")
    
    mh = MinHash(num_perm=num_perm)
    
    # Add each field value to the MinHash
    for field in key_fields:
        if field in row and pd.notna(row[field]):
            # Convert to string and encode
            value = str(row[field]).encode('utf-8')
            mh.update(value)
    
    return mh

def _create_simhash_signature(
    row: pd.Series,
    key_fields: List[str],
    num_bits: int = 64
) -> int:
    """
    Create a SimHash signature for a row using specified key fields.
    
    Args:
        row: Row from DataFrame
        key_fields: List of column names to use for signature
        num_bits: Number of bits for SimHash
        
    Returns:
        SimHash value as integer
    """
    # Initialize a vector of zeros
    v = [0] * num_bits
    
    # Process each field
    for field in key_fields:
        if field in row and pd.notna(row[field]):
            # Convert to string
            value = str(row[field])
            
            # Generate shingles (character n-grams)
            shingles = [value[i:i+3] for i in range(len(value)-2)]
            
            # Hash each shingle and update the vector
            for shingle in shingles:
                # Hash the shingle
                h = int(hashlib.md5(shingle.encode('utf-8')).hexdigest(), 16)
                
                # Update the vector based on the hash
                for i in range(num_bits):
                    bit = (h >> i) & 1
                    if bit == 1:
                        v[i] += 1
                    else:
                        v[i] -= 1
    
    # Convert the vector to a binary hash
    simhash = 0
    for i in range(num_bits):
        if v[i] > 0:
            simhash |= (1 << i)
    
    return simhash

def _hamming_distance(hash1: int, hash2: int) -> int:
    """
    Calculate Hamming distance between two hash values.
    
    Args:
        hash1: First hash value
        hash2: Second hash value
        
    Returns:
        Hamming distance (number of differing bits)
    """
    xor = hash1 ^ hash2
    return bin(xor).count('1')

def minhash_lsh_blocking(
    df: pd.DataFrame,
    key_fields: List[str],
    threshold: float = 0.5,
    num_perm: int = 128
) -> List[Tuple[int, int]]:
    """
    Use MinHash LSH to find candidate pairs.
    
    Args:
        df: Input DataFrame
        key_fields: List of column names to use for blocking
        threshold: Similarity threshold (0.0 to 1.0)
        num_perm: Number of permutations for MinHash
        
    Returns:
        List of candidate pairs as (index1, index2) tuples
    """
    if not HAS_DATASKETCH:
        raise ImportError("datasketch package is required for MinHash LSH")
    
    # Create LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    
    # Store MinHash signatures for each row
    minhashes = {}
    
    # Add each row to the LSH index
    for idx, row in df.iterrows():
        mh = _create_minhash_signature(row, key_fields, num_perm)
        minhashes[idx] = mh
        lsh.insert(idx, mh)
    
    # Find candidate pairs
    candidate_pairs = set()
    
    for idx, mh in minhashes.items():
        # Query the LSH index for similar items
        similar_indices = lsh.query(mh)
        
        # Add pairs (ensuring no duplicates and no self-pairs)
        for similar_idx in similar_indices:
            if similar_idx != idx:
                # Sort indices to avoid duplicates
                pair = tuple(sorted([idx, similar_idx]))
                candidate_pairs.add(pair)
    
    return list(candidate_pairs)

def simhash_blocking(
    df: pd.DataFrame,
    key_fields: List[str],
    threshold: int = 10,  # Maximum Hamming distance
    num_bits: int = 64
) -> List[Tuple[int, int]]:
    """
    Use SimHash to find candidate pairs.
    
    Args:
        df: Input DataFrame
        key_fields: List of column names to use for blocking
        threshold: Maximum Hamming distance threshold
        num_bits: Number of bits for SimHash
        
    Returns:
        List of candidate pairs as (index1, index2) tuples
    """
    # Calculate SimHash for each row
    simhashes = {}
    
    for idx, row in df.iterrows():
        simhashes[idx] = _create_simhash_signature(row, key_fields, num_bits)
    
    # Find candidate pairs
    candidate_pairs = set()
    
    # Compare all pairs (this can be optimized further)
    for idx1, idx2 in combinations(simhashes.keys(), 2):
        # Calculate Hamming distance
        distance = _hamming_distance(simhashes[idx1], simhashes[idx2])
        
        # If distance is below threshold, add to candidates
        if distance <= threshold:
            # Sort indices to avoid duplicates
            pair = tuple(sorted([idx1, idx2]))
            candidate_pairs.add(pair)
    
    return list(candidate_pairs)

async def run_blocking(
    df: pd.DataFrame,
    method: str,
    key_fields: List[str],
    params: Dict[str, Any],
    dataset_id: int,
    user_id: int,
    output_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run blocking to generate candidate pairs.
    
    Args:
        df: Input DataFrame
        method: Blocking method ('minhash_lsh' or 'simhash')
        key_fields: List of column names to use for blocking
        params: Additional parameters for the blocking method
        dataset_id: ID of the dataset
        user_id: ID of the user
        
    Returns:
        Dictionary with blocking results and metadata
    """
    try:
        # Run the appropriate blocking method
        if method == 'minhash_lsh':
            threshold = params.get('threshold', 0.5)
            num_perm = params.get('num_perm', 128)
            
            candidate_pairs = minhash_lsh_blocking(
                df, 
                key_fields, 
                threshold=threshold, 
                num_perm=num_perm
            )
            
        elif method == 'simhash':
            threshold = params.get('threshold', 10)
            num_bits = params.get('num_bits', 64)
            
            candidate_pairs = simhash_blocking(
                df, 
                key_fields, 
                threshold=threshold, 
                num_bits=num_bits
            )
            
        else:
            return {
                "status": "error",
                "message": f"Unknown blocking method: {method}"
            }
        
        # Save candidate pairs to JSON
        filename_base = output_name or "candidate_pairs"
        json_path = _get_blocking_artifact_path(
            settings.DATASET_UPLOAD_DIR,
            dataset_id,
            user_id,
            f"{filename_base}.json"
        )
        serializable_pairs = [list(pair) for pair in candidate_pairs]
        with open(json_path, 'w') as f:
            json.dump(serializable_pairs, f)

        # Save candidate pairs to CSV (pair_id, record1_id, record2_id)
        import pandas as pd
        pairs_df = pd.DataFrame(
            [(i, pair[0], pair[1]) for i, pair in enumerate(candidate_pairs)],
            columns=["pair_id", "record1_id", "record2_id"]
        )
        csv_path = _get_blocking_artifact_path(
            settings.DATASET_UPLOAD_DIR,
            dataset_id,
            user_id,
            f"{filename_base}.csv"
        )
        pairs_df.to_csv(csv_path, index=False)

        # Web paths
        web_json_path = json_path if json_path.startswith(('/', '\\')) else f"/{json_path}"
        web_csv_path = csv_path if csv_path.startswith(('/', '\\')) else f"/{csv_path}"

        # Create a summary of blocking
        summary = {
            "method": method,
            "key_fields": key_fields,
            "params": params,
            "total_records": len(df),
            "candidate_pairs": len(candidate_pairs),
            "reduction_ratio": 1.0 - (len(candidate_pairs) / (len(df) * (len(df) - 1) / 2)),
            "json_path": web_json_path,
            "csv_path": web_csv_path,
            "filename_base": filename_base
        }
        
        # Create a preview of candidate pairs
        preview_pairs = []
        for i, (idx1, idx2) in enumerate(candidate_pairs[:10]):  # Show first 10 pairs
            pair_info = {
                "pair_id": i,
                "record1_id": int(idx1),
                "record2_id": int(idx2),
                "record1_data": df.iloc[idx1].to_dict(),
                "record2_data": df.iloc[idx2].to_dict()
            }
            preview_pairs.append(pair_info)
        
        return {
            "status": "success",
            "message": "Blocking completed successfully",
            "summary": summary,
            "candidate_pairs_json_path": web_json_path,
            "candidate_pairs_csv_path": web_csv_path,
            "preview": preview_pairs
        }
        
    except Exception as e:
        logger.error(f"Error in blocking: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error in blocking: {str(e)}",
            "error": str(e)
        }
