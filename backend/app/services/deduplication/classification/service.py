"""
Classification service for data deduplication.

This module implements machine learning approaches for duplicate classification:
- Random Forest: Tree-based ensemble method
- XGBoost: Gradient boosting implementation
- Siamese Network: Neural network for similarity learning
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import os
import json
import logging
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Try to import XGBoost, but provide a fallback if not available
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Try to import TensorFlow for Siamese Network, but provide a fallback if not available
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from app.config.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

def _get_classification_artifact_path(
    base_path: str,
    dataset_id: int,
    user_id: int,
    artifact_name: str
) -> str:
    """Create and return the path for classification artifacts."""
    dir_path = os.path.join(base_path, f"user_{user_id}", f"dataset_{dataset_id}", "deduplication", "classification")
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, artifact_name)

def _prepare_features_from_similarity(
    similarity_results: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Prepare features for classification from similarity results.
    
    Args:
        similarity_results: List of similarity results
        
    Returns:
        DataFrame with features and list of record pairs
    """
    features = []
    record_pairs = []
    
    for result in similarity_results:
        # Extract features from similarity results
        feature_dict = {
            "composite_similarity": result["similarity"]
        }
        
        # Add individual field similarities as features
        for field, similarity in result.get("field_similarities", {}).items():
            feature_dict[f"similarity_{field}"] = similarity
        
        features.append(feature_dict)
        
        # Keep track of record pairs
        record_pair = {
            "record1_id": result["record1_id"],
            "record2_id": result["record2_id"],
            "record1_data": result["record1_data"],
            "record2_data": result["record2_data"]
        }
        record_pairs.append(record_pair)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features)
    
    return features_df, record_pairs

def random_forest_classification(
    features: pd.DataFrame,
    record_pairs: List[Dict[str, Any]],
    params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Classify duplicate pairs using Random Forest.
    
    Args:
        features: DataFrame with features
        record_pairs: List of record pairs
        params: Parameters for Random Forest
        
    Returns:
        List of classified pairs with confidence scores
    """
    # Extract parameters
    confidence_threshold = params.get('confidence_threshold', 0.7)
    n_estimators = params.get('n_estimators', 100)
    max_depth = params.get('max_depth', None)
    class_weight = params.get('class_weight', 'balanced')
    criterion = params.get('criterion', 'gini')
    min_samples_leaf = params.get('min_samples_leaf', 1)
    max_features = params.get('max_features', 'sqrt')
    
    # Create and configure the classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        criterion=criterion,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )
    
    # Since we don't have labeled data, we'll use a heuristic approach
    # We'll assume pairs with high composite similarity are duplicates
    high_similarity_threshold = 0.9
    y_pseudo = (features['composite_similarity'] > high_similarity_threshold).astype(int)
    
    # Fit the classifier
    clf.fit(features, y_pseudo)
    
    # Predict probabilities
    probas = clf.predict_proba(features)
    
    # Create results
    classified_pairs = []
    
    for i, (record_pair, proba) in enumerate(zip(record_pairs, probas)):
        # Get duplicate probability (class 1)
        duplicate_confidence = proba[1] if len(proba) > 1 else proba[0]
        
        # Classify as duplicate if confidence is above threshold
        is_duplicate = duplicate_confidence >= confidence_threshold
        
        result = {
            "record1_id": record_pair["record1_id"],
            "record2_id": record_pair["record2_id"],
            "record1_data": record_pair["record1_data"],
            "record2_data": record_pair["record2_data"],
            "confidence": float(duplicate_confidence),
            "is_duplicate": bool(is_duplicate),
            "features": features.iloc[i].to_dict()
        }
        
        classified_pairs.append(result)
    
    # Sort by confidence (descending)
    classified_pairs.sort(key=lambda x: x["confidence"], reverse=True)
    
    return classified_pairs

def xgboost_classification(
    features: pd.DataFrame,
    record_pairs: List[Dict[str, Any]],
    params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Classify duplicate pairs using XGBoost.
    
    Args:
        features: DataFrame with features
        record_pairs: List of record pairs
        params: Parameters for XGBoost
        
    Returns:
        List of classified pairs with confidence scores
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost is required for this classification method")
    
    # Extract parameters
    confidence_threshold = params.get('confidence_threshold', 0.7)
    learning_rate = params.get('learning_rate', 0.1)
    n_estimators = params.get('n_estimators', 100)
    max_depth = params.get('max_depth', 3)
    subsample = params.get('subsample', 0.8)
    colsample_bytree = params.get('colsample_bytree', 0.8)
    
    # Create and configure the classifier
    clf = xgb.XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42
    )
    
    # Since we don't have labeled data, we'll use a heuristic approach
    # We'll assume pairs with high composite similarity are duplicates
    high_similarity_threshold = 0.9
    y_pseudo = (features['composite_similarity'] > high_similarity_threshold).astype(int)
    
    # If only a single class present, skip training to avoid XGBoost class error
    if y_pseudo.nunique() < 2:
        # Use composite similarity as a proxy confidence (clip to [0,1])
        positive_conf = features['composite_similarity'].clip(0, 1).astype(float).values
        probas = np.column_stack([1 - positive_conf, positive_conf])
    else:
        # Fit the classifier
        clf.fit(features, y_pseudo)
        # Predict probabilities
        probas = clf.predict_proba(features)
    
    # Create results
    classified_pairs = []
    
    for i, (record_pair, proba) in enumerate(zip(record_pairs, probas)):
        # Get duplicate probability (class 1)
        duplicate_confidence = proba[1] if len(proba) > 1 else proba[0]
        
        # Classify as duplicate if confidence is above threshold
        is_duplicate = duplicate_confidence >= confidence_threshold
        
        result = {
            "record1_id": record_pair["record1_id"],
            "record2_id": record_pair["record2_id"],
            "record1_data": record_pair["record1_data"],
            "record2_data": record_pair["record2_data"],
            "confidence": float(duplicate_confidence),
            "is_duplicate": bool(is_duplicate),
            "features": features.iloc[i].to_dict()
        }
        
        classified_pairs.append(result)
    
    # Sort by confidence (descending)
    classified_pairs.sort(key=lambda x: x["confidence"], reverse=True)
    
    return classified_pairs

def siamese_network_classification(
    features: pd.DataFrame,
    record_pairs: List[Dict[str, Any]],
    params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Classify duplicate pairs using a simplified Siamese Network approach.
    
    Args:
        features: DataFrame with features
        record_pairs: List of record pairs
        params: Parameters for Siamese Network
        
    Returns:
        List of classified pairs with confidence scores
    """
    if not HAS_TENSORFLOW:
        raise ImportError("TensorFlow is required for this classification method")
    
    # Extract parameters
    confidence_threshold = params.get('confidence_threshold', 0.7)
    
    # Since we don't have a real Siamese Network implementation,
    # we'll use a simple neural network on the similarity features
    
    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Create a simple neural network
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(features.shape[1],)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Since we don't have labeled data, we'll use a heuristic approach
    # We'll assume pairs with high composite similarity are duplicates
    high_similarity_threshold = 0.9
    y_pseudo = (features['composite_similarity'] > high_similarity_threshold).astype(int)
    
    # If only one class present, skip training and derive probabilities directly
    if y_pseudo.nunique() < 2:
        probas = features['composite_similarity'].clip(0, 1).astype(float).values
    else:
        # Fit the model
        model.fit(scaled_features, y_pseudo, epochs=5, verbose=0)
        # Predict probabilities
        probas = model.predict(scaled_features).flatten()
    
    # Create results
    classified_pairs = []
    
    for i, (record_pair, proba) in enumerate(zip(record_pairs, probas)):
        # Get duplicate probability
        duplicate_confidence = float(proba)
        
        # Classify as duplicate if confidence is above threshold
        is_duplicate = duplicate_confidence >= confidence_threshold
        
        result = {
            "record1_id": record_pair["record1_id"],
            "record2_id": record_pair["record2_id"],
            "record1_data": record_pair["record1_data"],
            "record2_data": record_pair["record2_data"],
            "confidence": float(duplicate_confidence),
            "is_duplicate": bool(is_duplicate),
            "features": features.iloc[i].to_dict()
        }
        
        classified_pairs.append(result)
    
    # Sort by confidence (descending)
    classified_pairs.sort(key=lambda x: x["confidence"], reverse=True)
    
    return classified_pairs

async def run_classification(
    similarity_results: List[Dict[str, Any]],
    method: str,
    params: Dict[str, Any],
    dataset_id: int,
    user_id: int,
    output_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run classification on similarity results.
    
    Args:
        similarity_results: List of similarity results
        method: Classification method ('random_forest', 'xgboost', 'siamese_network')
        params: Parameters for the classification method
        dataset_id: ID of the dataset
        user_id: ID of the user
        
    Returns:
        Dictionary with classification results and metadata
    """
    try:
        # Prepare features from similarity results
        features, record_pairs = _prepare_features_from_similarity(similarity_results)
        
        # Run the appropriate classification method
        if method == 'random_forest':
            classified_pairs = random_forest_classification(features, record_pairs, params)
            
        elif method == 'xgboost':
            if not HAS_XGBOOST:
                return {
                    "status": "error",
                    "message": "XGBoost is not available. Please install it with 'pip install xgboost'."
                }
            
            classified_pairs = xgboost_classification(features, record_pairs, params)
            
        elif method == 'siamese_network':
            if not HAS_TENSORFLOW:
                return {
                    "status": "error",
                    "message": "TensorFlow is not available. Please install it with 'pip install tensorflow'."
                }
            
            classified_pairs = siamese_network_classification(features, record_pairs, params)
            
        else:
            return {
                "status": "error",
                "message": f"Unknown classification method: {method}"
            }
        
        
        
        # ---------------------------
        # Persist outputs (JSON & CSV)
        # ---------------------------
        timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        base_name = output_name if output_name else f"classification_{method}_{timestamp}"

        # JSON
        json_path = _get_classification_artifact_path(
            settings.DATASET_UPLOAD_DIR,
            dataset_id,
            user_id,
            f"{base_name}.json"
        )
        with open(json_path, "w", encoding="utf-8") as f_out:
            json.dump(classified_pairs, f_out, ensure_ascii=False, indent=2)

        # CSV
        csv_path = _get_classification_artifact_path(
            settings.DATASET_UPLOAD_DIR,
            dataset_id,
            user_id,
            f"{base_name}.csv"
        )
        import csv as _csv
        csv_fields = ["record1_id", "record2_id", "confidence", "is_duplicate"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
            writer = _csv.DictWriter(f_csv, fieldnames=csv_fields)
            writer.writeheader()
            for pair in classified_pairs:
                writer.writerow({k: pair[k] for k in csv_fields})

        # ---------------------------
        # Evaluation metrics (pseudo)
        # ---------------------------
        confidence_threshold = params.get("confidence_threshold", 0.7)
        pseudo_labels = [pair["features"].get("composite_similarity", 0.0) > 0.9 for pair in classified_pairs]
        preds = [pair["is_duplicate"] for pair in classified_pairs]
        tp = sum(p and y for p, y in zip(preds, pseudo_labels))
        fp = sum(p and not y for p, y in zip(preds, pseudo_labels))
        fn = sum((not p) and y for p, y in zip(preds, pseudo_labels))
        tn = sum((not p) and (not y) for p, y in zip(preds, pseudo_labels))
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * precision * recall) / (precision + recall) if (precision and recall and (precision + recall)) else None

        summary = {
            "method": method,
            "params": params,
            "total_pairs": len(classified_pairs),
            "duplicate_pairs": sum(preds),
            "confidence_threshold": confidence_threshold,
            "evaluation_metrics": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn
            },
            "json_path": json_path,
            "csv_path": csv_path
        }

        # Convert local paths to web-style paths for frontend convenience
        from pathlib import Path as _Path
        web_json_path = "/" + _Path(json_path).as_posix()
        web_csv_path = "/" + _Path(csv_path).as_posix()

        return {
            "status": "success",
            "message": "Classification completed successfully",
            "summary": summary,
            "classification_results_path": web_csv_path,
            "preview": classified_pairs[:10]
        }
        
    except Exception as e:
        logger.error(f"Error in classification: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error in classification: {str(e)}",
            "error": str(e)
        }
