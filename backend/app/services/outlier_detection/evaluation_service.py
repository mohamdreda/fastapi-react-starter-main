# app/services/evaluation_service.py
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score
)
from typing import List, Dict, Any, Optional

from app.schemas.outliers import GroundTruthDataPoint, EvaluationMetricsSchema, ConfusionMatrixSchema
# from app.db.models import OutlierDetectionRun # Not strictly needed in this service directly if paths are constructed
from app.config.config import get_settings
settings = get_settings()
import os

class OutlierEvaluationService:
    def __init__(self, outlier_run_id: int, dataset_id: int, user_id: int):
        self.outlier_run_id = outlier_run_id
        self.dataset_id = dataset_id
        self.user_id = user_id
        self.ARTIFACT_BASE = settings.OUTLIER_ARTIFACTS_BASE_PATH
        self.DATASET_BASE = settings.DATASET_UPLOAD_DIR
        print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Initialized for dataset {self.dataset_id}, user {self.user_id}")

    def _load_predictions(self) -> Optional[pd.DataFrame]:
        predicted_results_path = os.path.join(
            self.ARTIFACT_BASE,
            f"user_{self.user_id}",
            f"dataset_{self.dataset_id}",
            "final_results", # Subdirectory where orchestrator saves it
            f"final_outlier_analysis_results.parquet" # Specific filename
        )
        print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Attempting to load predictions from: {predicted_results_path}")
        if not os.path.exists(predicted_results_path):
            print(f"EVAL_SERVICE ERROR [Run {self.outlier_run_id}]: Predicted results file NOT FOUND at {predicted_results_path}")
            return None
        try:
            predictions_df = pd.read_parquet(predicted_results_path)
            print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Predictions loaded successfully. Shape: {predictions_df.shape}")
            print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Prediction columns: {predictions_df.columns.tolist()}")
            if not predictions_df.empty:
                print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Prediction head:\n{predictions_df.head()}")
            return predictions_df
        except Exception as e:
            print(f"EVAL_SERVICE ERROR [Run {self.outlier_run_id}]: Error loading predictions from {predicted_results_path}: {e}")
            return None
            
    def _load_original_dataset(self) -> Optional[pd.DataFrame]:
        """Load the original dataset to extract the Class column for ground truth"""
        try:
            # Try to find the original dataset file
            dataset_dir = os.path.join(
                self.DATASET_BASE,
                f"user_{self.user_id}",
                f"dataset_{self.dataset_id}"
            )
            
            # Look for CSV files in the dataset directory
            csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
            if not csv_files:
                print(f"EVAL_SERVICE WARNING [Run {self.outlier_run_id}]: No CSV files found in {dataset_dir}")
                return None
                
            # Use the first CSV file found (assuming it's the original dataset)
            original_dataset_path = os.path.join(dataset_dir, csv_files[0])
            print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Loading original dataset from {original_dataset_path}")
            
            # Load the dataset
            original_df = pd.read_csv(original_dataset_path)
            
            # Check if 'Class' column exists
            if 'Class' not in original_df.columns:
                print(f"EVAL_SERVICE WARNING [Run {self.outlier_run_id}]: No 'Class' column found in original dataset")
                return None
                
            print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Original dataset loaded successfully. Shape: {original_df.shape}")
            print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Class column values: {original_df['Class'].value_counts()}")
            
            return original_df
        except Exception as e:
            print(f"EVAL_SERVICE ERROR [Run {self.outlier_run_id}]: Error loading original dataset: {e}")
            return None

    def calculate_metrics_from_original_dataset(self) -> EvaluationMetricsSchema:
        """Calculate evaluation metrics using the Class column from the original dataset as ground truth"""
        print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Starting metric calculation using original dataset Class column.")
        
        # Load predictions and original dataset
        predictions_df = self._load_predictions()
        original_df = self._load_original_dataset()
        
        if predictions_df is None:
            print(f"EVAL_SERVICE ERROR [Run {self.outlier_run_id}]: Failed to load predictions. Aborting metric calculation.")
            return EvaluationMetricsSchema(message="Failed to load model predictions for evaluation.")
            
        if original_df is None:
            print(f"EVAL_SERVICE ERROR [Run {self.outlier_run_id}]: Failed to load original dataset. Aborting metric calculation.")
            return EvaluationMetricsSchema(message="Failed to load original dataset with Class column for evaluation.")
            
        # Check if original dataset has an index column that can be used to match with predictions
        if 'original_index' not in predictions_df.columns:
            print(f"EVAL_SERVICE ERROR [Run {self.outlier_run_id}]: Predictions DataFrame missing 'original_index' column.")
            return EvaluationMetricsSchema(message="Predictions data is missing 'original_index' column.")
            
        # Create ground truth from Class column (assuming Class=1 is normal, others are outliers)
        # Add an index column to original_df if it doesn't have one
        if 'original_index' not in original_df.columns:
            original_df['original_index'] = original_df.index
            
        # Create a ground truth DataFrame with the required columns
        gt_df = pd.DataFrame()
        gt_df['original_index'] = original_df['original_index']
        
        # Convert Class column to binary outlier labels (Class=1 is normal, others are outliers)
        # This is a common convention in outlier detection datasets
        gt_df['true_is_outlier'] = original_df['Class'] != 1
        
        print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Created ground truth from Class column. Outlier count: {gt_df['true_is_outlier'].sum()}")
        
        # Merge predictions with ground truth
        print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Merging predictions with ground truth on 'original_index'.")
        
        # Determine which score column to use based on the outlier detection method
        score_column = 'if_score'  # Default to isolation forest score
        
        # Check for One-Class SVM score specifically
        if 'ocsvm_score' in predictions_df.columns:
            score_column = 'ocsvm_score'
            print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Using One-Class SVM score column: {score_column}")
        else:
            # Use the first alternative score column found
            for col in predictions_df.columns:
                if col.endswith('_score') and col != 'if_score':
                    score_column = col
                    print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Using score column: {score_column}")
                    break
                
        merged_df = pd.merge(
            predictions_df[['original_index', 'is_outlier', score_column]],
            gt_df[['original_index', 'true_is_outlier']],
            on='original_index',
            how='inner'
        )
        
        print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Merged DataFrame shape: {merged_df.shape}")
        if merged_df.empty:
            print(f"EVAL_SERVICE ERROR [Run {self.outlier_run_id}]: Merged DataFrame is empty. No matching 'original_index' between predictions and ground truth.")
            return EvaluationMetricsSchema(message="No matching records found between predictions and ground truth based on 'original_index'.")
            
        # Calculate metrics
        y_true = merged_df['true_is_outlier'].astype(bool)
        y_pred = merged_df['is_outlier'].astype(bool)
        y_scores = merged_df[score_column]
        
        print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: y_true unique values and counts:\n{y_true.value_counts(dropna=False)}")
        print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: y_pred unique values and counts:\n{y_pred.value_counts(dropna=False)}")
        
        try:
            # Calculate AUC-ROC if possible
            auc_roc_val = None
            if len(np.unique(y_true)) > 1:  # Check if more than one class in true labels
                if not y_scores.isnull().all():  # Check if scores are not all NaN
                    # For most methods, higher score = more outlier-like
                    # For isolation forest, lower score = more outlier-like, so we negate it
                    score_for_auc = y_scores if not score_column == 'if_score' else -y_scores
                    auc_roc_val = roc_auc_score(y_true, score_for_auc)
                    print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: AUC-ROC calculated: {auc_roc_val}")
            
            # Calculate confusion matrix and metrics
            cm = confusion_matrix(y_true, y_pred, labels=[False, True])
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate metrics, handling division by zero
            precision_val = precision_score(y_true, y_pred, zero_division=0.0, pos_label=True)
            recall_val = recall_score(y_true, y_pred, zero_division=0.0, pos_label=True)
            f1_val = f1_score(y_true, y_pred, zero_division=0.0, pos_label=True)
            
            # Calculate accuracy
            accuracy_val = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            
            print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Metrics calculated from original dataset Class column:")
            print(f"  Accuracy: {accuracy_val}")
            print(f"  Precision: {precision_val}")
            print(f"  Recall: {recall_val}")
            print(f"  F1: {f1_val}")
            print(f"  AUC-ROC: {auc_roc_val}")
            
            # Determine if this is One-Class SVM based on the score column
            is_one_class_svm = 'ocsvm_score' in predictions_df.columns
            
            return EvaluationMetricsSchema(
                accuracy=accuracy_val,
                precision=precision_val,
                recall=recall_val,
                f1_score=f1_val,
                auc_roc=auc_roc_val,
                source="original_dataset",  # Add source information
                confusion_matrix=ConfusionMatrixSchema(
                    true_positives=int(tp),
                    false_positives=int(fp),
                    true_negatives=int(tn),
                    false_negatives=int(fn)
                ),
                message="Evaluation using original dataset Class column completed successfully" + 
                        " for One-Class SVM" if is_one_class_svm else "."
            )
        except Exception as e:
            print(f"EVAL_SERVICE ERROR [Run {self.outlier_run_id}]: Error during metric calculation: {e}")
            import traceback
            traceback.print_exc()
            return EvaluationMetricsSchema(message=f"An error occurred during evaluation: {str(e)}")
    
    def calculate_metrics(self, ground_truth_data: List[GroundTruthDataPoint]) -> EvaluationMetricsSchema:
        print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Starting metric calculation.")
        
        predictions_df = self._load_predictions()
        if predictions_df is None:
            print(f"EVAL_SERVICE ERROR [Run {self.outlier_run_id}]: Failed to load predictions. Aborting metric calculation.")
            return EvaluationMetricsSchema(message="Failed to load model predictions for evaluation.")

        if not ground_truth_data:
            print(f"EVAL_SERVICE ERROR [Run {self.outlier_run_id}]: No ground truth data provided. Aborting.")
            return EvaluationMetricsSchema(message="No ground truth data provided.")

        print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Received {len(ground_truth_data)} ground truth data points.")
        gt_df = pd.DataFrame([item.model_dump() for item in ground_truth_data])
        print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Ground truth DataFrame shape: {gt_df.shape}")
        if not gt_df.empty:
            print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Ground truth DataFrame head:\n{gt_df.head()}")


        if 'original_index' not in gt_df.columns or 'true_is_outlier' not in gt_df.columns:
            print(f"EVAL_SERVICE ERROR [Run {self.outlier_run_id}]: Ground truth data missing required columns. Columns found: {gt_df.columns.tolist()}")
            return EvaluationMetricsSchema(message="Ground truth data must contain 'original_index' and 'true_is_outlier' columns.")

        # Check required columns in predictions_df
        required_pred_cols = ['original_index', 'is_outlier', 'if_score']
        for col in required_pred_cols:
            if col not in predictions_df.columns:
                print(f"EVAL_SERVICE ERROR [Run {self.outlier_run_id}]: Predictions DataFrame missing required column '{col}'. Columns found: {predictions_df.columns.tolist()}")
                return EvaluationMetricsSchema(message=f"Predictions data is missing column: {col}")
        
        print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Merging predictions with ground truth on 'original_index'.")
        merged_df = pd.merge(
            predictions_df[['original_index', 'is_outlier', 'if_score']],
            gt_df[['original_index', 'true_is_outlier']], # Ensure only necessary columns from GT are merged
            on='original_index',
            how='inner'
        )

        print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Merged DataFrame shape: {merged_df.shape}")
        if merged_df.empty:
            print(f"EVAL_SERVICE ERROR [Run {self.outlier_run_id}]: Merged DataFrame is empty. No matching 'original_index' between predictions and ground truth.")
            print(f"  Prediction original_index sample: {predictions_df['original_index'].head().tolist() if not predictions_df.empty else 'N/A'}")
            print(f"  Ground truth original_index sample: {gt_df['original_index'].head().tolist() if not gt_df.empty else 'N/A'}")
            return EvaluationMetricsSchema(message="No matching records found between predictions and ground truth based on 'original_index'.")
        if not merged_df.empty:
            print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Merged DataFrame head:\n{merged_df.head()}")

        y_true = merged_df['true_is_outlier'].astype(bool)
        y_pred = merged_df['is_outlier'].astype(bool)
        y_scores = merged_df['if_score'] # Lower score = more anomalous for Isolation Forest

        print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: y_true unique values and counts:\n{y_true.value_counts(dropna=False)}")
        print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: y_pred unique values and counts:\n{y_pred.value_counts(dropna=False)}")
        print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: y_scores sample (first 5, using -y_scores for AUC logic):\n{(-y_scores).head().tolist()}")


        try:
            auc_roc_val = None
            if len(np.unique(y_true)) > 1: # Check if more than one class in true labels
                if not y_scores.isnull().all(): # Check if scores are not all NaN
                    # Using -y_scores because lower IF score indicates outlier (positive class)
                    # roc_auc_score expects higher scores for the positive class.
                    auc_roc_val = roc_auc_score(y_true, -y_scores)
                    print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: AUC-ROC calculated: {auc_roc_val}")
                else:
                    print("EVAL_SERVICE WARNING [Run {self.outlier_run_id}]: All 'if_score' values are NaN. AUC-ROC cannot be computed.")
                    auc_roc_val = None # Explicitly set to None
            else:
                print("EVAL_SERVICE WARNING [Run {self.outlier_run_id}]: Ground truth 'y_true' has only one class. AUC-ROC is not defined and will be None.")
                auc_roc_val = None # Explicitly set to None

            # Confusion Matrix: specified labels ensure correct order for tn, fp, fn, tp
            # labels=[False, True] means:
            #   False (Inlier) is the "negative" class (index 0 of matrix axis)
            #   True (Outlier) is the "positive" class (index 1 of matrix axis)
            # Resulting matrix: [[TN, FP], [FN, TP]]
            cm = confusion_matrix(y_true, y_pred, labels=[False, True])
            tn, fp, fn, tp = cm.ravel()
            print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Confusion Matrix (TN, FP, FN, TP): ({tn}, {fp}, {fn}, {tp})")

            # Calculate metrics, handling division by zero
            precision_val = precision_score(y_true, y_pred, zero_division=0.0, pos_label=True)
            recall_val = recall_score(y_true, y_pred, zero_division=0.0, pos_label=True)
            f1_val = f1_score(y_true, y_pred, zero_division=0.0, pos_label=True)
            print(f"EVAL_SERVICE [Run {self.outlier_run_id}]: Precision: {precision_val}, Recall: {recall_val}, F1: {f1_val}")


            return EvaluationMetricsSchema(
                precision=precision_val,
                recall=recall_val,
                f1_score=f1_val,
                accuracy=accuracy_val,  # Add accuracy metric
                auc_roc=auc_roc_val,
                source="provided_ground_truth",  # Add source information
                confusion_matrix=ConfusionMatrixSchema(
                    true_positives=int(tp),
                    false_positives=int(fp),
                    true_negatives=int(tn),
                    false_negatives=int(fn)
                ),
                message="Evaluation completed successfully using provided ground truth data."
            )
        except ValueError as ve:
            print(f"EVAL_SERVICE ERROR [Run {self.outlier_run_id}]: ValueError during metric calculation: {ve}")
            return EvaluationMetricsSchema(message=f"Error during metric calculation: {ve}")
        except Exception as e:
            print(f"EVAL_SERVICE ERROR [Run {self.outlier_run_id}]: Unexpected error during metric calculation: {e}")
            import traceback
            traceback.print_exc()
            return EvaluationMetricsSchema(message=f"An unexpected error occurred during evaluation: {str(e)}")