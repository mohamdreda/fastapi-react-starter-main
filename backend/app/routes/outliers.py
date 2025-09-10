# backend/app/routes/outliers.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Path, Query, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, select, func, desc, and_, or_, text
import os
import json
import shutil
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import joblib
import uuid
from uuid import UUID
import asyncio
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, auc, classification_report,
    mean_squared_error
)

# Auth dependencies
from app.services.auth import get_current_user, get_current_user_id

from app.db import get_db
from app.db.database import AsyncSessionLocal
from app.db.models import Dataset as DatasetModel, User as UserModel, OutlierDetectionRun
from app.schemas.outliers import (
    DetectOutliersRequest,
    DetectOutliersResponse,
    TaskStatusResponse,
    OutlierDetectionRunSchema,
    EvaluateOutliersRequest, 
    EvaluateOutliersResponse,
    EvaluationMetricsSchema,
    GroundTruthDataPoint,
    ConfusionMatrixSchema
)

# Import anomaly detection services
from app.services.outlier_detection.anomaly_detection.isolation_forest import IsolationForestService
from app.services.outlier_detection.anomaly_detection.lof_service import LOFService
from app.services.outlier_detection.anomaly_detection.ocsvm_service import OCSVMService
from app.services.outlier_detection.evaluation_service import OutlierEvaluationService
from app.schemas.user import User as UserSchema

from app.config.config import get_settings
settings = get_settings()

# Session service for workflow capture
from app.services import sessions as session_service

from fastapi.responses import Response
# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

router = APIRouter(
    tags=["Outlier Detection"],
    responses={
        404: {"description": "Not found"},
        403: {"description": "Not authorized"},
    }
)

# In-memory status tracking (consider Redis/Celery for production)
background_task_status_dict: dict = {}

# Sync fallback to update session steps to avoid async issues in background tasks
def _update_session_step_sync(
    step_id: str,
    status: str,
    error: Optional[str] = None,
    run_ref_type: Optional[str] = None,
    run_ref_id: Optional[str] = None,
):
    try:
        from app.db.database import SessionLocal
        from app.db.models.workflow import SessionStep as SyncSessionStep
        from uuid import UUID as _UUID
        from datetime import datetime as _dt
        db = SessionLocal()
        try:
            step = db.query(SyncSessionStep).filter(SyncSessionStep.id == _UUID(step_id)).first()
            if not step:
                return
            if status:
                step.status = status
            if error is not None:
                step.error = error
            if run_ref_type is not None:
                step.run_ref_type = run_ref_type
            if run_ref_id is not None:
                step.run_ref_id = run_ref_id
            if status in ("success", "failed"):
                step.finished_at = _dt.utcnow()
            db.commit()
        finally:
            db.close()
    except Exception as e:
        print(f"TASK_WRAPPER: Sync fallback step update failed: {e}")

async def run_outlier_detection_pipeline_task(
    run_id: int,
    dataset_id: int,
    user_id: int,
    file_path: str,
    run_parameters: Dict[str, Any],
    session_step_id: Optional[str] = None,
):
    task_id_for_status = f"outlier_run_{run_id}" 
    print(f"TASK_WRAPPER: Background task started for run_id: {run_id}, task_id: {task_id_for_status}")
    background_task_status_dict[task_id_for_status] = "processing_setup"

    async_db: AsyncSession = AsyncSessionLocal()
    try:
        stmt_update_status = (
            update(OutlierDetectionRun)
            .where(OutlierDetectionRun.id == run_id)
            .values(status="processing_setup", task_id=task_id_for_status, started_at=datetime.utcnow())
        )
        await async_db.execute(stmt_update_status)
        await async_db.commit()

        background_task_status_dict[task_id_for_status] = "loading_data"
        print(f"TASK_WRAPPER: Loading data from {file_path}")
        df = pd.read_csv(file_path)
        numerical_df = df.select_dtypes(include=['int64', 'float64'])
        
        if numerical_df.empty:
            raise ValueError("No numerical features found in dataset. Outlier detection requires numerical data.")
        
        # Initialize results and artifact paths
        results_summary = {}
        artifact_paths = {}
        
        # Get outlier detection method from parameters
        outlier_detection_method = run_parameters.get('outlier_detection_method', 'isolation_forest')
        print(f"TASK_WRAPPER: Using outlier detection method: {outlier_detection_method}")
        
        # Get cluster labels if available (for per-cluster outlier detection)
        # For now, we'll create a dummy cluster label of 0 for all points
        # In a real implementation, this would come from a clustering step or user input
        cluster_labels = pd.Series(0, index=numerical_df.index)
        
        background_task_status_dict[task_id_for_status] = "detecting_outliers"
        
        # Initialize outlier detection results
        outlier_results_df = None
        
        # Run the selected outlier detection algorithm
        if outlier_detection_method == 'isolation_forest':
            print(f"TASK_WRAPPER: Running Isolation Forest on dataset with {len(numerical_df)} samples")
            # Get parameters for Isolation Forest
            if_n_estimators = run_parameters.get('if_n_estimators', 100)
            if_contamination = run_parameters.get('if_contamination', 0.05)
            
            # Initialize service
            if_service = IsolationForestService(
                dataset_id=dataset_id,
                user_id=user_id,
                n_estimators=if_n_estimators,
                contamination=if_contamination,
                random_state=run_parameters.get('random_state', 42)
            )
            
            # Detect outliers
            outlier_results_df = if_service.detect_outliers_per_cluster(numerical_df, cluster_labels)
            artifact_paths.update(if_service.get_artifact_paths())
            
        elif outlier_detection_method == 'lof' or outlier_detection_method == 'local_outlier_factor':
            print(f"TASK_WRAPPER: Running Local Outlier Factor on dataset with {len(numerical_df)} samples")
            # Get parameters for LOF - support both naming conventions
            # First try with standard naming, then fall back to prefixed naming
            lof_n_neighbors = run_parameters.get('n_neighbors', run_parameters.get('lof_n_neighbors', 20))
            lof_contamination = run_parameters.get('contamination', run_parameters.get('lof_contamination', 0.05))
            
            # Initialize service
            lof_service = LOFService(
                dataset_id=dataset_id,
                user_id=user_id,
                n_neighbors=lof_n_neighbors,
                contamination=lof_contamination,
                random_state=run_parameters.get('random_state', 42)
            )
            
            # Detect outliers
            outlier_results_df = lof_service.detect_outliers_per_cluster(numerical_df, cluster_labels)
            artifact_paths.update(lof_service.get_artifact_paths())
            
        elif outlier_detection_method == 'ocsvm' or outlier_detection_method == 'one_class_svm':
            print(f"TASK_WRAPPER: Running One-Class SVM on dataset with {len(numerical_df)} samples")
            # Get parameters for OCSVM - support both naming conventions
            # First try with standard naming, then fall back to prefixed naming
            ocsvm_nu = run_parameters.get('nu', run_parameters.get('ocsvm_nu', 0.05))
            ocsvm_kernel = run_parameters.get('kernel', run_parameters.get('ocsvm_kernel', 'rbf'))
            ocsvm_gamma = run_parameters.get('gamma', run_parameters.get('ocsvm_gamma', 'scale'))
            
            # Initialize service
            ocsvm_service = OCSVMService(
                dataset_id=dataset_id,
                user_id=user_id,
                nu=ocsvm_nu,
                kernel=ocsvm_kernel,
                gamma=ocsvm_gamma,
                random_state=run_parameters.get('random_state', 42)
            )
            
            # Detect outliers
            outlier_results_df = ocsvm_service.detect_outliers_per_cluster(numerical_df, cluster_labels)
            artifact_paths.update(ocsvm_service.get_artifact_paths())
            
        else:
            raise ValueError(f"Unknown outlier detection method: {outlier_detection_method}")
        
        # Prepare results summary
        results_summary['pipeline_status'] = 'completed'
        results_summary['total_points_processed'] = len(numerical_df)
        results_summary['num_numerical_features_used'] = len(numerical_df.columns)
        results_summary['outlier_detection_method'] = outlier_detection_method
        results_summary['total_outliers_detected'] = int(outlier_results_df['is_outlier'].sum())
        results_summary['outlier_results_list'] = outlier_results_df.to_dict('records')
        
        # Calculate evaluation metrics using the original dataset's Class column as ground truth
        try:
            print(f"TASK_WRAPPER: Attempting to calculate evaluation metrics using original dataset Class column...")
            evaluation_service = OutlierEvaluationService(
                outlier_run_id=run_id,
                dataset_id=dataset_id,
                user_id=user_id
            )
            
            # Calculate metrics using the original dataset's Class column
            metrics_result = evaluation_service.calculate_metrics_from_original_dataset()
            
            # Update results summary with calculated metrics
            if metrics_result:
                results_summary['evaluation_metrics'] = {
                    'accuracy': metrics_result.accuracy,
                    'precision': metrics_result.precision,
                    'recall': metrics_result.recall,
                    'f1': metrics_result.f1_score,
                    'roc_auc': metrics_result.auc_roc,
                    'average_precision': None  # Not calculated in our service
                }
                results_summary['evaluation_source'] = 'original_dataset_class_column'
                print(f"TASK_WRAPPER: Evaluation metrics calculated successfully using original dataset Class column.")
            else:
                # Fallback to placeholder metrics if calculation fails
                results_summary['evaluation_metrics'] = {
                    'accuracy': None,
                    'precision': None,
                    'recall': None,
                    'f1': None,
                    'roc_auc': None,
                    'average_precision': None
                }
                results_summary['evaluation_source'] = None
                print(f"TASK_WRAPPER: Failed to calculate evaluation metrics from original dataset.")
        except Exception as e:
            print(f"TASK_WRAPPER ERROR: Error calculating evaluation metrics: {e}")
            # Fallback to placeholder metrics if an exception occurs
            results_summary['evaluation_metrics'] = {
                'accuracy': None,
                'precision': None,
                'recall': None,
                'f1': None,
                'roc_auc': None,
                'average_precision': None
            }
            results_summary['evaluation_source'] = None

        background_task_status_dict[task_id_for_status] = "generating_visualizations"
        scatter_plot_path_val = None
        outlier_distribution_path_val = None
        pca_plot_path_val = None
        
        if results_summary.get("pipeline_status") == "completed" and outlier_results_df is not None:
            try:
                print(f"TASK_WRAPPER: Attempting to generate visualizations for run_id {run_id}...")
                viz_paths_dict = {}
                
                # Use a consistent, report-friendly theme and palette
                sns.set_theme(style='whitegrid', context='notebook', font_scale=1.1)
                palette = {'Inlier': '#2563eb', 'Outlier': '#ef4444'}  # blue, red
                
                # 1. Generate scatter plot of the first two features with outliers highlighted
                plt.figure(figsize=(10, 8))
                if len(numerical_df.columns) >= 2:
                    # Use the first two features for visualization
                    feature_cols = numerical_df.columns[:2].tolist()
                    plt.figure(figsize=(10, 8))
                    
                    # Create a copy of the numerical DataFrame with outlier information
                    plot_df = numerical_df.copy()
                    plot_df['is_outlier'] = outlier_results_df['is_outlier']
                    plot_df['label'] = np.where(plot_df['is_outlier'], 'Outlier', 'Inlier')
                    
                    # Plot inliers and outliers with different colors and markers
                    sns.scatterplot(
                        x=feature_cols[0],
                        y=feature_cols[1],
                        hue='label',
                        style='label',
                        markers={'Inlier': 'o', 'Outlier': 'X'},
                        palette=palette,
                        data=plot_df,
                        s=40,
                        alpha=0.85,
                        edgecolor='white',
                        linewidth=0.3,
                        legend=True
                    )
                    plt.title(f'Outlier Detection using {results_summary["outlier_detection_method"].replace("_", " ").title()}')
                    plt.legend(title='Class', loc='upper right', frameon=True)
                    plt.tight_layout()
                else:
                    # If only one feature, create a 1D scatter plot
                    feature_col = numerical_df.columns[0]
                    plot_df = numerical_df.copy()
                    plot_df['is_outlier'] = outlier_results_df['is_outlier']
                    plot_df['label'] = np.where(plot_df['is_outlier'], 'Outlier', 'Inlier')
                    plot_df['y'] = 0  # Dummy y value for visualization
                    
                    sns.scatterplot(
                        x=feature_col,
                        y='y',
                        hue='label',
                        style='label',
                        markers={'Inlier': 'o', 'Outlier': 'X'},
                        palette=palette,
                        data=plot_df,
                        s=40,
                        alpha=0.85,
                        edgecolor='white',
                        linewidth=0.3,
                        legend=True
                    )
                    plt.title(f'Outlier Detection using {results_summary["outlier_detection_method"].replace("_", " ").title()}')
                    plt.legend(title='Class', loc='upper right', frameon=True)
                    plt.yticks([])
                    plt.tight_layout()
                
                bytes_io = BytesIO()
                plt.savefig(bytes_io, format='png', dpi=200, bbox_inches='tight')
                bytes_io.seek(0)
                scatter_plot_path_val = base64.b64encode(bytes_io.read()).decode('utf-8')
                viz_paths_dict['scatter_plot_path'] = scatter_plot_path_val
                plt.close()
                
                # 2. Generate outlier distribution plot
                plt.figure(figsize=(10, 6))
                
                # Get the score column based on the algorithm used
                if results_summary['outlier_detection_method'] == 'isolation_forest':
                    score_col = 'if_score'
                elif results_summary['outlier_detection_method'] == 'lof':
                    score_col = 'lof_score'
                elif results_summary['outlier_detection_method'] == 'ocsvm':
                    score_col = 'ocsvm_score'
                else:
                    score_col = None
                
                if score_col and score_col in outlier_results_df.columns:
                    # Create histogram of outlier scores with different colors for inliers and outliers
                    score_df = outlier_results_df.copy()
                    score_df['label'] = np.where(score_df['is_outlier'], 'Outlier', 'Inlier')
                    sns.histplot(
                        data=score_df,
                        x=score_col,
                        hue='label',
                        palette=palette,
                        kde=True,
                        bins=30,
                        alpha=0.75
                    )
                    plt.title(f'Distribution of Outlier Scores ({results_summary["outlier_detection_method"].replace("_", " ").title()})')
                    plt.xlabel('Outlier Score')
                    plt.ylabel('Count')
                    plt.legend(title='Class', loc='upper right', frameon=True)
                    plt.tight_layout()
                    
                    bytes_io = BytesIO()
                    plt.savefig(bytes_io, format='png', dpi=200, bbox_inches='tight')
                    bytes_io.seek(0)
                    outlier_distribution_path_val = base64.b64encode(bytes_io.read()).decode('utf-8')
                    viz_paths_dict['outlier_distribution_path'] = outlier_distribution_path_val
                    plt.close()
                
                # 3. Generate PCA plot if there are more than 2 features
                if len(numerical_df.columns) > 2:
                    from sklearn.decomposition import PCA
                    from sklearn.preprocessing import StandardScaler
                    
                    # Standardize the data
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(numerical_df)
                    
                    # Apply PCA to reduce to 2 dimensions
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(scaled_data)
                    
                    # Create DataFrame with PCA results
                    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
                    pca_df['is_outlier'] = outlier_results_df['is_outlier']
                    pca_df['label'] = np.where(pca_df['is_outlier'], 'Outlier', 'Inlier')
                    
                    # Plot PCA results with outliers highlighted
                    plt.figure(figsize=(10, 8))
                    sns.scatterplot(
                        x='PC1',
                        y='PC2',
                        hue='label',
                        style='label',
                        markers={'Inlier': 'o', 'Outlier': 'X'},
                        palette=palette,
                        data=pca_df,
                        s=40,
                        alpha=0.85,
                        edgecolor='white',
                        linewidth=0.3,
                        legend=True
                    )
                    plt.title(f'PCA Visualization of Outliers ({results_summary["outlier_detection_method"].replace("_", " ").title()})')
                    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                    plt.legend(title='Class', loc='upper right', frameon=True)
                    plt.tight_layout()
                    
                    bytes_io = BytesIO()
                    plt.savefig(bytes_io, format='png', dpi=200, bbox_inches='tight')
                    bytes_io.seek(0)
                    pca_plot_path_val = base64.b64encode(bytes_io.read()).decode('utf-8')
                    viz_paths_dict['pca_plot_path'] = pca_plot_path_val
                    plt.close()
                
                print(f"TASK_WRAPPER: Visualizations generated. Paths: {list(viz_paths_dict.keys())}")
            except Exception as viz_e:
                print(f"TASK_WRAPPER: Error generating visualizations for run_id {run_id}: {viz_e}")
                import traceback
                print(traceback.format_exc())
                results_summary['visualization_error'] = str(viz_e)
        else:
            print(f"TASK_WRAPPER: Skipping visualization for run_id {run_id} due to pipeline status or missing outlier results.")

        background_task_status_dict[task_id_for_status] = "saving_results"
        # Use raw SQL to update only the columns we know exist in the database
        # Store visualization paths in the results JSON instead of separate columns
        results_json = results_summary.get("outlier_results_list", [])
        
        # Create a metadata object to store in the parameters column
        metadata = {
            "visualizations": {
                "scatter_plot": scatter_plot_path_val,
                "outlier_distribution": outlier_distribution_path_val,
                "pca_plot": pca_plot_path_val
            },
            "metrics": results_summary.get("evaluation_metrics", {}),
            "summary": {
                "total_points_processed": results_summary.get("total_points_processed"),
                "num_numerical_features_used": results_summary.get("num_numerical_features_used"),
                "total_outliers_detected": results_summary.get("total_outliers_detected")
            }
        }
        
        # Get the existing parameters to update them with new results
        try:
            # First, get the current parameters from the database
            get_params_query = text("SELECT parameters FROM outlier_detection_runs WHERE id = :run_id")
            result = await async_db.execute(get_params_query, {"run_id": run_id})
            row = result.mappings().first()
            
            if row and row["parameters"]:
                # Parse existing parameters
                existing_params = row["parameters"]
                if isinstance(existing_params, str):
                    existing_params = json.loads(existing_params)
            else:
                existing_params = {}
                
            # Update the parameters with results and metadata
            if results_json:
                existing_params["results"] = results_json
            if metadata:
                existing_params.update(metadata)
                
            # Store all visualization paths in the parameters
            if artifact_paths:
                existing_params["visualization_paths"] = artifact_paths
                
            # Prepare values to update - only use columns we know exist
            values_to_update = {
                "status": results_summary.get("pipeline_status", "unknown_completion_status"),
                "completed_at": datetime.utcnow(),
                "error_message": results_summary.get("error_message", None),
                "parameters": json.dumps(existing_params)
            }
        except Exception as e:
            print(f"Error getting/updating parameters: {str(e)}")
            # Fallback to minimal update
            values_to_update = {
                "status": results_summary.get("pipeline_status", "unknown_completion_status"),
                "completed_at": datetime.utcnow(),
                "error_message": results_summary.get("error_message", None)
            }
            
        values_to_update = {k: v for k, v in values_to_update.items() if v is not None}

        # Use raw SQL update to ensure compatibility with database schema
        # This avoids ORM issues with potentially mismatched columns
        # Build the update query dynamically based on available values
        update_parts = []
        update_values = {}
        
        # Always include these fields
        update_parts.append("status = :status")
        update_values["status"] = values_to_update.get("status")
        
        update_parts.append("completed_at = :completed_at")
        update_values["completed_at"] = values_to_update.get("completed_at")
        
        # Only include fields that have values
        if values_to_update.get("error_message") is not None:
            update_parts.append("error_message = :error_message")
            update_values["error_message"] = values_to_update.get("error_message")
        
        if values_to_update.get("results") is not None:
            update_parts.append("results = :results")
            update_values["results"] = values_to_update.get("results")
        
        if values_to_update.get("parameters") is not None:
            update_parts.append("parameters = :parameters")
            update_values["parameters"] = values_to_update.get("parameters")
        
        # Add run_id for the WHERE clause
        update_values["run_id"] = run_id
        
        # Simplified update query - only include columns we know exist in the database
        update_query = """
        UPDATE outlier_detection_runs
        SET status = :status, 
            completed_at = :completed_at, 
            error_message = :error_message,
            parameters = :parameters
        WHERE id = :run_id
        """
        
        # Prepare update values - only include parameters if we have them
        update_values = {
            "status": values_to_update.get("status", "completed"),
            "completed_at": values_to_update.get("completed_at", datetime.utcnow()),
            "error_message": values_to_update.get("error_message"),
            "parameters": values_to_update.get("parameters"),
            "run_id": run_id
        }
        
        try:
            await async_db.execute(text(update_query), update_values)
            await async_db.commit()
            print(f"TASK_WRAPPER: Successfully updated outlier detection run {run_id}")
        except Exception as e:
            await async_db.rollback()
            print(f"TASK_WRAPPER: Error updating outlier detection run: {str(e)}")
            # Try a minimal update with just status and error message as fallback
            try:
                # Minimal update with only essential fields
                minimal_update = """
                UPDATE outlier_detection_runs
                SET status = :status,
                    completed_at = :completed_at,
                    error_message = :error_message
                WHERE id = :run_id
                """
                
                minimal_update_values = {
                    "status": "error",
                    "completed_at": datetime.utcnow(),
                    "error_message": f"Failed to update results: {str(e)}",
                    "run_id": run_id
                }
                
                await async_db.execute(text(minimal_update), minimal_update_values)
                await async_db.commit()
                print(f"TASK_WRAPPER: Completed minimal fallback update for run {run_id}")
            except Exception as inner_e:
                await async_db.rollback()
                print(f"TASK_WRAPPER: Even minimal update failed: {str(inner_e)}")

        # Update session step to success if applicable
        if session_step_id:
            try:
                # Primary path: async service update
                upd = await session_service.update_step(
                    async_db,
                    None,
                    UUID(session_step_id),
                    status="success",
                    finished_at=datetime.utcnow(),
                    run_ref_type="outliers",
                    run_ref_id=str(run_id),
                )
                print(f"TASK_WRAPPER: async session step update->success OK step_id={session_step_id} (row exists={bool(upd)})")
            except Exception as upd_e:
                print(f"TASK_WRAPPER: async update_step failed on success: {upd_e}. Trying direct SQL...")
                try:
                    await async_db.execute(
                        text(
                            """
                            UPDATE session_steps
                            SET status = :status, run_ref_type = :run_ref_type, run_ref_id = :run_ref_id, finished_at = :finished_at
                            WHERE id = :step_id
                            """
                        ),
                        {
                            "status": "success",
                            "run_ref_type": "outliers",
                            "run_ref_id": str(run_id),
                            "finished_at": datetime.utcnow(),
                            "step_id": UUID(session_step_id),
                        },
                    )
                    await async_db.commit()
                    print(f"TASK_WRAPPER: direct SQL session step update->success OK step_id={session_step_id}")
                except Exception as sql_e:
                    print(f"TASK_WRAPPER: direct SQL update failed on success: {sql_e}. Falling back to sync SessionLocal...")
                    try:
                        await asyncio.to_thread(_update_session_step_sync, session_step_id, "success", None, "outliers", str(run_id))
                        print(f"TASK_WRAPPER: sync fallback session step update->success OK step_id={session_step_id}")
                    except Exception as sync_e:
                        print(f"TASK_WRAPPER: sync fallback failed on success: {sync_e}")
        
        final_status = results_summary.get("pipeline_status", "completed_with_unknown_status")
        background_task_status_dict.pop(task_id_for_status, None)
        print(f"TASK_WRAPPER: Background task completed for run_id: {run_id} with status: {final_status}")
    except Exception as e:
        print(f"TASK_WRAPPER: Error in background task for run_id {run_id}: {e}")
        import traceback
        print(traceback.format_exc())
        
        try:
            # Use raw SQL for error update to avoid ORM issues
            error_update_query = """
            UPDATE outlier_detection_runs
            SET status = :status,
                completed_at = :completed_at,
                error_message = :error_message
            WHERE id = :run_id
            """
            
            error_update_values = {
                "status": "failed",
                "completed_at": datetime.utcnow(),
                "error_message": str(e),
                "run_id": run_id
            }
            
            await async_db.execute(text(error_update_query), error_update_values)
            await async_db.commit()
        except Exception as db_e:
            print(f"TASK_WRAPPER: Error updating database with failure status: {db_e}")
        
        background_task_status_dict[task_id_for_status] = "failed"
        # Update session step to failed if applicable (robust with fallbacks)
        if session_step_id:
            try:
                # Primary path: async service update
                upd = await session_service.update_step(
                    async_db,
                    None,
                    UUID(session_step_id),
                    status="failed",
                    error=str(e),
                    finished_at=datetime.utcnow(),
                    run_ref_type="outliers",
                    run_ref_id=str(run_id),
                )
                print(f"TASK_WRAPPER: async session step update->failed OK step_id={session_step_id} (row exists={bool(upd)})")
            except Exception as upd_e:
                print(f"TASK_WRAPPER: async update_step failed on error: {upd_e}. Trying direct SQL...")
                try:
                    await async_db.execute(
                        text(
                            """
                            UPDATE session_steps
                            SET status = :status, error = :error, run_ref_type = :run_ref_type, run_ref_id = :run_ref_id, finished_at = :finished_at
                            WHERE id = :step_id
                            """
                        ),
                        {
                            "status": "failed",
                            "error": str(e),
                            "run_ref_type": "outliers",
                            "run_ref_id": str(run_id),
                            "finished_at": datetime.utcnow(),
                            "step_id": UUID(session_step_id),
                        },
                    )
                    await async_db.commit()
                    print(f"TASK_WRAPPER: direct SQL session step update->failed OK step_id={session_step_id}")
                except Exception as sql_e:
                    print(f"TASK_WRAPPER: direct SQL update failed on error path: {sql_e}. Falling back to sync SessionLocal...")
                    try:
                        await asyncio.to_thread(_update_session_step_sync, session_step_id, "failed", str(e), "outliers", str(run_id))
                        print(f"TASK_WRAPPER: sync fallback session step update->failed OK step_id={session_step_id}")
                    except Exception as sync_e:
                        print(f"TASK_WRAPPER: sync fallback failed on error path: {sync_e}")
        
        final_status = "failed"
        background_task_status_dict.pop(task_id_for_status, None)
        print(f"TASK_WRAPPER: Background task completed for run_id: {run_id} with status: {final_status}")
    finally:
        await async_db.close()


@router.post("/datasets/{dataset_id}/detect", response_model=DetectOutliersResponse)
async def trigger_outlier_detection(
    request_params: DetectOutliersRequest,
    background_tasks: BackgroundTasks,
    dataset_id: int = Path(..., ge=1),
    db: AsyncSession = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
    session_id: Optional[UUID] = Query(None, description="Optional session to record this run as a step"),
):
    user_id_for_query = int(current_user.id) 
    dataset_stmt = select(DatasetModel).where(DatasetModel.id == dataset_id, DatasetModel.user_id == user_id_for_query)
    result = await db.execute(dataset_stmt)
    dataset_record = result.scalars().first()

    if not dataset_record:
        raise HTTPException(status_code=404, detail="Dataset not found or access denied.")
    
    file_path_from_db = dataset_record.file_path
    print(f"DEBUG: Original file path from DB: '{file_path_from_db}'")
    
    possible_paths = []
    
    possible_paths.append(("Original DB path", file_path_from_db))
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    project_relative_path = os.path.join(project_root, file_path_from_db)
    possible_paths.append(("Project root relative", project_relative_path))
    
    explicit_path = os.path.join(r'C:\Users\HP\OneDrive\Desktop\proejct_pfe\fastapi-react-starter-main', file_path_from_db)
    possible_paths.append(("Explicit project path", explicit_path))
    
    backslash_path = file_path_from_db.replace('/', '\\')
    possible_paths.append(("Backslash converted", backslash_path))
    
    forward_slash_path = file_path_from_db.replace('\\', '/')
    possible_paths.append(("Forward slash converted", forward_slash_path))
    
    found_file = False
    for path_type, path in possible_paths:
        exists = os.path.exists(path)
        print(f"DEBUG: Checking {path_type}: '{path}' - Exists: {exists}")
        if exists:
            file_path_from_db = path
            found_file = True
            print(f"DEBUG: Found file at: '{path}'")
            break
    
    if not found_file:
        error_msg = f"Dataset file path '{file_path_from_db}' for dataset ID {dataset_id} not found on server after trying multiple approaches."
        print(f"ERROR: {error_msg}")
        raise HTTPException(status_code=404, detail="Dataset file not found on server. Please check dataset integrity.")

    run_params_dict = request_params.model_dump()
    # Optionally create a session step prior to run creation
    session_step_id = None
    if session_id is not None:
        try:
            step_row = await session_service.add_step(
                db,
                current_user,
                session_id,
                tool="outliers",
                step="detect",
                algorithm=request_params.outlier_detection_method,
                params=run_params_dict,
            )
            session_step_id = step_row.id
        except PermissionError:
            raise HTTPException(status_code=403, detail="Session not found or access denied")
        except Exception as e:
            print(f"WARNING: Unable to create session step: {e}")
    new_run = OutlierDetectionRun(
        dataset_id=dataset_id,
        user_id=user_id_for_query,
        status="queued",
        parameters_json=run_params_dict, 
        started_at=datetime.utcnow()
    )
    db.add(new_run)
    await db.commit()
    await db.refresh(new_run)

    if new_run.id is None: 
        raise HTTPException(status_code=500, detail="Failed to create outlier detection run record.")

    # Link session step to the created run and mark running
    if session_step_id is not None:
        try:
            await session_service.update_step(
                db,
                current_user,
                step_id=session_step_id,
                status="running",
                run_ref_type="outliers",
                run_ref_id=str(new_run.id),
            )
        except Exception as e:
            print(f"WARNING: Failed to update session step after run creation: {e}")

    task_id_for_status_dict = f"outlier_run_{new_run.id}"
    background_task_status_dict[task_id_for_status_dict] = "queued"

    background_tasks.add_task(
        run_outlier_detection_pipeline_task,
        run_id=new_run.id,
        dataset_id=dataset_id,
        user_id=user_id_for_query,
        file_path=file_path_from_db,
        run_parameters=run_params_dict,
        session_step_id=str(session_step_id) if session_step_id else None,
    )

    print(f"ROUTE: Outlier detection task {task_id_for_status_dict} queued for dataset {dataset_id}, run_id {new_run.id}")
    return DetectOutliersResponse(
        message="Outlier detection pipeline initiated.",
        task_id=task_id_for_status_dict, 
        outlier_run_id=new_run.id,
        run_id=new_run.id  # Include run_id as well for compatibility
    )

@router.get("/runs/{run_id}/status", response_model=Dict[str, Any])
async def get_outlier_detection_status(
    run_id: int = Path(..., ge=1, description="The ID of the outlier detection run."),
    current_user_id: int = Depends(get_current_user_id)
):
    user_id_for_query = int(current_user_id)
    
    # Use raw SQL to only query columns that actually exist in the database
    query = text("""
        SELECT id, dataset_id, user_id, task_id, status, parameters, 
               started_at, completed_at, error_message
        FROM outlier_detection_runs
        WHERE id = :run_id
    """)
    
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(query, {"run_id": run_id})
            run_record = result.mappings().first()
    except Exception as e:
        # Graceful degraded response when DB is temporarily unavailable
        task_id_for_status = f"outlier_run_{run_id}"
        current_mem_status = background_task_status_dict.get(task_id_for_status, "unknown")
        # IMPORTANT: Return a plain dict to avoid Pydantic validation on missing fields
        response_dict: Dict[str, Any] = {
            "task_id": task_id_for_status,
            "status": current_mem_status,
            "run_details": {
                "id": run_id,
                "user_id": user_id_for_query,
                "task_id": task_id_for_status,
                "status": current_mem_status,
                # Provide minimal placeholders; frontend mainly needs status while DB is down
                "dataset_id": None,
                "parameters_json": None,
                "started_at": None,
                "completed_at": None,
                "error_message": f"Database temporarily unavailable: {str(e)}",
            },
        }
        # Copy critical fields also to top-level for frontend compatibility
        for key, value in list(response_dict.get("run_details", {}).items()):
            if key not in response_dict:
                response_dict[key] = value
        print("STATUS ENDPOINT: DB unavailable, returning in-memory status only")
        return response_dict
    
    if not run_record:
        raise HTTPException(status_code=404, detail=f"Outlier detection run with ID {run_id} not found.")

    if run_record["user_id"] != user_id_for_query:
        raise HTTPException(status_code=403, detail="Not authorized to view this outlier detection run.")

    task_id_for_status = run_record["task_id"] if run_record["task_id"] else f"outlier_run_{run_id}"
    current_mem_status = background_task_status_dict.get(task_id_for_status)

    display_status = run_record["status"]
    if run_record["status"] in ["queued", "processing_setup", "orchestrator_running", "generating_visualizations", "saving_results"] and current_mem_status:
        display_status = current_mem_status
    
    # Create a run details object with all necessary fields for the frontend
    run_details = {
        "id": run_record["id"],
        "dataset_id": run_record["dataset_id"],
        "user_id": run_record["user_id"],
        "task_id": run_record["task_id"],
        "status": run_record["status"],
        "parameters_json": run_record["parameters"],  # Map to the expected field name
        "started_at": run_record["started_at"],
        "completed_at": run_record["completed_at"],
        "error_message": run_record["error_message"]
    }
    
    # Extract additional fields from parameters JSON for frontend compatibility
    if run_record["parameters"]:
        try:
            # Parse parameters if it's a string
            params = run_record["parameters"]
            if isinstance(params, str):
                params = json.loads(params)
                
            # Add fields that the frontend expects
            # First add the run_id and outlier_run_id for compatibility
            run_details["run_id"] = run_record["id"]
            run_details["outlier_run_id"] = run_record["id"]
            
            # Extract results data if available - this is critical for frontend display
            if "results" in params:
                results = params["results"]
                if isinstance(results, str):
                    results = json.loads(results)
                run_details["outlier_results"] = results
            elif "outlier_results_list" in params:
                # Alternative field name that might be used
                results = params["outlier_results_list"]
                if isinstance(results, str):
                    results = json.loads(results)
                run_details["outlier_results"] = results
            
            # Look for metrics in different possible locations
            if "evaluation_metrics" in params:
                run_details["evaluation_metrics"] = params["evaluation_metrics"]
            elif "metrics" in params:
                run_details["evaluation_metrics"] = params["metrics"]
            
            # Extract metadata fields from different possible locations
            if "summary" in params and isinstance(params["summary"], dict):
                for key, value in params["summary"].items():
                    run_details[key] = value
            
            # Also check for direct metadata fields
            for key in ["total_points_processed", "total_outliers_detected", "num_numerical_features_used"]:
                if key in params:
                    run_details[key] = params[key]
            
            # Extract visualization paths from different possible locations
            
            # Check for visualization_paths dictionary
            if "visualization_paths" in params and isinstance(params["visualization_paths"], dict):
                for path_key, path_value in params["visualization_paths"].items():
                    run_details[path_key] = path_value
            
            # Check for visualizations nested dictionary
            if "visualizations" in params and isinstance(params["visualizations"], dict):
                for viz_key, viz_value in params["visualizations"].items():
                    # Map to expected frontend keys
                    if viz_key == "scatter_plot":
                        run_details["scatter_plot_path"] = viz_value
                    elif viz_key == "pca_plot":
                        run_details["scatter_plot_pca_path"] = viz_value
                    elif viz_key == "outlier_distribution":
                        run_details["outlier_distribution_path"] = viz_value
                    else:
                        # For any other visualization keys
                        run_details[viz_key + "_path"] = viz_value
            
            # Direct path checks
            for path_key in ["scatter_plot_pca_path", "scatter_plot_path", "outlier_distribution_path"]:
                if path_key in params:
                    run_details[path_key] = params[path_key]
        except Exception as e:
            print(f"Error extracting data from parameters JSON: {str(e)}")
    
    # Create the response with enhanced details
    response = {
        "task_id": task_id_for_status,
        "status": display_status,
        "run_details": run_details
    }
    response_dict = response
    
    # Add key fields at top level for frontend compatibility
    if run_details:
        # Copy all run_details fields to top level for maximum compatibility
        for key, value in run_details.items():
            # Don't overwrite existing top-level fields
            if key not in response_dict:
                response_dict[key] = value
        
        # Ensure critical fields are definitely present at top level
        critical_fields = [
            "outlier_results", "evaluation_metrics", "scatter_plot_pca_path", 
            "outlier_distribution_path", "scatter_plot_path", "pca_plot_path",
            "total_points_processed", "total_outliers_detected"
        ]
        
        # Add each critical field if it exists in run_details
        for key in critical_fields:
            if key in run_details:
                response_dict[key] = run_details[key]
    
    # Debug logging
    print("\n\n==== DEBUG: OUTLIER DETECTION STATUS RESPONSE ====")
    print(f"Response structure: {json.dumps({k: type(v).__name__ for k, v in response_dict.items()}, indent=2)}")
    print(f"Has outlier_results: {'outlier_results' in response_dict}")
    print(f"Has evaluation_metrics: {'evaluation_metrics' in response_dict}")
    print(f"Has scatter_plot_pca_path: {'scatter_plot_pca_path' in response_dict}")
    print(f"Has run_details: {'run_details' in response_dict}")
    if 'run_details' in response_dict:
        print(f"run_details keys: {list(response_dict['run_details'].keys())}")
    print("==== END DEBUG ====\n\n")
    
    return response_dict



@router.get("/runs/{run_id}/download")
async def download_outlier_run_results(
    run_id: int = Path(..., ge=1, description="The ID of the outlier detection run."),
    db: AsyncSession = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    """
    Download the outlier detection results for a given run as a CSV.

    Results are read from the run's parameters JSON under either:
    - 'results' (preferred), or
    - legacy 'outlier_results_list'
    """
    # Fetch run and verify ownership
    result = await db.execute(
        text(
            """
            SELECT id, user_id, parameters
            FROM outlier_detection_runs
            WHERE id = :run_id
            """
        ),
        {"run_id": run_id},
    )
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    if int(row["user_id"]) != int(current_user.id):
        # Hide existence from unauthorized users
        raise HTTPException(status_code=404, detail="Run not found")

    params = row.get("parameters")
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except Exception:
            params = None

    # Extract results list from parameters
    results_list = None
    if isinstance(params, dict):
        # Preferred key
        val = params.get("results")
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except Exception:
                pass
        if isinstance(val, list):
            results_list = val

        # Legacy key fallback
        if results_list is None:
            val2 = params.get("outlier_results_list")
            if isinstance(val2, str):
                try:
                    val2 = json.loads(val2)
                except Exception:
                    pass
            if isinstance(val2, list):
                results_list = val2

    if not results_list:
        raise HTTPException(status_code=404, detail="No results available for this run")

    # Convert to CSV
    try:
        df = pd.DataFrame(results_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to construct results DataFrame: {e}")

    csv_text = df.to_csv(index=False)
    filename = f"outliers_run_{run_id}.csv"
    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return Response(content=csv_text, media_type="text/csv", headers=headers)


@router.get("/runs")
async def list_outlier_detection_runs(
    db: AsyncSession = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
    dataset_id: Optional[int] = Query(None, ge=1),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List outlier detection runs for the current user, optionally filtered by dataset ID."""
    user_id = int(current_user.id)
    
    # Build the SQL query with parameters
    params = {"user_id": user_id, "limit": limit, "offset": offset}
    
    # Use raw SQL to only query columns that actually exist in the database
    sql_query = """
        SELECT id, dataset_id, user_id, task_id, status, parameters, 
               started_at, completed_at, error_message
        FROM outlier_detection_runs
        WHERE user_id = :user_id
    """
    
    if dataset_id:
        sql_query += " AND dataset_id = :dataset_id"
        params["dataset_id"] = dataset_id
    
    sql_query += " ORDER BY id DESC LIMIT :limit OFFSET :offset"
    
    result = await db.execute(text(sql_query), params)
    rows = result.mappings().all()
    
    # Convert rows to dictionaries with enhanced fields for frontend compatibility
    runs = []
    for row in rows:
        run_dict = {
            "id": row["id"],
            "dataset_id": row["dataset_id"],
            "user_id": row["user_id"],
            "task_id": row["task_id"],
            "status": row["status"],
            "parameters": row["parameters"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "error_message": row["error_message"],
            # Add these fields for frontend compatibility
            "run_id": row["id"],
            "outlier_run_id": row["id"]
        }
        
        # Extract additional fields from parameters JSON
        if row["parameters"]:
            try:
                # Parse parameters if it's a string
                params_data = row["parameters"]
                if isinstance(params_data, str):
                    params_data = json.loads(params_data)
                
                # Extract results data if available
                if "results" in params_data:
                    results = params_data["results"]
                    if isinstance(results, str):
                        results = json.loads(results)
                    run_dict["outlier_results"] = results
                
                # Extract metrics if available
                if "evaluation_metrics" in params_data:
                    run_dict["evaluation_metrics"] = params_data["evaluation_metrics"]
                
                # Extract metadata fields
                for key in ["total_points_processed", "total_outliers_detected"]:
                    if key in params_data:
                        run_dict[key] = params_data[key]
                
                # Extract visualization paths
                if "visualization_paths" in params_data:
                    viz_paths = params_data["visualization_paths"]
                    # Map specific paths to expected frontend fields
                    if "scatter_plot_path" in viz_paths:
                        run_dict["scatter_plot_pca_path"] = viz_paths["scatter_plot_path"]
                    if "outlier_distribution_path" in viz_paths:
                        run_dict["outlier_distribution_path"] = viz_paths["outlier_distribution_path"]
                    if "pca_plot_path" in viz_paths:
                        run_dict["pca_plot_path"] = viz_paths["pca_plot_path"]
            except Exception as e:
                print(f"Error extracting data from parameters JSON for run {row['id']}: {str(e)}")
        
        runs.append(run_dict)
    
    return runs

@router.post("/runs/{run_id}/evaluate", response_model=EvaluationMetricsSchema)
async def evaluate_outlier_detection(
    run_id: int,
    ground_truth_data: List[GroundTruthDataPoint] = None,
    current_user: UserSchema = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    use_original_dataset: bool = Query(True, description="Whether to use the original dataset's Class column as ground truth")
):
    """Evaluate outlier detection results against ground truth data"""
    try:
        # Get the run details to check if it's One-Class SVM
        run = await db.execute(select(OutlierDetectionRun).where(
            OutlierDetectionRun.id == run_id,
            OutlierDetectionRun.user_id == current_user.id
        ))
        run = run.scalar_one_or_none()
        
        if not run:
            raise HTTPException(status_code=404, detail=f"Outlier detection run {run_id} not found for current user")
        
        # Check if this is a One-Class SVM run
        is_one_class_svm = False
        if run.parameters_json:
            params = run.parameters_json
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except:
                    params = {}
            
            if params.get("outlier_detection_method") in ["ocsvm", "one_class_svm"]:
                is_one_class_svm = True
                print(f"Detected One-Class SVM run, prioritizing original dataset evaluation")
        
        evaluation_service = OutlierEvaluationService(
            outlier_run_id=run_id,
            dataset_id=run.dataset_id,
            user_id=current_user.id
        )
        
        # For One-Class SVM, always try to use the original dataset first
        # For other algorithms, respect the use_original_dataset parameter
        if is_one_class_svm or use_original_dataset:
            print(f"Attempting to evaluate using original dataset's Class column...")
            metrics = evaluation_service.calculate_metrics_from_original_dataset()
            
            # If metrics were successfully calculated, return them
            if metrics and any([metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score, metrics.auc_roc]):
                print(f"Successfully evaluated using original dataset's Class column")
                # Ensure source is set correctly
                metrics.source = "original_dataset"
                return metrics
            else:
                print(f"Failed to evaluate using original dataset's Class column, falling back to provided ground truth data")
        
        # Fall back to using provided ground truth data if original dataset evaluation failed or wasn't requested
        if ground_truth_data:
            print(f"Evaluating using provided ground truth data...")
            metrics = evaluation_service.calculate_metrics(ground_truth_data)
            # Set source for metrics from provided ground truth
            metrics.source = "provided_ground_truth"
            return metrics
        else:
            # If we get here, both methods failed or weren't available
            return EvaluationMetricsSchema(
                message="No ground truth data available. Please provide ground truth data or ensure the original dataset contains a 'Class' column."
            )
    except Exception as e:
        print(f"Error in evaluate_outlier_detection: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error evaluating outlier detection: {str(e)}")

# --- ENDPOINT FOR FILE UPLOAD OUTLIER DETECTION ---
@router.post("/upload/detect", response_model=DetectOutliersResponse)
async def detect_outliers_from_upload(
    background_tasks: BackgroundTasks,
    algorithm: str = Form(...),
    parameters: str = Form(...),
    file: UploadFile = File(...),
    true_labels_file: Optional[UploadFile] = File(None),
    save_visualizations: bool = Form(True),
    include_visualizations: bool = Form(True),
    db: AsyncSession = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
    session_id: Optional[UUID] = Query(None, description="Optional session to record this run as a step"),
):
    """
    Detect outliers from an uploaded file.
    
    This endpoint allows users to upload a dataset file and perform outlier detection on it.
    Optionally, users can also upload a file with ground truth labels for evaluation.
    """
    try:
        # Create a unique directory for this upload
        upload_dir = os.path.join(settings.UPLOAD_DIR, f"upload_{uuid.uuid4().hex}")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the uploaded file
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Save the true labels file if provided
        true_labels_path = None
        if true_labels_file:
            true_labels_path = os.path.join(upload_dir, true_labels_file.filename)
            with open(true_labels_path, "wb") as buffer:
                shutil.copyfileobj(true_labels_file.file, buffer)
        
        # Parse parameters
        try:
            parsed_parameters = json.loads(parameters)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid parameters format")
        
        # Create a temporary dataset record
        dataset = DatasetModel(
            filename=file.filename,
            file_path=file_path,
            user_id=current_user.id,
            file_type=os.path.splitext(file.filename)[1][1:],  # Extract extension without dot
            analysis_metadata={"temporary": True}  # Use analysis_metadata to mark as temporary
        )
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)
        
        # Create request params object for compatibility with schema (not used downstream)
        request_params = DetectOutliersRequest(
            outlier_detection_method=algorithm,
            save_visualizations=save_visualizations,
        )
        
        # Optionally create a session step prior to run creation
        session_step_id = None
        if session_id is not None:
            try:
                step_row = await session_service.add_step(
                    db,
                    current_user,
                    session_id,
                    tool="outliers",
                    step="detect",
                    algorithm=algorithm,
                    params={
                        "outlier_detection_method": algorithm,
                        "parameters": parsed_parameters,
                        "save_visualizations": save_visualizations,
                        "include_visualizations": include_visualizations,
                    },
                )
                session_step_id = step_row.id
            except PermissionError:
                raise HTTPException(status_code=403, detail="Session not found or access denied")
            except Exception as e:
                print(f"WARNING: Unable to create session step: {e}")
        
        # Use raw SQL to insert only the minimal required columns
        # This avoids issues with model/database schema mismatches
        
        # Create parameters JSON
        parameters_json = {
            "outlier_detection_method": algorithm,
            "parameters": parsed_parameters,
            "save_visualizations": save_visualizations,
            "include_visualizations": include_visualizations
        }
        
        # Convert Python dict to JSON string for PostgreSQL
        parameters_json_str = json.dumps(parameters_json)
        
        # Insert with only the essential columns we know exist
        # Use text() for raw SQL with proper parameter binding
        query = text("""
            INSERT INTO outlier_detection_runs 
            (dataset_id, user_id, status, parameters) 
            VALUES (:dataset_id, :user_id, :status, :parameters)
            RETURNING id
        """)
        
        # PostgreSQL will automatically convert the JSON string to JSONB
        result = await db.execute(
            query, 
            {
                "dataset_id": dataset.id, 
                "user_id": current_user.id,
                "status": "pending",
                "parameters": parameters_json_str
            }
        )
        run_id = result.scalar_one()
        await db.commit()
        
        # Create a minimal run object for the response
        run = OutlierDetectionRun(id=run_id, dataset_id=dataset.id, user_id=current_user.id, status="pending")
        
        # Link session step to the created run and mark running
        if session_step_id is not None:
            try:
                await session_service.update_step(
                    db,
                    current_user,
                    step_id=session_step_id,
                    status="running",
                    run_ref_type="outliers",
                    run_ref_id=str(run.id),
                )
            except Exception as e:
                print(f"WARNING: Failed to update session step after run creation: {e}")
        
        # Add true labels path to run parameters if provided
        run_parameters = {
            "outlier_detection_method": algorithm,
            "parameters": parsed_parameters,
            "save_visualizations": save_visualizations,
            "include_visualizations": include_visualizations
        }
        
        if true_labels_path:
            run_parameters["true_labels_path"] = true_labels_path
        
        # Start background task
        background_tasks.add_task(
            run_outlier_detection_pipeline_task,
            run_id=run.id,
            dataset_id=dataset.id,
            user_id=current_user.id,
            file_path=file_path,
            run_parameters=run_parameters,
            session_step_id=str(session_step_id) if session_step_id else None,
        )
        
        # Return both outlier_run_id and run_id for better frontend compatibility
        # The frontend may be looking for either field
        return {
            "outlier_run_id": run.id,
            "run_id": run.id,  # Include run_id as well for compatibility
            "message": "Outlier detection task started",
            "task_id": f"outlier_run_{run.id}"  # Include a task_id for tracking
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")