# backend/app/services/outlier_detection/pipeline_service.py
import pandas as pd
import numpy as np
import time
import os
from typing import Dict, Any, Optional, List, Tuple
import asyncio
import json

from app.config.config import get_settings
from app.services.outlier_detection.feature_extraction.autoencoder_service import AutoencoderService
from app.services.outlier_detection.feature_extraction.pca_service import PCAService
from app.services.outlier_detection.feature_extraction.isomap_service import IsomapService
from app.services.outlier_detection.clustering.dbscan_service import DBSCANService
from app.services.outlier_detection.clustering.denclue_service import DenclueService
from app.services.outlier_detection.clustering.optics_service import OPTICSService
from app.services.outlier_detection.anomaly_detection.isolation_forest_service import IsolationForestService
from app.services.outlier_detection.anomaly_detection.lof_service import LOFService
from app.services.outlier_detection.anomaly_detection.ocsvm_service import OCSVMService
from app.services.outlier_detection.evaluation_service import EvaluationService
from app.utils.visualization import generate_and_save_outlier_visualizations

settings = get_settings()

class OutlierDetectionPipelineService:
    def __init__(
        self,
        dataset_id: int,
        user_id: int,
        config: Dict[str, Any],
        labeled_data_path: Optional[str] = None,
        base_path: Optional[str] = None
    ):
        self.dataset_id = dataset_id
        self.user_id = user_id
        self.config = config
        self.labeled_data_path = labeled_data_path
        self.base_path = base_path or settings.ARTIFACTS_PATH
        
        # Extract configuration
        self.feature_extraction_config = config.get('featureExtraction', {})
        self.clustering_config = config.get('clustering', {})
        self.anomaly_detection_config = config.get('anomalyDetection', {})
        self.general_config = config.get('general', {})
        
        # Initialize services based on configuration
        self._init_feature_extraction()
        self._init_clustering()
        self._init_anomaly_detection()
        
        # Initialize evaluation service if labeled data is provided
        self.evaluation_service = None
        if self.labeled_data_path:
            self.evaluation_service = EvaluationService(
                self.dataset_id,
                self.user_id,
                self.labeled_data_path,
                base_path=self.base_path
            )
    
    def _init_feature_extraction(self):
        algorithm = self.feature_extraction_config.get('algorithm', 'pca')
        params = self.feature_extraction_config.get('parameters', {})
        
        if algorithm == 'autoencoder':
            self.feature_extraction = AutoencoderService(
                self.dataset_id,
                self.user_id,
                latent_dim=params.get('latentDim', 16),
                epochs=params.get('epochs', 25),
                batch_size=params.get('batchSize', 64),
                random_state=self.general_config.get('random_state', 42),
                base_path=self.base_path
            )
        elif algorithm == 'pca':
            self.feature_extraction = PCAService(
                self.dataset_id,
                self.user_id,
                n_components=params.get('pcaComponents', 2),
                random_state=self.general_config.get('random_state', 42),
                base_path=self.base_path
            )
        elif algorithm == 'isomap':
            self.feature_extraction = IsomapService(
                self.dataset_id,
                self.user_id,
                n_components=params.get('isomapComponents', 2),
                n_neighbors=params.get('isomapNeighbors', 5),
                base_path=self.base_path
            )
        else:
            raise ValueError(f"Unsupported feature extraction algorithm: {algorithm}")
    
    def _init_clustering(self):
        algorithm = self.clustering_config.get('algorithm', 'dbscan')
        params = self.clustering_config.get('parameters', {})
        
        if algorithm == 'dbscan':
            self.clustering = DBSCANService(
                self.dataset_id,
                self.user_id,
                eps=params.get('eps', 0.5),
                min_samples=params.get('minSamples', 5),
                base_path=self.base_path
            )
        elif algorithm == 'denclue':
            self.clustering = DenclueService(
                self.dataset_id,
                self.user_id,
                h=params.get('denclueH', 0.1),
                eps=params.get('denclueEps', 0.0001),
                base_path=self.base_path
            )
        elif algorithm == 'optics':
            self.clustering = OPTICSService(
                self.dataset_id,
                self.user_id,
                min_samples=params.get('opticsMinSamples', 5),
                max_eps=params.get('opticsMaxEps', float('inf')),
                xi=params.get('opticsXi', 0.05),
                base_path=self.base_path
            )
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
    
    def _init_anomaly_detection(self):
        algorithm = self.anomaly_detection_config.get('algorithm', 'isolation_forest')
        params = self.anomaly_detection_config.get('parameters', {})
        
        if algorithm == 'isolation_forest':
            self.anomaly_detection = IsolationForestService(
                self.dataset_id,
                self.user_id,
                n_estimators=params.get('nEstimators', 100),
                contamination=params.get('contamination', 0.05),
                max_samples=params.get('maxSamples', 256),
                random_state=self.general_config.get('random_state', 42),
                base_path=self.base_path
            )
        elif algorithm == 'lof':
            self.anomaly_detection = LOFService(
                self.dataset_id,
                self.user_id,
                n_neighbors=params.get('lofNeighbors', 20),
                contamination=params.get('lofContamination', 0.1),
                random_state=self.general_config.get('random_state', 42),
                base_path=self.base_path
            )
        elif algorithm == 'one_class_svm':
            self.anomaly_detection = OCSVMService(
                self.dataset_id,
                self.user_id,
                nu=params.get('ocsvmNu', 0.1),
                kernel=params.get('ocsvmKernel', 'rbf'),
                gamma=params.get('ocsvmGamma', 'scale'),
                base_path=self.base_path
            )

        else:
            raise ValueError(f"Unsupported anomaly detection algorithm: {algorithm}")
    
    async def run_pipeline(
        self,
        data_path: str,
        run_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the complete outlier detection pipeline
        
        Args:
            data_path: Path to the input data file
            run_id: Optional run ID for tracking
            
        Returns:
            Dictionary with pipeline results and metadata
        """
        start_time = time.time()
        
        print(f"PIPELINE: Starting outlier detection pipeline for dataset {self.dataset_id}")
        print(f"PIPELINE: Configuration: {json.dumps(self.config, indent=2)}")
        
        # Step 1: Feature Extraction
        print("PIPELINE: Step 1 - Feature Extraction")
        latent_features_df, fe_metadata = self.feature_extraction.extract_features(data_path)
        
        # Step 2: Clustering
        print("PIPELINE: Step 2 - Clustering")
        clustering_results_df, clustering_metadata = self.clustering.cluster(latent_features_df)
        
        # Step 3: Anomaly Detection (per cluster if applicable)
        print("PIPELINE: Step 3 - Anomaly Detection")
        
        # Merge clustering results with latent features
        merged_df = pd.merge(
            latent_features_df,
            clustering_results_df[['original_index', 'cluster_label']],
            left_index=True,
            right_on='original_index'
        )
        
        # Get unique clusters
        unique_clusters = merged_df['cluster_label'].unique()
        
        # Initialize results containers
        all_outlier_results = []
        all_outlier_metadata = {}
        cluster_outlier_counts = {}
        
        # Process each cluster
        for cluster_id in unique_clusters:
            # Skip noise cluster (-1) if configured to do so
            if cluster_id == -1 and not self.general_config.get('process_noise_cluster', True):
                continue
                
            # Filter data for this cluster
            cluster_data = merged_df[merged_df['cluster_label'] == cluster_id].drop(columns=['cluster_label'])
            
            # Set index back to original_index for proper alignment
            cluster_features = cluster_data.set_index('original_index')
            
            # Skip if cluster is too small
            min_cluster_size = self.general_config.get('min_cluster_size', 10)
            if len(cluster_features) < min_cluster_size:
                print(f"PIPELINE: Skipping cluster {cluster_id} with only {len(cluster_features)} samples (min: {min_cluster_size})")
                continue
                
            print(f"PIPELINE: Processing cluster {cluster_id} with {len(cluster_features)} samples")
            
            try:
                # Run anomaly detection on this cluster
                cluster_results, cluster_metadata = self.anomaly_detection.fit_predict(
                    cluster_features,
                    cluster_id=cluster_id
                )
                
                # Add cluster label to results
                cluster_results['final_cluster_label'] = cluster_id
                
                # Store results
                all_outlier_results.append(cluster_results)
                all_outlier_metadata[f"cluster_{cluster_id}"] = cluster_metadata
                cluster_outlier_counts[cluster_id] = cluster_metadata.get('outlier_count', 0)
                
            except Exception as e:
                print(f"PIPELINE: Error processing cluster {cluster_id}: {e}")
        
        # Combine results from all clusters
        if all_outlier_results:
            combined_results = pd.concat(all_outlier_results, ignore_index=False)
        else:
            raise ValueError("PIPELINE: No valid clusters to process")
            
        # Calculate overall metrics
        total_samples = sum(metadata.get('total_samples', 0) for metadata in all_outlier_metadata.values())
        total_outliers = sum(metadata.get('outlier_count', 0) for metadata in all_outlier_metadata.values())
        
        # Generate visualizations
        vis_paths = await generate_and_save_outlier_visualizations(
            self.dataset_id,
            run_id or 0,
            latent_features_df,
            combined_results,
            self.base_path
        )
        
        # Evaluate if labeled data is available
        evaluation_metrics = {}
        if self.evaluation_service:
            try:
                evaluation_metrics = await self.evaluation_service.evaluate(
                    combined_results,
                    run_id=run_id
                )
                print(f"PIPELINE: Evaluation metrics: {evaluation_metrics}")
            except Exception as e:
                print(f"PIPELINE: Error during evaluation: {e}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare final results
        pipeline_results = {
            'total_samples': total_samples,
            'outlier_count': total_outliers,
            'outlier_percentage': (total_outliers / total_samples * 100) if total_samples > 0 else 0,
            'processing_time': processing_time,
            'feature_extraction': fe_metadata,
            'clustering': clustering_metadata,
            'anomaly_detection': {
                'cluster_outlier_counts': cluster_outlier_counts,
                'total_outliers': total_outliers
            },
            'evaluation_metrics': evaluation_metrics,
            'visualization_paths': vis_paths,
            'outlier_results_df': combined_results
        }
        
        print(f"PIPELINE: Pipeline completed in {processing_time:.2f} seconds")
        print(f"PIPELINE: Found {total_outliers} outliers out of {total_samples} samples")
        
        return pipeline_results
