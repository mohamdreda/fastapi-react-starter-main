"""
Clustering service for data deduplication.

This module implements clustering techniques for grouping duplicate records:
- Graph-based: Connected components, community detection
- Density-based: Integration with existing DBSCAN, OPTICS, DENCLUE algorithms
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Set, Optional
import os
import json
import logging
import networkx as nx
from sklearn.cluster import DBSCAN, OPTICS
import matplotlib.pyplot as plt
import io
import base64

from app.config.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

def _get_clustering_artifact_path(
    base_path: str,
    dataset_id: int,
    user_id: int,
    artifact_name: str
) -> str:
    """Create and return the path for clustering artifacts."""
    dir_path = os.path.join(base_path, f"user_{user_id}", f"dataset_{dataset_id}", "deduplication", "clustering")
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, artifact_name)

def graph_based_clustering(
    classified_pairs: List[Dict[str, Any]],
    method: str = 'connected_components',
    confidence_threshold: float = 0.7
) -> List[List[int]]:
    """
    Cluster duplicate records using graph-based methods.
    
    Args:
        classified_pairs: List of classified pairs
        method: Clustering method ('connected_components', 'community_detection')
        confidence_threshold: Minimum confidence for including edges
        
    Returns:
        List of clusters, where each cluster is a list of record IDs
    """
    # Create a graph
    G = nx.Graph()
    
    # Add edges for duplicate pairs
    for pair in classified_pairs:
        score = pair.get("confidence")
        if score is None:
            score = pair.get("similarity")
        if score is None:
            continue  # skip rows without a usable score
        if score >= confidence_threshold:
            G.add_edge(pair["record1_id"], pair["record2_id"], weight=score)
    
    # Apply the appropriate clustering method
    if method == 'connected_components':
        # Find connected components
        clusters = list(nx.connected_components(G))
        
    elif method == 'community_detection':
        # Use Louvain community detection
        try:
            from community import best_partition
            
            # Find communities
            partition = best_partition(G)
            
            # Group nodes by community
            communities = {}
            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(node)
            
            # Convert to list of clusters
            clusters = list(communities.values())
            
        except ImportError:
            # Fall back to connected components if python-louvain is not available
            logger.warning("python-louvain package not available, falling back to connected components")
            clusters = list(nx.connected_components(G))
    
    else:
        # Default to connected components
        clusters = list(nx.connected_components(G))
    
    # Convert sets to lists for JSON serialization
    clusters = [list(cluster) for cluster in clusters]
    
    # Filter out singleton clusters (not duplicates)
    clusters = [cluster for cluster in clusters if len(cluster) > 1]
    
    return clusters

def density_based_clustering(
    classified_pairs: List[Dict[str, Any]],
    method: str = 'dbscan',
    params: Dict[str, Any] = None
) -> List[List[int]]:
    """
    Cluster duplicate records using density-based methods.
    
    Args:
        classified_pairs: List of classified pairs
        method: Clustering method ('dbscan', 'optics')
        params: Additional parameters for the clustering method
        
    Returns:
        List of clusters, where each cluster is a list of record IDs
    """
    if params is None:
        params = {}
    
    # Extract record IDs and confidence scores
    records = set()
    for pair in classified_pairs:
        records.add(pair["record1_id"])
        records.add(pair["record2_id"])
    
    # Create a distance matrix (1 - confidence)
    n = len(records)
    record_to_idx = {rec: i for i, rec in enumerate(records)}
    idx_to_record = {i: rec for rec, i in record_to_idx.items()}
    
    # Initialize distance matrix with 1.0 (maximum distance)
    distance_matrix = np.ones((n, n))
    
    # Fill in distances for known pairs
    for pair in classified_pairs:
        i = record_to_idx[pair["record1_id"]]
        j = record_to_idx[pair["record2_id"]]
        
        # Distance is 1 - confidence
        distance = 1.0 - pair["confidence"]
        
        # Set distance in both directions
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance
    
    # Set diagonal to 0 (distance to self)
    np.fill_diagonal(distance_matrix, 0)
    
    # Apply the appropriate clustering method
    if method == 'dbscan':
        # Extract parameters
        eps = params.get('eps', 0.3)
        min_samples = params.get('min_samples', 2)
        
        # Run DBSCAN
        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='precomputed'
        ).fit(distance_matrix)
        
        # Get cluster labels
        labels = clustering.labels_
        
    elif method == 'optics':
        # Extract parameters
        min_samples = params.get('min_samples', 2)
        max_eps = params.get('max_eps', 0.5)
        
        # Run OPTICS
        clustering = OPTICS(
            min_samples=min_samples,
            max_eps=max_eps,
            metric='precomputed'
        ).fit(distance_matrix)
        
        # Get cluster labels
        labels = clustering.labels_
        
    else:
        # Default to DBSCAN
        clustering = DBSCAN(
            eps=0.3,
            min_samples=2,
            metric='precomputed'
        ).fit(distance_matrix)
        
        # Get cluster labels
        labels = clustering.labels_
    
    # Group records by cluster
    clusters = {}
    for i, label in enumerate(labels):
        # Skip noise points (label -1)
        if label == -1:
            continue
        
        if label not in clusters:
            clusters[label] = []
        
        # Add record ID to cluster
        record_id = idx_to_record[i]
        clusters[label].append(record_id)
    
    # Convert to list of clusters
    cluster_list = list(clusters.values())
    
    # Filter out singleton clusters (not duplicates)
    cluster_list = [cluster for cluster in cluster_list if len(cluster) > 1]
    
    return cluster_list

def generate_cluster_visualization(
    df: pd.DataFrame,
    clusters: List[List[int]],
    output_path: str
) -> str:
    """
    Generate a visualization of the clusters.
    
    Args:
        df: Input DataFrame
        clusters: List of clusters
        output_path: Path to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    try:
        # Create a graph
        G = nx.Graph()
        
        # Add nodes for all records in clusters
        for cluster_id, cluster in enumerate(clusters):
            for record_id in cluster:
                G.add_node(record_id, cluster=cluster_id)
        
        # Add edges within clusters
        for cluster in clusters:
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    G.add_edge(cluster[i], cluster[j])
        
        # Create a figure
        plt.figure(figsize=(12, 8))
        
        # Get node colors based on cluster
        node_colors = [G.nodes[n].get('cluster', 0) for n in G.nodes()]
        
        # Draw the graph
        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            node_color=node_colors,
            cmap=plt.cm.tab10,
            with_labels=True,
            node_size=100,
            font_size=8
        )
        
        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating cluster visualization: {str(e)}", exc_info=True)
        return None

async def run_clustering(
    df: pd.DataFrame,
    classified_pairs: List[Dict[str, Any]],
    method: str,
    params: Dict[str, Any],
    dataset_id: int,
    user_id: int
) -> Dict[str, Any]:
    """
    Run clustering on classified pairs.
    
    Args:
        df: Input DataFrame
        classified_pairs: List of classified pairs
        method: Clustering method ('graph_connected_components', 'graph_community_detection', 'dbscan', 'optics')
        params: Parameters for the clustering method
        dataset_id: ID of the dataset
        user_id: ID of the user
        
    Returns:
        Dictionary with clustering results and metadata
    """
    try:
        # Run the appropriate clustering method
        if method.startswith('graph_'):
            # Extract the specific graph method
            graph_method = method.replace('graph_', '')
            
            # Extract parameters
            confidence_threshold = params.get('confidence_threshold', 0.7)
            
            # Run graph-based clustering
            clusters = graph_based_clustering(
                classified_pairs,
                method=graph_method,
                confidence_threshold=confidence_threshold
            )
            
        elif method in ['dbscan', 'optics']:
            # Run density-based clustering
            clusters = density_based_clustering(
                classified_pairs,
                method=method,
                params=params
            )
            
        else:
            return {
                "status": "error",
                "message": f"Unknown clustering method: {method}"
            }
        
        # Save the clustering results as CSV assignments
        raw_name = str(params.get('output_name', '')).strip()
        output_filename = raw_name if raw_name else 'clustering_results.csv'
        if not output_filename.lower().endswith('.csv'):
            output_filename += '.csv'

        output_path = _get_clustering_artifact_path(
            settings.DATASET_UPLOAD_DIR,
            dataset_id,
            user_id,
            output_filename
        )

        assignment_rows = []
        for cid, cluster in enumerate(clusters):
            for rid in cluster:
                assignment_rows.append({"cluster_id": cid, "record_id": rid})
        pd.DataFrame(assignment_rows).to_csv(output_path, index=False)

        
        # Generate a visualization
        viz_output_path = _get_clustering_artifact_path(
            settings.DATASET_UPLOAD_DIR,
            dataset_id,
            user_id,
            "cluster_visualization.png"
        )
        
        viz_path = generate_cluster_visualization(df, clusters, viz_output_path)
        
        # Create a summary of clustering
        summary = {
            "method": method,
            "params": params,
            "total_clusters": len(clusters),
            "total_records_in_clusters": sum(len(cluster) for cluster in clusters),
            "output_path": output_path,
            "visualization_path": viz_path
        }
        
        # Create a preview of clusters
        preview_clusters = []
        for i, cluster in enumerate(clusters[:10]):  # Show first 10 clusters
            cluster_info = {
                "cluster_id": i,
                "size": len(cluster),
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
            
            preview_clusters.append(cluster_info)
        
        return {
            "status": "success",
            "message": "Clustering completed successfully",
            "summary": summary,
            "clustering_results_path": output_path,
            "visualization_path": viz_path,
            "preview": preview_clusters
        }
        
    except Exception as e:
        logger.error(f"Error in clustering: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error in clustering: {str(e)}",
            "error": str(e)
        }
