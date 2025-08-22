# Modular Deduplication Pipeline

This directory contains a modular, step-by-step data deduplication pipeline that replaces the legacy monolithic deduplication code. The pipeline is designed to be flexible, transparent, and user-controlled, with the ability to download intermediate results at each step.

## Pipeline Structure

The deduplication pipeline consists of six distinct steps, each implemented as a separate service:

1. **Preprocessing**: Prepares data for deduplication by cleaning text, scaling numeric values, and encoding categorical values.
2. **Blocking**: Reduces the O(n²) comparison problem by generating candidate pairs using techniques like MinHash LSH and SimHash.
3. **Similarity Calculation**: Computes similarity between candidate pairs using field-specific metrics.
4. **Classification**: Uses machine learning to classify pairs as duplicates or non-duplicates.
5. **Clustering**: Groups duplicate records into clusters using graph-based or density-based methods.
6. **Resolution**: Applies strategies to resolve duplicate clusters (keep first, keep most complete, merge, or manual review).

## API Endpoints

Each step of the pipeline has its own API endpoint:

- `POST /api/v1/deduplication/pipeline/preprocessing`: Preprocess a dataset
- `POST /api/v1/deduplication/pipeline/blocking`: Generate candidate pairs
- `POST /api/v1/deduplication/pipeline/similarity`: Calculate similarity between pairs
- `POST /api/v1/deduplication/pipeline/classification`: Classify pairs as duplicates
- `POST /api/v1/deduplication/pipeline/clustering`: Group duplicates into clusters
- `POST /api/v1/deduplication/pipeline/resolution`: Resolve duplicate clusters
- `POST /api/v1/deduplication/pipeline/manual-resolution`: Apply manual resolution decisions
- `GET /api/v1/deduplication/pipeline/algorithms`: List available algorithms for each step

## Features

- **Modular Design**: Each step is independent and can be executed separately.
- **Intermediate Results**: Results from each step can be downloaded and inspected.
- **Multiple Algorithms**: Various algorithms are available for each step.
- **Configurable Parameters**: Each algorithm can be fine-tuned with custom parameters.
- **Visualization**: Clustering results include visualizations for better understanding.
- **Manual Review**: Option for manual review and resolution of duplicate clusters.

## Legacy Algorithms

The legacy fuzzy matching and deep entity resolution algorithms are retained for comparison purposes. These can be accessed through the original deduplication endpoint:

- `POST /api/v1/deduplication`: Run legacy deduplication algorithms

## Usage Example

A typical workflow using the modular pipeline:

1. Preprocess the dataset:
```json
POST /api/v1/deduplication/pipeline/preprocessing
{
  "dataset_id": 123,
  "text_columns": ["name", "description"],
  "numeric_columns": ["price", "quantity"],
  "categorical_columns": ["category", "brand"]
}
```

2. Generate candidate pairs using blocking:
```json
POST /api/v1/deduplication/pipeline/blocking
{
  "dataset_id": 123,
  "method": "minhash_lsh",
  "key_fields": ["name", "brand"],
  "params": {
    "num_perm": 128,
    "threshold": 0.7
  }
}
```

3. Calculate similarity between candidate pairs:
```json
POST /api/v1/deduplication/pipeline/similarity
{
  "dataset_id": 123,
  "candidate_pairs_path": "/path/to/candidate_pairs.json",
  "field_configs": {
    "name": {
      "type": "text",
      "method": "jaro_winkler",
      "weight": 2.0
    },
    "price": {
      "type": "numeric",
      "method": "normalized_distance",
      "weight": 1.0
    }
  },
  "threshold": 0.7
}
```

4. Classify pairs as duplicates:
```json
POST /api/v1/deduplication/pipeline/classification
{
  "dataset_id": 123,
  "similarity_results_path": "/path/to/similarity_results.json",
  "method": "random_forest",
  "params": {
    "n_estimators": 100,
    "max_depth": 10
  }
}
```

5. Cluster duplicate records:
```json
POST /api/v1/deduplication/pipeline/clustering
{
  "dataset_id": 123,
  "classification_results_path": "/path/to/classification_results.json",
  "method": "graph_connected_components",
  "params": {}
}
```

6. Resolve duplicate clusters:
```json
POST /api/v1/deduplication/pipeline/resolution
{
  "dataset_id": 123,
  "clustering_results_path": "/path/to/clustering_results.json",
  "method": "keep_most_complete",
  "params": {}
}
```

## Dependencies

The pipeline has several dependencies, some of which are optional:

- **Required**: pandas, numpy, scikit-learn
- **Optional**:
  - `datasketch`: For MinHash LSH blocking
  - `thefuzz`: For fuzzy string matching (fallback implemented)
  - `xgboost`: For XGBoost classification
  - `tensorflow`: For Siamese network classification
  - `networkx`: For graph-based clustering
  - `matplotlib`: For cluster visualization
  - `community` (python-louvain): For community detection clustering

## Error Handling

Each service includes comprehensive error handling and logging. If a step fails, it returns a detailed error message that can help diagnose and fix the issue.
