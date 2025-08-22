"""
Deduplication Pipeline Package.

This package contains the modular components for the data deduplication pipeline:
1. Preprocessing: Text/numeric/categorical field standardization
2. Blocking: MinHash/LSH for candidate pair generation
3. Similarity: Field-specific and composite similarity calculation
4. Classification: ML-based duplicate classification
5. Clustering: Graph-based and density-based duplicate clustering
6. Resolution: Strategies for handling identified duplicates
"""
