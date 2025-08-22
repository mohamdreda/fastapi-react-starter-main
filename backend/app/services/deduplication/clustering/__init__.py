"""
Clustering module for data deduplication.

This module implements clustering techniques for grouping duplicate records:
- Graph-based: Connected components, community detection
- Density-based: Integration with existing DBSCAN, OPTICS, DENCLUE algorithms

These methods resolve transitivity issues and group duplicates into coherent clusters.
"""
