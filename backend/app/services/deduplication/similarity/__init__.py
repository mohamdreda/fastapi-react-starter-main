"""
Similarity calculation module for data deduplication.

This module implements field-specific similarity metrics:
- Text similarity: Jaro-Winkler, TF-IDF + Cosine
- Numeric similarity: Normalized distance
- Categorical similarity: Exact match, Jaccard similarity

It also provides composite similarity calculation with field weights.
"""
