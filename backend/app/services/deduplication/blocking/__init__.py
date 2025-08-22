"""
Blocking module for data deduplication.

This module implements efficient blocking techniques to reduce the number of comparisons:
- MinHash LSH: Locality-Sensitive Hashing with MinHash signatures
- SimHash: Hash-based similarity estimation for text data

These techniques help avoid the O(n²) comparison problem by grouping similar records.
"""
