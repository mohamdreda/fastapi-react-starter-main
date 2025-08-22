"""
Resolution module for data deduplication.

This module implements strategies for handling identified duplicate clusters:
- Keep First Record: Keep only the first record in each cluster
- Keep Most Complete Record: Keep the record with fewest missing values
- Merge Records: Create a new record by combining values from duplicates
- Manual Review: Tools for user-guided resolution

These methods provide options for cleaning datasets after duplicate detection.
"""
