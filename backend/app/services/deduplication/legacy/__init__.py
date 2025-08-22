"""
Legacy deduplication algorithms module.

This module integrates the legacy fuzzy matching and deep entity resolution algorithms
into the new modular deduplication pipeline structure. These algorithms are retained
for comparison purposes with the new modular approach.
"""

from .fuzzy_matching import fuzzy_matching_deduplication
from .deep_er import deep_er_deduplication
from .service import run_legacy_deduplication, get_legacy_algorithms

__all__ = [
    'fuzzy_matching_deduplication',
    'deep_er_deduplication',
    'run_legacy_deduplication',
    'get_legacy_algorithms'
]
