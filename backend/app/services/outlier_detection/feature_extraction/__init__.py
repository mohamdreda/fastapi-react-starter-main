"""
Feature Extraction Module.

This module contains implementations of various feature extraction algorithms:
- Autoencoder
- PCA (Principal Component Analysis)
- ISOMAP
"""

# IMPORTANT: Do not import AutoencoderService here to avoid importing TensorFlow
# at package import time. Import it lazily where needed, e.g.:
#     from app.services.outlier_detection.feature_extraction.autoencoder import AutoencoderService

from .pca_service import PCAService  # lightweight
from .isomap_service import IsomapService  # lightweight

__all__ = ["PCAService", "IsomapService"]