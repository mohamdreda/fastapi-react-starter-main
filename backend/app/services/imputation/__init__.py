"""Data Imputation service package.

Exposes a single helper ``run_imputation`` so that other modules (router,
Celery task) can call ``from app.services.imputation import run_imputation``
without touching internal implementation details.
"""

from .service import run_imputation  # noqa: F401
