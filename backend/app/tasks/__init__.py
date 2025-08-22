"""Celery tasks package.

This file enables autodiscovery for the 'app.tasks' package.
"""

# Optional: explicit imports so autodiscovery finds modules quickly
from . import workflow_tasks  # noqa: F401
from . import imputation  # noqa: F401
