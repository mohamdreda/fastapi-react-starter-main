"""Celery application instance for background tasks.

This standalone module allows both FastAPI routes and task modules to import the
same Celery application via:

    from app.celery_app import celery_app

Only minimal configuration is included so that the backend can boot even when
RabbitMQ / Redis is not running during local development. The broker/backend
URLs can be supplied via environment variables; sensible defaults are provided
for Docker deployments (see README).
"""
from __future__ import annotations

import os
try:
    from celery import Celery  # type: ignore
except ImportError:  # Celery not installed in the local env
    Celery = None  # type: ignore

# Broker and backend URLs can be configured via environment variables
BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

if Celery:
    celery_app = Celery(
        "data_cleaning_backend",
        broker=BROKER_URL,
        backend=RESULT_BACKEND,
    )
    celery_app.autodiscover_tasks(["app"])
    # Configure periodic watchdog (Celery beat) to clean stale 'running' session steps
    try:
        interval = int(os.getenv("WATCHDOG_INTERVAL_SECONDS", "60"))  # run every 60s by default
        celery_app.conf.timezone = os.getenv("CELERY_TIMEZONE", "UTC")
        celery_app.conf.beat_schedule = {
            "watchdog-mark-stale-session-steps": {
                "task": "app.tasks.workflow_tasks.mark_stale_session_steps",
                "schedule": interval,
            },
        }
    except Exception:
        # Beat config is best-effort; avoid breaking app if env incomplete
        pass
else:
    class _DummyCelery:
        """Fallback Celery replacement so imports succeed when Celery not installed."""
        def task(self, *args, **kwargs):
            # No-op decorator to allow modules to define tasks without Celery
            def _decorator(func):
                return func
            return _decorator
        def send_task(self, *_, **__):
            raise RuntimeError("Celery is not installed. Async tasks are unavailable.")

    celery_app = _DummyCelery()  # type: ignore
