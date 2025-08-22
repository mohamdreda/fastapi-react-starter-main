import os
from datetime import datetime
from app.config.config import get_settings

settings = get_settings()


def get_artifact_path(dataset_id: int, user_id: int, filename: str) -> str:
    """Return a full path under the standard uploads directory for imputation
    artefacts and make sure the directory exists.
    """
    base_dir = os.path.join(
        settings.DATASET_UPLOAD_DIR, f"user_{user_id}", f"dataset_{dataset_id}", "imputation"
    )
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, filename)


def timestamped_name(base: str, ext: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    return f"{base}_{ts}.{ext.lstrip('.')}"
