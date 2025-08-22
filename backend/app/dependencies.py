# File: backend/app/dependencies.py
from app.db.database import get_db
from app.services.auth import get_current_user

__all__ = ["get_db", "get_current_user"]