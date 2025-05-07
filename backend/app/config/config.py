import os
from pathlib import Path
from functools import lru_cache
from typing import List, Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field, field_validator
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

class Settings(BaseSettings):
    # Configuration de base
    APP_NAME: str = "Data Cleaning Platform"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Enterprise-grade Data Cleaning Solution"
    ENVIRONMENT: Literal["development", "production"] = "development"
    API_PREFIX: str = "/api/v1"
    
    # Configuration PostgreSQL
    DB_NAME: str = "cleaning_db"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "1234"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_POOL_SIZE: int = 25
    DB_MAX_OVERFLOW: int = 15
    DB_POOL_TIMEOUT: int = 30
    DB_ECHO: bool = False
    DB_SSL_MODE: Optional[str] = None
    
    # Sécurité
    SECRET_KEY: str = os.getenv("SECRET_KEY", "default-secret-key")
    ALGORITHM: str = "HS256-512"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24h
    ADMIN_SECRET: str 
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # React frontend
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ]
    CORS_ALLOW_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # Configuration des fichiers
    MAX_FILE_SIZE_MB: int = 1024  # 1GB
    ALLOWED_FILE_TYPES: List[str] = ["csv", "xlsx", "json"]
    UPLOAD_DIR: Path = Path("uploads")
    
    # Configuration du logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
    LOG_RETENTION_DAYS: int = 7
    LOG_ROTATION_SIZE: str = "10 MB"

    # Configuration Pydantic v2
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        """Construit l'URL de connexion PostgreSQL de manière sécurisée"""
        return (
            f"postgresql+asyncpg://{self.DB_USER}:"
            f"{self.DB_PASSWORD}@{self.DB_HOST}:"
            f"{self.DB_PORT}/{self.DB_NAME}"
        )

    @field_validator("CORS_ORIGINS", mode="before")
    def parse_cors_origins(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",")]
        return value

# Configuration de logging manquante
CURRENT_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "verbose",
            "filename": "logs/app.log",
            "maxBytes": 10 * 1024 * 1024,  # 10 MB
            "backupCount": 5
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}

@lru_cache
def get_settings() -> Settings:
    return Settings()

# Configuration des répertoires
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = BASE_DIR / "logs"
UPLOAD_DIR = BASE_DIR / "uploads"

# Création des répertoires
for directory in [LOGS_DIR, UPLOAD_DIR]:
    directory.mkdir(exist_ok=True, parents=True)