import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from app.config import get_settings

settings = get_settings()

def setup_logger(name: str = "DataCleaner") -> logging.Logger:
    """Configure and return a structured logger instance"""
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if settings.ENVIRONMENT == "development" else logging.INFO)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatting
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (development)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # File handler (production)
    file_handler = RotatingFileHandler(
        filename=logs_dir / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # Add handlers based on environment
    if settings.ENVIRONMENT == "development":
        logger.addHandler(console_handler)
    else:
        logger.addHandler(file_handler)

    # Add a separate error log handler
    error_handler = RotatingFileHandler(
        filename=logs_dir / "error.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    return logger