from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import AsyncAdaptedQueuePool, NullPool
from app.utils.logger import setup_logger
from app.config import get_settings
from urllib.parse import quote_plus
from typing import AsyncGenerator
from tenacity import retry, stop_after_attempt, wait_fixed
import logging
import os
from sqlalchemy import text

settings = get_settings()   
logger = setup_logger(__name__)

# Configuration avancée du pool de connexions
POSTGRES_POOL_CONFIG = {
    "pool_size": int(os.getenv("DB_POOL_SIZE", 20)),
    "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", 10)),
    "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", 30)),
    "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", 3600)),
    "pool_pre_ping": True
}

# Configuration des tentatives de connexion
DB_CONNECTION_RETRIES = 5
RETRY_WAIT_SECONDS = 3

def get_database_url() -> str:
    """Construit l'URL de connexion PostgreSQL avec validation des paramètres"""
    required_fields = [settings.DB_USER, settings.DB_NAME, settings.DB_HOST]
    if not all(required_fields):
        logger.critical("Configuration PostgreSQL manquante dans les variables d'environnement")
        raise ValueError("Configuration de base de données incomplète")

    password = quote_plus(settings.DB_PASSWORD) if settings.DB_PASSWORD else ""
    return (
        f"postgresql+asyncpg://{settings.DB_USER}:{password}@"
        f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    )

@retry(stop=stop_after_attempt(DB_CONNECTION_RETRIES), wait=wait_fixed(RETRY_WAIT_SECONDS))
def create_engine_with_retry():
    """Crée le moteur de base de données avec mécanisme de réessai"""
    database_url = get_database_url()
    
    try:
        engine = create_async_engine(
            database_url,
            poolclass=AsyncAdaptedQueuePool,
            **POSTGRES_POOL_CONFIG,
            connect_args={
                "server_settings": {
                    "application_name": settings.APP_NAME,
                    "jit": "off"
                }
            },
            echo=settings.ENVIRONMENT == "development"
        )
        logger.info(f"Connecté à PostgreSQL sur {settings.DB_HOST}:{settings.DB_PORT}")
        return engine
    except Exception as e:
        logger.error(f"Échec de la connexion à la base de données: {str(e)}")
        raise

# Initialisation du moteur avec mécanisme de réessai
engine = create_engine_with_retry()

# Configuration de la session asynchrone
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    future=True
)

class Base(DeclarativeBase):
    """Classe de base pour tous les modèles SQLAlchemy"""
    __allow_unmapped__ = True

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Fournit une session de base de données avec gestion de transaction"""
    session = AsyncSessionLocal()
    try:
        logger.debug("Début de la session de base de données")
        yield session
        await session.commit()
    except Exception as e:
        logger.error(f"Erreur de transaction: {str(e)}")
        await session.rollback()
        raise
    finally:
        logger.debug("Fermeture de la session de base de données")
        await session.close()

async def init_db() -> None:
    """Initialise la base de données avec vérification de connexion"""
    logger.info("Démarrage de l'initialisation de la base de données")
    
    try:
        async with engine.begin() as conn:
            # Vérifie la connexion avant de créer les tables
            await conn.execute(text("SELECT 1"))
            logger.info("Connexion à PostgreSQL vérifiée")
            
            # # Ne PAS supprimer les tables — juste créer si elles n'existent pas
            #await conn.run_sync(Base.metadata.drop_all)  # <-- Ajoutez cette ligne
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Schéma de base de données réinitialisé avec succès")
            
    except Exception as e:
        logger.critical(f"Échec de l'initialisation de la base de données: {str(e)}")
        raise