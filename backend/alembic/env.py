import sys
import asyncio
import logging
from pathlib import Path
from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

# Configuration ABSOLUE du chemin
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Racine du projet (4 niveaux au-dessus)
sys.path.insert(0, str(PROJECT_ROOT / "backend"))  # Chemin explicite vers le backend

# Imports APRÈS configuration des chemins
from app.db.database import Base, engine  # noqa
from app.config import get_settings  # noqa
from app.db.models import *  # noqa

# Configuration du logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger('alembic')

# Chargement de la configuration
settings = get_settings()

def run_migrations_online():
    """Exécute les migrations en mode asynchrone"""
    
    async def run_async_migrations():
        async with engine.connect() as connection:
            await connection.run_sync(do_run_migrations)
            logger.info("Migrations asynchrones terminées avec succès")

    try:
        asyncio.run(run_async_migrations())
    except Exception as e:
        logger.error(f"Erreur lors des migrations : {str(e)}")
        raise

def do_run_migrations(connection):
    """Configuration du contexte de migration"""
    context.configure(
        connection=connection,
        target_metadata=Base.metadata,
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
        version_table_schema=Base.metadata.schema,
        user_module_prefix='sa.'
    )

    with context.begin_transaction():
        context.run_migrations()
        logger.debug("Transaction de migration terminée")

def run_migrations_offline():
    """Mode hors ligne pour génération SQL"""
    context.configure(
        url=engine.url.render_as_string(hide_password=False),
        target_metadata=Base.metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True
    )

    with context.begin_transaction():
        context.run_migrations()
        logger.info("Script SQL généré avec succès")

if context.is_offline_mode():
    logger.info("Exécution en mode hors ligne")
    run_migrations_offline()
else:
    logger.info("Exécution en mode connecté")
    run_migrations_online()