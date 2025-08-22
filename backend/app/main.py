from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import time
import os
from .utils.logger import setup_logger
from .routes.workflow import router as workflow_router
from .routes.auth import router as auth_router
from .routes.upload import router as upload_router
from .routes.diagnosis import router as diagnosis_router
from .routes.datasets import router as datasets_router
from .routes.users import router as users_router
from .routes.proxy import router as proxy_router
from .db.database import init_db, engine
from .config import get_settings
from .db import models
from .routes.outliers import router as actual_outliers_router
from .routes.transformation import router as transformation_router
from .routes.feature_engineering import router as feature_engineering_router
from .routes.clustering import router as clustering_router
from .routes.sessions import router as sessions_router
from .routes.deduplication_pipeline import router as deduplication_pipeline_router
from .routes.artifacts import router as artifacts_router
# Imputation router
from .routes.imputation import router as imputation_router

logger = setup_logger(__name__)
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(models.Base.metadata.create_all)
            logger.info("Database tables created")
        
        # Initialize database
        await init_db()
        logger.info("Database initialized")
        
        yield
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down...")
        await engine.dispose()
        logger.info("Database connection closed")

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url=None
)

# Register routers
app.include_router(workflow_router)
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(upload_router, prefix="/api/v1/upload", tags=["File Upload"])
app.include_router(diagnosis_router, prefix="/api/v1/diagnosis", tags=["Data Diagnosis"])
app.include_router(datasets_router, prefix="/api/v1/datasets", tags=["Datasets"])
app.include_router(users_router, prefix="/api/v1/users", tags=["Users"])
app.include_router(artifacts_router)
app.include_router(proxy_router, prefix="/api/v1/proxy", tags=["Proxy"])
app.include_router(actual_outliers_router, prefix="/api/v1/outliers", tags=["Outlier Detection"])
app.include_router(transformation_router, prefix="/api/v1/transformation", tags=["Data Transformation"])
app.include_router(feature_engineering_router, prefix="/api/v1/feature-engineering", tags=["Feature Engineering"])
app.include_router(clustering_router, prefix="/api/v1/clustering", tags=["Clustering"])
app.include_router(imputation_router, prefix="/api/v1", tags=["Data Imputation"])
app.include_router(sessions_router)
# Modular deduplication pipeline router
# Register modular deduplication pipeline router once, API version handled here;
# the router itself already includes '/deduplication/pipeline'
app.include_router(
    deduplication_pipeline_router,
    prefix="/api/v1",
    tags=["Modular Deduplication Pipeline"]
)

from app.routes.feature_sets import router as feature_sets_router
app.include_router(feature_sets_router, prefix="/api/v1/feature-sets", tags=["Feature Sets"])

# Configure CORS middleware


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)

# DEBUG: Print all registered routes

# DEBUG: Print all registered routes after routers are included
for route in app.routes:
    if hasattr(route, "methods"):
        print(f"ROUTE: {route.path} METHODS: {route.methods}")
    else:  # Likely an APIWebSocketRoute or others without 'methods'
        print(f"ROUTE: {route.path} (non-HTTP route)")

# Mount uploads directory to serve dataset files
uploads_dir = os.path.abspath(str(settings.UPLOAD_DIR)) if hasattr(settings, "UPLOAD_DIR") else os.path.join(os.getcwd(), "uploads")
if os.path.exists(uploads_dir):
    app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")
    logger.info(f"Mounted uploads directory: {uploads_dir}")
else:
    logger.warning(f"Uploads directory not found: {uploads_dir}")

# Mount static files directory for data artifacts
data_artifacts_dir = os.path.join(os.getcwd(), "data_artifacts")
if os.path.exists(data_artifacts_dir):
    app.mount("/data_artifacts", StaticFiles(directory=data_artifacts_dir), name="data_artifacts")
    logger.info(f"Mounted static files directory: {data_artifacts_dir}")
else:
    logger.warning(f"Static files directory not found: {data_artifacts_dir}")

# Mount temporary files directory for visualizations
temp_dir = "/tmp"
if os.name == 'nt':  # Windows
    temp_dir = os.environ.get('TEMP', 'C:\\Windows\\Temp')
app.mount("/tmp", StaticFiles(directory=temp_dir), name="temp_files")
logger.info(f"Mounted temporary files directory: {temp_dir}")

# Middleware for logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    path = request.url.path
    method = request.method
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        logger.info(f"{method} {path} completed in {duration:.2f}s")
        return response
    except Exception as e:
        logger.error(f"{method} {path} failed: {str(e)}")
        raise

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

# Status endpoint
@app.get("/api/status")
async def get_status():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "operational"
    }

# Root redirect to docs
@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.APP_NAME} API. Visit /api/docs for documentation."}

logger.info(f"Application ready: {settings.APP_NAME} v{settings.APP_VERSION}")