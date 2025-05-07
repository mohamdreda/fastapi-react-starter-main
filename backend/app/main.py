from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
from .utils.logger import setup_logger
from .routes.auth import router as auth_router
from .routes.upload import router as upload_router
from .routes.diagnosis import router as diagnosis_router
from .routes.datasets import router as datasets_router
from .routes.users import router as users_router
from .routes.proxy import router as proxy_router
from .db.database import init_db, engine
from .config import get_settings
from .db import models

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

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin", "Access-Control-Request-Method", "Access-Control-Request-Headers"],
    expose_headers=["Content-Length", "Access-Control-Allow-Origin", "Access-Control-Allow-Headers", "Cache-Control", "Content-Language", "Content-Type"],
    max_age=86400  # Cache preflight requests for 24 hours
)

# Include routers with proper prefixes
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(upload_router, prefix="/api/v1/upload", tags=["File Upload"])
app.include_router(diagnosis_router, prefix="/api/v1/diagnosis", tags=["Data Diagnosis"])
app.include_router(datasets_router, prefix="/api/v1/datasets", tags=["Datasets"])
app.include_router(users_router, prefix="/api/v1/users", tags=["Users"])
app.include_router(proxy_router, prefix="/api/v1/proxy", tags=["Proxy"])

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