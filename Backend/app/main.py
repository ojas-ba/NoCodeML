"""Main FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.config import settings
from app.api import api_router
from app.db.session import async_engine
from app.models import Base
from app.core.model_cache import initialize_model_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown events."""
    # Note: Table creation is handled by Alembic migrations
    # Run: docker exec fastapi_app alembic upgrade head
    print("✓ Application started - using Alembic for database migrations")
    
    # Initialize model cache on startup
    initialize_model_cache()
    print("✓ Model cache initialized")
    
    yield
    
    await async_engine.dispose()
    print("✓ Database connections closed")


app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan,
    description="NoCodeML API - Build ML models without code, powered by custom JWT authentication",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    """API welcome message."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": "2.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "NoCodeML API"}

app.include_router(api_router, prefix=settings.API_V1_STR)