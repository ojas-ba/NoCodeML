from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    PROJECT_NAME: str = "NoCodeML API"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str
    
    # Redis (for Celery)
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    
    # JWT Authentication
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # CORS - comma-separated list of allowed origins
    BACKEND_CORS_ORIGINS: str = (
        "https://www.nocodeml.cloud,"
        "https://nocodeml.cloud,"
        "http://localhost:5173,"
        "http://localhost:3000,"
        "http://localhost:8080"
    )

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.BACKEND_CORS_ORIGINS.split(",") if o.strip()]

    class Config:
        env_file = ".env"

settings = Settings()