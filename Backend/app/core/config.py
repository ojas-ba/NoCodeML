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

    class Config:
        env_file = ".env"

settings = Settings()