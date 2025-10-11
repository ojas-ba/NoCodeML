from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core.config import settings
from typing import AsyncGenerator

# psycopg3 async driver - URL should be postgresql+psycopg://...
# The create_async_engine will use psycopg in async mode automatically
async_engine = create_async_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    echo=False,
    future=True  # Use SQLAlchemy 2.0 style
)
AsyncSessionLocal = async_sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database sessions."""
    async with AsyncSessionLocal() as session:
        yield session