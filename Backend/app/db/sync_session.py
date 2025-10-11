from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# psycopg3 supports both sync and async with the same driver
# For sync, we use create_engine with postgresql+psycopg URL (not create_async_engine)
# The psycopg[binary] package includes both sync and async support
sync_engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600,
    echo=False,
    future=True  # Use SQLAlchemy 2.0 style
)
SyncSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)

def get_sync_db():
    db = SyncSessionLocal()
    try:
        yield db
    finally:
        db.close()