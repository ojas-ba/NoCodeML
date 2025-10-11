"""Database initialization script.

Run this to create all tables in the database.
Usage: python -m app.db.init_db
"""
import asyncio
from app.db.session import async_engine
from app.models import Base


async def init_db():
    """Create all tables in the database.
    
    Why: This creates the users table and any other tables defined in models.
    Run this once before starting the application for the first time.
    """
    async with async_engine.begin() as conn:
        # Drop all tables (be careful in production!)
        # await conn.run_sync(Base.metadata.drop_all)
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
    
    print("Database tables created successfully!")


if __name__ == "__main__":
    asyncio.run(init_db())
