from fastapi import FastAPI, Depends
import asyncpg
import os
from contextlib import asynccontextmanager

DATABASE_URL = os.getenv("DATABASE_URL")

# Global variable for connection pool
db_pool = None

async def intialize_tables():
    # Creates tables if they don't exist
    with open("app/db/tables.sql", "r") as f:
        sql_script = f.read()
    
    statements = [stmt.strip() for stmt in sql_script.split(";") if stmt.strip()]#extract all the sql commands
        
    async with db_pool.acquire() as conn:
        if conn is not None:
            for sql_query in statements: # Execute queries one by one.
                await conn.execute(sql_query)
        

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown tasks using FastAPI lifespan."""
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
    await intialize_tables()
    yield  # Application runs here
    await db_pool.close()

async def get_db_connection():
    """Dependency to get a connection from the pool."""
    async with db_pool.acquire() as conn:
        yield conn
