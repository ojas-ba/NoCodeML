from fastapi import FastAPI, Depends
import asyncpg
from .db.init_db import lifespan,get_db_connection
from .auth.middleware import AuthMiddleware
from .auth.routes import router as auth_routes
from .projects.routes import router as project_routes
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(lifespan=lifespan,docs_url="/docs_swagger")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuthMiddleware)
app.include_router(auth_routes,prefix="/auth")
app.include_router(project_routes,prefix="/projects")

@app.get("/")
async def root(db: asyncpg.Connection = Depends(get_db_connection)):
    """Sample endpoint using the database connection."""
    result = await db.fetchval("SELECT 'FastAPI with PostgreSQL Running!'")
    return {"message": result}