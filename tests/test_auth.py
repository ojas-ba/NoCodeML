import sys
import os
import pytest
import pytest_asyncio
from httpx import AsyncClient
from unittest.mock import AsyncMock

# Add the parent directory to sys.path to allow importing from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app
from app.db.init_db import get_db_connection
from app.auth.auth_funtions import verify_token, hashing_password

# Create a mock database connection
@pytest_asyncio.fixture
async def mock_db():
    mock_conn = AsyncMock()
    mock_conn.fetchval.return_value = None  # Default return value
    return mock_conn

# Override FastAPI dependency with mock DB
@pytest_asyncio.fixture
async def override_db(mock_db):
    app.dependency_overrides[get_db_connection] = lambda: mock_db
    yield
    app.dependency_overrides.pop(get_db_connection, None)

# Create an Async test client (FIXED: Removed 'app=app')
@pytest_asyncio.fixture
async def client():
    async with AsyncClient(base_url="http://localhost:8000") as ac:
        yield ac

# Test User Registration
@pytest.mark.asyncio
async def test_register_user(client, override_db, mock_db):
    response = await client.post(
        "/auth/register",
        json={
            "email": "test1@example.com",
            "firstName": "Johnnny",
            "lastName": "Doe",
            "password": "testpassword",
            "secretKey": 1234
        },
    )
    assert response.status_code == 200
    assert response.json()["message"] == "User created successfully"

# Test Login
@pytest.mark.asyncio
async def test_login_user(client, mock_db, override_db):
    # Simulate a real hashed password
    mock_db.fetchval.return_value = hashing_password("testpassword")
    
    response = await client.post(
        "/auth/login",
        json={
            "email": "test1@example.com",
            "password": "testpassword"
        },
    )
    assert response.status_code == 200
    assert response.json()["message"] == "success"
    assert "access_token" in response.cookies

# Test Protected Route (/me) Without Token
@pytest.mark.asyncio
async def test_protected_route_without_token(client):
    response = await client.get("/auth/me")
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


# Test Logout
@pytest.mark.asyncio
async def test_logout_user(client):
    response = await client.post("/auth/logout")
    assert response.status_code == 200
    assert response.json()["message"] == "Logout successful"
    assert "access_token" not in response.cookies
