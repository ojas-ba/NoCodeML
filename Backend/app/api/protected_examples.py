"""Example of protected API routes.

This demonstrates how to create API endpoints that require authentication.
Use this as a template for your own protected routes.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.core.auth import current_active_user
from app.models import User
from app.db.session import get_db

router = APIRouter()


@router.get("/protected-example")
async def protected_route_example(
    user: User = Depends(current_active_user)
):
    """Example of a simple protected route.
    
    This endpoint requires authentication. The user object is automatically
    injected by FastAPI based on the JWT token in the Authorization header.
    
    Frontend usage:
        axios.get('/api/v1/protected-example', {
            headers: { Authorization: `Bearer ${token}` }
        })
    """
    return {
        "message": f"Hello {user.email}!",
        "user_id": user.id,
        "is_superuser": user.is_superuser
    }


@router.get("/user-profile")
async def get_user_profile(
    user: User = Depends(current_active_user)
):
    """Get the current user's profile.
    
    Returns detailed information about the authenticated user.
    """
    return {
        "id": user.id,
        "email": user.email,
        "is_active": user.is_active,
        "is_verified": user.is_verified,
        "created_at": user.created_at,
        "updated_at": user.updated_at
    }


@router.post("/admin-only")
async def admin_only_route(
    user: User = Depends(current_active_user)
):
    """Example of an admin-only route.
    
    This checks if the user is a superuser before allowing access.
    """
    if not user.is_superuser:
        raise HTTPException(
            status_code=403,
            detail="Only administrators can access this endpoint"
        )
    
    return {"message": "Welcome, admin!"}


@router.get("/user-data-with-db")
async def get_user_data_with_db(
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Example of accessing the database in a protected route.
    
    This shows how to use both authentication and database access.
    You can query related data, create records owned by the user, etc.
    """
    # Example: Query some user-specific data from another table
    # result = await db.execute(select(UserData).where(UserData.user_id == user.id))
    # user_data = result.scalars().all()
    
    return {
        "user_email": user.email,
        "message": "You can now query the database here"
        # "user_data": user_data
    }


# How to include this in your main API router:
# In app/api/__init__.py, add:
# from app.api.protected_examples import router as protected_router
# api_router.include_router(protected_router, prefix="/examples", tags=["examples"])
