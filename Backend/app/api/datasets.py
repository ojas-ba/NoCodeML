"""Dataset API endpoints."""
from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.db.session import get_db
from app.models import User
from app.schemas import (
    DatasetCreate,
    DatasetUpdate,
    DatasetResponse,
    DatasetListResponse,
    DatasetPreviewResponse
)
from app.services import dataset_service


router = APIRouter()


@router.post("/", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    name: str = Form(..., description="Dataset name"),
    description: Optional[str] = Form(None, description="Optional dataset description"),
    file: UploadFile = File(..., description="Dataset file (CSV, Excel, or Parquet)"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload a new dataset."""
    dataset = await dataset_service.create_dataset(
        db=db,
        file=file,
        name=name,
        description=description,
        user_id=current_user.id
    )
    return dataset


@router.get("/", response_model=DatasetListResponse)
async def list_datasets(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all datasets owned by the current user."""
    datasets, total = await dataset_service.get_user_datasets(
        db=db,
        user_id=current_user.id,
        skip=skip,
        limit=limit
    )
    
    return DatasetListResponse(
        datasets=datasets,
        total=total,
        skip=skip,
        limit=limit
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed information about a specific dataset."""
    dataset = await dataset_service.get_dataset_by_id(
        db=db,
        dataset_id=dataset_id,
        user_id=current_user.id
    )
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    return dataset


@router.get("/{dataset_id}/preview", response_model=DatasetPreviewResponse)
async def preview_dataset(
    dataset_id: UUID,
    rows: int = 10,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Preview the contents of a dataset."""
    preview = await dataset_service.get_dataset_preview(
        db=db,
        dataset_id=dataset_id,
        user_id=current_user.id,
        rows=rows
    )
    
    if not preview:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    return preview


@router.put("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: UUID,
    dataset_update: DatasetUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update dataset metadata (name and description only)."""
    dataset = await dataset_service.update_dataset(
        db=db,
        dataset_id=dataset_id,
        user_id=current_user.id,
        name=dataset_update.name,
        description=dataset_update.description
    )
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    return dataset


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a dataset permanently."""
    success = await dataset_service.delete_dataset(
        db=db,
        dataset_id=dataset_id,
        user_id=current_user.id
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    return None
