"""Dataset service layer for business logic."""
import os
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import pandas as pd

from app.models.dataset import Dataset
from app.core.config import settings


UPLOAD_DIR = Path("/app/datasets")
MAX_FILE_SIZE = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.parquet'}
MAX_PREVIEW_ROWS = 50


async def create_dataset(
    db: AsyncSession,
    file: UploadFile,
    name: str,
    description: Optional[str],
    user_id: int
) -> Dataset:
    """Create a new dataset from uploaded file."""
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '{file_ext}' not supported. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    dataset_id = uuid.uuid4()
    user_dir = UPLOAD_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    
    storage_filename = f"{dataset_id}_{file.filename}"
    storage_path = user_dir / storage_filename
    
    try:
        file_size = await save_upload_file(file, storage_path)
        
        if file_size > MAX_FILE_SIZE:
            storage_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.0f}MB"
            )
    except Exception as e:
        storage_path.unlink(missing_ok=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )
    
    try:
        metadata = await extract_file_metadata(str(storage_path))
    except Exception as e:
        storage_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file: {str(e)}. Please ensure the file is valid and not corrupted."
        )
    
    dataset = Dataset(
        id=dataset_id,
        user_id=user_id,
        name=name,
        description=description,
        storage_path=str(storage_path),
        file_name=file.filename,
        file_size_bytes=file_size,
        row_count=metadata['row_count'],
        column_count=metadata['column_count'],
        column_info=metadata['column_info']
    )
    
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    
    return dataset


async def get_user_datasets(
    db: AsyncSession,
    user_id: int,
    skip: int = 0,
    limit: int = 100
) -> tuple[List[Dataset], int]:
    """Fetch all datasets for a user with pagination."""
    count_query = select(func.count()).select_from(Dataset).where(Dataset.user_id == user_id)
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    query = (
        select(Dataset)
        .where(Dataset.user_id == user_id)
        .order_by(Dataset.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(query)
    datasets = result.scalars().all()
    
    return list(datasets), total


async def get_dataset_by_id(
    db: AsyncSession,
    dataset_id: uuid.UUID,
    user_id: int
) -> Optional[Dataset]:
    """Fetch a single dataset and verify ownership."""
    query = select(Dataset).where(
        Dataset.id == dataset_id,
        Dataset.user_id == user_id
    )
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def update_dataset(
    db: AsyncSession,
    dataset_id: uuid.UUID,
    user_id: int,
    name: str,
    description: Optional[str]
) -> Optional[Dataset]:
    """Update dataset name and description."""
    dataset = await get_dataset_by_id(db, dataset_id, user_id)
    if not dataset:
        return None
    
    dataset.name = name
    dataset.description = description
    
    await db.commit()
    await db.refresh(dataset)
    
    return dataset


async def check_dataset_dependencies(
    db: AsyncSession,
    dataset_id: uuid.UUID,
    user_id: int
) -> None:
    """
    Check if dataset is used by any experiments.
    
    Args:
        db: Database session
        dataset_id: ID of the dataset
        user_id: ID of the user
    
    Raises:
        HTTPException: 409 if experiments depend on this dataset
    """
    from app.models.experiment import Experiment
    
    # Query for experiments using this dataset
    query = select(Experiment).where(
        Experiment.dataset_id == dataset_id,
        Experiment.user_id == user_id
    )
    result = await db.execute(query)
    experiments = result.scalars().all()
    
    if experiments:
        experiment_names = [exp.name for exp in experiments]
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot delete dataset. It is used by {len(experiments)} experiment(s): {', '.join(experiment_names)}"
        )


async def delete_dataset(
    db: AsyncSession,
    dataset_id: uuid.UUID,
    user_id: int
) -> bool:
    """Delete a dataset (file and database record)."""
    # Fetch and verify ownership
    dataset = await get_dataset_by_id(db, dataset_id, user_id)
    if not dataset:
        return False
    
    # Check for dependencies (experiments using this dataset)
    await check_dataset_dependencies(db, dataset_id, user_id)
    
    # Delete physical file
    try:
        delete_file(dataset.storage_path)
    except Exception as e:
        # Log the error but continue with database deletion
        print(f"Warning: Failed to delete file {dataset.storage_path}: {str(e)}")
    
    # Delete database record
    await db.delete(dataset)
    await db.commit()
    
    return True


async def get_dataset_preview(
    db: AsyncSession,
    dataset_id: uuid.UUID,
    user_id: int,
    rows: int = 10
) -> Optional[Dict[str, Any]]:
    """Get a preview of dataset contents."""
    # Fetch and verify ownership
    dataset = await get_dataset_by_id(db, dataset_id, user_id)
    if not dataset:
        return None
    
    # Limit preview rows
    rows = min(rows, MAX_PREVIEW_ROWS)
    
    # Read file and extract preview
    try:
        file_ext = Path(dataset.storage_path).suffix.lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(dataset.storage_path, nrows=rows)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(dataset.storage_path, nrows=rows)
        elif file_ext == '.parquet':
            df = pd.read_parquet(dataset.storage_path)
            df = df.head(rows)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        df_filled = df.where(pd.notna(df), None)
        
        return {
            "columns": df.columns.tolist(),
            "data": df_filled.values.tolist(),
            "row_count": dataset.row_count,
            "preview_rows": len(df)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read dataset preview: {str(e)}"
        )


async def save_upload_file(file: UploadFile, destination: Path) -> int:
    """Save uploaded file to filesystem."""
    file_size = 0
    
    with open(destination, 'wb') as buffer:
        while chunk := await file.read(8192):
            buffer.write(chunk)
            file_size += len(chunk)
    
    return file_size


async def extract_file_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata from a dataset file."""
    file_ext = Path(file_path).suffix.lower()
    
    # Read file based on extension
    if file_ext == '.csv':
        df = pd.read_csv(file_path)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif file_ext == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")
    
    row_count, column_count = df.shape
    
    columns_info = []
    for col in df.columns:
        non_null_count = int(df[col].count())
        null_count = int(df[col].isna().sum())
        dtype = str(df[col].dtype)
        
        columns_info.append({
            "name": str(col),
            "dtype": dtype,
            "non_null_count": non_null_count,
            "null_count": null_count
        })
    
    return {
        "row_count": row_count,
        "column_count": column_count,
        "column_info": {"columns": columns_info}
    }


def delete_file(storage_path: str) -> bool:
    """Delete a file from the filesystem."""
    path = Path(storage_path)
    if path.exists():
        path.unlink()
        return True
    return False
