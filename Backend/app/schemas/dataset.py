"""Pydantic schemas for dataset request/response validation."""
from datetime import datetime
from typing import Optional, List, Any
from uuid import UUID
from pydantic import BaseModel, Field, field_validator


class DatasetCreate(BaseModel):
    """Schema for dataset creation request."""
    name: str = Field(..., min_length=1, max_length=255, description="Dataset name")
    description: Optional[str] = Field(None, description="Optional dataset description")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate dataset name is not just whitespace."""
        if not v.strip():
            raise ValueError('Dataset name cannot be empty or just whitespace')
        return v.strip()


class DatasetUpdate(BaseModel):
    """Schema for updating dataset metadata."""
    name: str = Field(..., min_length=1, max_length=255, description="Dataset name")
    description: Optional[str] = Field(None, description="Optional dataset description")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Dataset name cannot be empty or just whitespace')
        return v.strip()


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    non_null_count: int
    null_count: int


class DatasetResponse(BaseModel):
    """Schema for dataset data in API responses."""
    id: UUID
    user_id: int
    name: str
    description: Optional[str] = None
    storage_path: str
    file_name: str
    file_size_bytes: int
    row_count: int
    column_count: int
    column_info: dict  # JSONB field with column metadata
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class DatasetListResponse(BaseModel):
    datasets: List[DatasetResponse]
    total: int
    skip: int
    limit: int


class DatasetPreviewResponse(BaseModel):
    columns: List[str] = Field(..., description="List of column names")
    data: List[List[Any]] = Field(..., description="List of rows, each row is a list of values")
    row_count: int = Field(..., description="Total number of rows in the dataset")
    preview_rows: int = Field(..., description="Number of rows in this preview")
