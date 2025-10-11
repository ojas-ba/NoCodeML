"""Pydantic schemas for request/response validation."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, field_validator


class UserCreate(BaseModel):
    """Schema for user registration request."""
    email: EmailStr
    password: str
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password meets minimum requirements."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class UserResponse(BaseModel):
    """Schema for user data in responses."""
    id: int
    email: str
    is_active: bool
    is_superuser: bool
    is_verified: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    """Schema for login request."""
    email: EmailStr
    password: str


class Token(BaseModel):
    """Schema for token response."""
    access_token: str
    token_type: str = "bearer"


from app.schemas.dataset import (
    DatasetCreate,
    DatasetUpdate,
    DatasetResponse,
    DatasetListResponse,
    DatasetPreviewResponse,
    ColumnInfo
)

from app.schemas.experiment import (
    ExperimentConfig,
    ExperimentCreate,
    ExperimentUpdate,
    ExperimentResponse,
    ExperimentListResponse
)

from app.schemas.model_config import (
    PreprocessingConfig,
    TrainingConfig,
    ModelConfigRequest,
    ModelConfigResponse,
    ModelSelectionRequest,
    ModelSelectionResponse
)

from app.schemas.training import (
    TrainingJobCreate,
    TrainingJobResponse,
    TrainingProgress,
    TrainingMetrics,
    FeatureImportance,
    ConfusionMatrix,
    TrainingResultResponse,
    TrainingLogResponse,
    StartTrainingRequest,
    StartTrainingResponse,
    JobStatusResponse,
    ExperimentTrainingStatus
)

__all__ = [
    "UserCreate",
    "UserResponse",
    "LoginRequest",
    "Token",
    "DatasetCreate",
    "DatasetUpdate",
    "DatasetResponse",
    "DatasetListResponse",
    "DatasetPreviewResponse",
    "ColumnInfo",
    "ExperimentConfig",
    "ExperimentCreate",
    "ExperimentUpdate",
    "ExperimentResponse",
    "ExperimentListResponse",
    "PreprocessingConfig",
    "TrainingConfig",
    "ModelConfigRequest",
    "ModelConfigResponse",
    "ModelSelectionRequest",
    "ModelSelectionResponse",
    "TrainingJobCreate",
    "TrainingJobResponse",
    "TrainingProgress",
    "TrainingMetrics",
    "FeatureImportance",
    "ConfusionMatrix",
    "TrainingResultResponse",
    "TrainingLogResponse",
    "StartTrainingRequest",
    "StartTrainingResponse",
    "JobStatusResponse",
    "ExperimentTrainingStatus"
]
