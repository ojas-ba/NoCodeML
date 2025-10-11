"""Pydantic schemas for experiment request/response validation."""
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from uuid import UUID
from pydantic import BaseModel, Field, field_validator


class FeatureTypes(BaseModel):
    """Feature type categorization."""
    numerical: List[str] = Field(default_factory=list, description="Numerical feature columns")
    categorical: List[str] = Field(default_factory=list, description="Categorical feature columns")


class ModelConfig(BaseModel):
    """Configuration for a single model."""
    model_config = {
        "protected_namespaces": (),  # Allow model_ prefix
        "populate_by_name": True  # Allow both camelCase and snake_case
    }
    
    model_type: str = Field(..., description="Model class name (e.g., 'LogisticRegression')", alias="modelType")
    display_name: str = Field(..., description="Human-readable model name", alias="displayName")
    preset: Literal["fast", "default"] = Field(
        default="default",
        description="Hyperparameter preset"
    )
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resolved hyperparameters (merged from preset + custom)"
    )
    custom_hyperparameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="User-provided custom hyperparameters (overrides preset)",
        alias="customHyperparameters"
    )


class ExperimentConfig(BaseModel):
    """Schema for experiment configuration with validation."""
    # Basic config
    taskType: Optional[Literal["classification", "regression"]] = None
    targetColumn: Optional[str] = None
    
    # Feature selection (NEW!)
    selectedFeatures: Optional[List[str]] = Field(
        default=None,
        description="Features selected for training"
    )
    featureTypes: Optional[FeatureTypes] = Field(
        default=None,
        description="Feature type categorization"
    )
    excludedColumns: Optional[List[str]] = Field(
        default_factory=list,
        description="Columns excluded from training (e.g., ID columns)"
    )
    
    # Training config
    trainTestSplit: Optional[float] = Field(
        None,
        ge=0.1,
        le=0.9,
        description="Train/test split ratio between 0.1 and 0.9"
    )
    randomSeed: int = Field(default=42, description="Random seed for reproducibility")
    
    # Hyperparameter optimization
    enableOptimization: Optional[bool] = Field(
        default=False,
        description="Enable automatic hyperparameter optimization"
    )
    
    # Model selection (UPDATED!)
    features: Optional[List[str]] = Field(
        default=None,
        description="DEPRECATED: Use 'selectedFeatures' instead"
    )
    selectedModels: Optional[List[str]] = Field(
        default=None,
        description="DEPRECATED: Use 'models' instead"
    )
    models: Optional[List[ModelConfig]] = Field(
        default=None,
        description="List of model configurations"
    )
    
    class Config:
        extra = "forbid"  # Reject unknown fields


class ExperimentCreate(BaseModel):
    """Schema for experiment creation request."""
    name: str = Field(..., min_length=1, max_length=255, description="Experiment name")
    datasetId: str = Field(..., description="ID of the dataset to use for this experiment")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate experiment name is not just whitespace."""
        if not v.strip():
            raise ValueError('Experiment name cannot be empty or just whitespace')
        return v.strip()


class ExperimentUpdate(BaseModel):
    """Schema for updating experiment."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Experiment name")
    config: Optional[ExperimentConfig] = Field(None, description="Experiment configuration")
    status: Optional[Literal["in_progress", "completed"]] = Field(None, description="Experiment status")
    results: Optional[Dict[str, Any]] = Field(None, description="Experiment results")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate experiment name is not just whitespace."""
        if v is not None and not v.strip():
            raise ValueError('Experiment name cannot be empty or just whitespace')
        return v.strip() if v else None


class ExperimentResponse(BaseModel):
    """Schema for experiment data in API responses."""
    id: str
    name: str
    datasetId: str
    datasetName: Optional[str] = None
    status: str
    config: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None
    createdAt: datetime
    updatedAt: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class ExperimentListResponse(BaseModel):
    """Schema for paginated experiment list response."""
    experiments: List[ExperimentResponse]
    total: int
    page: int
    pageSize: int
    totalPages: int
