"""Pydantic schemas for model configuration and training."""
from typing import Dict, Any, Optional, Literal, List
from pydantic import BaseModel, Field


class PreprocessingConfig(BaseModel):
    """Configuration for AutoClean preprocessing."""
    duplicates: bool = Field(default=True, description="Remove duplicate rows")
    missing_num: str = Field(default="auto", description="Strategy for numerical missing values")
    missing_categ: str = Field(default="auto", description="Strategy for categorical missing values")
    encode_categ: List[str] = Field(default=["onehot"], description="Categorical encoding methods")
    outliers: str = Field(default="auto", description="Outlier handling strategy")
    extract_datetime: bool = Field(default=False, description="Extract datetime features")


class TrainingConfig(BaseModel):
    """Configuration for training process."""
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test set size ratio")
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    stratify: bool = Field(default=True, description="Stratify train/test split for classification")
    cv_folds: int = Field(default=3, ge=2, le=10, description="Number of cross-validation folds")
    scaling: bool = Field(default=True, description="Apply standard scaling to features")
    feature_selection: bool = Field(default=False, description="Apply feature selection")


class ModelConfigRequest(BaseModel):
    """Request schema for model configuration."""
    model_type: str = Field(..., description="Model type (e.g., 'LogisticRegression')")
    preset: Literal["fast", "default"] = Field(default="default", description="Hyperparameter preset")
    custom_hyperparameters: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Custom hyperparameters to override defaults"
    )
    preprocessing: Optional[PreprocessingConfig] = Field(
        default=None,
        description="Preprocessing configuration (uses defaults if not provided)"
    )
    training: Optional[TrainingConfig] = Field(
        default=None,
        description="Training configuration (uses defaults if not provided)"
    )


class ModelConfigResponse(BaseModel):
    """Response schema for model configuration."""
    model_type: str
    display_name: str
    preset: str
    hyperparameters: Dict[str, Any]
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    supports_feature_importance: bool
    preprocessing_required: bool


class ModelSelectionRequest(BaseModel):
    """Request schema for selecting models for training."""
    models: List[ModelConfigRequest] = Field(..., description="List of model configurations")
    task_type: Literal["classification", "regression"] = Field(..., description="Task type")


class ModelSelectionResponse(BaseModel):
    """Response schema for model selection."""
    models: List[ModelConfigResponse]
    task_type: str
    total_models: int