"""Pydantic schemas for training jobs, results, and logs."""
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID
from pydantic import BaseModel, Field


class TrainingJobCreate(BaseModel):
    """Schema for creating a new training job."""
    experiment_id: UUID = Field(..., description="ID of the experiment to train")
    model_type: str = Field(..., description="Type of model to train")
    config: Dict[str, Any] = Field(..., description="Training configuration including hyperparameters")


class TrainingJobResponse(BaseModel):
    """Schema for training job response."""
    id: UUID
    experiment_id: UUID
    model_type: str
    status: str
    config_json: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    celery_task_id: Optional[str] = None
    
    class Config:
        from_attributes = True


class TrainingProgress(BaseModel):
    """Schema for training progress updates."""
    job_id: UUID
    status: str
    progress_percent: Optional[float] = Field(None, ge=0, le=100)
    current_epoch: Optional[int] = None
    current_metrics: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class TrainingMetrics(BaseModel):
    """Schema for training metrics."""
    # Common metrics
    training_time_seconds: float
    
    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    
    # Regression metrics
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    
    # Cross-validation
    cv_scores: Optional[List[float]] = None
    mean_cv_score: Optional[float] = None
    std_cv_score: Optional[float] = None


class FeatureImportance(BaseModel):
    """Schema for feature importance data."""
    feature_importances: Dict[str, float] = Field(..., description="Feature name to importance mapping")
    top_features: List[Dict[str, Any]] = Field(..., description="Top N most important features")


class ConfusionMatrix(BaseModel):
    """Schema for confusion matrix data."""
    matrix: List[List[int]] = Field(..., description="Confusion matrix as 2D array")
    labels: List[str] = Field(..., description="Class labels")
    accuracy: float = Field(..., description="Overall accuracy")


class TrainingResultResponse(BaseModel):
    """Schema for complete training results."""
    id: UUID
    job_id: UUID
    model_path: str
    metrics: TrainingMetrics
    feature_importance: Optional[FeatureImportance] = None
    confusion_matrix: Optional[ConfusionMatrix] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class TrainingLogResponse(BaseModel):
    """Schema for training log entries."""
    id: UUID
    job_id: UUID
    epoch: Optional[int] = None
    progress_percent: Optional[float] = None
    metrics_json: Dict[str, Any]
    message: Optional[str] = None
    timestamp: datetime
    
    class Config:
        from_attributes = True


class StartTrainingRequest(BaseModel):
    """Schema for starting training on an experiment."""
    # Model types are read from experiment's saved configuration
    pass
    
    
class StartTrainingResponse(BaseModel):
    """Schema for start training response."""
    jobs: List[TrainingJobResponse] = Field(..., description="Created training jobs")
    total_jobs: int = Field(..., description="Total number of jobs created")


class JobStatusResponse(BaseModel):
    """Schema for job status check response."""
    job: TrainingJobResponse
    progress: Optional[TrainingProgress] = None
    latest_logs: List[TrainingLogResponse] = Field(default_factory=list, description="Recent log entries")


class ExperimentTrainingStatus(BaseModel):
    """Schema for experiment training status overview."""
    experiment_id: UUID
    training_status: str
    active_jobs: List[TrainingJobResponse]
    completed_jobs: List[TrainingJobResponse]
    failed_jobs: List[TrainingJobResponse]
    total_jobs: int
    progress_percent: float = Field(..., ge=0, le=100, description="Overall training progress")