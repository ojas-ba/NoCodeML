"""Training API endpoints for starting and monitoring ML training jobs."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import List, Optional, Any
import uuid
import copy
from datetime import datetime, date
from decimal import Decimal

from app.db.session import get_db
from app.core.deps import get_current_user
from app.models import User
from app.models.experiment import Experiment, TrainingStatus
from app.models.training import TrainingJob, TrainingResult, TrainingLog
from app.services import training_service
from app.services import training_service_runs
from app.schemas.training import (
    TrainingJobResponse,
    TrainingResultResponse,
    StartTrainingResponse,
    StartTrainingRequest,
    JobStatusResponse,
    ExperimentTrainingStatus,
    TrainingLogResponse
)

router = APIRouter()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_progress(result: dict) -> dict:
    """Calculate progress info from run result with safe nested access."""
    if not isinstance(result, dict):
        return {'percent': 0, 'message': 'Invalid result data'}
    
    status = result.get('status', 'pending')
    
    if status == 'completed':
        return {'percent': 100, 'message': 'Completed'}
    elif status == 'failed':
        error_msg = result.get('error_message', 'Training failed')
        return {'percent': 0, 'message': f'Failed: {error_msg}'}
    elif status == 'running':
        # Try to get progress from results if available
        results = result.get('results')
        if results and isinstance(results, dict):
            progress_data = results.get('progress')
            if progress_data and isinstance(progress_data, dict):
                current = progress_data.get('current', 0)
                total = progress_data.get('total', 1)
                current_model = progress_data.get('current_model', 'model')
                
                # Safe division with validation
                if isinstance(current, (int, float)) and isinstance(total, (int, float)) and total > 0:
                    percent = int((current / total) * 100)
                    percent = max(0, min(100, percent))  # Clamp to 0-100
                    return {
                        'percent': percent,
                        'message': f"Training {current_model}... ({current}/{total})"
                    }
        
        return {'percent': 50, 'message': 'Training in progress...'}
    else:  # pending or unknown
        return {'percent': 0, 'message': 'Pending'}


# ============================================================================
# RUN-BASED TRAINING ENDPOINTS (NEW ARCHITECTURE)
# ============================================================================

@router.post("/experiments/{experiment_id}/runs")
async def start_training_run(
    experiment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Start a new training run for the experiment's current config.
    All selected models will be trained as one run.
    """
    result = await training_service_runs.start_training_run(
        db=db,
        experiment_id=experiment_id,
        user_id=current_user.id
    )
    return result


@router.get("/experiments/{experiment_id}/runs")
async def list_training_runs(
    experiment_id: uuid.UUID,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get paginated list of training runs for an experiment.
    """
    result = await training_service_runs.get_experiment_runs(
        db=db,
        experiment_id=experiment_id,
        user_id=current_user.id,
        page=page,
        page_size=page_size
    )
    return result


@router.get("/runs/{run_id}")
async def get_run_details(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get complete details and results for a specific training run.
    """
    result = await training_service_runs.get_run_details(
        db=db,
        run_id=run_id,
        user_id=current_user.id
    )
    return result


@router.get("/runs/{run_id}/status")
async def get_run_status(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get current status of a training run (for polling).
    """
    result = await training_service_runs.get_run_status(
        db=db,
        run_id=run_id,
        user_id=current_user.id
    )
    return result


# ============================================================================
# LEGACY JOB-BASED ENDPOINTS (KEPT FOR BACKWARD COMPATIBILITY)
# ============================================================================


def _sanitize_for_json(value: Any) -> Any:
    """Recursively convert complex Python objects into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value

@router.post("/experiments/{experiment_id}/train", response_model=StartTrainingResponse)
async def start_training(
    experiment_id: uuid.UUID,
    request: StartTrainingRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Start training jobs for selected model types.
    
    Args:
        experiment_id: ID of the experiment to train
        model_types: List of model types to train (e.g. ['LogisticRegression', 'RandomForest'])
        
    Returns:
        List of created training jobs
    """
    # Verify experiment exists and belongs to user
    experiment_query = select(Experiment).where(
        and_(
            Experiment.id == experiment_id,
            Experiment.user_id == current_user.id
        )
    )
    result = await db.execute(experiment_query)
    experiment = result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found or access denied"
        )
    
    # Read model types from experiment's saved configuration
    experiment_config = experiment.config or {}
    models = experiment_config.get('models', [])
    
    if not models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No models configured in experiment. Please configure models first."
        )
    
    # Extract model types from configured models
    model_types = [model.get('modelType') or model.get('model_type') for model in models]
    model_types = [mt for mt in model_types if mt]  # Filter out None values
    
    if not model_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid model types found in experiment configuration."
        )
    
    # Convert config to a JSON-serializable snapshot so it can be stored in JSONB
    try:
        config_snapshot = _sanitize_for_json(experiment_config)
    except Exception as exc:  # pragma: no cover - defensive catch for unexpected data types
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to serialize experiment config: {exc}"
        )

    # Create model configs for each model type with full config snapshot
    model_configs = []
    for model_type in model_types:
        model_configs.append({
            "model_type": model_type,
            "experiment_id": str(experiment_id),  # Convert UUID to string for JSON serialization
            # Deep copy ensures per-job isolation of the config snapshot
            "config": copy.deepcopy(config_snapshot)
        })
    
    # Start training jobs
    response = await training_service.start_training_jobs(
        db=db,
        experiment_id=experiment_id,
        model_configs=model_configs,
        user_id=current_user.id
    )
    
    # Update experiment training status
    experiment.training_status = TrainingStatus.TRAINING
    await db.commit()
    
    return response


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed status of a training job.
    
    Args:
        job_id: ID of the training job
        
    Returns:
        Job status with progress and recent logs
    """
    # Get job with experiment to verify ownership
    job_query = select(TrainingJob).join(Experiment).where(
        and_(
            TrainingJob.id == job_id,
            Experiment.user_id == current_user.id
        )
    )
    result = await db.execute(job_query)
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found or access denied"
        )
    
    # Get recent logs (last 10)
    logs_query = select(TrainingLog).where(
        TrainingLog.job_id == job_id
    ).order_by(TrainingLog.created_at.desc()).limit(10)
    logs_result = await db.execute(logs_query)
    logs = logs_result.scalars().all()
    
    # Create response
    job_response = TrainingJobResponse.model_validate(job)
    log_responses = [TrainingLogResponse.model_validate(log) for log in logs]
    
    return JobStatusResponse(
        job=job_response,
        latest_logs=log_responses
    )


@router.get("/experiments/{experiment_id}/jobs", response_model=List[TrainingJobResponse])
async def list_training_jobs(
    experiment_id: uuid.UUID,
    status_filter: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List all training jobs for an experiment.
    
    Args:
        experiment_id: ID of the experiment
        status_filter: Optional status filter (queued, running, completed, failed)
        
    Returns:
        List of training jobs
    """
    # Verify experiment belongs to user
    experiment_query = select(Experiment).where(
        and_(
            Experiment.id == experiment_id,
            Experiment.user_id == current_user.id
        )
    )
    result = await db.execute(experiment_query)
    experiment = result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found or access denied"
        )
    
    # Build query for jobs
    jobs_query = select(TrainingJob).where(TrainingJob.experiment_id == experiment_id)
    
    if status_filter:
        jobs_query = jobs_query.where(TrainingJob.status == status_filter)
    
    jobs_query = jobs_query.order_by(TrainingJob.created_at.desc())
    
    result = await db.execute(jobs_query)
    jobs = result.scalars().all()
    
    return [TrainingJobResponse.model_validate(job) for job in jobs]


@router.get("/experiments/{experiment_id}/status", response_model=ExperimentTrainingStatus)
async def get_experiment_training_status(
    experiment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get overall training status for an experiment.
    
    Args:
        experiment_id: ID of the experiment
        
    Returns:
        Training status overview with job counts and progress
    """
    # Verify experiment belongs to user
    experiment_query = select(Experiment).where(
        and_(
            Experiment.id == experiment_id,
            Experiment.user_id == current_user.id
        )
    )
    result = await db.execute(experiment_query)
    experiment = result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found or access denied"
        )
    
    # Get all jobs for experiment
    jobs_query = select(TrainingJob).where(
        TrainingJob.experiment_id == experiment_id
    ).order_by(TrainingJob.created_at.desc())
    
    result = await db.execute(jobs_query)
    jobs = result.scalars().all()
    
    # Categorize jobs
    active_jobs = [job for job in jobs if job.status in ["queued", "running"]]
    completed_jobs = [job for job in jobs if job.status == "completed"]
    failed_jobs = [job for job in jobs if job.status == "failed"]
    
    # Calculate progress
    total_jobs = len(jobs)
    progress_percent = 0.0
    if total_jobs > 0:
        progress_percent = (len(completed_jobs) / total_jobs) * 100
    
    return ExperimentTrainingStatus(
        experiment_id=experiment_id,
        training_status=experiment.training_status.value,
        active_jobs=[TrainingJobResponse.model_validate(job) for job in active_jobs],
        completed_jobs=[TrainingJobResponse.model_validate(job) for job in completed_jobs],
        failed_jobs=[TrainingJobResponse.model_validate(job) for job in failed_jobs],
        total_jobs=total_jobs,
        progress_percent=progress_percent
    )


@router.get("/results/{result_id}", response_model=TrainingResultResponse)
async def get_training_result(
    result_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed training result.
    
    Args:
        result_id: ID of the training result
        
    Returns:
        Training result with metrics and feature importance
    """
    # Get result with job and experiment to verify ownership
    result_query = select(TrainingResult).join(TrainingJob).join(Experiment).where(
        and_(
            TrainingResult.id == result_id,
            Experiment.user_id == current_user.id
        )
    )
    result = await db.execute(result_query)
    training_result = result.scalar_one_or_none()
    
    if not training_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training result not found or access denied"
        )
    
    return TrainingResultResponse.model_validate(training_result)


@router.get("/results/job/{job_id}", response_model=TrainingResultResponse)
async def get_result_by_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get training result by job ID.
    
    Args:
        job_id: ID of the training job
        
    Returns:
        Training result for the job
    """
    # Get result with job and experiment to verify ownership
    result_query = select(TrainingResult).join(TrainingJob).join(Experiment).where(
        and_(
            TrainingResult.job_id == job_id,
            Experiment.user_id == current_user.id
        )
    )
    result = await db.execute(result_query)
    training_result = result.scalar_one_or_none()
    
    if not training_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training result not found or job not completed"
        )
    
    return TrainingResultResponse.model_validate(training_result)
