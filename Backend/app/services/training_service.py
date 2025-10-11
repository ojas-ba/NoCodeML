"""Training service layer for managing ML training jobs."""
import uuid
import json
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_

from app.models.training import TrainingJob, TrainingResult, TrainingLog, TrainingJobStatus
from app.models.experiment import Experiment, TrainingStatus
from app.models.dataset import Dataset
from app.schemas.training import (
    TrainingJobCreate, 
    TrainingJobResponse, 
    StartTrainingRequest,
    StartTrainingResponse,
    JobStatusResponse,
    ExperimentTrainingStatus
)
from app.core.model_cache import get_model_instance_config, get_model_info
from app.core.model_defaults import get_available_models, get_preprocessing_config, get_training_config
from app.services.task_manager import TaskManager


async def create_training_job(
    db: AsyncSession,
    experiment_id: uuid.UUID,
    model_type: str,
    config: Dict[str, Any],
    user_id: int
) -> TrainingJob:
    """
    Create a new training job for an experiment.
    
    Args:
        db: Database session
        experiment_id: ID of the experiment to train
        model_type: Type of model to train
        config: Training configuration including hyperparameters
        user_id: ID of the user creating the job
    
    Returns:
        Created training job
    
    Raises:
        HTTPException: If experiment not found or invalid model type
    """
    # Verify experiment exists and belongs to user
    experiment_query = select(Experiment).where(
        and_(
            Experiment.id == experiment_id,
            Experiment.user_id == user_id
        )
    )
    experiment_result = await db.execute(experiment_query)
    experiment = experiment_result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found"
        )
    
    # Determine task type from experiment config
    task_type = experiment.config.get('taskType', 'classification')
    
    # Validate model type
    available_models = get_available_models(task_type)
    if model_type not in available_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_type}' not available for {task_type}"
        )
    
    # Create training job
    training_job = TrainingJob(
        experiment_id=experiment_id,
        model_type=model_type,
        config_json=config,
        status=TrainingJobStatus.QUEUED
    )
    
    db.add(training_job)
    await db.commit()
    await db.refresh(training_job)
    
    return training_job


async def start_training_jobs(
    db: AsyncSession,
    experiment_id: uuid.UUID,
    model_configs: List[Dict[str, Any]],
    user_id: int
) -> StartTrainingResponse:
    """
    Start multiple training jobs for an experiment with Celery tasks.
    
    Args:
        db: Database session
        experiment_id: ID of the experiment
        model_configs: List of model configurations
        user_id: ID of the user
    
    Returns:
        Response with created training jobs
    """
    # Get experiment info
    experiment_query = select(Experiment).where(
        and_(
            Experiment.id == experiment_id,
            Experiment.user_id == user_id
        )
    )
    experiment_result = await db.execute(experiment_query)
    experiment = experiment_result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found"
        )
    
    # Get dataset info
    dataset_query = select(Dataset).where(Dataset.id == experiment.dataset_id)
    dataset_result = await db.execute(dataset_query)
    dataset = dataset_result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Determine task type and target column from experiment config
    task_type = experiment.config.get('taskType', 'classification')
    target_column = experiment.config.get('targetColumn')
    
    if not target_column:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Target column not specified in experiment configuration"
        )
    
    created_jobs = []
    task_manager = TaskManager()
    
    # Create training jobs and start Celery tasks for each model
    for config in model_configs:
        model_type = config.get('model_type')
        if not model_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="model_type is required in each config"
            )
        
        # Create training job for this model
        # Extract only the config portion (already sanitized in API layer)
        job_config = config.get('config', {})
        
        job = await create_training_job(
            db=db,
            experiment_id=experiment_id,
            model_type=model_type,
            config=job_config,
            user_id=user_id
        )
        
        # Start individual Celery task for this model
        try:
            celery_task_id = task_manager.start_training_task(
                job_id=str(job.id),
                experiment_id=str(experiment_id),
                dataset_id=str(dataset.id),
                model_type=model_type,
                task_type=task_type
            )
            
            # Update job with Celery task ID
            await update_training_job_status(
                db=db,
                job_id=job.id,
                status=TrainingJobStatus.QUEUED,
                celery_task_id=celery_task_id
            )
            await db.refresh(job)
            
        except Exception as e:
            # If Celery task failed to start, mark job as failed
            await update_training_job_status(
                db=db,
                job_id=job.id,
                status=TrainingJobStatus.FAILED,
                error_message=f"Failed to start training task: {str(e)}"
            )
            await db.refresh(job)
        
        created_jobs.append(job)
    
    # Update experiment training status
    await update_experiment_training_status(db, experiment_id, TrainingStatus.TRAINING)
    
    # Convert to response format
    job_responses = [
        TrainingJobResponse(
            id=job.id,
            experiment_id=job.experiment_id,
            model_type=job.model_type,
            status=job.status.value,
            config_json=job.config_json,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            celery_task_id=job.celery_task_id
        )
        for job in created_jobs
    ]
    
    return StartTrainingResponse(
        jobs=job_responses,
        total_jobs=len(job_responses)
    )


async def get_training_job(
    db: AsyncSession,
    job_id: uuid.UUID,
    user_id: int
) -> Optional[TrainingJob]:
    """
    Get a training job by ID.
    
    Args:
        db: Database session
        job_id: ID of the training job
        user_id: ID of the user
    
    Returns:
        Training job if found and accessible
    """
    query = select(TrainingJob).join(Experiment).where(
        and_(
            TrainingJob.id == job_id,
            Experiment.user_id == user_id
        )
    )
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def get_job_status(
    db: AsyncSession,
    job_id: uuid.UUID,
    user_id: int
) -> Optional[JobStatusResponse]:
    """
    Get detailed status of a training job.
    
    Args:
        db: Database session
        job_id: ID of the training job
        user_id: ID of the user
    
    Returns:
        Job status response with progress and logs
    """
    job = await get_training_job(db, job_id, user_id)
    if not job:
        return None
    
    # Get recent logs (last 10)
    logs_query = select(TrainingLog).where(
        TrainingLog.job_id == job_id
    ).order_by(TrainingLog.timestamp.desc()).limit(10)
    logs_result = await db.execute(logs_query)
    logs = logs_result.scalars().all()
    
    # Create response
    job_response = TrainingJobResponse(
        id=job.id,
        experiment_id=job.experiment_id,
        model_type=job.model_type,
        status=job.status.value,
        config_json=job.config_json,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        celery_task_id=job.celery_task_id
    )
    
    # Get latest progress if available
    progress = None
    if logs:
        latest_log = logs[0]
        progress = {
            "job_id": job_id,
            "status": job.status.value,
            "progress_percent": latest_log.progress_percent,
            "current_epoch": latest_log.epoch,
            "current_metrics": latest_log.metrics_json,
            "message": latest_log.message
        }
    
    from app.schemas.training import TrainingLogResponse
    log_responses = [
        TrainingLogResponse(
            id=log.id,
            job_id=log.job_id,
            epoch=log.epoch,
            progress_percent=log.progress_percent,
            metrics_json=log.metrics_json,
            message=log.message,
            timestamp=log.timestamp
        )
        for log in reversed(logs)  # Reverse to show chronological order
    ]
    
    return JobStatusResponse(
        job=job_response,
        progress=progress,
        latest_logs=log_responses
    )


async def get_experiment_training_status(
    db: AsyncSession,
    experiment_id: uuid.UUID,
    user_id: int
) -> Optional[ExperimentTrainingStatus]:
    """
    Get training status overview for an experiment.
    
    Args:
        db: Database session
        experiment_id: ID of the experiment
        user_id: ID of the user
    
    Returns:
        Experiment training status overview
    """
    # Verify experiment access
    experiment_query = select(Experiment).where(
        and_(
            Experiment.id == experiment_id,
            Experiment.user_id == user_id
        )
    )
    experiment_result = await db.execute(experiment_query)
    experiment = experiment_result.scalar_one_or_none()
    
    if not experiment:
        return None
    
    # Get all training jobs for this experiment
    jobs_query = select(TrainingJob).where(
        TrainingJob.experiment_id == experiment_id
    ).order_by(TrainingJob.created_at.desc())
    jobs_result = await db.execute(jobs_query)
    jobs = jobs_result.scalars().all()
    
    # Group jobs by status
    active_jobs = [job for job in jobs if job.status in [TrainingJobStatus.QUEUED, TrainingJobStatus.RUNNING]]
    completed_jobs = [job for job in jobs if job.status == TrainingJobStatus.COMPLETED]
    failed_jobs = [job for job in jobs if job.status in [TrainingJobStatus.FAILED, TrainingJobStatus.CANCELLED]]
    
    # Calculate overall progress
    total_jobs = len(jobs)
    completed_count = len(completed_jobs)
    progress_percent = (completed_count / total_jobs * 100) if total_jobs > 0 else 0
    
    # Convert to response format
    def job_to_response(job: TrainingJob) -> TrainingJobResponse:
        return TrainingJobResponse(
            id=job.id,
            experiment_id=job.experiment_id,
            model_type=job.model_type,
            status=job.status.value,
            config_json=job.config_json,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            celery_task_id=job.celery_task_id
        )
    
    return ExperimentTrainingStatus(
        experiment_id=experiment_id,
        training_status=experiment.training_status.value,
        active_jobs=[job_to_response(job) for job in active_jobs],
        completed_jobs=[job_to_response(job) for job in completed_jobs],
        failed_jobs=[job_to_response(job) for job in failed_jobs],
        total_jobs=total_jobs,
        progress_percent=progress_percent
    )


async def update_training_job_status(
    db: AsyncSession,
    job_id: uuid.UUID,
    status: TrainingJobStatus,
    error_message: Optional[str] = None,
    celery_task_id: Optional[str] = None
) -> None:
    """
    Update training job status.
    
    Args:
        db: Database session
        job_id: ID of the training job
        status: New status
        error_message: Error message if failed
        celery_task_id: Celery task ID for tracking
    """
    update_data = {"status": status}
    
    if status == TrainingJobStatus.RUNNING and not celery_task_id:
        update_data["started_at"] = datetime.utcnow()
    elif status in [TrainingJobStatus.COMPLETED, TrainingJobStatus.FAILED, TrainingJobStatus.CANCELLED]:
        update_data["completed_at"] = datetime.utcnow()
    
    if error_message:
        update_data["error_message"] = error_message
    
    if celery_task_id:
        update_data["celery_task_id"] = celery_task_id
    
    await db.execute(
        update(TrainingJob)
        .where(TrainingJob.id == job_id)
        .values(**update_data)
    )
    await db.commit()


async def update_experiment_training_status(
    db: AsyncSession,
    experiment_id: uuid.UUID,
    status: TrainingStatus
) -> None:
    """
    Update experiment training status.
    
    Args:
        db: Database session
        experiment_id: ID of the experiment
        status: New training status
    """
    await db.execute(
        update(Experiment)
        .where(Experiment.id == experiment_id)
        .values(training_status=status)
    )
    await db.commit()


async def cancel_training_job(
    db: AsyncSession,
    job_id: uuid.UUID,
    user_id: int
) -> bool:
    """
    Cancel a training job and its Celery task.
    
    Args:
        db: Database session
        job_id: ID of the training job
        user_id: ID of the user
    
    Returns:
        True if job was cancelled, False if not found or cannot be cancelled
    """
    job = await get_training_job(db, job_id, user_id)
    if not job:
        return False
    
    # Can only cancel queued or running jobs
    if job.status not in [TrainingJobStatus.QUEUED, TrainingJobStatus.RUNNING]:
        return False
    
    # Cancel Celery task if it exists
    if job.celery_task_id:
        try:
            task_manager = TaskManager()
            cancel_result = task_manager.cancel_task(job.celery_task_id)
            
            if not cancel_result['success']:
                print(f"Failed to cancel Celery task: {cancel_result.get('error')}")
        except Exception as e:
            print(f"Error cancelling Celery task: {e}")
    
    # Update job status to cancelled
    await update_training_job_status(
        db=db,
        job_id=job_id,
        status=TrainingJobStatus.CANCELLED
    )
    
    return True


async def get_celery_task_status(
    db: AsyncSession,
    job_id: uuid.UUID,
    user_id: int
) -> Optional[Dict[str, Any]]:
    """
    Get Celery task status for a training job.
    
    Args:
        db: Database session
        job_id: ID of the training job
        user_id: ID of the user
    
    Returns:
        Celery task status information or None
    """
    job = await get_training_job(db, job_id, user_id)
    if not job or not job.celery_task_id:
        return None
    
    try:
        task_manager = TaskManager()
        return task_manager.get_task_status(job.celery_task_id)
    except Exception as e:
        return {
            'error': f'Failed to get task status: {str(e)}',
            'task_id': job.celery_task_id
        }


async def sync_job_with_celery_status(
    db: AsyncSession,
    job_id: uuid.UUID
) -> bool:
    """
    Synchronize training job status with Celery task status.
    
    Args:
        db: Database session
        job_id: ID of the training job
    
    Returns:
        True if synchronized successfully
    """
    try:
        # Get job
        job_query = select(TrainingJob).where(TrainingJob.id == job_id)
        job_result = await db.execute(job_query)
        job = job_result.scalar_one_or_none()
        
        if not job or not job.celery_task_id:
            return False
        
        # Get Celery task status
        task_manager = TaskManager()
        task_status = task_manager.get_task_status(job.celery_task_id)
        
        # Map Celery status to job status
        celery_to_job_status = {
            'PENDING': TrainingJobStatus.QUEUED,
            'STARTED': TrainingJobStatus.RUNNING,
            'PROGRESS': TrainingJobStatus.RUNNING,
            'SUCCESS': TrainingJobStatus.COMPLETED,
            'FAILURE': TrainingJobStatus.FAILED,
            'REVOKED': TrainingJobStatus.CANCELLED,
            'RETRY': TrainingJobStatus.RUNNING,
        }
        
        celery_status = task_status.get('status', 'PENDING')
        new_job_status = celery_to_job_status.get(celery_status, job.status)
        
        # Update job status if different
        if new_job_status != job.status:
            error_message = None
            if new_job_status == TrainingJobStatus.FAILED:
                error_message = task_status.get('error', 'Task failed')
            
            await update_training_job_status(
                db=db,
                job_id=job_id,
                status=new_job_status,
                error_message=error_message
            )
        
        return True
        
    except Exception as e:
        print(f"Failed to sync job {job_id} with Celery: {e}")
        return False