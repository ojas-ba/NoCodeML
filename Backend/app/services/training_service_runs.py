"""Training service layer - run-based architecture for ML training."""
import uuid
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from app.models.training import TrainingRun
from app.models.experiment import Experiment
from app.models.dataset import Dataset
from app.services.task_manager import TaskManager


async def start_training_run(
    db: AsyncSession,
    experiment_id: uuid.UUID,
    user_id: int
) -> Dict[str, Any]:
    """
    Start a new training run for entire config.
    
    Args:
        db: Database session
        experiment_id: Experiment ID
        user_id: User ID
        
    Returns:
        {
            'run_id': str,
            'run_number': int,
            'job_id': str,
            'status': 'pending',
            'created_at': str
        }
        
    Raises:
        HTTPException: If experiment not found or config invalid
    """
    # 1. Verify experiment exists and belongs to user
    experiment_query = select(Experiment).where(
        and_(Experiment.id == experiment_id, Experiment.user_id == user_id)
    )
    result = await db.execute(experiment_query)
    experiment = result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found"
        )
    
    # 2. Validate config completeness
    config = experiment.config
    if not config or not config.get('taskType') or not config.get('targetColumn'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Experiment config is incomplete. Please configure task type and target column."
        )
    
    if not config.get('selectedFeatures') or len(config.get('selectedFeatures', [])) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No features selected for training"
        )
    
    if not config.get('models') or len(config.get('models', [])) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No models selected for training"
        )
    
    # 3. Get dataset
    dataset_query = select(Dataset).where(Dataset.id == experiment.dataset_id)
    dataset_result = await db.execute(dataset_query)
    dataset = dataset_result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # 4. Generate run_number atomically (prevent race condition)
    # Lock the experiment row to prevent concurrent run number conflicts
    lock_query = select(Experiment).where(Experiment.id == experiment_id).with_for_update()
    await db.execute(lock_query)
    
    # Get max run_number for this experiment
    max_run_query = select(func.coalesce(func.max(TrainingRun.run_number), 0)).where(
        TrainingRun.experiment_id == experiment_id
    )
    max_run_result = await db.execute(max_run_query)
    run_number = (max_run_result.scalar() or 0) + 1
    
    # 5. Create training_run record with config snapshot
    training_run = TrainingRun(
        experiment_id=experiment_id,
        run_number=run_number,
        status='pending',
        config_snapshot=config  # Save immutable snapshot
    )
    
    db.add(training_run)
    await db.commit()
    await db.refresh(training_run)
    
    # 6. Start Celery task for entire run
    task_manager = TaskManager()
    try:
        job_id = task_manager.start_training_run_task(
            run_id=str(training_run.id),
            experiment_id=str(experiment_id),
            dataset_id=str(dataset.id)
        )
        
        # 7. Update run with job_id
        training_run.job_id = job_id
        await db.commit()
        
    except Exception as e:
        # If Celery dispatch fails, mark run as failed
        training_run.status = 'failed'
        training_run.error_message = f"Failed to dispatch training task: {str(e)}"
        training_run.completed_at = datetime.now(timezone.utc)
        await db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to start training: {str(e)}"
        )
    
    return {
        'run_id': str(training_run.id),
        'run_number': run_number,
        'job_id': job_id,
        'status': 'pending',
        'created_at': training_run.created_at.isoformat()
    }


async def get_experiment_runs(
    db: AsyncSession,
    experiment_id: uuid.UUID,
    user_id: int,
    page: int = 1,
    page_size: int = 20
) -> Dict[str, Any]:
    """
    Get paginated list of training runs for an experiment.
    
    Args:
        db: Database session
        experiment_id: Experiment ID
        user_id: User ID
        page: Page number (1-indexed)
        page_size: Items per page
        
    Returns:
        Paginated list of training runs
    """
    # Verify ownership
    experiment_query = select(Experiment).where(
        and_(Experiment.id == experiment_id, Experiment.user_id == user_id)
    )
    result = await db.execute(experiment_query)
    experiment = result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found"
        )
    
    # Get total count
    count_query = select(func.count(TrainingRun.id)).where(
        TrainingRun.experiment_id == experiment_id
    )
    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0
    
    # Get paginated runs
    offset = (page - 1) * page_size
    runs_query = (
        select(TrainingRun)
        .where(TrainingRun.experiment_id == experiment_id)
        .order_by(TrainingRun.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    runs_result = await db.execute(runs_query)
    runs = runs_result.scalars().all()
    
    return {
        'runs': [
            {
                'id': str(run.id),
                'run_number': run.run_number,
                'status': run.status,
                'started_at': run.started_at.isoformat() if run.started_at else None,
                'completed_at': run.completed_at.isoformat() if run.completed_at else None,
                'duration_seconds': run.duration_seconds,
                'results_summary': {
                    **run.results.get('summary', {}),
                    'best_model': run.results.get('best_model')
                } if run.results else None,
                'created_at': run.created_at.isoformat()
            }
            for run in runs
        ],
        'total': total,
        'page': page,
        'page_size': page_size,
        'total_pages': (total + page_size - 1) // page_size if total > 0 else 0
    }


async def get_run_details(
    db: AsyncSession,
    run_id: uuid.UUID,
    user_id: int
) -> Dict[str, Any]:
    """
    Get complete details of a training run including all results.
    
    Args:
        db: Database session
        run_id: Training run ID
        user_id: User ID
        
    Returns:
        Complete run details with results
    """
    # Get run with experiment to verify ownership
    run_query = (
        select(TrainingRun)
        .join(Experiment, TrainingRun.experiment_id == Experiment.id)
        .where(
            and_(
                TrainingRun.id == run_id,
                Experiment.user_id == user_id
            )
        )
    )
    result = await db.execute(run_query)
    run = result.scalar_one_or_none()
    
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training run not found"
        )
    
    return {
        'id': str(run.id),
        'experiment_id': str(run.experiment_id),
        'run_number': run.run_number,
        'status': run.status,
        'config_snapshot': run.config_snapshot,
        'results': run.results or {},
        'artifacts': run.artifacts or {},
        'started_at': run.started_at.isoformat() if run.started_at else None,
        'completed_at': run.completed_at.isoformat() if run.completed_at else None,
        'duration_seconds': run.duration_seconds,
        'error_message': run.error_message,
        'created_at': run.created_at.isoformat()
    }


async def get_run_status(
    db: AsyncSession,
    run_id: uuid.UUID,
    user_id: int
) -> Dict[str, Any]:
    """
    Get lightweight status of a training run (for polling).
    
    Args:
        db: Database session
        run_id: Training run ID
        user_id: User ID
        
    Returns:
        Run status info
    """
    # Get run with experiment to verify ownership
    run_query = (
        select(TrainingRun)
        .join(Experiment, TrainingRun.experiment_id == Experiment.id)
        .where(
            and_(
                TrainingRun.id == run_id,
                Experiment.user_id == user_id
            )
        )
    )
    result = await db.execute(run_query)
    run = result.scalar_one_or_none()
    
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training run not found"
        )
    
    # Calculate progress
    progress = 0
    if run.status == 'completed':
        progress = 100
    elif run.status == 'running':
        # Could enhance with real-time progress from Celery
        progress = 50
    elif run.status == 'pending':
        progress = 0
    
    return {
        'run_id': str(run.id),
        'run_number': run.run_number,
        'status': run.status,
        'progress': progress,
        'error_message': run.error_message,
        'started_at': run.started_at.isoformat() if run.started_at else None
    }
