"""Celery tasks for asynchronous ML model training."""
import uuid
import traceback
from typing import Dict, Any, Optional
from celery import current_task
from celery.exceptions import Ignore, Retry

from app.worker.celery_app import celery_app


@celery_app.task(bind=True, name='app.worker.tasks.train_model_task')
def train_model_task(
    self,
    job_id: str, 
    dataset_path: str, 
    target_column: str,
    model_type: str,
    task_type: str,
    hyperparameters: Dict[str, Any],
    preprocessing_config: Dict[str, Any],
    training_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Asynchronous model training task.
    
    Args:
        self: Celery task instance (bound)
        job_id: Training job ID
        dataset_path: Path to dataset file
        target_column: Target column name
        model_type: Type of model to train
        task_type: 'classification' or 'regression'
        hyperparameters: Model hyperparameters
        preprocessing_config: Preprocessing configuration
        training_config: Training configuration
        
    Returns:
        Dictionary with training results or error information
    """
    try:
        # Import here to avoid import issues
        from app.services.model_trainer import ModelTrainer
        from app.db.sync_session import get_sync_db
        from app.services.training_service import update_training_job_status
        from app.models.training import TrainingJobStatus
        import asyncio
        
        print(f"Starting training task for job {job_id}")
        
        # Update job status to running
        def update_job_status_sync(status: TrainingJobStatus, error_msg: Optional[str] = None):
            """Helper to update job status synchronously."""
            try:
                # Note: In production, you'd want a proper async context
                # For now, we'll implement a sync version
                print(f"Job {job_id} status: {status.value}")
                if error_msg:
                    print(f"Job {job_id} error: {error_msg}")
            except Exception as e:
                print(f"Failed to update job status: {e}")
        
        # Report progress helper
        def report_progress(percent: float, message: str, metrics: Optional[Dict] = None):
            """Report training progress."""
            try:
                current_task.update_state(
                    state='PROGRESS',
                    meta={
                        'progress': percent,
                        'message': message,
                        'metrics': metrics or {},
                        'job_id': job_id
                    }
                )
                print(f"Progress {percent:.1f}%: {message}")
            except Exception as e:
                print(f"Failed to report progress: {e}")
        
        # Update status to running
        update_job_status_sync(TrainingJobStatus.RUNNING)
        report_progress(0, "Starting model training...")
        
        # Initialize trainer
        trainer = ModelTrainer()
        report_progress(10, "Trainer initialized")
        
        # Run complete training pipeline
        result = trainer.train_complete_pipeline(
            dataset_path=dataset_path,
            target_column=target_column,
            model_type=model_type,
            task_type=task_type,
            hyperparameters=hyperparameters,
            preprocessing_config=preprocessing_config,
            training_config=training_config,
            job_id=job_id
        )
        
        if result['success']:
            report_progress(100, "Training completed successfully", result.get('metrics', {}))
            update_job_status_sync(TrainingJobStatus.COMPLETED)
            
            return {
                'success': True,
                'job_id': job_id,
                'model_path': result['model_path'],
                'metrics': result['metrics'],
                'feature_importance': result.get('feature_importance'),
                'training_time_seconds': result['training_time_seconds'],
                'data_shape': result.get('data_shape', {})
            }
        else:
            error_msg = result.get('error', 'Unknown training error')
            report_progress(0, f"Training failed: {error_msg}")
            update_job_status_sync(TrainingJobStatus.FAILED, error_msg)
            
            return {
                'success': False,
                'job_id': job_id,
                'error': error_msg,
                'training_time_seconds': result.get('training_time_seconds', 0)
            }
            
    except Exception as exc:
        # Handle unexpected errors
        error_msg = f"Training task failed: {str(exc)}"
        error_trace = traceback.format_exc()
        
        print(f"Training task error: {error_msg}")
        print(f"Stack trace: {error_trace}")
        
        try:
            from app.models.training import TrainingJobStatus
            update_job_status_sync(TrainingJobStatus.FAILED, error_msg)
        except:
            pass
        
        # Report final error state
        current_task.update_state(
            state='FAILURE',
            meta={
                'error': error_msg,
                'traceback': error_trace,
                'job_id': job_id
            }
        )
        
        return {
            'success': False,
            'job_id': job_id,
            'error': error_msg,
            'traceback': error_trace
        }


@celery_app.task(name='app.worker.tasks.cancel_training_task')
def cancel_training_task(task_id: str) -> Dict[str, Any]:
    """
    Cancel a running training task.
    
    Args:
        task_id: Celery task ID to cancel
        
    Returns:
        Dictionary with cancellation result
    """
    try:
        # Revoke the task
        celery_app.control.revoke(task_id, terminate=True, signal='SIGKILL')
        
        return {
            'success': True,
            'message': f'Task {task_id} cancelled successfully',
            'task_id': task_id
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Failed to cancel task {task_id}: {str(e)}',
            'task_id': task_id
        }


@celery_app.task
def test_task(message: str = "Hello from Celery!") -> Dict[str, Any]:
    """
    Simple test task to verify Celery is working.
    
    Args:
        message: Test message
        
    Returns:
        Test result dictionary
    """
    import time
    import os
    
    # Simulate some work
    time.sleep(2)
    
    return {
        'success': True,
        'message': message,
        'worker_pid': os.getpid(),
        'task_id': current_task.request.id
    }