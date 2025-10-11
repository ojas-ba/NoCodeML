"""Task management service for handling Celery training tasks."""
import uuid
from typing import Dict, Any, Optional, List
from celery.result import AsyncResult
from sqlalchemy.orm import Session

from app.worker.celery_app import celery_app
from app.worker.tasks import train_model_task, cancel_training_task, test_task
# Avoid circular import by importing the task by name instead
# from app.worker.training_tasks import train_models
from app.models.training import TrainingJob, TrainingResult, TrainingLog, TrainingJobStatus
from app.models.experiment import Experiment, TrainingStatus
from app.db.sync_session import get_sync_db
from app.core.model_cache import get_model_instance_config
from app.core.model_defaults import get_preprocessing_config, get_training_config


class TaskManager:
    """Manages Celery tasks for ML training."""
    
    def __init__(self):
        """Initialize task manager."""
        self.celery_app = celery_app
    
    def start_training_task(
        self,
        job_id: str,
        experiment_id: str,
        dataset_id: str,
        model_type: str,
        task_type: str
    ) -> str:
        """
        Start an asynchronous training task for a single model.
        
        Args:
            job_id: Training job ID
            experiment_id: Experiment ID
            dataset_id: Dataset ID
            model_type: Model type to train
            task_type: 'classification' or 'regression'
            
        Returns:
            Celery task ID
        """
        # Start async task - train ONE model per task
        # Use send_task to avoid circular import
        print(f"[TaskManager] Dispatching task for job {job_id}, model: {model_type}")
        task_result = self.celery_app.send_task(
            'app.worker.training_tasks.train_models',
            kwargs={
                'job_id': job_id,
                'experiment_id': experiment_id,
                'dataset_id': dataset_id,
                'model_types': [model_type],  # Single model as list
                'task_type': task_type
            }
        )
        
        print(f"[TaskManager] Task dispatched with ID: {task_result.id}")
        return task_result.id
    
    def start_training_run_task(
        self,
        run_id: str,
        experiment_id: str,
        dataset_id: str
    ) -> str:
        """
        Start an asynchronous training run task (trains all models in config).
        
        Args:
            run_id: Training run ID
            experiment_id: Experiment ID
            dataset_id: Dataset ID
            
        Returns:
            Celery task ID
        """
        print(f"[TaskManager] Dispatching training run task for run {run_id}")
        task_result = self.celery_app.send_task(
            'app.worker.training_tasks.train_config_run',
            kwargs={
                'run_id': run_id,
                'experiment_id': experiment_id,
                'dataset_id': dataset_id
            }
        )
        
        print(f"[TaskManager] Training run task dispatched with ID: {task_result.id}")
        return task_result.id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of a Celery task.
        
        Args:
            task_id: Celery task ID
            
        Returns:
            Task status information
        """
        try:
            result = AsyncResult(task_id, app=self.celery_app)
            
            status_info = {
                'task_id': task_id,
                'status': result.status,
                'ready': result.ready(),
                'successful': result.successful() if result.ready() else None,
                'failed': result.failed() if result.ready() else None,
            }
            
            if result.ready():
                # Task is complete
                if result.successful():
                    status_info['result'] = result.result
                else:
                    status_info['error'] = str(result.result) if result.result else 'Unknown error'
                    status_info['traceback'] = getattr(result.result, 'traceback', None)
            else:
                # Task is still running, check for progress
                if result.status == 'PROGRESS':
                    status_info['progress'] = result.info
                else:
                    status_info['info'] = result.info
            
            return status_info
            
        except Exception as e:
            return {
                'task_id': task_id,
                'status': 'ERROR',
                'error': f'Failed to get task status: {str(e)}',
                'ready': False
            }
    
    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """
        Cancel a running task.
        
        Args:
            task_id: Celery task ID to cancel
            
        Returns:
            Cancellation result
        """
        try:
            # Revoke the task
            self.celery_app.control.revoke(task_id, terminate=True)
            
            # Also run the cancel task for cleanup
            cancel_result = cancel_training_task.delay(task_id)
            
            return {
                'success': True,
                'message': f'Task {task_id} cancellation initiated',
                'task_id': task_id,
                'cancel_task_id': cancel_result.id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to cancel task: {str(e)}',
                'task_id': task_id
            }
    
    def test_celery_connection(self) -> Dict[str, Any]:
        """
        Test Celery connection with a simple task.
        
        Returns:
            Test result
        """
        try:
            # Send test task
            task_result = test_task.delay("Celery connection test")
            
            return {
                'success': True,
                'message': 'Test task sent successfully',
                'task_id': task_result.id,
                'status': task_result.status
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Celery connection test failed: {str(e)}'
            }
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """
        Get list of active tasks.
        
        Returns:
            List of active task information
        """
        try:
            # Get active tasks from Celery
            inspect = self.celery_app.control.inspect()
            active_tasks = inspect.active()
            
            if not active_tasks:
                return []
            
            # Flatten task information
            all_tasks = []
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    all_tasks.append({
                        'worker': worker,
                        'task_id': task['id'],
                        'name': task['name'],
                        'args': task.get('args', []),
                        'kwargs': task.get('kwargs', {}),
                        'time_start': task.get('time_start')
                    })
            
            return all_tasks
            
        except Exception as e:
            return [{
                'error': f'Failed to get active tasks: {str(e)}'
            }]


def update_training_job_status_sync(
    job_id: str,
    status: TrainingJobStatus,
    celery_task_id: Optional[str] = None,
    error_message: Optional[str] = None
) -> bool:
    """
    Update training job status synchronously (for Celery tasks).
    
    Args:
        job_id: Training job ID
        status: New status
        celery_task_id: Celery task ID
        error_message: Error message if failed
        
    Returns:
        True if updated successfully
    """
    try:
        db_gen = get_sync_db()
        db = next(db_gen)
        
        try:
            # Find the job
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if not job:
                print(f"Training job {job_id} not found")
                return False
            
            # Update status
            job.status = status
            
            if celery_task_id:
                job.celery_task_id = celery_task_id
            
            if error_message:
                job.error_message = error_message
            
            # Update timestamps
            from datetime import datetime
            if status == TrainingJobStatus.RUNNING:
                job.started_at = datetime.utcnow()
            elif status in [TrainingJobStatus.COMPLETED, TrainingJobStatus.FAILED, TrainingJobStatus.CANCELLED]:
                job.completed_at = datetime.utcnow()
            
            db.commit()
            return True
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"Failed to update job status: {e}")
        return False


def save_training_result_sync(
    job_id: str,
    result_data: Dict[str, Any]
) -> bool:
    """
    Save training results synchronously (for Celery tasks).
    
    Args:
        job_id: Training job ID
        result_data: Training result data
        
    Returns:
        True if saved successfully
    """
    try:
        db_gen = get_sync_db()
        db = next(db_gen)
        
        try:
            # Create training result
            training_result = TrainingResult(
                job_id=job_id,
                model_path=result_data['model_path'],
                metrics_json=result_data.get('metrics', {}),
                feature_importance_json=result_data.get('feature_importance'),
                confusion_matrix_json=result_data.get('metrics', {}).get('confusion_matrix'),
                training_time_seconds=result_data.get('training_time_seconds', 0),
                cross_val_scores=result_data.get('metrics', {}).get('cv_scores')
            )
            
            db.add(training_result)
            db.commit()
            return True
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"Failed to save training result: {e}")
        return False


def log_training_progress_sync(
    job_id: str,
    progress_percent: Optional[float] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None
) -> bool:
    """
    Log training progress synchronously (for Celery tasks).
    
    Args:
        job_id: Training job ID
        progress_percent: Progress percentage (0-100)
        epoch: Current epoch number
        metrics: Current metrics
        message: Progress message
        
    Returns:
        True if logged successfully
    """
    try:
        db_gen = get_sync_db()
        db = next(db_gen)
        
        try:
            # Create training log entry
            log_entry = TrainingLog(
                job_id=job_id,
                progress_percent=progress_percent,
                epoch=epoch,
                metrics_json=metrics or {},
                message=message
            )
            
            db.add(log_entry)
            db.commit()
            return True
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"Failed to log training progress: {e}")
        return False