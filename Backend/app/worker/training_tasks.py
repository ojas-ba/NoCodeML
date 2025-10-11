"""
Celery training tasks for asynchronous ML model training.
"""

import traceback
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from app.worker.celery_app import celery_app
from app.db.sync_session import SyncSessionLocal
from app.models.training import TrainingJob, TrainingJobStatus, TrainingResult, TrainingLog
from app.models.dataset import Dataset
from app.models.experiment import Experiment
from app.core.model_defaults import DEFAULT_HYPERPARAMETERS, PREPROCESSING_CONFIG
from app.services.model_trainer import ModelTrainer

from sqlalchemy import select, and_
from sqlalchemy.orm import Session


def get_sync_session() -> Session:
    """Get synchronous database session for Celery tasks"""
    return SyncSessionLocal()


def log_training_progress(db: Session, job_id: uuid.UUID, message: str, level: str = "INFO"):
    """Log training progress to database"""
    try:
        log_entry = TrainingLog(
            job_id=job_id,
            message=f"[{level}] {message}",
            metrics_json={}
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        print(f"Failed to log training progress: {e}")


def update_job_status(db: Session, job_id: uuid.UUID, status: TrainingJobStatus, error_message: Optional[str] = None):
    """Update training job status"""
    try:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job:
            job.status = status
            if error_message:
                job.error_message = error_message
            db.commit()
    except Exception as e:
        print(f"Failed to update job status: {e}")


@celery_app.task(bind=True, name='app.worker.training_tasks.train_models')
def train_models(self, job_id: str, experiment_id: str, dataset_id: str, model_types: List[str], task_type: str):
    """
    Train multiple ML models asynchronously.
    
    Args:
        job_id: Training job ID
        experiment_id: Experiment ID
        dataset_id: Dataset ID  
        model_types: List of model types to train
        task_type: 'classification' or 'regression'
    
    Returns:
        Dict with training results
    """
    print(f"[Celery Worker] TASK RECEIVED: job_id={job_id}, models={model_types}")
    db = get_sync_session()
    job_uuid = uuid.UUID(job_id)
    
    try:
        # Update job status to running
        update_job_status(db, job_uuid, TrainingJobStatus.RUNNING)
        log_training_progress(db, job_uuid, f"Starting training for {len(model_types)} models")
        
        # Get job with saved config snapshot
        job = db.query(TrainingJob).filter(TrainingJob.id == job_uuid).first()
        if not job:
            raise ValueError(f"Training job {job_id} not found")
        
        # Get dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Use config from job snapshot (not from experiment table)
        config = job.config_json
        if not config:
            raise ValueError(f"Job config not found")
        
        log_training_progress(db, job_uuid, f"Using dataset: {dataset.name} with saved config")
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Get dataset file path
        dataset_path = dataset.storage_path
        if not dataset_path:
            raise ValueError(f"Dataset {dataset.name} has no storage path")
        
        # Get configuration from job config snapshot
        target_column = config.get('targetColumn')
        if not target_column:
            raise ValueError(f"Target column not specified in job configuration")
        
        # Get feature selection from config
        selected_features = config.get('selectedFeatures', [])
        feature_types = config.get('featureTypes', {})
        
        # Get training parameters from config
        train_test_split = config.get('trainTestSplit', 0.2)
        random_seed = config.get('randomSeed', 42)
        
        log_training_progress(db, job_uuid, f"Loading dataset from: {dataset_path}")
        log_training_progress(db, job_uuid, f"Target column: {target_column}, Task type: {task_type}")
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'current': 1, 'total': len(model_types) + 1, 'status': 'Starting training'}
        )
        
        results = []
        
        # Train each model using complete pipeline
        for i, model_type in enumerate(model_types, 1):
            try:
                # Update progress
                self.update_state(
                    state='PROGRESS',
                    meta={'current': i, 'total': len(model_types) + 1, 'status': f'Training {model_type}'}
                )
                
                log_training_progress(db, job_uuid, f"Training {model_type} ({i}/{len(model_types)})")
                
                # Get model-specific hyperparameters from config or defaults
                models_config = config.get('models', [])
                model_config_obj = next((m for m in models_config if m.get('modelType') == model_type), None)
                
                if model_config_obj and model_config_obj.get('customHyperparameters'):
                    model_hyperparameters = model_config_obj['customHyperparameters']
                else:
                    # Fall back to defaults
                    if task_type not in DEFAULT_HYPERPARAMETERS:
                        raise ValueError(f"Unsupported task type: {task_type}")
                    
                    if model_type not in DEFAULT_HYPERPARAMETERS[task_type]:
                        raise ValueError(f"Unsupported model type: {model_type}")
                    
                    model_hyperparameters = DEFAULT_HYPERPARAMETERS[task_type][model_type]
                
                # Build training config from job config
                training_config = {
                    'test_size': train_test_split,
                    'cv_folds': 3,
                    'random_state': random_seed
                }
                
                # Use the complete training pipeline
                training_result = trainer.train_complete_pipeline(
                    dataset_path=dataset_path,
                    target_column=target_column,
                    model_type=model_type,
                    task_type=task_type,
                    hyperparameters=model_hyperparameters,
                    preprocessing_config=PREPROCESSING_CONFIG,
                    training_config=training_config,
                    selected_features=selected_features,
                    feature_types=feature_types,
                    job_id=str(job_uuid)
                )
                
                if not training_result.get('success'):
                    raise ValueError(training_result.get('error', 'Training failed'))
                
                log_training_progress(db, job_uuid, f"Model {model_type} trained successfully")
                
                # Extract results
                model_path = training_result['model_path']
                metrics = training_result['metrics']
                
                # Create training result record
                result = TrainingResult(
                    job_id=job_uuid,
                    model_path=model_path,
                    metrics_json=metrics,
                    training_time_seconds=training_result.get('training_time_seconds', 0.0),
                    feature_importance_json=training_result.get('feature_importance'),
                    confusion_matrix_json=metrics.get('confusion_matrix'),
                    cross_val_scores=metrics.get('cv_scores')
                )
                
                db.add(result)
                db.commit()
                
                results.append({
                    'model_type': model_type,
                    'metrics': metrics,
                    'model_path': model_path
                })
                
                log_training_progress(db, job_uuid, f"Model {model_type} saved to {model_path}")
                
            except Exception as e:
                error_msg = f"Failed to train {model_type}: {str(e)}"
                log_training_progress(db, job_uuid, error_msg, "ERROR")
                print(f"Error training {model_type}: {e}")
                print(traceback.format_exc())
                # Continue with other models
                continue
        
        # Final progress update
        self.update_state(
            state='PROGRESS',
            meta={'current': len(model_types) + 1, 'total': len(model_types) + 1, 'status': 'Completing'}
        )
        
        # Update job status to completed
        update_job_status(db, job_uuid, TrainingJobStatus.COMPLETED)
        log_training_progress(db, job_uuid, f"Training completed successfully for {len(results)} models")
        
        return {
            'status': 'SUCCESS',
            'job_id': job_id,
            'models_trained': len(results),
            'results': results
        }
        
    except Exception as e:
        error_message = f"Training failed: {str(e)}"
        traceback_str = traceback.format_exc()
        
        print(f"Training task failed: {e}")
        print(f"Traceback: {traceback_str}")
        
        # Update job status to failed
        update_job_status(db, job_uuid, TrainingJobStatus.FAILED, error_message)
        log_training_progress(db, job_uuid, f"Training failed: {error_message}", "ERROR")
        
        # Raise the exception to mark Celery task as failed
        raise Exception(error_message)
        
    finally:
        db.close()


@celery_app.task(bind=True, name='app.worker.training_tasks.train_config_run')
def train_config_run(self, run_id: str, experiment_id: str, dataset_id: str):
    """
    Train all models in config as a single run.
    
    This task:
    1. Loads config snapshot from training_run
    2. Prepares data once
    3. Trains all models sequentially
    4. Aggregates results
    5. Updates training_run with final results
    
    Args:
        run_id: Training run ID
        experiment_id: Experiment ID
        dataset_id: Dataset ID
    
    Returns:
        Dict with aggregated training results
    """
    from app.models.training import TrainingRun
    
    print(f"[Celery Worker] TRAINING RUN TASK RECEIVED: run_id={run_id}")
    db = get_sync_session()
    run_uuid = uuid.UUID(run_id)
    training_run = None
    
    try:
        # Get training run
        training_run = db.query(TrainingRun).filter(TrainingRun.id == run_uuid).first()
        if not training_run:
            raise ValueError(f"Training run {run_id} not found")
        
        # Update status to running
        training_run.status = 'running'
        training_run.started_at = datetime.now(timezone.utc)
        db.commit()
        
        print(f"[Training Run] Started run #{training_run.run_number}")
        
        # Get dataset
        dataset = db.query(Dataset).filter(Dataset.id == uuid.UUID(dataset_id)).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        dataset_path = dataset.storage_path
        
        # Get config snapshot
        config = training_run.config_snapshot
        task_type = config.get('taskType', 'classification')
        target_column = config.get('targetColumn')
        models_config = config.get('models', [])
        selected_features = config.get('selectedFeatures', [])
        feature_types = config.get('featureTypes', {})
        enable_optimization = config.get('enableOptimization', False)
        
        if not target_column:
            raise ValueError("No target column in config")
        
        if not models_config:
            raise ValueError("No models in config")
        
        print(f"[Training Run] Training {len(models_config)} models: {[m['model_type'] for m in models_config]}")
        if enable_optimization:
            print(f"[Training Run] ✨ Hyperparameter optimization ENABLED")
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Progress tracking
        total_models = len(models_config)
        model_results = []
        dataset_info_for_run = None  # Will be populated from first successful model
        
        # Train each model
        for idx, model_cfg in enumerate(models_config, 1):
            model_type = model_cfg['model_type']
            display_name = model_cfg.get('display_name', model_type)
            
            # Update progress in database (not just Celery state)
            training_run.results = {
                'progress': {
                    'current': idx,
                    'total': total_models,
                    'current_model': display_name
                }
            }
            db.commit()
            
            # Also update Celery state for compatibility
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': idx,
                    'total': total_models,
                    'status': f'Training {display_name}',
                    'run_id': run_id,
                    'run_number': training_run.run_number
                }
            )
            
            print(f"[Training Run] [{idx}/{total_models}] Training {model_type}...")
            
            try:
                # Train model with complete pipeline
                result = trainer.train_complete_pipeline(
                    dataset_path=dataset_path,
                    target_column=target_column,
                    model_type=model_type,
                    task_type=task_type,
                    hyperparameters=model_cfg.get('config', {}).get('hyperparameters', {}),
                    selected_features=selected_features,
                    feature_types=feature_types,
                    job_id=f"{run_id}_{model_type}",
                    enable_optimization=enable_optimization
                )
                
                if result.get('success'):
                    # Store dataset_info from first successful model (all models use same split)
                    if not model_results and result.get('dataset_info'):
                        dataset_info_for_run = result.get('dataset_info')
                    
                    model_results.append({
                        'model_type': model_type,
                        'display_name': display_name,
                        'metrics': result.get('metrics', {}),
                        'feature_importance': result.get('feature_importance'),
                        'confusion_matrix': result.get('confusion_matrix'),
                        'model_path': result.get('model_path'),
                        'training_time': result.get('training_time', 0),
                        'hyperparameters': result.get('hyperparameters'),
                        'hyperparameter_tuning': result.get('hyperparameter_tuning')
                    })
                    print(f"[Training Run] ✓ {model_type} completed successfully")
                else:
                    model_results.append({
                        'model_type': model_type,
                        'display_name': display_name,
                        'error': result.get('error', 'Training failed')
                    })
                    print(f"[Training Run] ✗ {model_type} failed: {result.get('error')}")
                    
            except Exception as e:
                error_msg = str(e)
                print(f"[Training Run] ✗ Error training {model_type}: {error_msg}")
                traceback.print_exc()
                model_results.append({
                    'model_type': model_type,
                    'display_name': display_name,
                    'error': error_msg
                })
        
        # Determine best model with robust metric selection (using test metrics)
        successful_models = [m for m in model_results if 'metrics' in m and m['metrics'] and 'test' in m['metrics']]
        best_model = None
        best_metric = None
        
        if successful_models:
            if task_type == 'classification':
                # Try accuracy first, fall back to f1_score, then precision
                for metric_name in ['accuracy', 'f1_score', 'precision']:
                    models_with_metric = [m for m in successful_models if m['metrics']['test'].get(metric_name) is not None]
                    if models_with_metric:
                        best_model = max(models_with_metric, key=lambda m: m['metrics']['test'].get(metric_name, 0))
                        best_metric = metric_name
                        break
            else:  # regression
                # Try r2_score first, fall back to negative MAE (higher is better)
                models_with_r2 = [m for m in successful_models if m['metrics']['test'].get('r2_score') is not None]
                if models_with_r2:
                    best_model = max(models_with_r2, key=lambda m: m['metrics']['test'].get('r2_score', -float('inf')))
                    best_metric = 'r2_score'
                else:
                    # Use negative MAE (lower MAE = higher score)
                    models_with_mae = [m for m in successful_models if m['metrics']['test'].get('mae') is not None]
                    if models_with_mae:
                        best_model = min(models_with_mae, key=lambda m: m['metrics']['test'].get('mae', float('inf')))
                        best_metric = 'mae'
        
        # Compile results
        results = {
            'task_type': task_type,
            'dataset_info': dataset_info_for_run,  # Dataset split information
            'models': model_results,
            'best_model': {
                'model_type': best_model['model_type'],
                'display_name': best_model['display_name'],
                'metric': best_metric,
                'value': best_model['metrics']['test'][best_metric]  # Use test metric for best model
            } if best_model else None,
            'summary': {
                'total_models': total_models,
                'successful': len(successful_models),
                'failed': total_models - len(successful_models)
            }
        }
        
        # Update training_run with results
        training_run.status = 'completed'
        training_run.completed_at = datetime.now(timezone.utc)
        training_run.duration_seconds = int((training_run.completed_at - training_run.started_at).total_seconds())
        training_run.results = results
        training_run.artifacts = {
            'models': {m['model_type']: m['model_path'] for m in model_results if 'model_path' in m}
        }
        
        db.commit()
        
        print(f"[Training Run] ✓ Run #{training_run.run_number} completed: {len(successful_models)}/{total_models} models successful")
        
        return {
            'status': 'SUCCESS',
            'run_id': run_id,
            'run_number': training_run.run_number,
            'results': results
        }
        
    except Exception as e:
        # Log full traceback
        error_trace = traceback.format_exc()
        print(f"[Training Run] ✗ Run {run_id} failed:")
        print(error_trace)
        
        # Sanitize error message (limit to 500 chars, avoid sensitive data)
        error_message = str(e)[:500]
        
        # Update run as failed
        if training_run:
            try:
                training_run.status = 'failed'
                training_run.completed_at = datetime.now(timezone.utc)
                if training_run.started_at:
                    training_run.duration_seconds = int((training_run.completed_at - training_run.started_at).total_seconds())
                training_run.error_message = error_message
                db.commit()
            except Exception as commit_error:
                print(f"[Training Run] Failed to update run status: {commit_error}")
                db.rollback()
        
        raise
        
    finally:
        # Always close DB session
        try:
            db.close()
        except Exception as close_error:
            print(f"[Training Run] Error closing DB session: {close_error}")


@celery_app.task(name='app.worker.training_tasks.health_check')
def health_check():
    """Simple health check task for Celery workers"""
    return {
        'status': 'healthy',
        'timestamp': str(uuid.uuid4())
    }