"""Experiment service layer for business logic."""
import uuid
from typing import List, Optional, Dict, Any, Tuple
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from sqlalchemy.orm.attributes import flag_modified

from app.models.experiment import Experiment, ExperimentStatus
from app.models.dataset import Dataset
from app.schemas.experiment import ExperimentConfig
from app.core.model_defaults import (
    get_default_hyperparameters,
    get_preset_hyperparameters,
    merge_hyperparameters,
    get_available_models
)


async def create_experiment(
    db: AsyncSession,
    name: str,
    dataset_id: uuid.UUID,
    user_id: int
) -> Dict[str, Any]:
    """
    Create a new experiment.
    
    Args:
        db: Database session
        name: Experiment name
        dataset_id: ID of the dataset to use
        user_id: ID of the user creating the experiment
    
    Returns:
        Created experiment data with dataset name
    
    Raises:
        HTTPException: 404 if dataset not found, 403 if no access, 409 if name exists
    """
    # Check if dataset exists and belongs to user
    dataset_query = select(Dataset).where(
        Dataset.id == dataset_id,
        Dataset.user_id == user_id
    )
    dataset_result = await db.execute(dataset_query)
    dataset = dataset_result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found or you don't have access to it"
        )
    
    # Check if experiment name is unique for this user
    name_query = select(Experiment).where(
        Experiment.user_id == user_id,
        Experiment.name == name
    )
    name_result = await db.execute(name_query)
    existing_experiment = name_result.scalar_one_or_none()
    
    if existing_experiment:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"An experiment named '{name}' already exists"
        )
    
    # Create new experiment with default values
    experiment = Experiment(
        id=uuid.uuid4(),
        user_id=user_id,
        dataset_id=dataset_id,
        name=name,
        status=ExperimentStatus.IN_PROGRESS,
        config={},
        results=None
    )
    
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)
    
    # Return dict with dataset name for consistency
    return {
        "id": str(experiment.id),
        "name": experiment.name,
        "datasetId": str(experiment.dataset_id),
        "datasetName": dataset.name,
        "status": experiment.status.value if hasattr(experiment.status, 'value') else experiment.status,
        "config": experiment.config or {},
        "results": experiment.results,
        "createdAt": experiment.created_at,
        "updatedAt": experiment.updated_at
    }


async def get_user_experiments(
    db: AsyncSession,
    user_id: int,
    skip: int = 0,
    limit: int = 20
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Get paginated list of user's experiments with dataset names.
    
    Args:
        db: Database session
        user_id: ID of the user
        skip: Number of records to skip
        limit: Maximum number of records to return
    
    Returns:
        Tuple of (list of experiments with dataset names, total count)
    """
    # Get total count
    count_query = select(func.count()).select_from(Experiment).where(Experiment.user_id == user_id)
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Get experiments with dataset names using join
    query = (
        select(Experiment, Dataset.name.label('dataset_name'))
        .join(Dataset, Experiment.dataset_id == Dataset.id)
        .where(Experiment.user_id == user_id)
        .order_by(Experiment.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(query)
    rows = result.all()
    
    # Build response with dataset names
    experiments = []
    for experiment, dataset_name in rows:
        exp_dict = {
            "id": str(experiment.id),
            "name": experiment.name,
            "datasetId": str(experiment.dataset_id),
            "datasetName": dataset_name,
            "status": experiment.status.value if hasattr(experiment.status, 'value') else experiment.status,
            "config": experiment.config or {},
            "results": experiment.results,
            "createdAt": experiment.created_at,
            "updatedAt": experiment.updated_at
        }
        experiments.append(exp_dict)
    
    return experiments, total


async def get_experiment_by_id(
    db: AsyncSession,
    experiment_id: uuid.UUID,
    user_id: int
) -> Dict[str, Any]:
    """
    Get single experiment with ownership check and dataset name.
    
    Args:
        db: Database session
        experiment_id: ID of the experiment
        user_id: ID of the user
    
    Returns:
        Experiment data with dataset name
    
    Raises:
        HTTPException: 404 if not found, 403 if no access
    """
    # Query experiment with dataset name
    query = (
        select(Experiment, Dataset.name.label('dataset_name'))
        .join(Dataset, Experiment.dataset_id == Dataset.id)
        .where(Experiment.id == experiment_id)
    )
    result = await db.execute(query)
    row = result.first()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found"
        )
    
    experiment, dataset_name = row
    
    # Check ownership
    if experiment.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this experiment"
        )
    
    return {
        "id": str(experiment.id),
        "name": experiment.name,
        "datasetId": str(experiment.dataset_id),
        "datasetName": dataset_name,
        "status": experiment.status.value if hasattr(experiment.status, 'value') else experiment.status,
        "config": experiment.config or {},
        "results": experiment.results,
        "createdAt": experiment.created_at,
        "updatedAt": experiment.updated_at
    }


async def update_experiment(
    db: AsyncSession,
    experiment_id: uuid.UUID,
    user_id: int,
    updates: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update experiment fields.
    
    Args:
        db: Database session
        experiment_id: ID of the experiment
        user_id: ID of the user
        updates: Dictionary of fields to update
    
    Returns:
        Updated experiment data with dataset name
    
    Raises:
        HTTPException: 404 if not found, 403 if no access, 409 if name conflict
    """
    # Fetch experiment and verify ownership
    query = select(Experiment).where(Experiment.id == experiment_id)
    result = await db.execute(query)
    experiment = result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found"
        )
    
    if experiment.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this experiment"
        )
    
    # If name is being updated, check uniqueness
    if 'name' in updates and updates['name'] != experiment.name:
        name_query = select(Experiment).where(
            Experiment.user_id == user_id,
            Experiment.name == updates['name'],
            Experiment.id != experiment_id
        )
        name_result = await db.execute(name_query)
        existing = name_result.scalar_one_or_none()
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"An experiment named '{updates['name']}' already exists"
            )
        
        experiment.name = updates['name']
    
    # Update config (REPLACE instead of merge to avoid JSONB tracking issues)
    if 'config' in updates:
        # Replace the entire config object
        experiment.config = updates['config']
        # Explicitly mark as modified for SQLAlchemy to detect JSONB changes
        flag_modified(experiment, 'config')
    
    # Update status if provided
    if 'status' in updates:
        experiment.status = updates['status']
    
    # Update results if provided
    if 'results' in updates:
        experiment.results = updates['results']
    
    await db.commit()
    await db.refresh(experiment)
    
    # Get dataset name for response
    dataset_query = select(Dataset.name).where(Dataset.id == experiment.dataset_id)
    dataset_result = await db.execute(dataset_query)
    dataset_name = dataset_result.scalar_one_or_none()
    
    return {
        "id": str(experiment.id),
        "name": experiment.name,
        "datasetId": str(experiment.dataset_id),
        "datasetName": dataset_name,
        "status": experiment.status.value if hasattr(experiment.status, 'value') else experiment.status,
        "config": experiment.config or {},
        "results": experiment.results,
        "createdAt": experiment.created_at,
        "updatedAt": experiment.updated_at
    }


async def delete_experiment(
    db: AsyncSession,
    experiment_id: uuid.UUID,
    user_id: int
) -> None:
    """
    Hard delete experiment and all associated files.
    
    Args:
        db: Database session
        experiment_id: ID of the experiment
        user_id: ID of the user
    
    Raises:
        HTTPException: 404 if not found, 403 if no access
    """
    from pathlib import Path
    import shutil
    from app.models.training import TrainingRun
    from app.models.prediction import PredictionBatch
    
    # Fetch experiment and verify ownership
    query = select(Experiment).where(Experiment.id == experiment_id)
    result = await db.execute(query)
    experiment = result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found"
        )
    
    if experiment.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this experiment"
        )
    
    # Get all training runs to delete model files
    training_runs_query = select(TrainingRun).where(TrainingRun.experiment_id == experiment_id)
    training_runs_result = await db.execute(training_runs_query)
    training_runs = training_runs_result.scalars().all()
    
    # Delete model files from training runs
    models_dir = Path("/app/models")
    for run in training_runs:
        if run.results and isinstance(run.results, dict):
            models = run.results.get('models', [])
            for model in models:
                model_path = model.get('model_path')
                if model_path:
                    try:
                        model_file = Path(model_path)
                        if model_file.exists():
                            model_file.unlink()
                            print(f"Deleted model file: {model_path}")
                    except Exception as e:
                        print(f"Failed to delete model file {model_path}: {e}")
    
    # Get all prediction batches to delete prediction files
    predictions_query = select(PredictionBatch).where(PredictionBatch.experiment_id == experiment_id)
    predictions_result = await db.execute(predictions_query)
    predictions = predictions_result.scalars().all()
    
    # Delete prediction files
    for pred in predictions:
        if pred.file_path:
            try:
                pred_file = Path(pred.file_path)
                if pred_file.exists():
                    pred_file.unlink()
                    print(f"Deleted prediction file: {pred.file_path}")
            except Exception as e:
                print(f"Failed to delete prediction file {pred.file_path}: {e}")
    
    # Permanently delete experiment (cascade will handle related records)
    await db.delete(experiment)
    await db.commit()
    
    print(f"Experiment {experiment_id} and all associated files deleted successfully")


async def duplicate_experiment(
    db: AsyncSession,
    experiment_id: uuid.UUID,
    user_id: int
) -> Dict[str, Any]:
    """
    Duplicate an experiment.
    
    Args:
        db: Database session
        experiment_id: ID of the experiment to duplicate
        user_id: ID of the user
    
    Returns:
        New experiment data with dataset name
    
    Raises:
        HTTPException: 404 if not found, 403 if no access
    """
    # Fetch source experiment and verify ownership
    query = (
        select(Experiment, Dataset.name.label('dataset_name'))
        .join(Dataset, Experiment.dataset_id == Dataset.id)
        .where(Experiment.id == experiment_id)
    )
    result = await db.execute(query)
    row = result.first()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found"
        )
    
    source_experiment, dataset_name = row
    
    if source_experiment.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this experiment"
        )
    
    # Generate unique name
    base_name = f"{source_experiment.name} (Copy)"
    new_name = base_name
    counter = 2
    
    while True:
        name_query = select(Experiment).where(
            Experiment.user_id == user_id,
            Experiment.name == new_name
        )
        name_result = await db.execute(name_query)
        existing = name_result.scalar_one_or_none()
        
        if not existing:
            break
        
        new_name = f"{source_experiment.name} (Copy {counter})"
        counter += 1
    
    # Create duplicate experiment
    new_experiment = Experiment(
        id=uuid.uuid4(),
        user_id=user_id,
        dataset_id=source_experiment.dataset_id,
        name=new_name,
        status=ExperimentStatus.IN_PROGRESS,
        config=source_experiment.config.copy() if source_experiment.config else {},
        results=None
    )
    
    db.add(new_experiment)
    await db.commit()
    await db.refresh(new_experiment)
    
    return {
        "id": str(new_experiment.id),
        "name": new_experiment.name,
        "datasetId": str(new_experiment.dataset_id),
        "datasetName": dataset_name,
        "status": new_experiment.status.value if hasattr(new_experiment.status, 'value') else new_experiment.status,
        "config": new_experiment.config or {},
        "results": new_experiment.results,
        "createdAt": new_experiment.created_at,
        "updatedAt": new_experiment.updated_at
    }


def validate_model_config(
    config: ExperimentConfig,
    dataset_columns: List[str],
    id_columns: List[str]
) -> List[str]:
    """
    Validate model configuration.
    
    Args:
        config: Experiment configuration to validate
        dataset_columns: List of column names in the dataset
        id_columns: List of detected ID columns
    
    Returns:
        List of error/warning messages (empty if valid)
    """
    errors = []
    
    # 1. Task type required
    if not config.taskType:
        errors.append("Task type is required (classification or regression)")
    
    # 2. Target column required
    if not config.targetColumn:
        errors.append("Target column is required")
    elif not config.targetColumn.strip():
        errors.append("Target column cannot be empty or whitespace")
    elif config.targetColumn not in dataset_columns:
        errors.append(f"Target column '{config.targetColumn}' not found in dataset")
    
    # 3. Features required
    if not config.selectedFeatures or len(config.selectedFeatures) == 0:
        errors.append("At least one feature must be selected")
    else:
        # Filter out empty feature names
        valid_features = [f for f in config.selectedFeatures if f and f.strip()]
        if len(valid_features) != len(config.selectedFeatures):
            errors.append("Feature names cannot be empty or whitespace")
        
        # Check features exist
        invalid_features = [f for f in valid_features if f not in dataset_columns]
        if invalid_features:
            errors.append(f"Invalid features not found in dataset: {', '.join(invalid_features)}")
        
        # Check target not in features
        if config.targetColumn and config.targetColumn in valid_features:
            errors.append("Target column cannot be a feature")
        
        # Warn about ID columns in features
        id_features = [f for f in valid_features if f in id_columns]
        if id_features:
            errors.append(f"Warning: ID columns detected in features: {', '.join(id_features)}. Consider excluding them.")
    
    # 4. Models required
    if config.models:
        if len(config.models) == 0:
            errors.append("At least one model must be selected")
        else:
            # Validate each model
            if config.taskType:
                available_models = get_available_models(config.taskType)
                for model in config.models:
                    if model.model_type not in available_models:
                        errors.append(
                            f"Model '{model.model_type}' not available for {config.taskType}"
                        )
    elif config.selectedModels:
        # Legacy field support
        if len(config.selectedModels) == 0:
            errors.append("At least one model must be selected")
    else:
        errors.append("At least one model must be selected")
    
    # 5. Train/test split validation
    if config.trainTestSplit:
        if not (0.1 <= config.trainTestSplit <= 0.9):
            errors.append("Train/test split must be between 0.1 and 0.9")
    
    return errors


def resolve_model_hyperparameters(config: ExperimentConfig) -> ExperimentConfig:
    """
    Resolve hyperparameters for models in config.
    Merges defaults -> presets -> custom overrides.
    
    Args:
        config: Experiment configuration
    
    Returns:
        Updated config with resolved hyperparameters
    """
    if not config.models or not config.taskType:
        return config
    
    for model_config in config.models:
        # Get defaults
        try:
            defaults = get_default_hyperparameters(
                model_config.model_type,
                config.taskType
            )
        except ValueError:
            # Model not found, skip
            defaults = {}
        
        # Get preset
        preset = get_preset_hyperparameters(
            model_config.model_type,
            model_config.preset
        )
        
        # Merge: custom > preset > defaults
        model_config.hyperparameters = merge_hyperparameters(
            defaults,
            preset,
            model_config.custom_hyperparameters
        )
    
    return config
