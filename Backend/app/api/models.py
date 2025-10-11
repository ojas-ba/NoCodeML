"""ML Models API endpoints."""
from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any
from app.core.model_cache import get_cached_models, get_model_info, get_model_instance_config
from app.core.model_defaults import (
    get_preprocessing_config, 
    get_training_config,
    get_available_models
)
from app.schemas.model_config import (
    ModelConfigRequest, 
    ModelConfigResponse, 
    ModelSelectionRequest,
    ModelSelectionResponse,
    PreprocessingConfig,
    TrainingConfig
)

router = APIRouter()


@router.get("/models")
async def get_available_models_api() -> Dict[str, List[Dict[str, Any]]]:
    """
    Get all available ML models grouped by task type.
    
    Returns:
        Dict containing 'classification' and 'regression' keys,
        each with a list of model objects containing:
        - model_type: The internal model identifier
        - display_name: Human-readable model name
        - description: Brief description of the model
        - supports_feature_importance: Whether model provides feature importance
        - preprocessing_required: Whether model needs feature scaling
    """
    return get_cached_models()


@router.get("/models/{task_type}")
async def get_models_by_task(task_type: str) -> List[Dict[str, Any]]:
    """
    Get available ML models for a specific task type.
    
    Args:
        task_type: Either 'classification' or 'regression'
        
    Returns:
        List of model objects for the specified task type
    """
    if task_type not in ["classification", "regression"]:
        raise HTTPException(status_code=400, detail="Task type must be 'classification' or 'regression'")
    
    models_cache = get_cached_models()
    return models_cache.get(task_type, [])


@router.get("/models/{task_type}/{model_type}/config")
async def get_model_config(
    task_type: str, 
    model_type: str, 
    preset: str = "default"
) -> ModelConfigResponse:
    """
    Get complete configuration for a specific model.
    
    Args:
        task_type: Either 'classification' or 'regression'
        model_type: Model identifier (e.g., 'LogisticRegression')
        preset: Hyperparameter preset ('fast' or 'default')
        
    Returns:
        Complete model configuration including hyperparameters, preprocessing, and training configs
    """
    if task_type not in ["classification", "regression"]:
        raise HTTPException(status_code=400, detail="Task type must be 'classification' or 'regression'")
    
    if preset not in ["fast", "default"]:
        raise HTTPException(status_code=400, detail="Preset must be 'fast' or 'default'")
    
    available_models = get_available_models(task_type)
    if model_type not in available_models:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_type}' not available for {task_type}"
        )
    
    # Get model configuration
    config = get_model_instance_config(model_type, task_type, preset)
    model_info = get_model_info(model_type)
    
    return ModelConfigResponse(
        model_type=model_type,
        display_name=model_info["display_name"],
        preset=preset,
        hyperparameters=config["hyperparameters"],
        preprocessing=PreprocessingConfig(**get_preprocessing_config()),
        training=TrainingConfig(**get_training_config()),
        supports_feature_importance=model_info["supports_feature_importance"],
        preprocessing_required=model_info["preprocessing_required"]
    )


@router.post("/models/configure")
async def configure_models(request: ModelSelectionRequest) -> ModelSelectionResponse:
    """
    Configure multiple models for training.
    
    Args:
        request: Model selection and configuration request
        
    Returns:
        Configured models with resolved hyperparameters and settings
    """
    if request.task_type not in ["classification", "regression"]:
        raise HTTPException(status_code=400, detail="Task type must be 'classification' or 'regression'")
    
    available_models = get_available_models(request.task_type)
    configured_models = []
    
    for model_config in request.models:
        if model_config.model_type not in available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_config.model_type}' not available for {request.task_type}"
            )
        
        # Get base configuration
        config = get_model_instance_config(
            model_config.model_type, 
            request.task_type, 
            model_config.preset
        )
        model_info = get_model_info(model_config.model_type)
        
        # Override with custom hyperparameters if provided
        final_hyperparameters = config["hyperparameters"].copy()
        if model_config.custom_hyperparameters:
            final_hyperparameters.update(model_config.custom_hyperparameters)
        
        # Use custom configs or defaults
        preprocessing_config = model_config.preprocessing or PreprocessingConfig(**get_preprocessing_config())
        training_config = model_config.training or TrainingConfig(**get_training_config())
        
        configured_models.append(ModelConfigResponse(
            model_type=model_config.model_type,
            display_name=model_info["display_name"],
            preset=model_config.preset,
            hyperparameters=final_hyperparameters,
            preprocessing=preprocessing_config,
            training=training_config,
            supports_feature_importance=model_info["supports_feature_importance"],
            preprocessing_required=model_info["preprocessing_required"]
        ))
    
    return ModelSelectionResponse(
        models=configured_models,
        task_type=request.task_type,
        total_models=len(configured_models)
    )
