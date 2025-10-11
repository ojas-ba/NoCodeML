"""In-memory model cache for fast model list retrieval."""
from typing import Dict, List, Any
from app.core.model_defaults import MODEL_METADATA, get_default_hyperparameters

# Global in-memory cache
_model_cache: Dict[str, Any] = {}


def initialize_model_cache() -> None:
    """Initialize the model cache from model_defaults.py."""
    global _model_cache
    
    classification_models = []
    regression_models = []
    
    for model_type, metadata in MODEL_METADATA.items():
        model_info = {
            "model_type": model_type,
            "display_name": metadata["display_name"],
            "description": metadata["description"],
            "pros": metadata.get("pros", []),
            "cons": metadata.get("cons", []),
            "supports_feature_importance": metadata.get("supports_feature_importance", False),
            "preprocessing_required": metadata.get("preprocessing_required", False)
        }
        
        if "classification" in metadata["task_types"]:
            classification_models.append(model_info)
        if "regression" in metadata["task_types"]:
            regression_models.append(model_info)
    
    _model_cache = {
        "classification": classification_models,
        "regression": regression_models,
        "all_models": {**{m["model_type"]: m for m in classification_models}, 
                      **{m["model_type"]: m for m in regression_models}}
    }


def get_cached_models() -> Dict[str, List[Dict[str, Any]]]:
    """Get the cached model list."""
    if not _model_cache:
        initialize_model_cache()
    return {
        "classification": _model_cache["classification"].copy(),
        "regression": _model_cache["regression"].copy()
    }


def get_cached_models_by_task(task_type: str) -> List[Dict[str, Any]]:
    """Get cached models for a specific task type."""
    if not _model_cache:
        initialize_model_cache()
    return _model_cache.get(task_type, [])


def get_model_info(model_type: str) -> Dict[str, Any]:
    """Get detailed model information from cache."""
    if not _model_cache:
        initialize_model_cache()
    
    return _model_cache.get("all_models", {}).get(model_type, {
        "model_type": model_type,
        "display_name": model_type,
        "description": "Unknown model",
        "pros": [],
        "cons": [],
        "supports_feature_importance": False,
        "preprocessing_required": False
    })


def get_model_instance_config(model_type: str, task_type: str, preset: str = "default") -> Dict[str, Any]:
    """Get complete model configuration for instantiation."""
    from app.core.model_defaults import get_preset_hyperparameters, merge_hyperparameters
    
    # Get default and preset hyperparameters
    defaults = get_default_hyperparameters(model_type, task_type)
    preset_params = get_preset_hyperparameters(model_type, preset)
    
    # Merge them
    final_params = merge_hyperparameters(defaults, preset_params)
    
    # Get model metadata
    model_info = get_model_info(model_type)
    
    return {
        "hyperparameters": final_params,
        "metadata": model_info,
        "model_class": model_type  # Will be used to import the actual class
    }
