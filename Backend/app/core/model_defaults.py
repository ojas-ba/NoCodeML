"""Default hyperparameters and presets for ML models."""
from typing import Dict, Any, Literal, List, Optional


# ===== FAST TRAINING CONFIGURATIONS =====
# Optimized for speed and quick results, not accuracy

DEFAULT_HYPERPARAMETERS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "classification": {
        "LogisticRegression": {
            "C": 1.0,
            "max_iter": 100,
            "solver": "lbfgs",
            "random_state": 42,
            "n_jobs": -1
        },
        "RandomForestClassifier": {
            "n_estimators": 50,
            "max_depth": 10,
            "min_samples_split": 2,
            "random_state": 42,
            "n_jobs": -1
        },
        "XGBClassifier": {
            "n_estimators": 50,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
            "verbosity": 0
        },
        "LGBMClassifier": {
            "n_estimators": 50,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }
    },
    "regression": {
        "LinearRegression": {
            "n_jobs": -1
        },
        "RandomForestRegressor": {
            "n_estimators": 50,
            "max_depth": 10,
            "min_samples_split": 2,
            "random_state": 42,
            "n_jobs": -1
        },
        "XGBRegressor": {
            "n_estimators": 50,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0
        },
        "LGBMRegressor": {
            "n_estimators": 50,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }
    }
}


# ===== AUTOCLEAN PREPROCESSING CONFIGURATION =====

PREPROCESSING_CONFIG: Dict[str, Any] = {
    "duplicates": True,
    "missing_num": "auto",
    "missing_categ": "auto",
    "encode_categ": ["onehot"],
    "outliers": "auto",
    "extract_datetime": False
}

# ===== TRAINING CONFIGURATION =====

TRAINING_CONFIG: Dict[str, Any] = {
    "test_size": 0.2,
    "random_state": 42,
    "stratify": True,  # Will be set based on task type
    "cv_folds": 3,  # For cross-validation
    "scaling": True,  # Whether to apply standard scaling
    "feature_selection": False  # Keep simple for now
}

# ===== HYPERPARAMETER PRESETS =====
# Simplified presets for fast training only

PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    # Classification presets - all optimized for speed
    "LogisticRegression": {
        "fast": {"C": 1.0, "max_iter": 50},
        "default": {"C": 1.0, "max_iter": 100}
    },
    "RandomForestClassifier": {
        "fast": {"n_estimators": 25, "max_depth": 5},
        "default": {"n_estimators": 50, "max_depth": 10}
    },
    "XGBClassifier": {
        "fast": {"n_estimators": 25, "max_depth": 3, "learning_rate": 0.2},
        "default": {"n_estimators": 50, "max_depth": 6, "learning_rate": 0.1}
    },
    "LGBMClassifier": {
        "fast": {"n_estimators": 25, "max_depth": 3, "learning_rate": 0.2},
        "default": {"n_estimators": 50, "max_depth": 6, "learning_rate": 0.1}
    },
    
    # Regression presets - all optimized for speed
    "LinearRegression": {
        "fast": {},
        "default": {}
    },
    "RandomForestRegressor": {
        "fast": {"n_estimators": 25, "max_depth": 5},
        "default": {"n_estimators": 50, "max_depth": 10}
    },
    "XGBRegressor": {
        "fast": {"n_estimators": 25, "max_depth": 3, "learning_rate": 0.2},
        "default": {"n_estimators": 50, "max_depth": 6, "learning_rate": 0.1}
    },
    "LGBMRegressor": {
        "fast": {"n_estimators": 25, "max_depth": 3, "learning_rate": 0.2},
        "default": {"n_estimators": 50, "max_depth": 6, "learning_rate": 0.1}
    }
}


# ===== MODEL METADATA =====

MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    # Classification models (4 total)
    "LogisticRegression": {
        "display_name": "Logistic Regression",
        "task_types": ["classification"],
        "description": "Linear model for binary and multi-class classification",
        "pros": ["Fast", "Interpretable", "Works well with linear data"],
        "cons": ["Limited to linear relationships"],
        "preprocessing_required": True,
        "supports_feature_importance": False
    },
    "RandomForestClassifier": {
        "display_name": "Random Forest",
        "task_types": ["classification"],
        "description": "Ensemble of decision trees with bagging",
        "pros": ["Handles non-linear data", "Feature importance", "Robust"],
        "cons": ["Can overfit", "Memory intensive"],
        "preprocessing_required": False,
        "supports_feature_importance": True
    },
    "XGBClassifier": {
        "display_name": "XGBoost",
        "task_types": ["classification"],
        "description": "Gradient boosting with regularization",
        "pros": ["High accuracy", "Handles missing data", "Feature importance"],
        "cons": ["Slower training", "Many hyperparameters"],
        "preprocessing_required": False,
        "supports_feature_importance": True
    },
    "LGBMClassifier": {
        "display_name": "LightGBM",
        "task_types": ["classification"],
        "description": "Fast gradient boosting framework",
        "pros": ["Very fast", "Memory efficient", "High accuracy"],
        "cons": ["Can overfit on small datasets"],
        "preprocessing_required": False,
        "supports_feature_importance": True
    },
    
    # Regression models (4 total)
    "LinearRegression": {
        "display_name": "Linear Regression",
        "task_types": ["regression"],
        "description": "Basic linear regression model",
        "pros": ["Fast", "Interpretable", "Simple"],
        "cons": ["Limited to linear relationships"],
        "preprocessing_required": True,
        "supports_feature_importance": False
    },
    "RandomForestRegressor": {
        "display_name": "Random Forest",
        "task_types": ["regression"],
        "description": "Ensemble of decision trees",
        "pros": ["Handles non-linear data", "Robust", "Feature importance"],
        "cons": ["Can overfit", "Memory intensive"],
        "preprocessing_required": False,
        "supports_feature_importance": True
    },
    "XGBRegressor": {
        "display_name": "XGBoost",
        "task_types": ["regression"],
        "description": "Gradient boosting for regression",
        "pros": ["High accuracy", "Handles missing data"],
        "cons": ["Slower training", "Many hyperparameters"],
        "preprocessing_required": False,
        "supports_feature_importance": True
    },
    "LGBMRegressor": {
        "display_name": "LightGBM",
        "task_types": ["regression"],
        "description": "Fast gradient boosting for regression",
        "pros": ["Very fast", "Memory efficient", "High accuracy"],
        "cons": ["Can overfit on small datasets"],
        "preprocessing_required": False,
        "supports_feature_importance": True
    }
}


# ===== HELPER FUNCTIONS =====

def get_default_hyperparameters(
    model_type: str,
    task_type: Literal["classification", "regression"]
) -> Dict[str, Any]:
    """Get default hyperparameters for a model."""
    if task_type not in DEFAULT_HYPERPARAMETERS:
        raise ValueError(f"Invalid task type: {task_type}")
    
    if model_type not in DEFAULT_HYPERPARAMETERS[task_type]:
        raise ValueError(f"Unknown model: {model_type} for {task_type}")
    
    return DEFAULT_HYPERPARAMETERS[task_type][model_type].copy()


def get_preset_hyperparameters(
    model_type: str,
    preset: Literal["fast", "default"] = "default"
) -> Dict[str, Any]:
    """Get preset hyperparameters for a model."""
    if model_type not in PRESETS:
        # Return empty dict if no preset defined
        return {}
    
    if preset not in PRESETS[model_type]:
        return {}
    
    return PRESETS[model_type][preset].copy()


def get_preprocessing_config() -> Dict[str, Any]:
    """Get AutoClean preprocessing configuration."""
    return PREPROCESSING_CONFIG.copy()


def get_training_config() -> Dict[str, Any]:
    """Get training configuration."""
    return TRAINING_CONFIG.copy()


def supports_feature_importance(model_type: str) -> bool:
    """Check if model supports feature importance."""
    metadata = MODEL_METADATA.get(model_type, {})
    return metadata.get("supports_feature_importance", False)


def requires_preprocessing(model_type: str) -> bool:
    """Check if model requires preprocessing (scaling)."""
    metadata = MODEL_METADATA.get(model_type, {})
    return metadata.get("preprocessing_required", False)


def merge_hyperparameters(
    defaults: Dict[str, Any],
    preset: Dict[str, Any],
    custom: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Merge hyperparameters with priority: custom > preset > defaults."""
    merged = defaults.copy()
    merged.update(preset)
    
    if custom:
        merged.update(custom)
    
    return merged


def get_available_models(
    task_type: Literal["classification", "regression"]
) -> List[str]:
    """Get list of available models for a task type."""
    if task_type not in DEFAULT_HYPERPARAMETERS:
        return []
    
    return list(DEFAULT_HYPERPARAMETERS[task_type].keys())


def get_model_info(model_type: str) -> Dict[str, Any]:
    """Get metadata for a model."""
    return MODEL_METADATA.get(model_type, {
        "display_name": model_type,
        "description": "No description available",
        "task_types": [],
        "pros": [],
        "cons": []
    })
