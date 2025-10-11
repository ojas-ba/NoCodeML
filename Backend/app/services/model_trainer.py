"""Model trainer for ML model training with AutoClean preprocessing."""
import os
import uuid
import time
import json
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

# Core ML libraries (always available)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Optional libraries with fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from autoclean import AutoClean
    AUTOCLEAN_AVAILABLE = True
except ImportError:
    AUTOCLEAN_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from app.core.model_defaults import get_preprocessing_config, get_training_config
from app.core.model_cache import get_model_info
from app.services.hyperparameter_optimizer import hyperparameter_optimizer


class ModelTrainer:
    """Handles ML model training with preprocessing and evaluation."""
    
    def __init__(self, models_dir: str = "/app/models"):
        """
        Initialize model trainer.
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.model_registry = {
            'classification': {
                'LogisticRegression': LogisticRegression,
                'RandomForestClassifier': RandomForestClassifier,
            },
            'regression': {
                'LinearRegression': LinearRegression,
                'RandomForestRegressor': RandomForestRegressor,
            }
        }
        
        # Add optional models if available
        if XGBOOST_AVAILABLE:
            self.model_registry['classification']['XGBClassifier'] = XGBClassifier
            self.model_registry['regression']['XGBRegressor'] = XGBRegressor
            
        if LIGHTGBM_AVAILABLE:
            self.model_registry['classification']['LGBMClassifier'] = LGBMClassifier
            self.model_registry['regression']['LGBMRegressor'] = LGBMRegressor
    
    def load_dataset(self, dataset_path: str) -> Optional[Any]:
        """
        Load dataset from file.
        
        Args:
            dataset_path: Path to dataset file
            
        Returns:
            Loaded dataset or None if pandas not available
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas is required for dataset loading")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        file_ext = Path(dataset_path).suffix.lower()
        
        if file_ext == '.csv':
            return pd.read_csv(dataset_path)
        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(dataset_path)
        elif file_ext == '.parquet':
            return pd.read_parquet(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def preprocess_data(self, df: Any, target_column: str, config: Dict[str, Any], task_type: str = 'classification') -> Tuple[Any, Any, Optional[Any]]:
        """
        Preprocess data using AutoClean or basic preprocessing.
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            config: Preprocessing configuration
            task_type: 'classification' or 'regression'
            
        Returns:
            Tuple of (features, target, label_encoder) - label_encoder is None for regression
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas is required for data preprocessing")
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        if AUTOCLEAN_AVAILABLE and config.get('use_autoclean', True):
            # Use AutoClean for preprocessing
            try:
                # Create a temporary dataframe with target for AutoClean
                temp_df = df.copy()
                
                pipeline = AutoClean(
                    temp_df,
                    target=target_column,
                    duplicates=config.get('duplicates', True),
                    missing_num=config.get('missing_num', 'auto'),
                    missing_categ=config.get('missing_categ', 'auto'),
                    encode_categ=config.get('encode_categ', ['onehot']),
                    outliers=config.get('outliers', 'auto'),
                    extract_datetime=config.get('extract_datetime', False)
                )
                
                cleaned_df = pipeline.output
                X_clean = cleaned_df.drop(columns=[target_column])
                y_clean = cleaned_df[target_column]
                
                # Encode target variable for classification if it's categorical
                label_encoder = None
                if task_type == 'classification' and y_clean.dtype == 'object':
                    label_encoder = LabelEncoder()
                    y_clean = pd.Series(label_encoder.fit_transform(y_clean), index=y_clean.index)
                
                return X_clean, y_clean, label_encoder
                
            except Exception as e:
                print(f"AutoClean failed, falling back to basic preprocessing: {e}")
        
        # Basic preprocessing fallback
        # Handle missing values
        X_numeric = X.select_dtypes(include=[np.number])
        X_categorical = X.select_dtypes(exclude=[np.number])
        
        # Fill numeric missing values with median
        if not X_numeric.empty:
            X_numeric = X_numeric.fillna(X_numeric.median())
        
        # Fill categorical missing values with mode
        if not X_categorical.empty:
            for col in X_categorical.columns:
                X_categorical[col] = X_categorical[col].fillna(X_categorical[col].mode()[0] if len(X_categorical[col].mode()) > 0 else 'unknown')
            
            # Simple label encoding for categorical variables
            le = LabelEncoder()
            for col in X_categorical.columns:
                X_categorical[col] = le.fit_transform(X_categorical[col].astype(str))
        
        # Combine back
        if not X_numeric.empty and not X_categorical.empty:
            X_processed = pd.concat([X_numeric, X_categorical], axis=1)
        elif not X_numeric.empty:
            X_processed = X_numeric
        else:
            X_processed = X_categorical
        
        # Handle target missing values
        y_processed = y.dropna()
        X_processed = X_processed.loc[y_processed.index]
        
        # Encode target variable for classification if it's categorical (string/object type)
        label_encoder = None
        if task_type == 'classification' and y_processed.dtype == 'object':
            label_encoder = LabelEncoder()
            y_processed = pd.Series(label_encoder.fit_transform(y_processed), index=y_processed.index)
        
        return X_processed, y_processed, label_encoder
    
    def split_and_scale_data(
        self, 
        X: Any, 
        y: Any, 
        config: Dict[str, Any], 
        task_type: str
    ) -> Tuple[Any, Any, Any, Any, Optional[Any]]:
        """
        Split data into train/test and apply scaling if needed.
        
        Args:
            X: Features
            y: Target
            config: Training configuration
            task_type: 'classification' or 'regression'
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, scaler)
        """
        test_size = config.get('test_size', 0.2)
        random_state = config.get('random_state', 42)
        stratify = y if (task_type == 'classification' and config.get('stratify', True)) else None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # Apply scaling if needed
        scaler = None
        if config.get('scaling', True):
            scaler = StandardScaler()
            # Fit scaler on training data only (no data leakage)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, scaler
    
    def train_model(
        self,
        model_type: str,
        task_type: str,
        X_train: Any,
        y_train: Any,
        hyperparameters: Dict[str, Any]
    ) -> Any:
        """
        Train a model with given hyperparameters.
        
        Args:
            model_type: Type of model to train
            task_type: 'classification' or 'regression'
            X_train: Training features
            y_train: Training target
            hyperparameters: Model hyperparameters
            
        Returns:
            Trained model
        """
        if task_type not in self.model_registry:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        if model_type not in self.model_registry[task_type]:
            raise ValueError(f"Model '{model_type}' not available for {task_type}")
        
        model_class = self.model_registry[task_type][model_type]
        model = model_class(**hyperparameters)
        
        # Train the model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        return model, training_time
    
    def evaluate_model(
        self,
        model: Any,
        X_train: Any,
        X_test: Any,
        y_train: Any,
        y_test: Any,
        task_type: str,
        cv_folds: int = 3,
        label_encoder: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate trained model and calculate metrics.
        
        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            task_type: 'classification' or 'regression'
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of evaluation metrics with separate train/test scores
        """
        # Make predictions on both train and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        metrics = {
            'train': {},
            'test': {},
            'confusion_matrix': None,
            'class_labels': None
        }
        
        if task_type == 'classification':
            # Training set metrics
            metrics['train']['accuracy'] = float(accuracy_score(y_train, y_train_pred))
            metrics['train']['precision'] = float(precision_score(y_train, y_train_pred, average='weighted', zero_division=0))
            metrics['train']['recall'] = float(recall_score(y_train, y_train_pred, average='weighted', zero_division=0))
            metrics['train']['f1_score'] = float(f1_score(y_train, y_train_pred, average='weighted', zero_division=0))
            
            # Test set metrics
            metrics['test']['accuracy'] = float(accuracy_score(y_test, y_test_pred))
            metrics['test']['precision'] = float(precision_score(y_test, y_test_pred, average='weighted', zero_division=0))
            metrics['test']['recall'] = float(recall_score(y_test, y_test_pred, average='weighted', zero_division=0))
            metrics['test']['f1_score'] = float(f1_score(y_test, y_test_pred, average='weighted', zero_division=0))
            
            # ROC AUC for binary classification
            try:
                if len(np.unique(y_test)) == 2:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    metrics['test']['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba))
            except:
                pass  # Skip if not supported
            
            # Confusion matrix with labels
            cm = confusion_matrix(y_test, y_test_pred)
            unique_labels = sorted(np.unique(y_test).tolist())
            
            # If we have a label encoder, use the original string labels
            if label_encoder is not None:
                original_labels = label_encoder.inverse_transform(unique_labels)
                label_strings = [str(label) for label in original_labels]
            else:
                label_strings = [str(label) for label in unique_labels]
            
            metrics['confusion_matrix'] = {
                'matrix': cm.tolist(),
                'labels': label_strings
            }
            
        else:  # regression
            # Training set metrics
            metrics['train']['mse'] = float(mean_squared_error(y_train, y_train_pred))
            metrics['train']['rmse'] = float(np.sqrt(metrics['train']['mse']))
            metrics['train']['mae'] = float(mean_absolute_error(y_train, y_train_pred))
            metrics['train']['r2_score'] = float(r2_score(y_train, y_train_pred))
            
            # Test set metrics
            metrics['test']['mse'] = float(mean_squared_error(y_test, y_test_pred))
            metrics['test']['rmse'] = float(np.sqrt(metrics['test']['mse']))
            metrics['test']['mae'] = float(mean_absolute_error(y_test, y_test_pred))
            metrics['test']['r2_score'] = float(r2_score(y_test, y_test_pred))
        
        # Cross-validation scores
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, n_jobs=-1)
            metrics['cv_scores'] = cv_scores.tolist()
            metrics['mean_cv_score'] = float(cv_scores.mean())
            metrics['std_cv_score'] = float(cv_scores.std())
        except Exception as e:
            print(f"Cross-validation failed: {e}")
        
        return metrics
    
    def get_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from model if supported.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importances or None
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
            else:
                return None
            
            # Create feature importance dictionary
            feature_importance = {
                name: float(importance) 
                for name, importance in zip(feature_names, importances)
            }
            
            # Sort by importance
            feature_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            return feature_importance
            
        except Exception as e:
            print(f"Failed to extract feature importance: {e}")
            return None
    
    def save_model(self, model: Any, model_id: str, scaler: Optional[Any] = None, 
                   label_encoder: Optional[Any] = None) -> str:
        """
        Save trained model, scaler, and label encoder to disk.
        
        Args:
            model: Trained model
            model_id: Unique identifier for the model
            scaler: Fitted scaler (optional)
            label_encoder: Fitted label encoder for classification (optional)
            
        Returns:
            Path to saved model file
        """
        model_path = self.models_dir / f"{model_id}.joblib"
        
        # Save model, scaler, and label_encoder together
        model_data = {
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'saved_at': time.time()
        }
        
        joblib.dump(model_data, model_path)
        return str(model_path)
    
    def load_model(self, model_path: str) -> Tuple[Any, Optional[Any]]:
        """
        Load saved model and scaler.
        
        Args:
            model_path: Path to saved model file
            
        Returns:
            Tuple of (model, scaler)
        """
        model_data = joblib.load(model_path)
        return model_data['model'], model_data.get('scaler')
    
    def train_complete_pipeline(
        self,
        dataset_path: str,
        target_column: str,
        model_type: str,
        task_type: str,
        hyperparameters: Dict[str, Any],
        preprocessing_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        selected_features: Optional[List[str]] = None,
        feature_types: Optional[Dict[str, str]] = None,
        job_id: Optional[str] = None,
        enable_optimization: bool = False
    ) -> Dict[str, Any]:
        """
        Complete training pipeline from data loading to model saving.
        
        Args:
            dataset_path: Path to dataset file
            target_column: Name of target column
            model_type: Type of model to train
            task_type: 'classification' or 'regression'
            hyperparameters: Model hyperparameters
            preprocessing_config: Preprocessing configuration
            training_config: Training configuration
            selected_features: List of feature column names to use (filters dataset)
            feature_types: Dictionary mapping feature names to types ('categorical' or 'numerical')
            job_id: Job ID for model saving
            
        Returns:
            Dictionary with training results
        """
        # Use default configs if not provided
        preprocessing_config = preprocessing_config or get_preprocessing_config()
        training_config = training_config or get_training_config()
        
        start_time = time.time()
        
        try:
            # Load dataset
            df = self.load_dataset(dataset_path)
            
            # Filter to selected features if specified
            if selected_features:
                # Include target column + selected features
                columns_to_keep = selected_features + [target_column]
                # Only keep columns that exist in the dataframe
                columns_to_keep = [col for col in columns_to_keep if col in df.columns]
                df = df[columns_to_keep]
            
            # Preprocess data
            X, y, label_encoder = self.preprocess_data(df, target_column, preprocessing_config, task_type)
            
            # Split and scale data
            X_train, X_test, y_train, y_test, scaler = self.split_and_scale_data(
                X, y, training_config, task_type
            )
            
            # Calculate class imbalance for classification (if optimization enabled)
            class_imbalance_ratio = None
            if task_type == 'classification' and enable_optimization:
                try:
                    from collections import Counter
                    class_counts = Counter(y_train)
                    if len(class_counts) == 2:
                        majority = max(class_counts.values())
                        minority = min(class_counts.values())
                        class_imbalance_ratio = majority / minority if minority > 0 else None
                except Exception:
                    pass
            
            # Get optimized hyperparameters if enabled
            tuning_metadata = None
            if enable_optimization:
                print(f"🔍 Optimizing hyperparameters for {model_type}...")
                optimization_result = hyperparameter_optimizer.optimize(
                    model_type=model_type,
                    task_type=task_type,
                    n_samples=len(X_train),
                    n_features=X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train.columns),
                    class_imbalance_ratio=class_imbalance_ratio,
                    user_params=hyperparameters
                )
                
                final_hyperparameters = optimization_result['params']
                tuning_metadata = optimization_result['metadata']
                
                print(f"✅ Optimization complete!")
                print(f"📊 Optimized parameters: {final_hyperparameters}")
            else:
                final_hyperparameters = hyperparameters
            
            # Train model
            model, training_time = self.train_model(
                model_type, task_type, X_train, y_train, final_hyperparameters
            )
            
            # Evaluate model
            metrics = self.evaluate_model(
                model, X_train, X_test, y_train, y_test, task_type,
                training_config.get('cv_folds', 3),
                label_encoder=label_encoder
            )
            
            # Get feature importance
            feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
            feature_importance_dict = self.get_feature_importance(model, feature_names)
            
            # Format feature importance for frontend visualization
            feature_importance = None
            if feature_importance_dict:
                # Convert dict to sorted lists for easy plotting
                sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
                feature_importance = {
                    'features': [f[0] for f in sorted_features],
                    'importance': [f[1] for f in sorted_features]
                }
            
            # Finalize tuning metadata with actual results (if optimization was enabled)
            if enable_optimization and tuning_metadata is None:
                # Generate metadata with actual test score
                test_score = metrics.get('test', {}).get('accuracy' if task_type == 'classification' else 'r2_score', 0.0)
                tuning_metadata = hyperparameter_optimizer.finalize_metadata(
                    optimized_params=final_hyperparameters,
                    test_score=test_score,
                    n_samples=len(X_train),
                    n_features=X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train.columns),
                    model_type=model_type
                )
            
            # Save model with scaler and label_encoder
            model_id = job_id or str(uuid.uuid4())
            model_path = self.save_model(model, model_id, scaler, label_encoder)
            
            total_time = time.time() - start_time
            
            return {
                'success': True,
                'model_path': model_path,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'confusion_matrix': metrics.get('confusion_matrix'),  # Already formatted in evaluate_model
                'training_time_seconds': training_time,
                'total_time_seconds': total_time,
                'dataset_info': {
                    'total_samples': len(X),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'n_features': X.shape[1],
                    'feature_names': feature_names
                },
                'preprocessing_config': preprocessing_config,
                'training_config': training_config,
                'hyperparameters': final_hyperparameters,
                'hyperparameter_tuning': tuning_metadata
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'training_time_seconds': time.time() - start_time
            }