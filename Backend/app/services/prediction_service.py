"""Prediction service for loading models and making predictions."""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import HTTPException, UploadFile
import uuid
import io

from app.db.sync_session import SyncSessionLocal
from app.models.training import TrainingRun
from app.models.prediction import PredictionBatch


class PredictionService:
    """Service for making predictions with trained models."""
    
    def __init__(self):
        self.models_dir = Path("/app/models")
        self.predictions_dir = Path("/app/predictions")
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
    
    def _preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same preprocessing that was done during training.
        This includes encoding categorical variables with label encoding.
        
        Args:
            df: DataFrame with raw feature values
            
        Returns:
            DataFrame with preprocessed (encoded) features
        """
        from sklearn.preprocessing import LabelEncoder
        
        # Separate numeric and categorical
        X_numeric = df.select_dtypes(include=[np.number])
        X_categorical = df.select_dtypes(exclude=[np.number])
        
        # Fill numeric missing values with median
        if not X_numeric.empty:
            X_numeric = X_numeric.fillna(X_numeric.median())
        
        # Fill categorical missing values with mode and encode
        if not X_categorical.empty:
            for col in X_categorical.columns:
                # Fill missing values
                mode_val = X_categorical[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'unknown'
                X_categorical[col] = X_categorical[col].fillna(fill_val)
                
                # Label encode each categorical column
                le = LabelEncoder()
                X_categorical[col] = le.fit_transform(X_categorical[col].astype(str))
        
        # Combine back in original column order
        if not X_numeric.empty and not X_categorical.empty:
            # Preserve original column order
            result = pd.DataFrame(index=df.index)
            for col in df.columns:
                if col in X_numeric.columns:
                    result[col] = X_numeric[col]
                elif col in X_categorical.columns:
                    result[col] = X_categorical[col]
            return result
        elif not X_numeric.empty:
            return X_numeric
        else:
            return X_categorical
    
    async def predict_single(self, experiment_id: uuid.UUID, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a single prediction.
        
        Args:
            experiment_id: Experiment ID
            features: Dictionary of feature values
            
        Returns:
            Prediction result with confidence scores
        """
        # Load best model from experiment
        model_data = self._load_best_model(experiment_id)
        model = model_data['model']
        scaler = model_data.get('scaler')
        label_encoder = model_data.get('label_encoder')
        training_config = model_data.get('config', {})
        
        # Get expected feature columns
        if scaler and hasattr(scaler, 'feature_names_in_'):
            feature_columns = list(scaler.feature_names_in_)
        elif 'selectedFeatures' in training_config:
            feature_columns = training_config['selectedFeatures']
        else:
            # Use all provided features
            feature_columns = list(features.keys())
        
        # Prepare input data with correct feature order
        try:
            feature_values = [features[col] for col in feature_columns]
            df = pd.DataFrame([feature_values], columns=feature_columns)
        except KeyError as e:
            missing_cols = [col for col in feature_columns if col not in features]
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_cols}"
            )
        
        # Apply preprocessing (categorical encoding, missing value handling)
        df_preprocessed = self._preprocess_features(df)
        
        # Apply scaling if exists
        if scaler:
            # Use .values to avoid feature name warnings
            scaled_values = scaler.transform(df_preprocessed.values)
        else:
            scaled_values = df_preprocessed.values
        
        # Make prediction (using numpy array to match training)
        prediction = model.predict(scaled_values)[0]
        
        # Get prediction probabilities if available
        probabilities = None
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(scaled_values)[0]
            probabilities = {
                str(i): float(p) for i, p in enumerate(proba)
            }
            confidence = float(max(proba))
        
        # Decode prediction if label encoder exists
        if label_encoder:
            prediction = label_encoder.inverse_transform([prediction])[0]
        
        return {
            'prediction': str(prediction),
            'probabilities': probabilities,
            'confidence': confidence
        }
    
    async def predict_batch(self, experiment_id: uuid.UUID, file: UploadFile, user_id: int) -> Dict[str, Any]:
        """
        Make batch predictions from CSV file and track ownership.
        
        Args:
            experiment_id: Experiment ID
            file: CSV file with features
            user_id: User ID (integer) for ownership tracking
            
        Returns:
            Dictionary with download URL for predictions CSV
        """
        # Load best model and config
        model_data = self._load_best_model(experiment_id)
        model = model_data['model']
        scaler = model_data.get('scaler')
        label_encoder = model_data.get('label_encoder')
        training_config = model_data.get('config', {})
        
        # Read uploaded CSV
        content = await file.read()
        df_original = pd.read_csv(io.BytesIO(content))
        
        # Get feature columns from training config or scaler
        if scaler and hasattr(scaler, 'feature_names_in_'):
            feature_columns = list(scaler.feature_names_in_)
        elif 'selectedFeatures' in training_config:
            feature_columns = training_config['selectedFeatures']
        else:
            # Fallback: use all columns except common non-feature columns
            exclude_cols = ['customerID', 'id', 'ID', training_config.get('targetColumn', 'target')]
            feature_columns = [col for col in df_original.columns if col not in exclude_cols]
        
        # Extract only the feature columns needed for prediction
        try:
            df_features = df_original[feature_columns].copy()
        except KeyError as e:
            missing_cols = [col for col in feature_columns if col not in df_original.columns]
            raise HTTPException(
                status_code=400,
                detail=f"Missing required feature columns: {missing_cols}"
            )
        
        # Apply preprocessing (categorical encoding, missing value handling)
        df_preprocessed = self._preprocess_features(df_features)
        
        # Apply scaling if exists
        if scaler:
            # Use .values to avoid feature name warnings (model was trained without feature names)
            scaled_values = scaler.transform(df_preprocessed.values)
            df_scaled = scaled_values
        else:
            df_scaled = df_preprocessed.values
        
        # Make predictions (using numpy arrays to match training)
        predictions = model.predict(df_scaled)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df_scaled)
            df_original['confidence'] = probabilities.max(axis=1)
        
        # Decode predictions if label encoder exists
        if label_encoder:
            predictions = label_encoder.inverse_transform(predictions)
        
        df_original['prediction'] = predictions
        
        # Save predictions to file (with all original columns + prediction + confidence)
        prediction_id = str(uuid.uuid4())
        output_path = self.predictions_dir / f"{prediction_id}.csv"
        df_original.to_csv(output_path, index=False)
        
        # Save to database for ownership tracking
        db = SyncSessionLocal()
        try:
            prediction_batch = PredictionBatch(
                id=uuid.UUID(prediction_id),
                user_id=user_id,
                experiment_id=experiment_id,
                file_path=str(output_path),
                total_predictions=len(df_original)
            )
            db.add(prediction_batch)
            db.commit()
        except Exception as e:
            db.rollback()
            # Log error but don't fail the prediction
            print(f"Warning: Failed to save prediction batch to database: {e}")
        finally:
            db.close()
        
        return {
            'prediction_id': prediction_id,
            'total_predictions': len(df_original),
            'download_url': f'/api/v1/predictions/download/{prediction_id}'
        }
    
    def _load_best_model(self, experiment_id: uuid.UUID) -> Dict[str, Any]:
        """
        Load the best trained model for an experiment by querying the database.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Dictionary with model, scaler, and label_encoder
        """
        db = SyncSessionLocal()
        try:
            # Query for completed training runs for this experiment
            from sqlalchemy import select, and_, desc
            
            # Get the most recent completed training run
            query = (
                select(TrainingRun)
                .filter(
                    and_(
                        TrainingRun.experiment_id == experiment_id,
                        TrainingRun.status == 'completed'
                    )
                )
                .order_by(desc(TrainingRun.created_at))
            )
            
            training_run = db.execute(query).scalars().first()
            
            if not training_run:
                raise HTTPException(
                    status_code=404,
                    detail="No trained models found for this experiment. Please train models first."
                )
            
            # Extract best model from run results
            results = training_run.results
            if not results or 'best_model' not in results:
                raise HTTPException(
                    status_code=500,
                    detail="Training run completed but no best model found in results."
                )
            
            best_model_info = results['best_model']
            
            # Find the full model data in the models array
            models = results.get('models', [])
            best_model_data = None
            for model in models:
                if model.get('model_type') == best_model_info.get('model_type'):
                    best_model_data = model
                    break
            
            if not best_model_data or 'model_path' not in best_model_data:
                raise HTTPException(
                    status_code=500,
                    detail="Best model path not found in training results."
                )
            
            # Load model from file
            model_path = Path(best_model_data['model_path'])
            
            if not model_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Model file not found at: {model_path}. The model may have been deleted."
                )
            
            # Load model data (includes model, scaler, label_encoder)
            model_data = joblib.load(model_path)
            
            # Include the training config in the returned data
            model_data['config'] = training_run.config_snapshot
            
            return model_data
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}"
            )
        finally:
            db.close()
