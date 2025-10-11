"""Prediction request/response schemas."""
from pydantic import BaseModel
from typing import Dict, Any, Optional


class SinglePredictionRequest(BaseModel):
    """Request for single prediction."""
    features: Dict[str, Any]


class SinglePredictionResponse(BaseModel):
    """Response for single prediction."""
    prediction: str
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    prediction_id: str
    total_predictions: int
    download_url: str
