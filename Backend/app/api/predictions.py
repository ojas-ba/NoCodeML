"""Prediction API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID
from pathlib import Path

from app.core.deps import get_db, get_current_user
from app.models import User
from app.models.experiment import Experiment
from app.services.prediction_service import PredictionService
from app.schemas.prediction import (
    SinglePredictionRequest,
    SinglePredictionResponse,
    BatchPredictionResponse
)

router = APIRouter()
prediction_service = PredictionService()


@router.post("/experiments/{experiment_id}/predict/single", response_model=SinglePredictionResponse)
async def predict_single(
    experiment_id: UUID,
    request: SinglePredictionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Make a single prediction using the best model from an experiment."""
    # Verify experiment ownership
    result = await db.execute(
        select(Experiment).filter(
            Experiment.id == experiment_id,
            Experiment.user_id == current_user.id
        )
    )
    experiment = result.scalar_one_or_none()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Make prediction
    prediction_result = await prediction_service.predict_single(
        experiment_id=experiment_id,
        features=request.features
    )
    
    return prediction_result


@router.post("/experiments/{experiment_id}/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    experiment_id: UUID,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Upload CSV for batch predictions."""
    # Verify experiment ownership
    result = await db.execute(
        select(Experiment).filter(
            Experiment.id == experiment_id,
            Experiment.user_id == current_user.id
        )
    )
    experiment = result.scalar_one_or_none()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    # Process batch predictions with user_id for ownership tracking
    prediction_result = await prediction_service.predict_batch(
        experiment_id=experiment_id,
        file=file,
        user_id=current_user.id
    )
    
    return prediction_result


@router.get("/experiments/{experiment_id}/history")
async def get_prediction_history(
    experiment_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get prediction history for an experiment."""
    from app.models.prediction import PredictionBatch
    from sqlalchemy import desc
    
    # Verify experiment ownership
    result = await db.execute(
        select(Experiment).filter(
            Experiment.id == experiment_id,
            Experiment.user_id == current_user.id
        )
    )
    experiment = result.scalar_one_or_none()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Get prediction history
    result = await db.execute(
        select(PredictionBatch).filter(
            PredictionBatch.experiment_id == experiment_id,
            PredictionBatch.user_id == current_user.id
        ).order_by(desc(PredictionBatch.created_at))
    )
    predictions = result.scalars().all()
    
    return {
        "predictions": [
            {
                "id": str(pred.id),
                "experiment_id": str(pred.experiment_id),
                "total_predictions": pred.total_predictions,
                "created_at": pred.created_at.isoformat(),
                "download_url": f"/api/v1/predictions/download/{pred.id}"
            }
            for pred in predictions
        ]
    }


@router.get("/download/{prediction_id}")
async def download_predictions(
    prediction_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Download batch prediction results with ownership validation."""
    from app.models.prediction import PredictionBatch
    
    # Validate prediction ID format
    try:
        pred_uuid = UUID(prediction_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid prediction ID format")
    
    # Check ownership in database
    result = await db.execute(
        select(PredictionBatch).filter(
            PredictionBatch.id == pred_uuid,
            PredictionBatch.user_id == current_user.id
        )
    )
    prediction_batch = result.scalar_one_or_none()
    
    if not prediction_batch:
        raise HTTPException(
            status_code=404, 
            detail="Prediction results not found or you don't have permission to access them"
        )
    
    # Get file path from database
    file_path = Path(prediction_batch.file_path)
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404, 
            detail="Prediction file has been deleted or moved"
        )
    
    return FileResponse(
        path=file_path,
        filename=f"predictions_{prediction_id}.csv",
        media_type="text/csv"
    )
