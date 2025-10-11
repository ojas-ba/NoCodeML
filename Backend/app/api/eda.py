"""EDA API endpoints."""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db, get_current_user
from app.models import User
from app.schemas.eda import EDAResponse, PlotRequest, PlotResponse
from app.services import eda_service


router = APIRouter(tags=["eda"])


@router.get("/datasets/{dataset_id}/eda", response_model=EDAResponse)
async def get_eda_summary(
    dataset_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get comprehensive EDA summary for a dataset.
    
    Returns:
        - Dataset information (rows, columns, size)
        - Column information (types, missing values, unique counts)
        - Numeric and categorical column lists
        - Auto-detected ID columns
        - Statistical summary for numeric columns
        - Correlation matrix
        - Missing data summary
    """
    user_id = current_user.id
    
    try:
        summary = await eda_service.get_eda_summary(dataset_id, user_id, db)
        return summary
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate EDA summary: {str(e)}")


@router.post("/datasets/{dataset_id}/plot", response_model=PlotResponse)
async def generate_plot(
    dataset_id: UUID,
    request: PlotRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate plot data dynamically based on user selection.
    
    Supports:
        - Histogram: Single numeric column distribution
        - Scatter: Two numeric columns relationship
        - Box: Numeric column with optional grouping
        - Correlation: Heatmap of all numeric columns
        - Bar: Categorical column frequency
    
    Automatically samples large datasets (>10,000 rows) for performance.
    """
    user_id = current_user.id
    
    try:
        plot_data = await eda_service.generate_plot_data(
            dataset_id=dataset_id,
            plot_type=request.plot_type,
            x_column=request.x_column,
            y_column=request.y_column,
            group_by=request.group_by,
            user_id=user_id,
            db=db
        )
        return plot_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate plot: {str(e)}")
