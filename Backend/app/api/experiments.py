"""Experiment API endpoints."""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.db.session import get_db
from app.models import User
from app.schemas import (
    ExperimentCreate,
    ExperimentUpdate,
    ExperimentResponse,
    ExperimentListResponse
)
from app.services import experiment_service
from app.services import eda_service


router = APIRouter()


@router.post("/", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    experiment_data: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new experiment."""
    try:
        dataset_id = UUID(experiment_data.datasetId)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID format"
        )
    
    experiment_dict = await experiment_service.create_experiment(
        db=db,
        name=experiment_data.name,
        dataset_id=dataset_id,
        user_id=current_user.id
    )
    
    return ExperimentResponse(**experiment_dict)


@router.get("/", response_model=ExperimentListResponse)
async def list_experiments(
    page: int = Query(1, ge=1, description="Page number starting from 1"),
    page_size: int = Query(20, ge=1, le=100, description="Number of items per page"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all experiments owned by the current user with pagination."""
    skip = (page - 1) * page_size
    
    experiments, total = await experiment_service.get_user_experiments(
        db=db,
        user_id=current_user.id,
        skip=skip,
        limit=page_size
    )
    
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0
    
    return ExperimentListResponse(
        experiments=[
            ExperimentResponse(**exp) for exp in experiments
        ],
        total=total,
        page=page,
        pageSize=page_size,
        totalPages=total_pages
    )


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed information about a specific experiment."""
    experiment = await experiment_service.get_experiment_by_id(
        db=db,
        experiment_id=experiment_id,
        user_id=current_user.id
    )
    
    return ExperimentResponse(**experiment)


@router.put("/{experiment_id}", response_model=ExperimentResponse)
async def update_experiment(
    experiment_id: UUID,
    experiment_data: ExperimentUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update an experiment's metadata, config, status, or results."""
    updates = {}
    
    if experiment_data.name is not None:
        updates['name'] = experiment_data.name
    
    if experiment_data.config is not None:
        config_dict = experiment_data.config.model_dump(exclude_none=True)
        
        # Validate config if it has required fields
        if experiment_data.config.taskType or experiment_data.config.targetColumn or experiment_data.config.selectedFeatures:
            # Get experiment to find dataset_id
            experiment = await experiment_service.get_experiment_by_id(
                db=db,
                experiment_id=experiment_id,
                user_id=current_user.id
            )
            
            try:
                # Get EDA data for validation
                eda_data = await eda_service.get_eda_summary(
                    dataset_id=UUID(experiment['datasetId']),
                    user_id=current_user.id,
                    db=db
                )
                
                # Extract column names
                dataset_columns = [col['name'] for col in eda_data['columns']]
                id_columns = eda_data['id_columns']
                
                # Validate config
                validation_errors = experiment_service.validate_model_config(
                    experiment_data.config,
                    dataset_columns,
                    id_columns
                )
                
                if validation_errors:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={"errors": validation_errors}
                    )
                
                # Resolve hyperparameters
                resolved_config = experiment_service.resolve_model_hyperparameters(
                    experiment_data.config
                )
                config_dict = resolved_config.model_dump(exclude_none=True)
                
            except HTTPException:
                raise
            except Exception as e:
                # If EDA fails, just update without validation (dataset might not be ready)
                pass
        
        updates['config'] = config_dict
    
    if experiment_data.status is not None:
        updates['status'] = experiment_data.status
    
    if experiment_data.results is not None:
        updates['results'] = experiment_data.results
    
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update"
        )
    
    experiment = await experiment_service.update_experiment(
        db=db,
        experiment_id=experiment_id,
        user_id=current_user.id,
        updates=updates
    )
    
    return ExperimentResponse(**experiment)


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_experiment(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete an experiment permanently."""
    await experiment_service.delete_experiment(
        db=db,
        experiment_id=experiment_id,
        user_id=current_user.id
    )
    return None


@router.post("/{experiment_id}/duplicate", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def duplicate_experiment(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Duplicate an experiment with a new name."""
    experiment = await experiment_service.duplicate_experiment(
        db=db,
        experiment_id=experiment_id,
        user_id=current_user.id
    )
    
    return ExperimentResponse(**experiment)
