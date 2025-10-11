"""API routes package.

Combines all API routers into a single main router.
This router is then included in the FastAPI app with the /api/v1 prefix.
"""
from fastapi import APIRouter
from app.api import auth, datasets, experiments, eda, models, training, predictions

# Create main API router
api_router = APIRouter()

# Authentication routes
# POST /auth/register - Register new user
# POST /auth/login - Login and get JWT token (accepts form data)
# GET /auth/me - Get current user info (protected)
# POST /auth/logout - Logout (optional, mostly client-side)
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"]
)

# Dataset routes
# POST /datasets - Upload new dataset
# GET /datasets - List user's datasets
# GET /datasets/{id} - Get dataset details
# GET /datasets/{id}/preview - Preview dataset contents
# PUT /datasets/{id} - Update dataset metadata
# DELETE /datasets/{id} - Delete dataset
api_router.include_router(
    datasets.router,
    prefix="/datasets",
    tags=["Datasets"]
)

# Experiment routes
# POST /experiments - Create new experiment
# GET /experiments - List user's experiments
# GET /experiments/{id} - Get experiment details
# PUT /experiments/{id} - Update experiment
# DELETE /experiments/{id} - Delete experiment
# POST /experiments/{id}/duplicate - Duplicate experiment
api_router.include_router(
    experiments.router,
    prefix="/experiments",
    tags=["Experiments"]
)

# EDA routes
# GET /datasets/{id}/eda - Get comprehensive EDA summary
# POST /datasets/{id}/plot - Generate plot data dynamically
api_router.include_router(
    eda.router,
    tags=["EDA"]
)

# ML Models routes
# GET /models - Get all available ML models grouped by task type
# GET /models/{task_type} - Get models for specific task type
api_router.include_router(
    models.router,
    tags=["ML Models"]
)

# Training routes
# POST /training/experiments/{id}/train - Start training jobs
# GET /training/jobs/{id} - Get job status
# GET /training/experiments/{id}/jobs - List jobs for experiment
# GET /training/experiments/{id}/status - Get training status overview
# GET /training/results/{id} - Get training result by result ID
# GET /training/results/job/{id} - Get training result by job ID
api_router.include_router(
    training.router,
    prefix="/training",
    tags=["Training"]
)

# Prediction routes
# POST /predictions/experiments/{id}/predict/single - Make single prediction
# POST /predictions/experiments/{id}/predict/batch - Make batch predictions
# GET /predictions/download/{id} - Download batch prediction results
api_router.include_router(
    predictions.router,
    prefix="/predictions",
    tags=["Predictions"]
)
