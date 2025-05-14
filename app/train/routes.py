import asyncpg
from fastapi import APIRouter, Depends, Query, Request
from ..db.init_db import get_db_connection
from ..models.projects import PlotForm
from ..models.train import TrainRequest
   
from ..tasks.plot import plot # Celery task for plotting
from ..tasks.train import train_classifier, train_regressor, train_clustering # Celery tasks for training

from .helper_functions import get_file_url, load_dataset_from_gcp, detect_column_types
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # Add numpy import at the top

router = APIRouter()


@router.get("/get_dataset", status_code=200)
async def get_dataset(project_id: int=Query(), user_id: int=Query(), db:asyncpg.Connection = Depends(get_db_connection)):
    
    file_url,dataset_name = await get_file_url(project_id, user_id, db) # Get the file URL from the database
    
    if isinstance(file_url, dict) and "message" in file_url: # Check if the response is a dictionary with a message key Meaning it failed to get the file URL
        return {"message": file_url["message"]}
    
    df = load_dataset_from_gcp(file_url) # Load the dataset from GCP
    
    
    if isinstance(df, dict) and "message" in df: # Check if the response is a dictionary with a message key Meaning it failed to load the dataset
        return {"message": df["message"]}
    df = df.dropna()
    
    col = detect_column_types(df)
    
    df = df.head(20)
    
    return {"dataset": df.to_dict(orient="records"),"columns": df.columns.tolist(),"file_url": file_url,"dataset_name":dataset_name, "numerical" : col["numerical"], "categorical":col["categorical"] } # Return the dataset as a dictionary with records and columns


@router.post("/plot", status_code=200)
async def create_plot(plot_form: PlotForm, db:asyncpg.Connection = Depends(get_db_connection)):
    try:
        file_url, dataset_name = await get_file_url(plot_form.project_id, plot_form.user_id, db)  # Unpack the tuple
        if isinstance(file_url, dict) and "message" in file_url:
            return {"message": file_url["message"]}
        
        # Plot configuration
        plot_config = {
            "scatter": { "use_case": "Two continuous variables", "args": ["x", "y", "hue"], "hue": True, "agg_needed": False },
            "line": { "use_case": "Time series or trends", "args": ["x", "y", "hue"], "hue": True, "agg_needed": False },
            "bar": { "use_case": "Category vs Value", "args": ["x", "y", "hue"], "hue": True, "agg_needed": True, "allowed_aggs": ["sum", "mean", "count", "median"] },
            "box": { "use_case": "Distribution by category", "args": ["x", "y", "hue"], "hue": True, "agg_needed": False },
            "violin": { "use_case": "Smoothed distribution by category", "args": ["x", "y", "hue"], "hue": True, "agg_needed": False },
            "hist": { "use_case": "Distribution of a single variable", "args": ["x", "hue"], "hue": True, "agg_needed": False },
            "kde": { "use_case": "Density estimation of a variable", "args": ["x", "hue"], "hue": True, "agg_needed": False },
            "pie": { "use_case": "Proportion of categories", "args": ["x", "y"], "hue": False, "agg_needed": True, "allowed_aggs": ["sum", "count"] },
            "pairplot": { "use_case": "Pairwise variable relationships", "args": ["df"], "hue": True, "agg_needed": False },
            "swarm": { "use_case": "Categorical scatter distribution", "args": ["x", "y", "hue"], "hue": True, "agg_needed": False },
            "strip": { "use_case": "Jittered categorical scatter", "args": ["x", "y", "hue"], "hue": True, "agg_needed": False },
            "count": { "use_case": "Frequency of categorical values", "args": ["x", "hue"], "hue": True, "agg_needed": False }
        }
        
        plot_type = plot_form.plotType.lower()
        if plot_type not in plot_config:
            return {"message": "Invalid plot type"}
        
        config = plot_config[plot_type]
        if config["agg_needed"]:
            if not plot_form.aggregation:
                return {"message": "Aggregation method required for this plot type"}
            if "allowed_aggs" in config and plot_form.aggregation not in config["allowed_aggs"]:
                return {"message": "Invalid aggregation method"}
        
        plot_id = plot.delay(file_url,plot_form.model_dump(), plot_type)  # Serialize arguments for Celery
        if not plot_id:
            return {"message": "Failed to create plot"}
        
        return {"message": "Started Ploting ...", "task_id": plot_id.id}
    except Exception as e:
        return {"message": str(e)}

@router.get("/plot_status", status_code=200)
async def get_plot_status(plot_id: str):
    try:
        task = plot.AsyncResult(plot_id)
        if task.ready():
            plot_result = task.result
            image = plot_result.get("image")
            if not image:
                return {"message": "No image found", "status": "failed"}
            return {"message": "Plot created successfully", "status": "success", "image": image}
        elif task.failed():
            return {"message": "Plot creation failed", "status": "failed"}
        else:
            return {"message": "Plot is still being created", "status": "pending"}
    except Exception as e:
        return {"message": str(e)}

@router.post("/train", status_code=200)
async def train_model(req:TrainRequest ,db:asyncpg.Connection = Depends(get_db_connection)):
    try:
        file_url, dataset_name = await get_file_url(req.project_id, req.user_id, db)  # Unpack the tuple
        if isinstance(file_url, dict) and "message" in file_url:
            return {"message": file_url["message"]}
        
        if req.type == "classification":
            train_id = train_classifier.delay(file_url, req.model_dump())
        elif req.type == "regression":
            train_id = train_regressor.delay(file_url, req.model_dump())
        elif req.type == "clustering":
            train_id = train_clustering.delay(file_url, req.model_dump())
        else:
            return {"message": "Invalid training type"}
        if not train_id:
            return {"message": "Failed to start training"}
        return {"message": "Started training ...", "task_id": train_id.id}
    except Exception as e:
        return {"message": str(e)}
    
async def fetch_result_from_db(project_id: int, user_id: int, db: asyncpg.Connection = Depends(get_db_connection)):
    try:
        query = """
        SELECT job_type, model_type, target_, features, metrics,
               plot_1, plot_2, plot_3, plot_4
        FROM ML_Results
        WHERE project_id = $1 AND user_id = $2;
        """

        result = await db.fetchrow(query, project_id, user_id)

        if result:
            return {
                "job_type": result["job_type"],
                "model_type": result["model_type"],
                "target": result["target_"],
                "features": result["features"],
                "metrics": result["metrics"],         # Already a Python dict (JSONB)
                "plots": [
                    result["plot_1"],
                    result["plot_2"],
                    result["plot_3"],
                    result["plot_4"]
                ]
            }
        else:
            raise Exception("No results found for the given project_id and user_id")

    except Exception as e:
        return e

@router.get("/train_status", status_code=200)
async def get_train_status(train_id: str, task_type: str = Query(...), db:asyncpg.Connection = Depends(get_db_connection)):
    try:
        if task_type == "classification":
            task = train_classifier.AsyncResult(train_id)
        elif task_type == "regression":
            task = train_regressor.AsyncResult(train_id)
        elif task_type == "clustering":
            task = train_clustering.AsyncResult(train_id)
        else:
            return {"message": "Invalid training type"}
        
        if task.ready():
            train_result = task.result
            
            project_id = train_result.get("project_id")
            user_id = train_result.get("user_id")
            
            results = await fetch_result_from_db(project_id, user_id, db)
            if isinstance(results, dict) and "message" in results:
                return {"message": results["message"], "status": "failed"}
            
            return {"message": "Training completed successfully", "result": results}
        elif task.failed():
            return {"message": "Training failed", "status": "failed"}
        else:
            return {"message": "Training is still in progress", "status": "pending"}
    except Exception as e:
        return {"message": str(e)}

@router.get("/train", status_code=200)
async def get_train_result(project_id: int, user_id: int, db:asyncpg.Connection = Depends(get_db_connection)):
    try:
        results = await fetch_result_from_db(project_id, user_id, db)
        return {"message": "Results fetched successfully", "result": results}
    
    except Exception as e:
        return {"message": str(e)}