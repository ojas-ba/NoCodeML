from fastapi import Depends
from ..worker import celery_app
from ..train.helper_functions import load_dataset_from_gcp
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from ..db.init_db import get_sync_db_connection
import json
from psycopg.types.json import Json
import os
import numpy as np

def sanitize_model(model):
    # 1. force any numeric arrays to float64 (as before)
    if hasattr(model, 'cluster_centers_'):
        model.cluster_centers_ = model.cluster_centers_.astype('float64')
    for attr in ('components_', 'reachability_'):
        if hasattr(model, attr):
            arr = getattr(model, attr)
            if isinstance(arr, np.ndarray) and arr.dtype != np.float64:
                setattr(model, attr, arr.astype('float64'))

    # 2. force labels_ to ints
    if hasattr(model, 'labels_'):
        # round in case it's e.g. array([0.,1.,2.])
        labels = np.round(model.labels_).astype(int)
        model.labels_ = labels

    return model


# This function is used to create a plot and convert it to base64 for rendering in the frontend.
# Each plot type is handled by a specific function from the pycaret library like its diff for classification, regression, and clustering.
# The function takes the model, plot type, and the specific plot function to be used which is passed as an argument.
# It creates the plot, saves it to a buffer, and then converts it to base64 for easy rendering in the frontend.
def get_base64_plot(model, plot_type,plot_function):
    # Create plot silently
    img_path = plot_function(model, plot=plot_type, save=True)  
    # img_path is usually something like 'Plots/Confusion Matrix.png'

    # read it back in and encode
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # clean up
    try:
        os.remove(img_path)
    except OSError:
        pass

    return img_b64

def save_result_to_db(req, metrics, plots):
    conn = get_sync_db_connection()
    cur = conn.cursor()
    try:
        # DELETE existing results
        cur.execute(
            "DELETE FROM ML_Results WHERE project_id=%s AND user_id=%s;",
            (req["project_id"], req["user_id"])
        )

        # Wrap metrics in psycopg3’s Json adapter
        metrics_adapter = Json(metrics)

        insert_query = """
            INSERT INTO ML_Results (
                project_id, user_id, job_type, model_type,
                target_, features, metrics,
                plot_1, plot_2, plot_3, plot_4
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING id;
        """
        cur.execute(insert_query, (
            req["project_id"],
            req["user_id"],
            req["type"],
            req["model"],
            req["target"],
            req.get("features"),
            metrics_adapter,      # ← use the Json wrapper here
            plots[0], plots[1], plots[2], plots[3]
        ))

        result_id = cur.fetchone()[0]
        conn.commit()
        return True

    except Exception:
        conn.rollback()
        raise

    finally:
        cur.close()
        conn.close()
    

@celery_app.task(bind=True)
def train_classifier(self,file_url, req):
    from pycaret.classification import setup, create_model, tune_model, predict_model, pull, plot_model

    try:
        
        df = load_dataset_from_gcp(file_url)
        if isinstance(df, dict) and "message" in df:
            return {"message": df["message"]}
        
        df = df.dropna(subset=[req["target"]])
        
        
        setup(data = df, target= req["target"], session_id=123, html=False)
        

        model = create_model(req["model"], fold=3, round=3, verbose=False)
        
        #model = tune_model(model,optimize="AUC",fold=3, n_iter=2, round=3, search_library="scikit-learn", search_algorithm="random", verbose=False)
        predict_model(model, verbose=False)
        metrics = pull()
        metrics = metrics.reset_index().to_dict(orient='records')
        
        confusion_matrix_plot = get_base64_plot(model, 'confusion_matrix',plot_model)
        feature_importance_plot = get_base64_plot(model, 'feature',plot_model)
        roc_curve_plot = get_base64_plot(model, 'auc',plot_model)
        learning_curve_plot = get_base64_plot(model, 'learning',plot_model)
        
        saved = save_result_to_db(req, metrics, [confusion_matrix_plot, feature_importance_plot, roc_curve_plot, learning_curve_plot])
        if not saved:
            raise Exception("Failed to save results to database")
        
        return {"message": "Classification model trained successfully","status": "success","project_id": req["project_id"],"user_id": req["user_id"]}
    
    except Exception as e:
        raise Exception(f"Error in training classifier: {str(e)}")

@celery_app.task(bind=True)
def train_regressor(self,file_url, req):
    from pycaret.regression import setup, create_model, tune_model, predict_model, pull, plot_model

    try:
        df = load_dataset_from_gcp(file_url)
        if isinstance(df, dict) and "message" in df:
            return {"message": df["message"]}
        
        df = df.dropna(subset=[req["target"]])
       
        setup(data = df, target= req["target"], session_id=123, html=False)
        model = create_model(req["model"], fold=3, round=3, verbose=False)
        # model = tune_model(model,optimize="RMSE",fold=3, n_iter=30, round=3, search_library="optuna",search_algorithm="tpe" , verbose=False)
        predict_model(model, verbose=False)
        metrics = pull()
        metrics = metrics.reset_index().to_dict(orient='records')
        
        prediction_plot = get_base64_plot(model, 'pipeline',plot_model)
        feature_importance_plot = get_base64_plot(model, 'feature',plot_model)
        residuals_plot = get_base64_plot(model, 'residuals',plot_model)
        learning_curve_plot = get_base64_plot(model, 'learning',plot_model)
    
        saved = save_result_to_db(req, metrics, [prediction_plot, feature_importance_plot, residuals_plot, learning_curve_plot])
        if not saved:
            raise Exception("Failed to save results to database")
        
        return {"message": "Regression model trained successfully","status": "success"}
    
    except Exception as e:
        raise Exception(f"Error in training regressor: {str(e)}")

@celery_app.task(bind=True)
def train_clustering(self,file_url, req):
    from pycaret.clustering import setup, create_model, pull, plot_model
    try:
        df = load_dataset_from_gcp(file_url)
        if isinstance(df, dict) and "message" in df:
            return {"message": df["message"]}
        
        
        df = df[req["features"]]
        
        setup(data = df, session_id=123, html=False)
        
        model = create_model(req["model"], **({"num_clusters": req["n_clusters"]} if req["model"] in ["kmeans", "ap", "meanshift", "sc", "hclust", "birch", "kmodes"] else {}), round=1)
        metrics = pull()
        metrics = metrics.reset_index().to_dict(orient='records')
        
        cluster_plot = get_base64_plot(model, 'cluster',plot_model)
        silhouette_plot = get_base64_plot(model, 'silhouette',plot_model)
        elbow_plot = get_base64_plot(model, 'elbow',plot_model)
        
        saved = save_result_to_db(req, metrics, [cluster_plot, silhouette_plot, elbow_plot," "])
        if not saved:
            raise Exception("Failed to save results to database")

        return {"message": "Clustering model trained successfully","status": "success"}
    
    except Exception as e:
        raise Exception(f"Error in training clustering model: {str(e)}")


