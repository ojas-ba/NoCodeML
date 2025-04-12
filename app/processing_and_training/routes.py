import asyncpg
from fastapi import APIRouter, Depends, Query, Request
from ..db.init_db import get_db_connection
from ..models.projects import PlotForm
from ..models.Training import TrainModelRequestFrontend    
from ..worker import plot
from .helper_functions import get_file_url, load_dataset_from_gcp, detect_column_types
import json
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # Add numpy import at the top level

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

@router.post("/preprocess", status_code=200)
async def preprocess_data(req:Request, db:asyncpg.Connection = Depends(get_db_connection)):
    try:
        data = await req.json()
        print(data)
        project_id = data.get("project_id")
        preprocess_config = data.get("preprocessing")
        preprocess_config = str(preprocess_config)# Convert single quotes to double quotes for JSON compatibility
        
        sql_query = f"""
        Insert Into PreprocessConfig (project_id, config) 
        Values ($1, $2)
        returning id
        """
        preprocess_id = await db.fetchval(sql_query, project_id, preprocess_config)
        if not preprocess_id:
            return {"message": "Failed to create preprocessing entry"}
        return {"message": "Preprocessing Config Saved", "preprocess_id": preprocess_id}
    except Exception as e:
        return {"message": str(e)}

@router.post("/train_model", status_code=200)
async def train_model(req: Request, db: asyncpg.Connection = Depends(get_db_connection)):
    try:
        import numpy as np  # Move numpy import inside function to ensure it's always available
        data = await req.json()
        project_id = data.get("project_id")
        model = data.get("model")
        target = data.get("target")
        preprocess_id = data.get("preprocess_id")
        hyperparameters = data.get("hyperparameters", {})
        prob_type = data.get("problem_type")  # Corrected key to match frontend payload
        user_id = data.get("user_id")
        
        file_url,dataset_name = await get_file_url(project_id, user_id, db) # Get the file URL from the database
    
        if isinstance(file_url, dict) and "message" in file_url: # Check if the response is a dictionary with a message key Meaning it failed to get the file URL
            return {"message": file_url["message"]}

        if not all([project_id, model, target, preprocess_id, prob_type]):
            return {"message": "Missing required parameters"}
        
        # Load and preprocess the dataset first - common for both types
        df = load_dataset_from_gcp(file_url)
        if isinstance(df, dict) and "message" in df:
            return {"message": df["message"]}
        df = df.dropna()
        
        if prob_type == "regression":
            # Load and preprocess the dataset
            df = pd.get_dummies(df)
            
            # New code: splitting dataset and training the model
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, mean_absolute_error

            X = df.drop(columns=[target])
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            hyperparams = hyperparameters or {}

            if model == "linear_regression":
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression(
                    fit_intercept = hyperparams.get('fit_intercept', True),
                    copy_X = hyperparams.get('copy_X', True),
                    n_jobs = hyperparams.get('n_jobs', None),
                    positive = hyperparams.get('positive', False)
                )
            elif model == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                reg = RandomForestRegressor(
                    n_estimators = hyperparams.get('n_estimators', 100),
                    max_depth = hyperparams.get('max_depth', None),
                    min_samples_split = hyperparams.get('min_samples_split', 2),
                    min_samples_leaf = hyperparams.get('min_samples_leaf', 1),
                    bootstrap = hyperparams.get('bootstrap', True),
                    oob_score = hyperparams.get('oob_score', False),
                    n_jobs = hyperparams.get('n_jobs', None),
                    random_state = hyperparams.get('random_state', 42),
                    verbose = hyperparams.get('verbose', 0),
                    max_samples = hyperparams.get('max_samples', None)
                )
            elif model == "xgb_regressor":
                from xgboost import XGBRegressor
                reg = XGBRegressor(
                    booster = hyperparams.get('booster', 'gbtree'),
                    n_estimators = hyperparams.get('n_estimators', 100),
                    learning_rate = hyperparams.get('learning_rate', 0.1),
                    max_depth = hyperparams.get('max_depth', 3),
                    min_child_weight = hyperparams.get('min_child_weight', 1),
                    subsample = hyperparams.get('subsample', 1),
                    colsample_bytree = hyperparams.get('colsample_bytree', 1),
                    colsample_bylevel = hyperparams.get('colsample_bylevel', 1),
                    colsample_bynode = hyperparams.get('colsample_bynode', 1),
                    gamma = hyperparams.get('gamma', 0),
                    reg_alpha = hyperparams.get('reg_alpha', 0),
                    reg_lambda = hyperparams.get('reg_lambda', 1),
                    objective = hyperparams.get('objective', 'reg:squarederror'),
                    eval_metric = hyperparams.get('eval_metric', 'rmse'),
                    random_state = hyperparams.get('random_state', 42),
                    n_jobs = hyperparams.get('n_jobs', 1)
                )
            elif model == "lightgbm":
                from lightgbm import LGBMRegressor
                reg = LGBMRegressor(
                    boosting_type = hyperparams.get('boosting_type', 'gbdt'),
                    objective = hyperparams.get('objective', 'regression'),
                    metric = hyperparams.get('metric', 'l2'),
                    n_estimators = hyperparams.get('n_estimators', 100),
                    learning_rate = hyperparams.get('learning_rate', 0.1),
                    num_leaves = hyperparams.get('num_leaves', 31),
                    max_depth = hyperparams.get('max_depth', -1),
                    min_child_samples = hyperparams.get('min_child_samples', 20),
                    subsample = hyperparams.get('subsample', 1.0),
                    colsample_bytree = hyperparams.get('colsample_bytree', 1.0),
                    reg_alpha = hyperparams.get('reg_alpha', 0.0),
                    reg_lambda = hyperparams.get('reg_lambda', 0.0),
                    random_state = hyperparams.get('random_state', 42),
                    n_jobs = hyperparams.get('n_jobs', -1),
                    verbosity = hyperparams.get('verbosity', 1)
                )
            else:
                return {"message": "Unsupported model type for regression"}

            reg.fit(X_train, y_train)
            preds = reg.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mse)

            return {"rmse": rmse, "mse": mse, "mae": mae, "message": "Model training completed"}
        elif prob_type == "classification":
            # Classification branch
            X = df.drop(columns=[target])
            y = df[target]
            
            # Encode target values before splitting
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            hyperparams = hyperparameters or {}
            if model == "logistic_regression":
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(
                    penalty = hyperparams.get('penalty', 'l2'),
                    C = hyperparams.get('C', 1.0),
                    solver = hyperparams.get('solver', 'lbfgs'),
                    multi_class = hyperparams.get('multi_class', 'auto'),
                    max_iter = hyperparams.get('max_iter', 100),
                    tol = hyperparams.get('tol', 1e-4),
                    fit_intercept = hyperparams.get('fit_intercept', True),
                    intercept_scaling = hyperparams.get('intercept_scaling', 1),
                    class_weight = hyperparams.get('class_weight', None),
                    random_state = hyperparams.get('random_state', None),
                    verbose = hyperparams.get('verbose', 0),
                    n_jobs = hyperparams.get('n_jobs', None),
                    l1_ratio = hyperparams.get('l1_ratio', None)
                )
            elif model == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier(
                    n_estimators = hyperparams.get('n_estimators', 100),
                    criterion = hyperparameters.get('criterion', 'gini'),
                    max_depth = hyperparameters.get('max_depth', None),
                    min_samples_split = hyperparameters.get('min_samples_split', 2),
                    min_samples_leaf = hyperparameters.get('min_samples_leaf', 1),
                    bootstrap = hyperparameters.get('bootstrap', True),
                    oob_score = hyperparameters.get('oob_score', False),
                    n_jobs = hyperparameters.get('n_jobs', None),
                    random_state = hyperparameters.get('random_state', None),
                    verbose = hyperparameters.get('verbose', 0),
                    class_weight = hyperparameters.get('class_weight', None),
                    max_samples = hyperparameters.get('max_samples', None)
                )
            elif model == "xgb_classifier":
                from xgboost import XGBClassifier
                clf = XGBClassifier(
                    booster = hyperparameters.get('booster', 'gbtree'),
                    n_estimators = hyperparameters.get('n_estimators', 100),
                    learning_rate = hyperparameters.get('learning_rate', 0.1),
                    max_depth = hyperparameters.get('max_depth', 3),
                    min_child_weight = hyperparameters.get('min_child_weight', 1),
                    subsample = hyperparameters.get('subsample', 1),
                    colsample_bytree = hyperparameters.get('colsample_bytree', 1),
                    colsample_bylevel = hyperparameters.get('colsample_bylevel', 1),
                    colsample_bynode = hyperparameters.get('colsample_bynode', 1),
                    gamma = hyperparameters.get('gamma', 0),
                    reg_alpha = hyperparameters.get('reg_alpha', 0),
                    reg_lambda = hyperparameters.get('reg_lambda', 1),
                    objective = hyperparameters.get('objective', 'binary:logistic'),
                    num_class = hyperparameters.get('num_class', None),
                    eval_metric = hyperparameters.get('eval_metric', 'logloss'),
                    random_state = hyperparameters.get('random_state', None),
                    n_jobs = hyperparameters.get('n_jobs', 1)
                )
            elif model == "lightgbm":
                from lightgbm import LGBMClassifier
                # Get number of unique classes using numpy
                n_classes = len(np.unique(y))
                # Set objective and metric based on number of classes
                objective = 'binary' if n_classes == 2 else 'multiclass'
                metric = 'binary_logloss' if n_classes == 2 else 'multi_logloss'
                
                clf = LGBMClassifier(
                    boosting_type = hyperparameters.get('boosting_type', 'gbdt'),
                    objective = objective,
                    num_class = None if n_classes == 2 else n_classes,
                    metric = metric,
                    n_estimators = hyperparameters.get('n_estimators', 100),
                    learning_rate = hyperparameters.get('learning_rate', 0.1),
                    num_leaves = hyperparameters.get('num_leaves', 31),
                    max_depth = hyperparameters.get('max_depth', -1),
                    min_child_samples = hyperparameters.get('min_child_samples', 20),
                    subsample = hyperparameters.get('subsample', 1.0),
                    colsample_bytree = hyperparameters.get('colsample_bytree', 1.0),
                    reg_alpha = hyperparameters.get('reg_alpha', 0.0),
                    reg_lambda = hyperparameters.get('reg_lambda', 0.0),
                    class_weight = hyperparameters.get('class_weight', None),
                    is_unbalance = hyperparameters.get('is_unbalance', False),
                    random_state = hyperparameters.get('random_state', None),
                    n_jobs = hyperparameters.get('n_jobs', -1),
                    verbosity = hyperparameters.get('verbosity', 1)
                )
            else:
                return {"message": "Unsupported model type for classification"}
            
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            from sklearn.metrics import accuracy_score, recall_score, f1_score
            accuracy = accuracy_score(y_test, preds)
            recall = recall_score(y_test, preds, average='weighted')
            f1 = f1_score(y_test, preds, average='weighted')
            return {"accuracy": accuracy, "recall": recall, "f1": f1, "message": "Model training completed"}
        else:
            return {"message": f"Unsupported problem type: {prob_type}"}

    except Exception as e:
        return {"message": str(e)}