from pydantic import BaseModel
from typing import Optional


class TrainModelRequestFrontend(BaseModel):
    dataset_display_name: str          # Display name for the dataset
    gcs_csv_url: str                   # GCS URI to the CSV dataset
    type: str                          # Type of model to train (e.g., "classification", "regression")
    target_column: str                 # Name of the column you're predicting
    model_display_name: str            # Display name for the trained model
    model_type: str                    # Type of model to train (e.g., "LinearRegression", "XGBoost", "RandomForest",etc)
    model_parameters: Optional[str]    # Model parameters (e.g., "learning_rate", "max_depth", etc.)
