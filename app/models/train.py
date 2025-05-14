from pydantic import BaseModel
from typing import Literal, Optional


class TrainRequest(BaseModel):
    type: Literal['classification', 'regression', 'clustering']
    model: str
    target: Optional[str] = None  # Required only for classification/regression
    project_id: int
    user_id: int
    features: Optional[list[str]] = None  # Required only for clustering that requires features
    n_clusters: Optional[int] = None      # only for clustering models that require n_clusters
    n_neighbors: Optional[int] = None     # only for clustering models that require n_neighbors

    