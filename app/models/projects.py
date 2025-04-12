from pydantic import BaseModel, Field, StrictInt
from typing import Optional
class project(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    description: str = Field(..., min_length=1, max_length=500)
    user_id: StrictInt = Field(..., gt=0, lt=10000)

class Datasets(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    description: str = Field(..., min_length=1, max_length=500)
    project_id: StrictInt = Field(..., gt=0, lt=10000)
    user_id: StrictInt = Field(..., gt=0, lt=10000)
    file_id: StrictInt = Field(..., gt=0, lt=10000)

class File_Model(BaseModel):
    id: StrictInt = Field(..., gt=0, lt=10000)
    file_path: str = Field(..., min_length=1, max_length=255)
    user_id: StrictInt = Field(..., gt=0, lt=10000)
class PlotForm(BaseModel):
    plotType: str
    x: Optional[str] = None
    y: Optional[str] = None
    hue: Optional[str] = None
    aggregation: Optional[str] = None
    user_id: int = Field(..., gt=0, lt=10000)
    project_id: int = Field(..., gt=0, lt=10000)
    