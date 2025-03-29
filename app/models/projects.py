from pydantic import BaseModel, Field, StrictInt

class project(BaseModel):
    id: StrictInt = Field(..., gt=0, lt=10000)
    name: str = Field(..., min_length=1, max_length=50)
    description: str = Field(..., min_length=1, max_length=500)
    user_id: StrictInt = Field(..., gt=0, lt=10000)

class Datasets(BaseModel):
    id: StrictInt = Field(..., gt=0, lt=10000)
    name: str = Field(..., min_length=1, max_length=50)
    description: str = Field(..., min_length=1, max_length=500)
    project_id: StrictInt = Field(..., gt=0, lt=10000)
    user_id: StrictInt = Field(..., gt=0, lt=10000)