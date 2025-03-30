from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from ..models.projects import project,Datasets
from ..db.init_db import get_db_connection
from ..models.user import User
import asyncpg
import os

# First we need to have a upload directory to save the files.
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Always make sure that the project belongs to the user who is creating it.
# Also make sure that the dataset belongs to the project and the user who is creating it.
# Because this can compromise the security of the system.

router = APIRouter()

@router.post("/create_project",status_code=201)
async def create_project(project:project, db:asyncpg.Connection = Depends(get_db_connection)):
    sql_query = """
    INSERT INTO Projects (name, description,user_id)
    VALUES($1,$2,$3)
    RETURNING id;
    """
    try:
        result = await db.fetchval(sql_query,project.name,project.description,project.user_id)
        if (result):
            return {"message": "success","project_id": result}
        else:
            return {"message": "Error while creating project"}
    except asyncpg.exceptions.UniqueViolationError:
        return {"message": "Project name already exists"}
    except Exception as e:
        return {"message": str(e)}



@router.post("/add_dataset_details",status_code=201)
async def add_dataset(dataset:Datasets, db:asyncpg.Connection = Depends(get_db_connection)):
    sql_query2 = """
    INSERT INTO Datasets (name, description,project_id,user_id,file_id)
    VALUES($1,$2,$3,$4,$5)
    RETURNING id;
    """
    try:
        result = await db.fetchval(sql_query2,dataset.name,dataset.description,dataset.project_id,dataset.user_id,dataset.file_id)
        if (result):
            return {"message": "Uploaded Successfully ","dataset_id": result}
        else:
            return {"message": "Error while adding dataset"}
    except asyncpg.exceptions.UniqueViolationError:
        return {"message": "Dataset name already exists"}
    except Exception as e:
        return {"message": str(e)}

@router.post("/upload_file",status_code=201)
async def upload_file(file: UploadFile = File(),user_id: int = Form(...), db:asyncpg.Connection = Depends(get_db_connection)):
    try:
        file_name = file.filename
        file_path = os.path.join(UPLOAD_DIR, file_name)
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())
        sql_query = """
        INSERT INTO Files (file_path, user_id)
        VALUES($1,$2)
        RETURNING id;
        """
        file_id = await db.fetchval(sql_query, file_name, user_id)
        if file_id:
            return {"message": "File uploaded successfully", "file_id": file_id}
        else:
            return {"message": "Error while uploading file"}
    except Exception as e:
        return {"message": str(e)}

@router.post("/delete_project",status_code=200)
async def delete_project(user:User,project_id: int, db: asyncpg.Connection = Depends(get_db_connection)):
    sql_query = """
    DELETE FROM Projects
    WHERE id = $1 and user_id = $2;
    """
    try:
        result = await db.execute(sql_query,user.id,project_id)
        if result:
            return {"message": "Project has been deleted successfully"}
        else:
            return {"message": "Error while deleting project"}
    except Exception as e:
        return {"message": str(e)}




# Since we do get project all projects of a user, So we can just use these details of the project in frontend. 
@router.get("/all_projects",status_code=200)
async def get_all_projects(user:User,db:asyncpg.Connection = Depends(get_db_connection)):
    sql_query = """
    SELECT name,description FROM Projects
    WHERE user_id = $1;
    """
    try:
        result = await db.fetch(sql_query,user.id)
        if result:
            return {"projects": result}
        else:
            return {"message": "No projects found"}
    except Exception as e:
        return {"message": str(e)}

