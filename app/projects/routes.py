import re
from fastapi import APIRouter, Depends, File, Form, UploadFile, Query
from pydantic import BaseModel

from ..train.routes import get_file_url
from ..models.projects import project,Datasets
from ..db.init_db import get_db_connection
from ..models.user import User
import asyncpg
import os
from google.cloud import storage

# Initialize Google Cloud Storage client

BUCKET_NAME = "my-fastapi-bucket" # Replace with your bucket name

# Always make sure that the project belongs to the user who is creating it.
# Also make sure that the dataset belongs to the project and the user who is creating it.
# Because this can compromise the security of the system.

router = APIRouter()

def get_gcs_client():
    return storage.Client()

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
async def upload_file(file: UploadFile = File(),user_id: int = Form(...), db:asyncpg.Connection = Depends(get_db_connection), client: storage.Client = Depends(get_gcs_client)):
    try:
        file_name = f"user_{user_id}_{file.filename}"
        #uploading file to Google Cloud Storage
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(file_name)
        blob.upload_from_file(file.file, content_type=file.content_type)
        # Save the file path/url to the database
        
        file_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{file_name}"
        
        sql_query = """
        INSERT INTO Files (file_path, user_id)
        VALUES($1,$2)
        RETURNING id;
        """
        file_id = await db.fetchval(sql_query,file_url, user_id)
        if file_id:
            return {"message": "File uploaded successfully", "file_id": file_id}
        else:
            return {"message": "Error while uploading file"}
    except Exception as e:
        return {"message": str(e)}
    
    
# THis shit is not secure. You have to extract user id from the jwt token not the query. 
# So change this later otherwise some nigga can enter random userids and delete everyones stuff.  
@router.post("/delete_project",status_code=200)
async def delete_project(project_id: int=Query(), user_id:int=Query(),db: asyncpg.Connection = Depends(get_db_connection)):
    
    sql_query = """
    DELETE FROM Projects
    WHERE id = $1 and user_id = $2;
    """
    try:
        # we have to delete dataset from GCP also
        err_msg = ""
        file_url, dataset_name  = await get_file_url(project_id,user_id,db)
        if file_url:
            # delete file from GCP
            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)
            match = re.match(r"https://storage\.googleapis\.com/([^/]+)/(.+)", file_url)
            if match:
                blob = bucket.blob(match.group(2))
                if blob.exists():
                    blob.delete()
                else:
                    err_msg += "File does not exist in GCP\n"
        result = await db.execute(sql_query,project_id,user_id)
        if result:
            return {"message": "Project has been deleted successfully","code":1,"file_status":err_msg}
        else:
            return {"message": "Error while deleting project","code":0,"file_status": err_msg}
    except Exception as e:
        return {"message": str(e)}




# Since we do get project all projects of a user, So we can just use these details of the project in frontend. 
@router.get("/all_projects",status_code=200)
async def get_all_projects(user_id:int = Query(),db:asyncpg.Connection = Depends(get_db_connection)):
    sql_query = """
    SELECT id,name,description FROM Projects
    WHERE user_id = $1;
    """
    try:
        result = await db.fetch(sql_query,user_id)
        if result:
            return {"projects": result}
        else:
            return {"message": "No projects found"}
    except Exception as e:
        return {"message": str(e)}


