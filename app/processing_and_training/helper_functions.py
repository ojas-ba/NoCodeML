
import io
import asyncpg
from fastapi import Depends
import re
from google.cloud import storage
import pandas as pd
from ..db.init_db import get_db_connection


async def get_file_url(project_id: int, user_id: int,db:asyncpg.Connection = Depends(get_db_connection)):
    sql_query = f"""
    SELECT f.file_path, d.name from Files f
    JOIN Datasets d on f.id = d.file_id
    JOIN Projects p on d.project_id = p.id
    WHERE p.id = ($1) AND p.user_id = ($2)
    """
    try:
        result = await db.fetchrow(sql_query, project_id, user_id)
        if not result:
            return {"message": "No file found for the given project ID and user ID"}
        return result["file_path"], result["name"]
           
    except Exception as e:
        return {"message": str(e)}

def load_dataset_from_gcp(file_url: str):
    try:
        match = re.match(r"https://storage\.googleapis\.com/([^/]+)/(.+)", file_url)
        if not match:
            return {"message": "Invalid GCS URL format"}
        
        bucket_name = match.group(1)
        file_name = match.group(2)
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        content = blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(content))
        
        return df

    except Exception as e:
        return {"message": str(e)}
def convert_to_gcs_uri(public_url: str) -> str:
    """
    Converts a GCS public URL to the format gs://bucket_name/object_path
    """
    if not public_url.startswith("https://storage.googleapis.com/"):
        raise ValueError("Not a valid GCS public URL")

    # Strip prefix
    stripped = public_url.replace("https://storage.googleapis.com/", "", 1)

    # Split bucket and object
    parts = stripped.split("/", 1)
    if len(parts) != 2:
        raise ValueError("Invalid GCS URL format")

    bucket_name, object_path = parts
    return f"gs://{bucket_name}/{object_path}"

def detect_column_types(df: pd.DataFrame,cardinality_threshold: int = 15):
    numerical = []
    categorical = []

    for col in df.columns:
        dtype = df[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            # Catch integer-based categorical features
            if df[col].nunique() < cardinality_threshold:
                categorical.append(col)
            else:
                numerical.append(col)
        else:
            categorical.append(col)

    return {
        "numerical": numerical,
        "categorical": categorical
    }

