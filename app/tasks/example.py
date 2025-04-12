# backend/app/tasks/example_task.py
from app.worker import celery_app
import time

@celery_app.task
def background_training_job(data):
    time.sleep(10)  # Simulate time-consuming job
    print("I am Celery Printting STuff")
    return {"status": "completed", "input": data}
