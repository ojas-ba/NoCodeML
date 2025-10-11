from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.worker.tasks", "app.worker.training_tasks"],  # Import both task modules
    broker_connection_retry_on_startup=True  # Retry connecting to Redis on startup
)

# Configure Celery for training tasks
celery_app.conf.update(
    # Use default 'celery' queue (no routing needed)
    # Task execution settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task timeouts and retries for ML training
    task_soft_time_limit=3300,  # 55 minutes soft limit
    task_time_limit=3600,       # 1 hour hard limit
    task_default_retry_delay=300,  # 5 minutes between retries
    task_max_retries=1,  # Only retry once for training tasks
    
    # Result backend settings
    result_expires=86400,  # Results expire after 24 hours
    task_track_started=True,
    task_send_sent_event=True,
    
    # Worker settings for ML training
    worker_prefetch_multiplier=1,  # One task at a time for training
    worker_max_tasks_per_child=5,  # Restart worker after 5 tasks to prevent memory leaks
    worker_disable_rate_limits=True,
    
    # Task acknowledgment
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)