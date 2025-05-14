from celery import Celery

celery_app = Celery(
    "nocodeml_tasks",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/1",
    include=[
        "app.tasks.train",
        "app.tasks.plot",
    ],
)

celery_app.conf.update(
    task_routes = {
        "app.tasks.train.train_classifier": {"queue": "train"},
        "app.tasks.train.train_regressor": {"queue": "train"},
        "app.tasks.train.train_clustering": {"queue": "train"},
        "app.tasks.plot.plot": {"queue": "plot"},
    }
)




