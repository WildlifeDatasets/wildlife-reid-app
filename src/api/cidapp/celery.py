import logging
import os

from celery import Celery

RABBITMQ_URL = os.environ["RABBITMQ_URL"]
REDIS_URL = os.environ["REDIS_URL"]

# set the default Django settings module for the 'celery' program
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "CarnivoreIDApp.settings")

logger = logging.getLogger("app")

logger.info("Creating celery app instance.")
app = Celery("cidapp", broker=RABBITMQ_URL, backend=REDIS_URL)
app.config_from_object("django.conf:settings", namespace="CELERY")

# update celery configuration
app.conf.task_routes = {
    # "upload": {"queue": "data_worker"},
    "predict": {"queue": "inference_worker"},
}
app.conf.update(task_track_started=True)

# load task modules
app.autodiscover_tasks(["cidapp"])
