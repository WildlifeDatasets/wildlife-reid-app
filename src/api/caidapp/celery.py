import logging
import os

from celery import Celery

RABBITMQ_URL = os.environ["RABBITMQ_URL"]
REDIS_URL = os.environ["REDIS_URL"]

# set the default Django settings module for the 'celery' program
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "CarnivoreIDApp.settings")

logger = logging.getLogger("app")

app = Celery("caidapp", broker=RABBITMQ_URL, backend=REDIS_URL)
app.config_from_object("django.conf:settings", namespace="CELERY")

# update celery configuration
app.conf.task_routes = {
    # recognition workflow
    "predict": {"queue": "taxon_worker"},
    # detection workflow
    "detect": {"queue": "detection_worker"},
    "detectionsimplelog": {"queue": "detection_worker"},
    # identification workflow
    "init_identification": {"queue": "identification_worker"},
    "train_identification": {"queue": "identification_worker"},
    "identify": {"queue": "identification_worker"},
}
app.conf.update(task_track_started=True)

# load task modules
app.autodiscover_tasks(["caidapp"])
