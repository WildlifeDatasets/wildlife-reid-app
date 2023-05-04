import logging
import os

from celery import Celery

RABBITMQ_URL = os.environ["RABBITMQ_URL"]
REDIS_URL = os.environ["REDIS_URL"]

logger = logging.getLogger("app")

logger.info("Creating celery app instance.")
tasks = Celery("api", broker=RABBITMQ_URL, backend=REDIS_URL)
tasks.conf.task_routes = {
    "upload": {"queue": "data_worker"},
    "predict": {"queue": "inference_worker"},
}
tasks.conf.update(task_track_started=True)  # TODO - what does this do?

logger.info(f"{RABBITMQ_URL=}; {REDIS_URL=}")  # TODO - tmp
