import logging
import os
import threading
from celery import Celery, signals

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
    "detect_identification_outliers": {"queue": "identification_worker"},
}
app.conf.update(task_track_started=True)


def _signal_context(sender=None, task_id=None, task=None, args=None, kwargs=None, **extra):
    parts = [
        f"pid={os.getpid()}",
        f"thread={threading.get_ident()}",
    ]
    if sender is not None:
        parts.append(f"sender={getattr(sender, 'hostname', sender)}")
    if task_id:
        parts.append(f"task_id={task_id}")
    if task is not None:
        parts.append(f"task_name={getattr(task, 'name', task)}")
    if args is not None:
        parts.append(f"args_len={len(args)}")
    if kwargs is not None:
        parts.append(f"kwargs_keys={sorted(kwargs.keys())}")
    if extra:
        parts.extend(f"{key}={value!r}" for key, value in extra.items())
    return " ".join(parts)


@signals.worker_ready.connect
def _on_worker_ready(sender=None, **kwargs):
    logger.info("Celery worker ready: %s", _signal_context(sender=sender, **kwargs))


@signals.worker_shutdown.connect
def _on_worker_shutdown(sender=None, **kwargs):
    logger.info("Celery worker shutdown: %s", _signal_context(sender=sender, **kwargs))


@signals.task_received.connect
def _on_task_received(sender=None, request=None, **kwargs):
    logger.info(
        "Celery task received: %s",
        _signal_context(
            sender=sender,
            task_id=getattr(request, "id", None),
            task=getattr(request, "task", None),
            args=getattr(request, "args", None),
            kwargs=getattr(request, "kwargs", None),
        ),
    )


@signals.task_prerun.connect
def _on_task_prerun(sender=None, task_id=None, task=None, args=None, kwargs=None, **extra):
    logger.info("Celery task prerun: %s", _signal_context(sender=sender, task_id=task_id, task=task, args=args, kwargs=kwargs))


@signals.task_postrun.connect
def _on_task_postrun(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **extra):
    logger.info(
        "Celery task postrun: %s state=%r retval_type=%s",
        _signal_context(sender=sender, task_id=task_id, task=task, args=args, kwargs=kwargs),
        state,
        type(retval).__name__ if retval is not None else None,
    )


@signals.task_failure.connect
def _on_task_failure(sender=None, task_id=None, exception=None, args=None, kwargs=None, einfo=None, **extra):
    logger.error(
        "Celery task failure: %s exception=%r traceback=%s",
        _signal_context(sender=sender, task_id=task_id, args=args, kwargs=kwargs),
        exception,
        getattr(einfo, "traceback", None),
    )

# load task modules
app.autodiscover_tasks(["caidapp"])
