"""Track installation-wide model downloads on the workers that use the weights."""

import logging
import uuid
from datetime import timedelta

from django.db import transaction
from django.utils import timezone

from .celery import app
from .embedding_runs import TERMINAL
from .models import IdentificationModel, TaskRun

logger = logging.getLogger(__name__)
KIND = "model_weights"
ACTIVE_KEY = "model-weights:active"
WORKERS = {"taxon": "Detection and species classification", "identification": "Individual identification"}
EXPIRES = timedelta(days=1)


def can_manage(user):
    """Use the same installation administrator access as the Admin Tools menu."""
    return user.is_active and user.is_staff and user.is_superuser


def start(user):
    """Share one active download run across administrators."""
    with transaction.atomic():
        run, created = TaskRun.objects.get_or_create(
            request_key=ACTIVE_KEY,
            defaults={
                "kind": KIND,
                "requested_by": user,
                "status": "queued",
                "message": "Waiting for the model workers.",
                "result": {
                    key: {"task_id": str(uuid.uuid4()), "status": "queued", "message": "Waiting for worker."}
                    for key in WORKERS
                },
            },
        )
        if created:
            transaction.on_commit(lambda: _publish(run))
    return run


def _identification_sources():
    sources = set()
    for model in IdentificationModel.objects.select_related("source_identification_model").exclude(model_path=""):
        source = model.get_runtime_model_source()
        if source:
            sources.add(source)
    return sorted(sources)


def _publish(run):
    errors = {}
    for key, step in run.result.items():
        try:
            app.send_task(
                f"download_{key}_model_weights",
                kwargs={"model_sources": _identification_sources()} if key == "identification" else {},
                queue=f"{key}_worker",
                task_id=step["task_id"],
                expires=EXPIRES.total_seconds(),
            )
        except Exception:
            logger.exception("Could not submit model weights task to %s worker", key)
            errors[key] = {**step, "status": "failed", "message": "Could not submit the task to this worker."}
    if errors:
        with transaction.atomic():
            current = TaskRun.objects.select_for_update().get(pk=run.pk)
            failures = {key: step for key, step in errors.items() if current.result[key]["status"] == "queued"}
            _save_progress(current, current.result | failures)


def _worker_progress(step, expired):
    if step["status"] in TERMINAL:
        return step
    try:
        meta = app.backend.get_task_meta(step["task_id"])
    except Exception:
        logger.warning("Could not read model download progress", exc_info=True)
        return step
    state, result = meta["status"], meta.get("result")
    if state == "SUCCESS":
        failures = result["failures"]
        total = result["total"]
        ready = total - len(failures)
        status = "succeeded" if not failures else ("partial" if ready else "failed")
        message = f"{ready} of {total} model entries ready."
        if failures:
            message += f" Failed: {', '.join(failures)}. See the worker log for details."
        if "SAM3" in failures:
            message += " Check the worker's Hugging Face access to facebook/sam3."
        return {**step, "status": status, "message": message}
    if state in {"FAILURE", "REVOKED"} or expired:
        message = "The worker stopped before confirming completion. See the worker log for details."
        return {**step, "status": "failed", "message": message}
    if state == "PROGRESS":
        message = f"Checking or downloading {result['current']} ({result['done'] + 1} of {result['total']})."
        return {**step, "status": "running", "message": message}
    if state == "STARTED" and step["status"] == "queued":
        return {**step, "status": "running", "message": "Preparing the download list."}
    return step


def _save_progress(run, steps):
    if steps == run.result or run.status in TERMINAL:
        return run
    statuses = {step["status"] for step in steps.values()}
    now = timezone.now()
    values = {"result": steps, "updated_at": now}
    if statuses <= TERMINAL:
        status = "failed"
        if statuses == {"succeeded"}:
            status = "succeeded"
        elif statuses & {"succeeded", "partial"}:
            status = "partial"
        values.update(
            status=status,
            message={
                "succeeded": "Model weights are ready.",
                "partial": "Some model weights could not be prepared.",
                "failed": "The model weight download failed.",
            }[status],
            finished_at=now,
            request_key=None,
        )
    elif statuses != {"queued"}:
        values.update(status="running", message="Preparing model weights.", started_at=run.started_at or now)
    # A status request and the background reconciler can read the same run concurrently.
    TaskRun.objects.filter(pk=run.pk, updated_at=run.updated_at).update(**values)
    run.refresh_from_db()
    return run


def refresh(run):
    """Persist the latest worker outcomes before the Celery results expire."""
    if run.status in TERMINAL:
        return run
    expired = timezone.now() - run.created_at > 2 * EXPIRES
    steps = {key: _worker_progress(step, expired) for key, step in run.result.items()}
    return _save_progress(run, steps)
