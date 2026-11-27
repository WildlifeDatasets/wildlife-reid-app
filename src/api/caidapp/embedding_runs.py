"""Track embedding work without changing the identification worker protocol."""

import errno
import shutil
import uuid
from collections import Counter
from datetime import timedelta
from pathlib import Path

from celery import signature
from celery.result import AsyncResult
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from .models import AnimalObservation, EmbeddingRunItem, TaskRun, WorkGroup
from .observation_embeddings import backfill_observations, embedding_generation

TERMINAL = {"succeeded", "partial", "failed", "cancelled"}
TASK_STAGES = {"prepare": "Preparing inputs", "compute": "Computing image features", "store": "Saving results"}


def task_ids(run):
    return {"prepare": run.prepare_task_id, "compute": run.compute_task_id, "store": run.store_task_id}


def failure_details(stage, task_id, *, error=None, state="FAILURE"):
    """Summarize exceptions without sending arbitrary worker messages to the browser."""
    error_type = type(error).__name__ if isinstance(error, BaseException) else ""
    detail = str(error).lower() if isinstance(error, BaseException) else ""
    if state == "REVOKED":
        reason = "The task was revoked."
    elif error_type in {"OutOfMemoryError", "MemoryError"} or "cuda out of memory" in detail:
        reason = "The worker ran out of memory."
    elif error_type in {"WorkerLostError", "Terminated"}:
        reason = "The worker process exited before completing the task. Check worker logs for a restart or memory limit."
    elif error_type in {"TimeLimitExceeded", "SoftTimeLimitExceeded", "TimeoutError"}:
        reason = "The task exceeded a time limit."
    elif isinstance(error, FileNotFoundError) or error_type in {"LocalEntryNotFoundError", "EntryNotFoundError"}:
        reason = "A required image, model file, or shared task file was unavailable. Check the task log for the file."
    elif isinstance(error, PermissionError):
        reason = "The process could not access a required file."
    elif isinstance(error, OSError) and error.errno in {errno.ENOSPC, errno.EDQUOT}:
        reason = "The storage volume ran out of free space or reached its quota."
    elif error_type == "OperationalError":
        reason = "The application database could not complete this operation."
    elif isinstance(error, ValidationError):
        reason = "The task inputs or results failed validation. Check the task log for details."
    elif detail in {
        "the model checkpoint changed after this run was prepared.",
        "source media changed after this run was prepared.",
        "the worker segmentation backend does not match the configured feature profile.",
    }:
        reason = str(error)
    else:
        reason = "The task failed. Use its Celery task ID to find the underlying error in the logs."
    return {
        "stage": stage,
        "stage_label": TASK_STAGES[stage],
        "task_id": task_id,
        "state": state,
        "error_type": error_type,
        "reason": reason,
    }


def record_failure(run_id, stage, task_id, *, error=None, state="FAILURE"):
    diagnostic = failure_details(stage, task_id, error=error, state=state)
    fail_run(run_id, f"{diagnostic['stage_label']} failed. {diagnostic['reason']}", diagnostic=diagnostic)


def fail_run(run_id, message, *, diagnostic=None, status="failed"):
    """Record a failure or cancellation without exposing worker tracebacks or paths."""
    with transaction.atomic():
        run = TaskRun.objects.select_for_update().filter(pk=run_id).first()
        if run is None or run.status in TERMINAL:
            return
        run.items.filter(status="pending").update(status=status, message=message)
        result = dict(Counter(run.items.values_list("status", flat=True)))
        if diagnostic:
            result["failure"] = diagnostic
        TaskRun.objects.filter(pk=run_id).update(
            status=status,
            message=message,
            finished_at=timezone.now(),
            updated_at=timezone.now(),
            result=result,
        )


def create_embedding_run(
    workgroup_id, observation_ids=None, *, mediafile_ids=None, requested_by=None, trigger="manual", request_key=None
):
    """Validate scope cheaply and prepare snapshots on the API worker after commit."""
    with transaction.atomic():
        # Lock only the workgroup: PostgreSQL cannot lock the nullable model side
        # of the outer join produced by select_related("identification_model").
        workgroup = WorkGroup.objects.select_for_update().get(pk=workgroup_id)
        if request_key:
            existing = TaskRun.objects.filter(request_key=request_key).first()
            if existing:
                if existing.workgroup_id != workgroup_id or existing.requested_by_id != getattr(
                    requested_by, "pk", None
                ):
                    raise ValidationError("This request key is already in use.")
                return existing
        model = workgroup.identification_model
        if model is None:
            raise ValidationError("Select an identification model first.")
        observations = AnimalObservation.objects.filter(mediafile__parent__owner__workgroup_id=workgroup_id)
        if observation_ids is not None:
            requested = set(observation_ids)
            observations = observations.filter(pk__in=requested)
            if set(observations.values_list("pk", flat=True)) != requested:
                raise ValidationError("Some requested observations are missing or outside this workgroup.")
        if mediafile_ids is not None:
            from .models import MediaFile

            requested = set(mediafile_ids)
            allowed = MediaFile.objects.filter(pk__in=requested, parent__owner__workgroup_id=workgroup_id)
            if set(allowed.values_list("pk", flat=True)) != requested:
                raise ValidationError("Some requested media files are missing or outside this workgroup.")
            observations = observations.filter(mediafile_id__in=requested)
        if trigger in {"automatic", "backfill"}:
            observations = observations.filter(pk__in=backfill_observations(workgroup_id))
        rows = list(observations.values_list("pk", "mediafile_id"))
        if not rows:
            raise ValidationError("No observations in the requested scope.")
        run = TaskRun.objects.create(
            workgroup=workgroup,
            requested_by=requested_by,
            trigger=trigger,
            request_key=request_key or None,
            prepare_task_id=str(uuid.uuid4()),
            model_snapshot={
                "model_id": model.pk,
                "model_source": model.get_runtime_model_source(),
                "weights_path": model.get_runtime_checkpoint_path(),
                "generation": embedding_generation(workgroup_id),
            },
        )
        EmbeddingRunItem.objects.bulk_create(
            [
                EmbeddingRunItem(run=run, observation_id=pk, observation_id_snapshot=pk, mediafile_id_snapshot=media)
                for pk, media in rows
            ]
        )

        def publish():
            try:
                signature("caidapp.tasks.prepare_embedding_run", args=[run.pk]).apply_async(task_id=run.prepare_task_id)
            except Exception as exc:
                record_failure(run.pk, "prepare", run.prepare_task_id, error=exc)
                raise

        transaction.on_commit(publish, robust=True)
    return run


def prepare_run(run_id):
    """Create the manifest in the background, preserving the requested model snapshot."""
    from .observation_embeddings import enqueue_embeddings

    with transaction.atomic():
        run = TaskRun.objects.select_for_update().get(pk=run_id)
        if run.status != "preparing" or run.compute_task_id or run.started_at:
            return
        run.started_at = timezone.now()
        run.save(update_fields=["started_at", "updated_at"])
    try:
        task = enqueue_embeddings(
            run.workgroup_id,
            list(run.items.values_list("observation_id_snapshot", flat=True)),
            skip_invalid=True,
            run_id=run.pk,
        )
        if task is None:
            now = timezone.now()
            TaskRun.objects.filter(pk=run.pk, status="preparing").update(
                status="partial",
                message="No eligible inputs; see individual observations.",
                result=dict(Counter(run.items.values_list("status", flat=True))),
                finished_at=now,
                updated_at=now,
            )
    except Exception as exc:
        record_failure(run_id, "prepare", run.prepare_task_id, error=exc)
        raise


def refresh_run(run):
    """Read transient execution evidence; only a committed finalizer marks success."""
    if run.status in TERMINAL:
        return run
    task_id = run.compute_task_id or run.prepare_task_id
    if not task_id:
        return run
    try:
        task = AsyncResult(task_id)
        state = task.state
        if state in {"FAILURE", "REVOKED"}:
            stage = "compute" if run.compute_task_id else "prepare"
            record_failure(run.pk, stage, task_id, error=task.result, state=state)
        elif run.compute_task_id and state == "SUCCESS":
            callback = AsyncResult(run.store_task_id) if run.store_task_id else None
            callback_state = callback.state if callback is not None else None
            if callback_state in {"FAILURE", "REVOKED"}:
                record_failure(run.pk, "store", run.store_task_id, error=callback.result, state=callback_state)
                run.refresh_from_db()
                return run
            if callback_state == "SUCCESS":
                run.status = "attention"
                run.message = "Computation ended but the result could not be confirmed in the database."
            else:
                run.status = "finalizing"
                run.message = "Saving and validating embedding results."
                first_seen = run.result.get("storage_observed_at")
                if first_seen is None:
                    run.result = {**run.result, "storage_observed_at": timezone.now().isoformat()}
                    TaskRun.objects.filter(pk=run.pk).exclude(status__in=TERMINAL).update(result=run.result)
                if first_seen and timezone.now() - timezone.datetime.fromisoformat(first_seen) > timedelta(minutes=10):
                    run.status = "attention"
                    run.message = "Result storage has not been confirmed; administrator review is needed."
            TaskRun.objects.filter(pk=run.pk).exclude(status__in=TERMINAL).update(
                status=run.status, message=run.message
            )
        elif run.compute_task_id and state in {"STARTED", "PROGRESS"}:
            TaskRun.objects.filter(pk=run.pk, status__in=["queued", "attention"]).update(status="running")
        elif timezone.now() - run.updated_at > timedelta(hours=2):
            TaskRun.objects.filter(pk=run.pk).exclude(status__in=TERMINAL).update(
                status="attention", message="Execution status cannot be confirmed. Administrator review is needed."
            )
    except Exception:
        # A temporary result-backend outage must not overwrite durable state.
        pass
    run.refresh_from_db()
    return run


def run_diagnostics(run):
    """Recover the cause on older failed runs while their Celery results are retained."""
    ids = task_ids(run)
    failure = run.result.get("failure")
    if not failure and run.status == "failed":
        for stage in ("store", "compute", "prepare"):
            if not ids[stage]:
                continue
            try:
                task = AsyncResult(ids[stage])
                state = task.state
                if state in {"FAILURE", "REVOKED"}:
                    failure = failure_details(stage, ids[stage], error=task.result, state=state)
                    break
            except Exception:
                break
    return {"task_ids": ids, "failure": failure}


def serialize_run(run, include_items=False):
    """Expose safe status data, excluding filesystem paths and model checkpoints."""
    data = {
        "uuid": str(run.uuid),
        "kind": run.kind,
        "status": run.status,
        "message": run.message,
        "trigger": run.trigger,
        "created_at": run.created_at.isoformat(),
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "result": run.result,
        "model_id": run.model_snapshot.get("model_id", run.model_snapshot.get("id")),
        "counts": dict(Counter(run.items.values_list("status", flat=True))),
    }
    if include_items:
        data["items"] = [
            {
                "observation_id": item.observation_id_snapshot,
                "mediafile_id": item.mediafile_id_snapshot,
                "status": item.status,
                "message": item.message,
            }
            for item in run.items.order_by("pk")
        ]
    return data


def cleanup_embedding_runs():
    """Remove only old terminal manifests inside their dedicated UUID directory."""
    root = (Path(settings.MEDIA_ROOT) / "observation_embeddings").resolve()
    removed = 0
    for run in TaskRun.objects.filter(
        kind="embedding", status__in=TERMINAL, finished_at__lt=timezone.now() - timedelta(days=7)
    ).exclude(manifest_path=""):
        path = Path(run.manifest_path)
        directory = path.parent
        try:
            uuid.UUID(directory.name)
            if path.name != "input.json" or directory.is_symlink() or directory.resolve().parent != root:
                continue
            if directory.exists():
                shutil.rmtree(directory)
            TaskRun.objects.filter(pk=run.pk).update(manifest_path="")
            removed += 1
        except (OSError, ValueError):
            continue
    return removed
