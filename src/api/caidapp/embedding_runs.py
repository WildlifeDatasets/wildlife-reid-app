"""Track embedding work without changing the identification worker protocol."""

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

TERMINAL = {"succeeded", "partial", "failed"}


def fail_run(run_id, message):
    """Record a public failure without exposing worker tracebacks or paths."""
    with transaction.atomic():
        run = TaskRun.objects.select_for_update().filter(pk=run_id).first()
        if run is None or run.status in TERMINAL:
            return
        run.items.filter(status="pending").update(status="failed", message=message)
        TaskRun.objects.filter(pk=run_id).update(
            status="failed",
            message=message,
            finished_at=timezone.now(),
            updated_at=timezone.now(),
            result=dict(Counter(run.items.values_list("status", flat=True))),
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
        if trigger == "automatic":
            observations = observations.filter(mediafile__parent__import_finished=True)
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
            except Exception:
                fail_run(run.pk, "Could not submit embedding preparation. Please try again.")

        transaction.on_commit(publish)
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
            run.refresh_from_db()
            run.status = "partial"
            run.message = "No eligible inputs; see individual observations."
            run.result = dict(Counter(run.items.values_list("status", flat=True)))
            run.finished_at = timezone.now()
            run.save()
    except Exception:
        fail_run(run_id, "Embedding preparation failed. Check source media files and the selected model.")
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
            fail_run(run.pk, "Embedding task failed or was revoked. Please try again.")
        elif run.compute_task_id and state == "SUCCESS":
            callback_state = AsyncResult(run.store_task_id).state if run.store_task_id else None
            if callback_state in {"FAILURE", "REVOKED", "SUCCESS"}:
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
        "model_id": run.model_snapshot.get("model_id"),
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
