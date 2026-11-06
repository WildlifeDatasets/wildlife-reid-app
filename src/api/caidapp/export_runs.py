"""Persist private export results independently of browser polling and Celery expiry."""

import uuid
from datetime import timedelta
from pathlib import Path

from celery import signature
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Q
from django.utils import timezone

from .models import ExportArtifact, TaskRun

EXPORT_TASKS = {"create_mediafiles_zip", "create_mediafiles_zip_with_metadata", "create_beta_stability_export"}


def export_root():
    """Keep exports outside the publicly served media directory."""
    return (Path(settings.PRIVATE_DATA_PATH) / "export_downloads").resolve()


def artifact_path(artifact):
    """Reject paths outside the export's dedicated UUID directory, including symlinks."""
    path = Path(artifact.path)
    expected = export_root() / str(artifact.run.uuid) / artifact.filename
    if path != expected or path.resolve() != expected or path.parent.is_symlink() or path.is_symlink():
        raise ValidationError("Invalid export destination.")
    return path


def start_export(user, task_name, args, output_index, *, filename, label, kwargs=None):
    """Create durable ownership before publishing a background export request."""
    if task_name not in EXPORT_TASKS or not user.workgroup_id:
        raise ValidationError("Export cannot be started.")
    filename = Path(filename).name
    if not filename or filename in {".", ".."}:
        raise ValidationError("Invalid export filename.")
    with transaction.atomic():
        run = TaskRun.objects.create(
            kind="export",
            workgroup_id=user.workgroup_id,
            requested_by=user,
            status="queued",
            prepare_task_id=str(uuid.uuid4()),
            result={"label": label},
        )
        artifact = ExportArtifact.objects.create(
            run=run,
            filename=filename,
            path=str(export_root() / str(run.uuid) / filename),
        )
        args = list(args)
        args[output_index] = artifact.path

        def publish():
            try:
                signature("caidapp.tasks.execute_export_run", args=[run.pk, task_name, args, kwargs or {}]).apply_async(
                    task_id=run.prepare_task_id
                )
            except Exception:
                TaskRun.objects.filter(pk=run.pk, status="queued").update(
                    status="failed",
                    message="Could not submit export. Please request a new export.",
                    finished_at=timezone.now(),
                )

        transaction.on_commit(publish)
    return run


def execute_export(run_id, task_name, args, kwargs):
    """Run the existing exporter and commit the artifact before announcing success."""
    from . import tasks

    if task_name not in EXPORT_TASKS:
        raise ValidationError("Unsupported export operation.")
    with transaction.atomic():
        run = TaskRun.objects.select_for_update().get(pk=run_id, kind="export")
        if run.status != "queued":
            return
        run.status = "running"
        run.started_at = timezone.now()
        run.save(update_fields=["status", "started_at", "updated_at"])
    artifact = run.artifact
    try:
        path = artifact_path(artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        result = getattr(tasks, task_name).run(*args, **kwargs)
        if Path(result).resolve() != path or not path.is_file():
            raise ValidationError("Export did not produce the expected file.")
        with transaction.atomic():
            artifact.size = path.stat().st_size
            artifact.expires_at = timezone.now() + timedelta(days=7)
            artifact.save(update_fields=["size", "expires_at"])
            TaskRun.objects.filter(pk=run.pk).update(
                status="succeeded",
                message="Export is ready to download.",
                finished_at=timezone.now(),
            )
    except Exception:
        TaskRun.objects.filter(pk=run.pk).update(
            status="failed",
            message="Export failed. Please check the selection and request a new export.",
            finished_at=timezone.now(),
        )
        raise


def refresh_export(run):
    """Keep persisted terminal outcomes even after Celery results expire."""
    if run.status in {"succeeded", "failed"}:
        return run
    from celery.result import AsyncResult

    try:
        state = AsyncResult(run.prepare_task_id).state
        if state in {"FAILURE", "REVOKED"}:
            TaskRun.objects.filter(pk=run.pk).exclude(status__in=["succeeded", "failed"]).update(
                status="failed",
                message="Export execution failed or was revoked.",
                finished_at=timezone.now(),
            )
        elif state == "SUCCESS" or timezone.now() - run.created_at > timedelta(hours=24):
            TaskRun.objects.filter(pk=run.pk).exclude(status__in=["succeeded", "failed"]).update(
                status="attention",
                message="Export completion could not be confirmed. Please request a new export.",
            )
    except Exception:
        pass  # A result-backend outage cannot change a durable export outcome.
    run.refresh_from_db()
    return run


def cleanup_exports():
    """Delete expired private artifacts while retaining task and artifact metadata."""
    removed = 0
    artifacts = ExportArtifact.objects.filter(
        Q(expires_at__lte=timezone.now(), run__status="succeeded")
        | Q(run__status="failed", run__finished_at__lt=timezone.now() - timedelta(days=7))
    ).select_related("run")
    for artifact in artifacts:
        try:
            path = artifact_path(artifact)
            if path.is_file():
                path.unlink()
                removed += 1
                path.parent.rmdir()
        except (OSError, ValidationError):
            continue
    return removed
