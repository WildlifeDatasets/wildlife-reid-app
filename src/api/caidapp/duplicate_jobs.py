"""Durable duplicate previews and confirmed deletions outside the lifetime of HTTP requests."""

import hashlib
import logging
import uuid
from datetime import timedelta
from functools import partial
from types import SimpleNamespace

from celery import signature
from django.core import signing
from django.db import transaction
from django.utils import timezone

from . import duplicate_services as service
from . import models

logger = logging.getLogger("app")
KINDS = ("duplicate_preview", "duplicate_removal")
TERMINAL = {"succeeded", "failed"}


def visible_proposals(user):
    """Reuse exactly the catalogue's current group and media-file access rules."""
    from .views_duplicates import _visible_proposals

    request = SimpleNamespace(user=SimpleNamespace(caiduser=user))
    return _visible_proposals(request, None, "pending")


def _publish(run_id):
    run = models.TaskRun.objects.get(pk=run_id)
    try:
        signature("caidapp.tasks.process_duplicate_job", args=[run.pk], queue="celery").apply_async(
            task_id=run.compute_task_id, retry=False
        )
        logger.info("Published duplicate job=%s kind=%s", run.uuid, run.kind)
    except Exception:
        logger.exception("Could not publish duplicate job=%s", run.uuid)
        models.TaskRun.objects.filter(pk=run.pk, status="queued").update(
            status="failed",
            message="Could not reach the API worker. Retry delivery.",
            finished_at=timezone.now(),
        )


@transaction.atomic
def start_preview(user, selected=None):
    """Enqueue metadata/source verification without performing any source I/O in the request."""
    models.CaIDUser.objects.select_for_update().get(pk=user.pk)
    if not user.workgroup_id:
        raise ValueError("Select a working group first")
    if selected is not None:
        selected = sorted({str(uuid.UUID(value)) for value in selected})
        if not 1 <= len(selected) <= service.MAX_BULK_PROPOSALS:
            raise ValueError(f"Select between 1 and {service.MAX_BULK_PROPOSALS} groups")
        if visible_proposals(user).filter(uuid__in=selected).count() != len(selected):
            raise ValueError("Some selected proposals are unavailable")
    inputs = {"selected": selected, "perfect_only": selected is None}
    existing = models.TaskRun.objects.filter(
        kind="duplicate_preview",
        requested_by=user,
        workgroup_id=user.workgroup_id,
        status__in=["queued", "running"],
        model_snapshot=inputs,
    ).first()
    if existing:
        return existing
    run = models.TaskRun.objects.create(
        kind="duplicate_preview",
        requested_by=user,
        workgroup_id=user.workgroup_id,
        status="queued",
        model_snapshot=inputs,
        compute_task_id=str(uuid.uuid4()),
        message="Waiting for the API worker to prepare the removal preview. No files have been deleted.",
    )
    transaction.on_commit(partial(_publish, run.pk), robust=True)
    return run


@transaction.atomic
def start_removal(user, token):
    """Accept a fresh signed confirmation once, then perform it in a durable worker task."""
    models.CaIDUser.objects.select_for_update().get(pk=user.pk)
    key = "duplicate-removal:" + hashlib.sha256(token.encode()).hexdigest()
    existing = models.TaskRun.objects.filter(request_key=key, requested_by=user, workgroup_id=user.workgroup_id).first()
    if existing:
        return existing
    data = signing.loads(token, salt="duplicate-bulk-v1", max_age=1800)
    plans = data["plans"]
    if data["user"] != user.pk or not 1 <= len(plans) <= service.MAX_BULK_PROPOSALS:
        raise ValueError("Invalid bulk preview")
    for plan in plans:
        signed = signing.loads(plan["token"], salt="duplicate-delete-v2", max_age=1800)
        if signed["user"] != user.pk:
            raise ValueError("Invalid bulk preview")
    allowed = set(visible_proposals(user).values_list("pk", flat=True))
    if not {plan["id"] for plan in plans}.issubset(allowed):
        raise ValueError("Access or proposal status changed. Prepare a new preview.")
    run, created = models.TaskRun.objects.get_or_create(
        request_key=key,
        defaults={
            "kind": "duplicate_removal",
            "requested_by": user,
            "workgroup_id": user.workgroup_id,
            "status": "queued",
            "compute_task_id": str(uuid.uuid4()),
            "message": "Confirmed. Waiting for the API worker to remove the reviewed duplicates.",
            "model_snapshot": {
                "token": token,
                "groups": len(plans),
                "remove_count": sum(len(plan["members"]) - 1 for plan in plans),
            },
        },
    )
    if created:
        transaction.on_commit(partial(_publish, run.pk), robust=True)
    return run


def _freeze_preview(preview):
    result = {key: preview[key] for key in ("token", "skipped", "limited", "remove_count")}
    result["previews"] = []
    for row in preview["previews"]:
        summary = row["summary"]
        frozen = {
            key: summary[key]
            for key in ("difference_count", "warnings", "perfect", "extra_differences", "total_difference_count")
        }
        frozen["members"] = [
            {
                "id": member["media"].pk,
                "name": member["name"],
                "role": member["role"],
                "upload": member["media"].parent.name if member["media"].parent else "—",
            }
            for member in summary["members"]
            if member["media"]
        ]
        result["previews"].append({"proposal_id": row["proposal"].pk, "summary": frozen})
    return service.serializable(result)


def preview_context(run, user):
    """Render frozen confirmation labels while resolving optional live metadata icons with current access."""
    result = {**run.result, "previews": []}
    ids = [member["id"] for row in run.result["previews"] for member in row["summary"]["members"]]
    media = {
        m.pk: m
        for m in models.MediaFile.objects.for_user(user)
        .filter(pk__in=ids)
        .select_related("parent", "locality", "sequence")
        .prefetch_related("observations__taxon", "observations__identity")
    }
    for row in run.result["previews"]:
        summary = {
            **row["summary"],
            "members": [{**member, "media": media.get(member["id"])} for member in row["summary"]["members"]],
        }
        result["previews"].append({"proposal": SimpleNamespace(pk=row["proposal_id"]), "summary": summary})
        summary.setdefault("extra_differences", [])
        summary.setdefault("total_difference_count", summary["difference_count"] + len(summary["extra_differences"]))
    return {**result, "perfect_only": run.model_snapshot["perfect_only"]}


def process(run_id):
    """Serialize redeliveries on the job row; commit deletion and its durable result together."""
    query = models.TaskRun.objects.filter(pk=run_id, kind__in=KINDS)
    if not query.exists():
        return
    query.filter(status="queued").update(
        status="running", started_at=timezone.now(), message="Processing in the API worker. You can close this page."
    )
    try:
        with transaction.atomic():
            run = query.select_for_update(of=("self",)).select_related("requested_by__user").get()
            if run.status in TERMINAL:
                return
            user = run.requested_by
            if user is None or user.workgroup_id != run.workgroup_id or not user.user.is_active:
                raise ValueError("Access to the working group changed. No files were removed.")
            logger.info("Starting duplicate job=%s kind=%s", run.uuid, run.kind)
            if run.kind == "duplicate_preview":
                rows = visible_proposals(user)
                selected = run.model_snapshot["selected"]
                if selected is not None:
                    rows = rows.filter(uuid__in=selected)
                    if rows.count() != len(selected):
                        raise ValueError("Some selected proposals changed. Prepare a new preview.")
                preview = service.bulk_preview(
                    rows.iterator(chunk_size=100), user, perfect_only=run.model_snapshot["perfect_only"]
                )
                run.result = _freeze_preview(preview)
                run.message = "Preview ready. Review and confirm before any files are deleted."
            else:
                # Confirmation age was checked when accepted into the durable queue.
                # Queueing/running time must not expire an already authorized job.
                count = service.apply_bulk(
                    user,
                    run.model_snapshot["token"],
                    visible_proposals(user).values_list("pk", flat=True),
                    preview_max_age=None,
                )
                run.result = {"removed": count}
                run.message = f"Removed {count} duplicate media files. Retained files were not changed."
            run.status, run.finished_at = "succeeded", timezone.now()
            run.save(update_fields=["result", "message", "status", "finished_at", "updated_at"])
            logger.info("Completed duplicate job=%s kind=%s", run.uuid, run.kind)
    except Exception as exc:
        logger.exception("Duplicate job error run_id=%s; see durable job status", run_id)
        message = (
            str(exc)
            if isinstance(exc, (ValueError, signing.BadSignature))
            else "Processing failed. No files were removed; prepare a new preview."
        )
        query.exclude(status__in=TERMINAL).update(status="failed", finished_at=timezone.now(), message=message)


def retry_delivery(run):
    """Republish lost deliveries; executing or completed jobs never run twice."""
    stale = timezone.now() - timedelta(minutes=2)
    updated = models.TaskRun.objects.filter(pk=run.pk, status="queued", updated_at__lt=stale).update(
        updated_at=timezone.now()
    )
    if run.status == "failed" and run.started_at is None:
        updated = models.TaskRun.objects.filter(pk=run.pk, status="failed", started_at__isnull=True).update(
            status="queued", finished_at=None, updated_at=timezone.now(), message="Retrying delivery to the API worker."
        )
    if updated:
        _publish(run.pk)
