"""Run and track the workgroup-wide refresh of identification results."""

import logging

from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Count
from django.utils import timezone

from . import embedding_runs, models, observation_comparisons
from .observation_embeddings import backfill_observations, embedding_generation
from .services.identity_review import model_version
from .services.shared_identification import reference_observations

KIND = "identification_refresh"
STEPS = {
    "features": "Calculate missing image features",
    "identify": "Suggest identities for unidentified observations",
    "unknown": "Find similar unidentified observations",
    "outliers": "Check for misassigned identities",
    "merges": "Find possible identity merges",
}
logger = logging.getLogger(__name__)


def active_refresh(workgroup_id):
    return models.TaskRun.objects.filter(workgroup_id=workgroup_id, kind=KIND).exclude(
        status__in=embedding_runs.TERMINAL
    ).first()


def create_refresh(workgroup, *, requested_by=None):
    """Create a refresh while the caller holds the workgroup lock."""
    model = model_version(workgroup.identification_model)
    if not model:
        raise ValidationError("Select an identification model first.")
    return models.TaskRun.objects.create(
        workgroup=workgroup, requested_by=requested_by, kind=KIND, status="queued",
        trigger="manual" if requested_by else "automatic",
        message="Waiting to calculate missing image features.",
        model_snapshot={**model, "generation": embedding_generation(workgroup.pk)},
        result={"steps": {step: {"status": "queued"} for step in STEPS}},
    )


@transaction.atomic
def request_refresh(workgroup_id, user):
    """Start now, or schedule one follow-up when a refresh already exists."""
    from .observation_embeddings import schedule_embeddings
    from .tasks import advance_identification_refresh

    group = models.WorkGroup.objects.select_for_update().get(pk=workgroup_id)
    run = active_refresh(group.pk)
    if run:
        schedule_embeddings(group.pk, delay_minutes=0)
        return run
    run = create_refresh(group, requested_by=user)
    models.ObservationEmbeddingSchedule.objects.filter(workgroup=group).update(task_id=None, eta=None)
    transaction.on_commit(lambda: advance_identification_refresh.delay(run.pk), robust=True)
    return run


def _launch(step, group):
    """Start the existing job and return its database record or a skip reason."""
    from . import tasks, views

    observations = backfill_observations(group.pk)
    if step == "features":
        if not observations.exists():
            return "No eligible observations with a saved bounding box."
        return embedding_runs.create_embedding_run(group.pk, trigger="backfill")
    if step == "unknown":
        if len(list(observations.filter(identity__isnull=True).values_list("pk", flat=True)[:2])) < 2:
            return "At least two unidentified observations with bounding boxes are required."
        return observation_comparisons.start(group, None, "compare_unknown", defer_submission=True)
    if step == "identify":
        if not reference_observations(group, include_non_representative=True).exists():
            return "No identified reference observations are available."
        unknown = observations.filter(identity__isnull=True)
        archive_ids = unknown.values("mediafile__parent_id")
        archives = models.UploadedArchive.objects.filter(pk__in=archive_ids)
        if not archives.exists():
            return "No unidentified observations with bounding boxes are available."
        return views.run_identification_bulk(
            group, uploaded_archives=archives, selection={
                "include_non_representative": True, "observation_ids": unknown.values_list("pk", flat=True),
            },
            defer_submission=True,
        ) or "No eligible unidentified observations for the selected taxon."

    known = models.AnimalObservation.objects.filter(
        mediafile__parent__owner__workgroup=group, mediafile__parent__import_finished=True,
        mediafile__media_type="image", identity__owner_workgroup=group, is_no_detection_placeholder=False,
    )
    if step == "outliers":
        enough_members = known.values("identity_id").annotate(members=Count("pk")).filter(members__gte=3)
        if len(list(enough_members.values_list("identity_id", flat=True)[:2])) < 2:
            return "At least two identities with three observations each are required."
        return tasks.run_identification_outlier_detection_for_workgroup(group, defer_submission=True)[0]
    if len(list(known.values_list("identity_id", flat=True).distinct()[:2])) < 2:
        return "At least two identified animals are required."
    return tasks.run_similar_identity_pairs_for_workgroup(group, defer_submission=True)


def _child_status(step, child_id):
    """Read completion from the same records used by standalone jobs."""
    if step in {"features", "unknown"}:
        child = models.TaskRun.objects.filter(pk=child_id).first()
        if child is None:
            return "failed"
        child = embedding_runs.refresh_run(child) if step == "features" else observation_comparisons.refresh(child)
        return child.status if child.status in embedding_runs.TERMINAL else None
    if step == "identify":
        child = models.IdentificationRunStatistic.objects.filter(pk=child_id).first()
        if child is None:
            return "failed"
        if child.finished_at is None:
            return None
        return "succeeded" if child.status == "finished" else "failed"
    model = models.IdentificationOutlierSuggestionResult if step == "outliers" else models.IdentificationSimilarPairResult
    child = model.objects.filter(pk=child_id).first()
    if child is None:
        return "failed"
    pixel_status = getattr(child, "pixel_status", "")
    if child.status in {"pending", "processing"} or pixel_status == "processing":
        return None
    if child.status == "done" and pixel_status != "error":
        return "succeeded"
    return "partial" if "done" in {child.status, pixel_status} else "failed"


@transaction.atomic
def advance(run_id):
    """Advance one step exactly once across concurrent or duplicate deliveries."""
    group_id = models.TaskRun.objects.filter(pk=run_id, kind=KIND).values_list("workgroup_id", flat=True).first()
    group = models.WorkGroup.objects.select_for_update(skip_locked=True).filter(pk=group_id).first()
    if group is None:
        return
    run = models.TaskRun.objects.select_for_update().filter(pk=run_id).exclude(
        status__in=embedding_runs.TERMINAL
    ).first()
    if run is None:
        return
    steps = run.result["steps"]
    now = timezone.now()
    changed = run.model_snapshot != {
        **model_version(group.identification_model), "generation": embedding_generation(group.pk),
    }
    for step, progress in steps.items():
        if progress["status"] != "running":
            continue
        status = _child_status(step, progress["record_id"])
        if status is None:
            return
        progress["status"] = status

    for step in STEPS:
        progress = steps[step]
        if progress["status"] != "queued":
            continue
        if changed or steps["features"]["status"] == "failed":
            progress.update(status="skipped", message=(
                "The model or feature generation changed." if changed else "Preparing image features failed."
            ))
            continue
        try:
            with transaction.atomic():
                child = _launch(step, group)
        except Exception:
            logger.exception("Could not start refresh step %s for run %s.", step, run.pk)
            progress.update(status="failed", message="Could not start this step. See the worker logs.")
            continue
        if isinstance(child, str):
            progress.update(status="skipped", message=child)
            continue
        progress.update(status="running", record_id=child.pk)
        run.status = "running"
        run.started_at = run.started_at or now
        run.message = STEPS[step]
        break
    else:
        statuses = {progress["status"] for progress in steps.values()}
        if changed or steps["features"]["status"] == "failed":
            run.status = "failed"
        elif statuses.intersection({"failed", "partial"}):
            run.status = "partial"
        else:
            run.status = "succeeded"
        run.finished_at = now
        if run.status == "failed":
            run.message = "Identification refresh could not finish."
        elif run.status == "partial":
            run.message = "Some identification steps need attention."
        else:
            run.message = "Identification refresh finished."
    run.save(update_fields=["result", "status", "started_at", "finished_at", "message", "updated_at"])
