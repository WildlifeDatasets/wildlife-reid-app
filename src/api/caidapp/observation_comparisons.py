"""Prepare, retain and review observation comparison runs."""

import csv
import uuid

from celery import shared_task
from celery.result import AsyncResult
from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from .models import AnimalObservation, IndividualIdentity, MediaFile, TaskRun, WorkGroup
from .observation_embeddings import backfill_observations, embedding_generation
from .services.identity_review import input_version, model_version

COMPARISON_LABELS = {
    "compare_unknown": "Group unidentified observations",
    "review_known": "Review assigned identities",
}
TERMINAL = {"succeeded", "failed"}


def start(workgroup, user, mode, *, include_non_representative=False, defer_submission=False):
    """Queue a durable run for the selected model and workgroup."""
    model = model_version(workgroup.identification_model)
    if not model or mode not in COMPARISON_LABELS:
        raise ValidationError("Select an identification model and comparison method.")
    run = TaskRun.objects.create(
        workgroup=workgroup, requested_by=user, kind=mode, status="queued",
        prepare_task_id=str(uuid.uuid4()),
        model_snapshot={**model, "generation": embedding_generation(workgroup.pk),
                        "include_non_representative": include_non_representative},
    )
    def publish():
        try:
            prepare.apply_async(
                args=[run.pk], task_id=run.prepare_task_id,
                link_error=fail.si(run.pk),
            )
        except Exception:
            fail(run.pk)
            run.refresh_from_db()

    if defer_submission:
        transaction.on_commit(publish)
    else:
        publish()
    return run


@shared_task(bind=True)
def prepare(self, run_id):
    """Build the query selection and reuse the shared feature pipeline."""
    from .tasks import identification_feature_job

    run = TaskRun.objects.select_related("workgroup__identification_model").get(pk=run_id)
    if run.status in TERMINAL:
        return
    group = run.workgroup
    if model_version(group.identification_model) != {
        key: run.model_snapshot[key] for key in ("id", "source", "checkpoint")
    }:
        raise ValidationError("The selected identification model changed.")
    queries = backfill_observations(group.pk)
    if run.kind == "compare_unknown":
        queries = queries.filter(identity__isnull=True)
    else:
        queries = queries.filter(identity__owner_workgroup=group)
    ids = list(queries.order_by("pk").values_list("pk", flat=True))
    if not ids:
        TaskRun.objects.filter(pk=run_id).update(
            status="succeeded", message="No eligible observations.", finished_at=timezone.now()
        )
        return
    # Keep each attempt's manifest independent of other deliveries.
    metadata = group.file_path(f"observation_comparisons/{run.uuid}/{uuid.uuid4().hex}/observations.csv")
    metadata.parent.mkdir(parents=True, exist_ok=True)
    with metadata.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["observation_id"])
        writer.writerows((pk,) for pk in ids)
    TaskRun.objects.filter(pk=run_id, status="queued").update(
        status="running", started_at=timezone.now(), updated_at=timezone.now()
    )
    comparison = identification_feature_job(
        group, group.identification_model, metadata, run.kind,
        generation=run.model_snapshot["generation"],
        include_non_representative=run.model_snapshot["include_non_representative"],
    )
    return self.replace(comparison | finish.s(run_id))


@shared_task
def finish(output, run_id):
    """Store the ranked observations and retain them for manual review."""
    rows = output["rows"]
    TaskRun.objects.filter(pk=run_id).exclude(status__in=TERMINAL).update(
        status="succeeded",
        message=f"Found comparison results for {len(rows)} observations.",
        result={"rows": rows},
        finished_at=timezone.now(),
        updated_at=timezone.now(),
    )


@shared_task
def fail(run_id):
    TaskRun.objects.filter(pk=run_id).exclude(status__in=TERMINAL).update(
        status="failed", message="Observation comparison failed. Check the inputs and start a new run.",
        finished_at=timezone.now(), updated_at=timezone.now(),
    )


def refresh(run):
    """Recover a terminal broker outcome when the error callback was lost."""
    if run.status in TERMINAL or not run.prepare_task_id:
        return run
    try:
        state = AsyncResult(run.prepare_task_id).state
    except Exception:
        return run
    if state in {"FAILURE", "REVOKED"}:
        fail(run.pk)
        run.refresh_from_db()
    return run


@transaction.atomic
def group_pair(run, query_id, candidate_id, user):
    """Create or extend a temporary identity after confirming one saved pair."""
    group = WorkGroup.objects.select_for_update().get(pk=run.workgroup_id)
    run = TaskRun.objects.select_for_update().get(pk=run.pk, kind="compare_unknown", status="succeeded")
    if run.model_snapshot["generation"] != embedding_generation(group.pk):
        raise ValidationError("The embeddings were reset. Start a new comparison.")
    evidence = {}
    for row in run.result.get("rows", []):
        if row["query"]["id"] == query_id:
            for candidate in row["candidates"]:
                if candidate["id"] == candidate_id:
                    evidence = {query_id: row["query"], candidate_id: candidate}
                    break
            break
    if len(evidence) != 2:
        raise ValidationError("This pair is no longer available.")
    # Bbox editors lock media files before individual observations.
    media_ids = AnimalObservation.objects.filter(pk__in=evidence).values("mediafile_id")
    locked_media = list(MediaFile.objects.select_for_update(of=("self",)).filter(
        pk__in=media_ids, parent__owner__workgroup=group
    ).order_by("pk").values_list("pk", flat=True))
    observations = list(AnimalObservation.objects.select_for_update(of=("self",)).filter(
        pk__in=evidence, mediafile_id__in=locked_media, is_no_detection_placeholder=False,
    ).select_related("mediafile").order_by("pk"))
    if len(observations) != 2:
        raise ValidationError("An observation is no longer available.")
    assigned = run.result.setdefault("assigned", {})
    for observation in observations:
        saved = assigned.get(str(observation.pk), evidence[observation.pk])
        if (observation.identity_id != saved.get("assigned_identity_id")
                or input_version(observation, observation.mediafile) != saved["version"]):
            raise ValidationError("An observation changed. Start a new comparison.")
    assigned_ids = {observation.identity_id for observation in observations if observation.identity_id}
    if len(assigned_ids) > 1:
        raise ValidationError("This pair is already assigned to two different identities.")
    identity = (
        IndividualIdentity.objects.get(pk=assigned_ids.pop(), owner_workgroup=group)
        if assigned_ids else IndividualIdentity.objects.create(
            owner_workgroup=group, name=f"Unknown-{uuid.uuid4().hex[:8]}"
        )
    )
    for observation in observations:
        if observation.identity_id is None:
            observation.identity = identity
            observation.identity_is_representative = False
            observation.updated_by = user
            observation.updated_at = timezone.now()
            observation.save(update_fields=["identity", "identity_is_representative", "updated_by", "updated_at"])
        assigned[str(observation.pk)] = {
            "assigned_identity_id": identity.pk, "version": input_version(observation, observation.mediafile),
        }
    run.save(update_fields=["result", "updated_at"])
    return identity
