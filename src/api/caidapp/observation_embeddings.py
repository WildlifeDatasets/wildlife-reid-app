"""Prepare versioned observation inputs and persist asynchronous encoder results."""

import hashlib
import json
import logging
import math
import uuid
from datetime import timedelta
from functools import partial
from pathlib import Path

from celery import signature
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from .models import AnimalObservation, EmbeddingProfile, ObservationEmbedding, ObservationEmbeddingSchedule, WorkGroup

logger = logging.getLogger(__name__)


def _publish_preparation(workgroup_id, task_id, eta):
    """Publish a timer after commit; release its reservation if the broker fails."""
    try:
        signature("caidapp.tasks.prepare_observation_embeddings", args=[workgroup_id, task_id]).apply_async(
            task_id=task_id, eta=eta
        )
    except Exception:
        ObservationEmbeddingSchedule.objects.filter(workgroup_id=workgroup_id, task_id=task_id).update(
            task_id=None, eta=None
        )
        raise


def schedule_embeddings(workgroup_id, delay_minutes=None):
    """Debounce input changes with one pending preparation task per workgroup."""
    if delay_minutes is None:
        delay_minutes = getattr(settings, "OBSERVATION_EMBEDDING_DELAY_MINUTES", 10)
    with transaction.atomic():
        workgroup = WorkGroup.objects.select_for_update().filter(pk=workgroup_id).first()
        if workgroup is None or not workgroup.identification_model_id:
            return None
        if not AnimalObservation.objects.filter(
            mediafile__parent__owner__workgroup_id=workgroup_id,
            mediafile__parent__import_finished=True,
            mediafile__media_type="image",
            is_no_detection_placeholder=False,
        ).exists():
            return None
        schedule, _ = ObservationEmbeddingSchedule.objects.get_or_create(workgroup=workgroup)
        pending = schedule.task_id
        task_id = pending or str(uuid.uuid4())
        eta = timezone.now() + timedelta(minutes=delay_minutes)
        ObservationEmbeddingSchedule.objects.filter(workgroup_id=workgroup_id).update(task_id=task_id, eta=eta)
        if not pending:
            transaction.on_commit(partial(_publish_preparation, workgroup_id, task_id, eta), robust=True)
    return task_id


def prepare_scheduled_embeddings(workgroup_id, task_id):
    """Claim a current timer, or defer it until the last change's quiet period ends."""
    with transaction.atomic():
        workgroup = WorkGroup.objects.select_for_update().filter(pk=workgroup_id).first()
        schedule = ObservationEmbeddingSchedule.objects.filter(workgroup_id=workgroup_id).first()
        if workgroup is None or schedule is None or schedule.task_id != task_id:
            return {"status": "superseded"}
        if schedule.eta and schedule.eta > timezone.now():
            next_id = str(uuid.uuid4())
            ObservationEmbeddingSchedule.objects.filter(workgroup_id=workgroup_id).update(task_id=next_id)
            transaction.on_commit(partial(_publish_preparation, workgroup_id, next_id, schedule.eta), robust=True)
            return {"status": "deferred", "task_id": next_id}
        from .embedding_runs import create_embedding_run

        # Reserve a durable run before consuming the timer, in the same transaction.
        try:
            run = create_embedding_run(workgroup_id, trigger="automatic")
        except ValidationError:
            run = None
        ObservationEmbeddingSchedule.objects.filter(workgroup_id=workgroup_id).update(task_id=None, eta=None)
    return {"status": "queued" if run else "empty", "task_id": run.prepare_task_id if run else None}


def fingerprint(specification):
    """Hash a canonical JSON specification, rejecting non-finite numbers."""
    return hashlib.sha256(
        json.dumps(specification, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def snapshot(observation):
    """Describe the exact image input; videos need an explicit future frame locator."""
    mediafile = observation.mediafile
    if observation.is_no_detection_placeholder or mediafile.media_type != "image":
        raise ValidationError("Embedding inputs must be real observations in image media files.")
    source = mediafile.image_file or mediafile.mediafile
    if not source:
        raise ValidationError("Observation has no source media file.")
    path = Path(source.path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    bbox = [observation.bbox_x_center, observation.bbox_y_center, observation.bbox_width, observation.bbox_height]
    if any(value is not None for value in bbox):
        if any(value is None or not math.isfinite(value) for value in bbox) or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValidationError("Embedding bbox must be complete, finite and have positive dimensions.")
    specification = {
        "mediafile_id": mediafile.pk,
        "source_sha256": digest.hexdigest(),
        "bbox_cxcywh": bbox,
        "orientation": observation.orientation,
    }
    return {
        "observation_id": observation.pk,
        "image_path": str(path),
        "input_specification": specification,
        "input_fingerprint": fingerprint(specification),
    }


def enqueue_embeddings(
    workgroup_id, observation_ids=None, *, skip_invalid=False, completed_uploads_only=False, run_id=None
):
    """Queue selected image observations and supply existing cache keys to the worker."""
    workgroup = WorkGroup.objects.select_related("identification_model").get(pk=workgroup_id)
    model = workgroup.identification_model
    if model is None and run_id is None:
        raise ValidationError("Select an identification model first.")
    observations = AnimalObservation.objects.filter(
        mediafile__parent__owner__workgroup=workgroup,
        mediafile__media_type="image",
        is_no_detection_placeholder=False,
    ).select_related("mediafile")
    if completed_uploads_only:
        observations = observations.filter(mediafile__parent__import_finished=True)
    if observation_ids is not None:
        requested = set(observation_ids)
        observations = observations.filter(pk__in=requested)
        if run_id is None and set(observations.values_list("pk", flat=True)) != requested:
            raise ValidationError("Some requested observations are missing, ineligible or outside this workgroup.")
    run = None
    if run_id is not None:
        from .models import TaskRun

        run = TaskRun.objects.get(pk=run_id)
        # Deleted/ineligible selected observations remain visible as skipped items.
        selected = set(observations.values_list("pk", flat=True))
        run.items.exclude(observation_id_snapshot__in=selected).update(
            status="skipped", message="Observation is missing, is a placeholder or belongs to a video."
        )
    inputs = []
    for observation in observations.order_by("pk"):
        try:
            row = snapshot(observation)
            inputs.append(row)
            if run:
                run.items.filter(observation_id_snapshot=observation.pk).update(
                    input_fingerprint=row["input_fingerprint"], input_specification=row["input_specification"]
                )
        except (OSError, ValidationError):
            if not skip_invalid:
                raise
            if run:
                run.items.filter(observation_id_snapshot=observation.pk).update(
                    status="skipped", message="Source media file or bounding box is unavailable or invalid."
                )
            logger.warning("Skipping unavailable embedding input for observation %s", observation.pk, exc_info=True)
    if not inputs:
        return None
    cached = {}
    for profile_key, observation_id, input_key in ObservationEmbedding.objects.filter(
        observation_id__in=[row["observation_id"] for row in inputs]
    ).values_list("profile__fingerprint", "observation_id", "input_fingerprint"):
        cached.setdefault(profile_key, []).append(f"{observation_id}:{input_key}")
    manifest = {
        "workgroup_id": workgroup.pk,
        "model_source": run.model_snapshot["model_source"] if run else model.get_runtime_model_source(),
        "weights_path": run.model_snapshot["weights_path"] if run else model.get_runtime_checkpoint_path(),
        "observations": inputs,
        "cached": cached,
    }
    directory = Path(settings.MEDIA_ROOT) / "observation_embeddings" / str(uuid.uuid4())
    directory.mkdir(parents=True)
    manifest_path = directory / "input.json"
    manifest_path.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")
    if run:
        run.manifest_path = str(manifest_path)
        run.compute_task_id = str(uuid.uuid4())
        run.store_task_id = str(uuid.uuid4())
        run.status = "queued"
        run.save()
    callback = signature(
        "caidapp.tasks.store_observation_embeddings",
        kwargs={"manifest_path": str(manifest_path), **({"run_id": run.pk} if run else {})},
    )
    if run:
        callback.set(task_id=run.store_task_id)
    return signature(
        "compute_observation_embeddings", kwargs={"manifest_path": str(manifest_path)}, queue="identification_worker"
    ).apply_async(link=callback, **({"task_id": run.compute_task_id} if run else {}))


def store_embeddings(output, manifest_path, run_id=None):
    """Validate worker results against submitted inputs and discard stale inputs."""
    if run_id is not None:
        from .models import TaskRun

        completed = TaskRun.objects.filter(pk=run_id, status__in=["succeeded", "partial", "failed"]).first()
        if completed:
            return completed.result
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    result_path = Path(manifest_path).with_name("output.json")
    if output.get("output_path") != str(result_path):
        raise ValidationError("Unexpected embedding output path.")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    spec = result["profile"]
    if spec["model_source"] != manifest["model_source"]:
        raise ValidationError("Encoder returned an unexpected model source.")
    submitted = {row["observation_id"]: row for row in manifest["observations"]}
    stored, stale = 0, 0
    with transaction.atomic():
        profile, _ = EmbeddingProfile.objects.get_or_create(
            fingerprint=fingerprint(spec), defaults={"specification": spec, "dimension": spec["dimension"]}
        )
        returned = {row["observation_id"]: row for row in result["embeddings"]}
        if len(returned) != len(result["embeddings"]) or not set(returned).issubset(submitted):
            raise ValidationError("Unexpected or duplicate result observations.")
        run = None
        if run_id is not None:
            from .models import TaskRun

            run = TaskRun.objects.select_for_update().get(pk=run_id)
            if run.status in {"succeeded", "partial", "failed"}:
                return run.result
        for observation_id, original in submitted.items():
            row = returned.get(observation_id)
            if row is None:
                cache_key = f"{observation_id}:{original['input_fingerprint']}"
                if cache_key not in manifest.get("cached", {}).get(profile.fingerprint, []):
                    raise ValidationError("Encoder result does not cover all submitted inputs.")
                row = {"observation_id": observation_id, "input_fingerprint": original["input_fingerprint"]}

            original = submitted[row["observation_id"]]
            if row["input_fingerprint"] != original["input_fingerprint"]:
                raise ValidationError("Encoder input fingerprint does not match the submitted input.")
            observation = (
                AnimalObservation.objects.select_for_update()
                .select_related("mediafile")
                .filter(pk=row["observation_id"], mediafile__parent__owner__workgroup_id=manifest["workgroup_id"])
                .first()
            )
            try:
                current = snapshot(observation) if observation is not None else None
            except (OSError, ValidationError):
                current = None
            if current is None or current["input_fingerprint"] != original["input_fingerprint"]:
                stale += 1
                if run:
                    run.items.filter(observation_id_snapshot=observation_id).update(
                        status="stale", message="Observation input changed or was removed during computation."
                    )
                continue
            if "vector" not in row:
                embedding = ObservationEmbedding.objects.filter(
                    observation=observation, profile=profile, input_fingerprint=original["input_fingerprint"]
                ).first()
                if embedding is None:
                    raise ValidationError("Cached embedding is no longer available.")
                if run:
                    run.items.filter(observation_id_snapshot=observation_id).update(
                        status="cached", embedding=embedding
                    )
                continue
            vector = row["vector"]
            if (
                not isinstance(vector, list)
                or len(vector) != profile.dimension
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector)
                or not any(v != 0 for v in vector)
            ):
                raise ValidationError("Encoder returned an invalid embedding vector.")
            embedding, created = ObservationEmbedding.objects.get_or_create(
                observation=observation,
                profile=profile,
                input_fingerprint=original["input_fingerprint"],
                defaults={"input_specification": original["input_specification"], "vector": vector},
            )
            stored += int(created)
            if run:
                run.items.filter(observation_id_snapshot=observation_id).update(status="computed", embedding=embedding)
        if run:
            from collections import Counter

            counts = dict(Counter(run.items.values_list("status", flat=True)))
            run.result = {**counts, "profile_uuid": str(profile.uuid)}
            run.status = "partial" if counts.get("stale", 0) or counts.get("skipped", 0) else "succeeded"
            run.finished_at = timezone.now()
            run.message = (
                "Embedding results saved." if run.status == "succeeded" else "Some inputs were skipped or changed."
            )
            run.save()
    return {"stored": stored, "stale": stale, "profile_uuid": str(profile.uuid)}
