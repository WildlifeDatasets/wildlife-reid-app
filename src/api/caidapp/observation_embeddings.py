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
        ObservationEmbeddingSchedule.objects.filter(workgroup_id=workgroup_id).update(task_id=None, eta=None)
    task = enqueue_embeddings(workgroup_id, skip_invalid=True, completed_uploads_only=True)
    return {"status": "queued" if task else "empty", "task_id": task.id if task else None}


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


def enqueue_embeddings(workgroup_id, observation_ids=None, *, skip_invalid=False, completed_uploads_only=False):
    """Queue selected image observations and supply existing cache keys to the worker."""
    workgroup = WorkGroup.objects.select_related("identification_model").get(pk=workgroup_id)
    model = workgroup.identification_model
    if model is None:
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
        if set(observations.values_list("pk", flat=True)) != requested:
            raise ValidationError("Some requested observations are missing, ineligible or outside this workgroup.")
    inputs = []
    for observation in observations.order_by("pk"):
        try:
            inputs.append(snapshot(observation))
        except (OSError, ValidationError):
            if not skip_invalid:
                raise
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
        "model_source": model.get_runtime_model_source(),
        "weights_path": model.get_runtime_checkpoint_path(),
        "observations": inputs,
        "cached": cached,
    }
    directory = Path(settings.MEDIA_ROOT) / "observation_embeddings" / str(uuid.uuid4())
    directory.mkdir(parents=True)
    manifest_path = directory / "input.json"
    manifest_path.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")
    return signature(
        "compute_observation_embeddings", kwargs={"manifest_path": str(manifest_path)}, queue="identification_worker"
    ).apply_async(
        link=signature("caidapp.tasks.store_observation_embeddings", kwargs={"manifest_path": str(manifest_path)})
    )


def store_embeddings(output, manifest_path):
    """Validate worker results against submitted inputs and discard stale inputs."""
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
        for row in result["embeddings"]:
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
                continue
            vector = row["vector"]
            if (
                not isinstance(vector, list)
                or len(vector) != profile.dimension
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector)
                or not any(v != 0 for v in vector)
            ):
                raise ValidationError("Encoder returned an invalid embedding vector.")
            _, created = ObservationEmbedding.objects.get_or_create(
                observation=observation,
                profile=profile,
                input_fingerprint=original["input_fingerprint"],
                defaults={"input_specification": original["input_specification"], "vector": vector},
            )
            stored += int(created)
    return {"stored": stored, "stale": stale, "profile_uuid": str(profile.uuid)}
