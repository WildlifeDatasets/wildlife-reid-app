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
from django.db.models import Q
from django.utils import timezone

from .models import (
    AnimalObservation,
    EmbeddingProfile,
    MediafilesForIdentification,
    ObservationEmbedding,
    ObservationEmbeddingSchedule,
    TaskRun,
    WorkGroup,
)
from .observation_geometry import HAS_BBOX

logger = logging.getLogger(__name__)


def embedding_generation(workgroup_id):
    """Read the current reset generation, including workgroups without a timer."""
    return (
        ObservationEmbeddingSchedule.objects.filter(workgroup_id=workgroup_id)
        .values_list("generation", flat=True)
        .first()
        or 0
    )


def _workgroup_image_inputs(workgroup_id):
    return AnimalObservation.objects.filter(
        is_no_detection_placeholder=False,
        mediafile__parent__owner__workgroup_id=workgroup_id,
        mediafile__parent__import_finished=True,
        mediafile__media_type="image",
    ).filter(Q(mediafile__image_file__gt="") | Q(mediafile__mediafile__gt=""))


def backfill_observations(workgroup_id):
    """Select valid image bbox inputs from finished uploads, regardless of labels."""
    return _workgroup_image_inputs(workgroup_id).filter(HAS_BBOX)


def _remove_feature_images(images):
    """Remove only generated crops, leaving original media files untouched."""
    root = (Path(settings.MEDIA_ROOT) / "identification_feature_images").resolve()
    for image in images:
        path = (Path(settings.MEDIA_ROOT) / image).resolve()
        if path.is_relative_to(root) and path.suffix == ".png":
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove an obsolete feature image.", exc_info=True)


def delete_embedding_rows(embeddings):
    """Delete vectors and ALIKED data together, then remove their crops after commit."""
    with transaction.atomic():
        images = list(embeddings.exclude(feature_image="").values_list("feature_image", flat=True))
        _, counts = embeddings.delete()
        if images:
            transaction.on_commit(partial(_remove_feature_images, images), robust=True)
    return counts.get(ObservationEmbedding._meta.label, 0)


def invalidate_observation_features(observations):
    """Discard features and pending comparisons using a changed observation."""
    with transaction.atomic():
        MediafilesForIdentification.objects.filter(
            Q(observation__in=observations) | Q(top_mediafiles__reference_observation__in=observations),
            provenance="reid",
        ).delete()
        delete_embedding_rows(ObservationEmbedding.objects.filter(observation__in=observations))


@transaction.atomic
def reset_workgroup_embeddings(workgroup_id):
    """Clear all feature profiles and cancel pending work from the old generation."""
    from .embedding_runs import TERMINAL, fail_run
    from .identification_refresh import KIND

    WorkGroup.objects.select_for_update().get(pk=workgroup_id)
    schedule, _ = ObservationEmbeddingSchedule.objects.get_or_create(workgroup_id=workgroup_id)
    schedule.generation += 1
    schedule.task_id = None
    schedule.eta = None
    schedule.save(update_fields=["generation", "task_id", "eta"])
    WorkGroup.objects.filter(pk=workgroup_id).update(
        identification_initialized_model=None,
        identification_init_status="Not initiated",
        identification_init_message="",
    )
    run_ids = (
        TaskRun.objects.filter(workgroup_id=workgroup_id, kind="embedding")
        .exclude(status__in=TERMINAL)
        .values_list("pk", flat=True)
    )
    for run_id in list(run_ids):
        fail_run(run_id, "Embeddings were reset. Start a new run to generate them again.")
    for run in TaskRun.objects.filter(workgroup_id=workgroup_id, kind=KIND).exclude(status__in=TERMINAL):
        for step in run.result["steps"].values():
            if step["status"] in {"queued", "running"}:
                step.update(status="skipped", message="Image features were reset.")
        run.status = "failed"
        run.message = "Image features were reset."
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "message", "result", "finished_at", "updated_at"])
    scope = {"observation__mediafile__parent__owner__workgroup_id": workgroup_id}
    MediafilesForIdentification.objects.filter(provenance="reid", **scope).delete()
    return delete_embedding_rows(ObservationEmbedding.objects.filter(**scope))


def _publish_preparation(workgroup_id, task_id, eta):
    """Publish a timer after commit, retaining it for recovery if the broker fails."""
    signature("caidapp.tasks.prepare_observation_embeddings", args=[workgroup_id, task_id]).apply_async(
        task_id=task_id, eta=eta
    )


def schedule_embeddings(workgroup_id, delay_minutes=None):
    """Debounce input changes with one pending preparation task per workgroup."""
    if delay_minutes is None:
        delay_minutes = getattr(settings, "OBSERVATION_EMBEDDING_DELAY_MINUTES", 10)
    with transaction.atomic():
        workgroup = WorkGroup.objects.select_for_update().filter(pk=workgroup_id).first()
        if workgroup is None or not workgroup.identification_model_id:
            return None
        if not _workgroup_image_inputs(workgroup_id).exists():
            return None
        schedule, _ = ObservationEmbeddingSchedule.objects.get_or_create(workgroup=workgroup)
        pending = schedule.task_id
        eta = timezone.now() + timedelta(minutes=delay_minutes)
        if pending and schedule.eta and eta < schedule.eta:
            pending = None
        task_id = pending or str(uuid.uuid4())
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
        from .identification_refresh import active_refresh, create_refresh
        from .tasks import advance_identification_refresh

        # Reserve a durable run before consuming the timer, in the same transaction.
        if active_refresh(workgroup_id):
            return {"status": "waiting"}
        try:
            run = create_refresh(workgroup)
        except ValidationError:
            run = None
        ObservationEmbeddingSchedule.objects.filter(workgroup_id=workgroup_id).update(task_id=None, eta=None)
        if run:
            transaction.on_commit(lambda: advance_identification_refresh.delay(run.pk), robust=True)
    return {"status": "queued" if run else "empty", "run_id": run.pk if run else None}


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
    paths = [Path(source.path) for source in (mediafile.image_file, mediafile.mediafile) if source]
    path = next((path for path in paths if path.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"Observation {observation.pk} has no available source image for media file {mediafile.pk}."
        )
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
    generation = embedding_generation(workgroup_id)
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
        run = TaskRun.objects.get(pk=run_id)
        if run.status != "preparing" or run.model_snapshot.get("generation", 0) != generation:
            return None
        if run.trigger in {"automatic", "backfill"}:
            observations = observations.filter(pk__in=backfill_observations(workgroup_id))
        # Deleted/ineligible selected observations remain visible as skipped items.
        pending_items = run.items.filter(status="pending")
        selected = set(observations.values_list("pk", flat=True))
        pending_items.exclude(observation_id_snapshot__in=selected).update(
            status="skipped", message="Observation is missing or no longer eligible for this run."
        )
    inputs = []
    for index, observation in enumerate(observations.order_by("pk")):
        if run and index % 50 == 0 and not TaskRun.objects.filter(pk=run.pk, status="preparing").exists():
            return None
        try:
            row = snapshot(observation)
            inputs.append(row)
            if run:
                pending_items.filter(observation_id_snapshot=observation.pk).update(
                    input_fingerprint=row["input_fingerprint"], input_specification=row["input_specification"]
                )
        except (OSError, ValidationError):
            if not skip_invalid:
                raise
            if run:
                pending_items.filter(observation_id_snapshot=observation.pk).update(
                    status="skipped", message="Source media file or bounding box is unavailable or invalid."
                )
            logger.warning("Skipping unavailable embedding input for observation %s", observation.pk, exc_info=True)
    if not inputs:
        return None
    manifest = {
        "workgroup_id": workgroup.pk,
        "generation": generation,
        "model_source": run.model_snapshot["model_source"] if run else model.get_runtime_model_source(),
        "weights_path": run.model_snapshot["weights_path"] if run else model.get_runtime_checkpoint_path(),
        "observations": inputs,
    }
    directory = Path(settings.MEDIA_ROOT) / "observation_embeddings" / str(uuid.uuid4())
    directory.mkdir(parents=True)
    from .services.shared_identification import prepare_feature_manifest

    prepare_feature_manifest(manifest)
    manifest_path = directory / "input.json"
    manifest_path.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")
    if run:
        run.manifest_path = str(manifest_path)
        run.compute_task_id = str(uuid.uuid4())
        run.store_task_id = str(uuid.uuid4())
        run.status = "queued"
        if not TaskRun.objects.filter(pk=run.pk, status="preparing").update(
            manifest_path=run.manifest_path,
            compute_task_id=run.compute_task_id,
            store_task_id=run.store_task_id,
            status="queued",
            updated_at=timezone.now(),
        ):
            manifest_path.unlink()
            directory.rmdir()
            return None
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
    from .embedding_runs import TERMINAL

    path = Path(manifest_path)
    if run_id is not None:
        completed = TaskRun.objects.filter(pk=run_id, status__in=TERMINAL).first()
        if completed:
            if (
                (completed.status == "cancelled"
                 or completed.model_snapshot.get("generation", 0) != embedding_generation(completed.workgroup_id))
                and path.is_file()
            ):
                manifest = json.loads(path.read_text(encoding="utf-8"))
                _remove_feature_images(
                    row["feature_image"] for row in manifest["observations"] if row.get("feature_image")
                )
            return completed.result
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result_path = path.with_name("output.json")
    if output.get("output_path") != str(result_path):
        raise ValidationError("Unexpected embedding output path.")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    spec = result["profile"]
    if spec["model_source"] != manifest["model_source"]:
        raise ValidationError("Encoder returned an unexpected model source.")
    if spec.get("feature_config") != manifest.get("feature_config"):
        raise ValidationError("Encoder returned an unexpected identification feature profile.")
    submitted = {row["observation_id"]: row for row in manifest["observations"]}
    stored, stale = 0, 0
    with transaction.atomic():
        workgroup_id = manifest["workgroup_id"]
        WorkGroup.objects.select_for_update().get(pk=workgroup_id)
        if manifest.get("generation", 0) != embedding_generation(workgroup_id):
            images = [row["feature_image"] for row in submitted.values() if row.get("feature_image")]
            transaction.on_commit(partial(_remove_feature_images, images), robust=True)
            return {"stored": 0, "stale": len(submitted), "profile_uuid": None}
        run = None
        if run_id is not None:
            run = TaskRun.objects.select_for_update().get(pk=run_id)
            if run.status == "cancelled":
                images = [row["feature_image"] for row in submitted.values() if row.get("feature_image")]
                transaction.on_commit(partial(_remove_feature_images, images), robust=True)
            if run.status in TERMINAL:
                return run.result
        profile, _ = EmbeddingProfile.objects.get_or_create(
            fingerprint=fingerprint(spec), defaults={"specification": spec, "dimension": spec["dimension"]}
        )
        returned = {row["observation_id"]: row for row in result["embeddings"]}
        if len(returned) != len(result["embeddings"]) or not set(returned).issubset(submitted):
            raise ValidationError("Unexpected or duplicate result observations.")
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
                if "feature_image" in original:
                    transaction.on_commit(partial(_remove_feature_images, [original["feature_image"]]), robust=True)
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
            delete_embedding_rows(observation.embeddings.exclude(input_fingerprint=current["input_fingerprint"]))
            vector = row["vector"]
            if (
                not isinstance(vector, list)
                or len(vector) != profile.dimension
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector)
                or not any(v != 0 for v in vector)
            ):
                raise ValidationError("Encoder returned an invalid embedding vector.")
            defaults = {"input_specification": original["input_specification"], "vector": vector}
            if "feature_config" in manifest:
                from .services.shared_identification import validated_local_features

                defaults.update(validated_local_features(row, original))
            embedding, created = ObservationEmbedding.objects.get_or_create(
                observation=observation,
                profile=profile,
                input_fingerprint=original["input_fingerprint"],
                defaults=defaults,
            )
            if "feature_config" in manifest and not created and (
                embedding.local_features is None
                or not embedding.feature_image
                or not (Path(settings.MEDIA_ROOT) / embedding.feature_image).is_file()
            ):
                for field in ("vector", "local_features", "feature_image"):
                    setattr(embedding, field, defaults[field])
                embedding.save(update_fields=["vector", "local_features", "feature_image"])
            if defaults.get("feature_image") and defaults["feature_image"] != embedding.feature_image:
                transaction.on_commit(
                    partial(_remove_feature_images, [defaults["feature_image"]]), robust=True
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
            if run.trigger == "backfill" and (counts.get("computed") or counts.get("cached")):
                model_id = run.model_snapshot["model_id"]
                WorkGroup.objects.filter(pk=workgroup_id, identification_model_id=model_id).update(
                    identification_initialized_model_id=model_id,
                    identification_init_at=run.finished_at,
                    identification_init_status="Finished",
                    identification_init_message=run.message,
                )
    return {"stored": stored, "stale": stale, "profile_uuid": str(profile.uuid)}
