"""BBox reconciliation shared by background detection and interactive proposals."""

import json
import logging
import math
import uuid
from collections import defaultdict
from functools import partial
from pathlib import Path

from celery import signature
from django.conf import settings
from django.core.serializers.json import DjangoJSONEncoder
from django.db import transaction
from django.utils import timezone
from scipy.optimize import linear_sum_assignment

from . import models

logger = logging.getLogger("app")

BBOX_FIELDS = ("bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height")
DEFAULT_OPTIONS = {"detector": "auto", "confidence": 0.5, "min_iou": 0.2, "create_new": True, "unmatched": "clear"}


def source_name(mediafile):
    """Return the image derivative name, falling back to the original media file."""
    return (mediafile.image_file or mediafile.mediafile).name


def observation_snapshot(mediafile):
    """Compare all persisted annotation fields, including edits that omit updated_at."""
    name = source_name(mediafile)
    path = (Path(settings.MEDIA_ROOT) / name).resolve()
    root = Path(settings.MEDIA_ROOT).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Image path is outside the media directory")
    stat = path.stat()
    snapshot = {
        "source": name,
        "size": stat.st_size,
        "mtime": stat.st_mtime_ns,
        "media_type": mediafile.media_type,
        "updated_at": mediafile.updated_at,
        "observations": list(mediafile.observations.order_by("pk").values()),
    }
    return json.loads(json.dumps(snapshot, cls=DjangoJSONEncoder))


def xyxy(observation):
    """Convert a valid center-size observation bbox to corners, or return None."""
    cx, cy, w, h = (getattr(observation, field) for field in BBOX_FIELDS)
    if any(v is None or not math.isfinite(v) for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
        return None
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def iou(a, b):
    """Return the intersection-over-union of two corner-coordinate bounding boxes."""
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0


def match_boxes(old, new, threshold):
    """Maximum total IoU with explicit unmatched columns and forbidden weak pairs."""
    if not old or not new:
        return []
    scores = [[iou(a, b) for b in new] for a in old]
    costs = [[-s if s >= threshold and s > 0 else 1 for s in row] + [0] * len(old) for row in scores]
    rows, columns = linear_sum_assignment(costs)
    return [(int(r), int(c), scores[r][c]) for r, c in zip(rows, columns) if c < len(new) and costs[r][c] < 0]


def validate_detections(result):
    """Return successful detections or raise ValueError for malformed boxes or confidence."""
    if result.get("status") != "ok" or not isinstance(result.get("detections"), list):
        raise ValueError(result.get("message") or "Detector did not return a successful result")
    detections = result["detections"]
    for detection in detections:
        box = detection.get("bbox", [])
        if len(box) != 4 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
            for v in box
        ):
            raise ValueError("Detector returned invalid coordinates")
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Detector returned an empty bounding box")
        confidence = detection.get("confidence")
        if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Detector returned invalid confidence")
    return detections


def create_jobs(caiduser, mediafiles, options, *, purpose="replace", batch=None, observation_ids=None):
    """Create an idempotent batch for images and dispatch queued jobs after commit."""
    batch = batch or uuid.uuid4()
    jobs = []
    with transaction.atomic():
        models.CaIDUser.objects.select_for_update().get(pk=caiduser.pk)
        existing = list(models.BboxDetectionJob.objects.filter(batch=batch, requested_by=caiduser))
        if existing:
            return batch, existing
        selected_by_mediafile = defaultdict(list)
        if purpose == "fill" and observation_ids is not None:
            for mediafile_id, observation_id in models.AnimalObservation.objects.filter(
                pk__in=observation_ids, mediafile__in=mediafiles
            ).values_list("mediafile_id", "pk"):
                selected_by_mediafile[mediafile_id].append(observation_id)
        for mediafile in mediafiles.order_by("pk").distinct():
            if mediafile.media_type != "image":
                continue
            job_options = dict(options)
            if purpose == "fill":
                from .observation_geometry import HAS_BBOX

                targets = mediafile.observations.exclude(HAS_BBOX)
                if observation_ids is not None:
                    targets = targets.filter(pk__in=selected_by_mediafile[mediafile.pk])
                target_ids = list(targets.order_by("pk").values_list("pk", flat=True))
                if not target_ids and mediafile.observations.exists():
                    continue
                job_options["observation_ids"] = target_ids
            job = models.BboxDetectionJob(
                batch=batch,
                requested_by=caiduser,
                mediafile=mediafile,
                options=job_options,
                purpose=purpose,
                task_id=str(uuid.uuid4()),
            )
            try:
                job.snapshot = observation_snapshot(mediafile)
            except (OSError, ValueError, AttributeError):
                job.status = "failed"
                job.message = "The source image is unavailable. No observations were changed."
                job.finished_at = timezone.now()
            job.save()
            jobs.append(job)
            if job.status == "queued":
                transaction.on_commit(partial(dispatch_job, job.pk))
    return batch, jobs


def dispatch_job(job_id):
    """Publish a detection task with completion callbacks, marking dispatch failures locally."""
    job = models.BboxDetectionJob.objects.get(pk=job_id)
    try:
        logger.info(
            "Dispatching bbox job job_id=%s task_id=%s mediafile_id=%s source=%s",
            job.pk,
            job.task_id,
            job.mediafile_id,
            Path(job.snapshot["source"]).name,
        )
        result = signature(
            "redetect_bboxes", args=[job.snapshot["source"], job.options], queue="taxon_worker"
        ).apply_async(
            task_id=job.task_id,
            link=signature("caidapp.tasks.finish_bbox_detection", args=[job.pk], queue="celery"),
            link_error=signature("caidapp.tasks.fail_bbox_detection", args=[job.pk], immutable=True, queue="celery"),
        )
        logger.info(
            "BBox job published job_id=%s task_id=%s celery_state=%s",
            job.pk,
            job.task_id,
            result.state,
        )
    except Exception:
        logger.exception("Cannot dispatch bbox job job_id=%s task_id=%s", job_id, job.task_id)
        fail_job(job_id, "Could not send the detection task to the worker. Please try again.")


def fail_job(job_id, message):
    """Mark a still-queued job as failed and record its completion time."""
    updated = models.BboxDetectionJob.objects.filter(pk=job_id, status="queued").update(
        status="failed",
        message=message,
        finished_at=timezone.now(),
    )
    logger.warning("BBox job marked failed job_id=%s updated=%s message=%s", job_id, bool(updated), message)


def _set_box(observation, box, user, timestamp):
    values = (
        (None,) * 4 if box is None else ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1])
    )
    for field, value in zip(BBOX_FIELDS, values):
        setattr(observation, field, value)
    observation.updated_by = user
    observation.updated_at = timestamp
    if box is not None:
        observation.is_no_detection_placeholder = False
    observation.save()


def _invalidate_reid(mediafile, observations):
    models.MediafilesForIdentification.objects.filter(mediafile=mediafile).delete()
    models.MediafileIdentificationSuggestion.objects.filter(mediafile=mediafile).delete()
    if mediafile.used_for_init_identification or any(o.identity_is_representative for o in observations):
        from .tasks import schedule_init_identification_for_workgroup

        workgroup = mediafile.parent.owner.workgroup
        if workgroup is not None:
            models.WorkGroup.objects.filter(pk=workgroup.pk).update(identification_initialized_model=None)
            transaction.on_commit(
                partial(schedule_init_identification_for_workgroup, workgroup, delay_minutes=10), robust=True
            )
    mediafile.used_for_init_identification = False
    # Do not expose the old import-time crop after changing its underlying bbox.
    metadata = dict(mediafile.metadata_json) if isinstance(mediafile.metadata_json, dict) else {}
    metadata["detection_crop_stale"] = True
    mediafile.metadata_json = metadata


@transaction.atomic
def finish_job(job_id, result):
    """Apply a result once. A retry or stale result must never modify annotations."""
    job = models.BboxDetectionJob.objects.select_for_update(of=("self",)).select_related("requested_by").get(pk=job_id)
    if job.status != "queued":
        logger.info("Ignoring duplicate bbox callback job_id=%s status=%s", job.pk, job.status)
        return job.status
    logger.info(
        "BBox callback received job_id=%s task_id=%s mediafile_id=%s result_status=%s detections=%s",
        job.pk,
        job.task_id,
        job.mediafile_id,
        result.get("status") if isinstance(result, dict) else "invalid",
        (
            len(result.get("detections", []))
            if isinstance(result, dict) and isinstance(result.get("detections"), list)
            else "invalid"
        ),
    )
    mediafile = (
        models.MediaFile.objects.select_for_update(of=("self",))
        .select_related("parent__owner__workgroup")
        .get(pk=job.mediafile_id)
    )
    observations = list(mediafile.observations.select_for_update().order_by("pk"))
    try:
        detections = validate_detections(result)
    except (ValueError, TypeError, AttributeError) as exc:
        job.status, job.message = "failed", str(exc)
    else:
        try:
            unchanged = observation_snapshot(mediafile) == job.snapshot
        except (OSError, ValueError, AttributeError):
            unchanged = False
        if not models.MediaFile.objects.for_user(job.requested_by).filter(pk=mediafile.pk).exists():
            job.status, job.message = "conflict", "Access to this media file changed. No observations were changed."
        elif not unchanged:
            job.status, job.message = "conflict", "Media file or observations changed since detection started."
        elif job.purpose == "suggest":
            job.status, job.result = "succeeded", result
        elif job.purpose == "fill":
            job.result = fill_missing(mediafile, observations, detections, job)
            job.result["detector"] = result.get("detector")
            job.status = "review" if job.result["counts"]["review"] else "succeeded"
            job.message = job.result["message"]
        else:
            job.result = reconcile(mediafile, observations, detections, job)
            job.result["detector"] = result.get("detector")
            job.status = "succeeded"
    job.finished_at = timezone.now()
    job.save(update_fields=["status", "result", "message", "finished_at"])
    logger.info(
        "BBox job completed job_id=%s task_id=%s mediafile_id=%s status=%s message=%s counts=%s",
        job.pk,
        job.task_id,
        job.mediafile_id,
        job.status,
        job.message,
        job.result.get("counts") if isinstance(job.result, dict) else None,
    )
    return job.status


def fill_missing(mediafile, observations, detections, job):
    """Fill selected missing boxes only when object assignment is unambiguous."""
    from .observation_geometry import HAS_BBOX

    boxed_ids = set(mediafile.observations.filter(HAS_BBOX).values_list("pk", flat=True))
    existing_boxes = [xyxy(o) for o in observations if o.pk in boxed_ids]
    # Every detection overlapping an existing box is reserved for that object;
    # do not let multiple detections of it be assigned to a missing observation.
    candidates = [
        d for d in detections if not any(iou(box, d["bbox"]) >= job.options["min_iou"] for box in existing_boxes)
    ]
    targets = [o for o in observations if o.pk in job.options["observation_ids"] and o.pk not in boxed_ids]
    missing_objects = [o for o in observations if not o.is_no_detection_placeholder and o.pk not in boxed_ids]
    counts = {
        "updated": 0,
        "created": 0,
        "cleared": 0,
        "deleted": 0,
        "kept": len(boxed_ids),
        "ignored": len(detections) - len(candidates),
        "review": 0,
    }
    audit = []
    timestamp = timezone.now()
    message = "No new animal detections. Existing observations were preserved."
    if candidates:
        if len(missing_objects) == 1 and len(candidates) == 1 and missing_objects[0] in targets:
            observation = missing_objects[0]
            _set_box(observation, candidates[0]["bbox"], job.requested_by, timestamp)
            counts["updated"] = 1
            audit.append({"observation_id": observation.pk, "detection_index": 0})
            message = "Filled the missing bbox. Identity and taxon were preserved."
        elif not missing_objects and (not observations or any(o.is_no_detection_placeholder for o in targets)):
            placeholder = next((o for o in targets if o.is_no_detection_placeholder), None)
            for index, detection in enumerate(candidates):
                observation = (
                    placeholder
                    if index == 0 and placeholder is not None
                    else models.AnimalObservation(mediafile=mediafile)
                )
                created = observation.pk is None
                _set_box(observation, detection["bbox"], job.requested_by, timestamp)
                counts["created" if created else "updated"] += 1
                audit.append({"observation_id": observation.pk, "detection_index": index, "created": created})
            message = "Created observations from detections, reusing the selected placeholder when present."
        else:
            counts["review"] = len(targets)
            audit.extend({"observation_id": o.pk, "review": True} for o in targets)
            message = (
                "Assignment is ambiguous. No observations were changed. "
                "Open the media file detail and use Suggest BBox to assign detections manually."
            )
    elif not observations:
        models.AnimalObservation.objects.create(mediafile=mediafile, is_no_detection_placeholder=True)
        message = "No animals found. Created a no-detection placeholder."
    if counts["updated"] or counts["created"]:
        _invalidate_reid(mediafile, observations)
        mediafile.updated_at, mediafile.updated_by = timestamp, job.requested_by
        mediafile.save(update_fields=["updated_at", "updated_by", "used_for_init_identification", "metadata_json"])
    return {
        "counts": counts,
        "matches": audit,
        "detections": candidates,
        "all_detections": detections,
        "message": message,
        "provenance": "automatic",
        "options": job.options,
    }


def reconcile(mediafile, observations, detections, job):
    """Apply matched detections and unmatched policies, reuse placeholders, and invalidate ReID data."""
    options = job.options
    old = [o for o in observations if not o.is_no_detection_placeholder and xyxy(o) is not None]
    old_boxes = [xyxy(o) for o in old]
    new = [d["bbox"] for d in detections]
    matches = match_boxes(old_boxes, new, options["min_iou"])
    matched_ids, matched_new = set(), set()
    counts = {"updated": 0, "created": 0, "cleared": 0, "deleted": 0, "kept": 0, "ignored": 0, "review": 0}
    audit = []
    timestamp = timezone.now()
    for old_index, new_index, score in matches:
        observation = old[old_index]
        # Flag near-ties for review, while still using the globally optimal assignment.
        ambiguous = any(
            abs(iou(old_boxes[old_index], box) - score) < 0.05 for index, box in enumerate(new) if index != new_index
        )
        ambiguous = ambiguous or any(
            abs(iou(box, new[new_index]) - score) < 0.05 for index, box in enumerate(old_boxes) if index != old_index
        )
        _set_box(observation, new[new_index], job.requested_by, timestamp)
        matched_ids.add(observation.pk)
        matched_new.add(new_index)
        counts["updated"] += 1
        counts["review"] += int(ambiguous)
        audit.append(
            {"observation_id": observation.pk, "detection_index": new_index, "iou": score, "review": ambiguous}
        )
    placeholder = next((o for o in observations if o.is_no_detection_placeholder), None)
    for index, box in enumerate(new):
        if index in matched_new:
            continue
        if placeholder is not None:
            _set_box(placeholder, box, job.requested_by, timestamp)
            matched_ids.add(placeholder.pk)
            audit.append({"observation_id": placeholder.pk, "detection_index": index, "reused_placeholder": True})
            placeholder = None
            counts["updated"] += 1
        elif options["create_new"]:
            observation = models.AnimalObservation(mediafile=mediafile)
            _set_box(observation, box, job.requested_by, timestamp)
            audit.append({"observation_id": observation.pk, "detection_index": index, "created": True})
            counts["created"] += 1
        else:
            counts["ignored"] += 1
    # Only rows that had a bbox at launch participate in the unmatched policy.
    # Existing bbox-less annotations are preserved because geometry cannot identify them.
    for observation in old:
        if observation.pk in matched_ids:
            continue
        if options["unmatched"] == "delete":
            observation.delete()
            counts["deleted"] += 1
        elif options["unmatched"] == "clear":
            _set_box(observation, None, job.requested_by, timestamp)
            counts["cleared"] += 1
        else:
            counts["kept"] += 1
    if not mediafile.observations.exists():
        models.AnimalObservation.objects.create(mediafile=mediafile, is_no_detection_placeholder=True)
    if any(counts[key] for key in ("updated", "created", "cleared", "deleted")):
        _invalidate_reid(mediafile, observations)
        mediafile.updated_at, mediafile.updated_by = timestamp, job.requested_by
        mediafile.save(update_fields=["updated_at", "updated_by", "used_for_init_identification", "metadata_json"])
    return {"counts": counts, "matches": audit, "detections": detections, "provenance": "automatic", "options": options}
