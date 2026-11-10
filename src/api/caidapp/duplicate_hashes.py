"""Workgroup-scoped, incremental exact hashing and persistent duplicate catalogues."""

import itertools
import json
import logging
import math
import uuid
from datetime import timedelta
from functools import partial
from pathlib import Path

from celery import signature
from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.db import transaction
from django.db.models import Count, F, Q
from django.utils import timezone

from . import duplicate_services as review
from . import models

logger = logging.getLogger(__name__)
HASH_VERSION = "exact-v1"
HASH_FIELDS = (
    "file_sha256",
    "pixel_sha256",
    "reference_sha256",
    "hash_version",
    "hash_status",
    "hash_revision",
    "hash_details",
    "hash_message",
    "hash_computed_at",
)
CLEAR_HASHES = dict(
    file_sha256="",
    pixel_sha256="",
    reference_sha256="",
    hash_version="",
    hash_status="",
    hash_details={},
    hash_message="",
    hash_computed_at=None,
)


def group_media(workgroup_id):
    """Scope to actual membership and finished imports, including images without detections."""
    return models.MediaFile.objects.filter(
        parent__owner__workgroup_id=workgroup_id,
        parent__import_finished=True,
        media_type__in=["image", "video"],
    )


def hash_input(media):
    """Describe the immutable source generation expected by a batch callback."""
    return {
        "id": media.pk,
        "name": media.original_filename,
        "source": media.mediafile.name or "",
        "reference": (media.image_file.name or "") if media.media_type == "video" else "",
        "media_type": media.media_type,
        "revision": media.hash_revision,
    }


def _same_input(record, media):
    current = hash_input(media)
    return all(record.get(field) == current[field] for field in ("id", "source", "reference", "media_type", "revision"))


def invalidate_hashes(queryset):
    """Advance a source generation so late worker results can never revive old evidence."""
    ids = queryset.values("pk")
    models.DuplicateProposal.objects.filter(status="pending", members__mediafile_id__in=ids).update(status="stale")
    return queryset.update(hash_revision=F("hash_revision") + 1, **CLEAR_HASHES)


def batch_directory(batch):
    """Keep manifests and results outside public media URLs."""
    root = getattr(settings, "DUPLICATE_SCAN_ROOT", Path(settings.SHARED_DATA_PATH) / "duplicate_scans")
    return Path(root) / "hash_batches" / str(batch.uuid)


def _publish_timer(workgroup_id, task_id, eta):
    try:
        logger.info("Scheduling exact hashes group=%s task=%s eta=%s", workgroup_id, task_id, eta)
        signature("caidapp.tasks.prepare_duplicate_hashes", args=[workgroup_id, task_id], queue="celery").apply_async(
            task_id=task_id,
            eta=eta,
        )
    except Exception:
        logger.exception("Could not schedule exact hashes for group %s", workgroup_id)
        models.DuplicateHashSchedule.objects.filter(workgroup_id=workgroup_id, task_id=task_id).update(
            task_id=None,
            eta=None,
            message="Could not contact the task queue. Automatic maintenance will retry.",
        )


@transaction.atomic
def schedule_hashes(workgroup_id, immediate=False):
    """Coalesce committed changes; keep hashing independent of any identification model."""
    workgroup = models.WorkGroup.objects.select_for_update().filter(pk=workgroup_id).first()
    if workgroup is None:
        return None
    if (
        not group_media(workgroup_id).exists()
        and not models.DuplicateScan.objects.filter(workgroup_id=workgroup_id).exists()
    ):
        return None
    schedule, _ = models.DuplicateHashSchedule.objects.get_or_create(workgroup=workgroup)
    schedule.dirty = True
    if schedule.active_scan_id and schedule.active_scan.status == "queued":
        schedule.save(update_fields=["dirty"])
        return schedule
    delay = 0 if immediate else getattr(settings, "DUPLICATE_HASH_DELAY_MINUTES", 2)
    schedule.eta = timezone.now() + timedelta(minutes=delay)
    publish = not schedule.task_id or immediate
    if publish:
        schedule.task_id = str(uuid.uuid4())
    schedule.message = ""
    schedule.save()
    if publish:
        transaction.on_commit(partial(_publish_timer, workgroup_id, schedule.task_id, schedule.eta), robust=True)
    return schedule


@transaction.atomic
def request_hashes(user, force=False):
    """Expose only whole-workgroup requests; administrators alone can invalidate all cached hashes."""
    if not user.workgroup_id:
        raise PermissionDenied("Select a working group first")
    models.WorkGroup.objects.select_for_update().get(pk=user.workgroup_id)
    if force:
        if not user.workgroup_admin:
            raise PermissionDenied("Only a workgroup administrator can force recalculation")
        invalidate_hashes(group_media(user.workgroup_id))
    logger.info("Exact hash check requested group=%s force=%s", user.workgroup_id, force)
    recover_group(user.workgroup_id)
    return schedule_hashes(user.workgroup_id, immediate=True)


def _needs_hash():
    # Failed inputs retry periodically, without looping on every ordinary edit.
    return (
        ~Q(hash_version=HASH_VERSION)
        | Q(hash_status="")
        | Q(hash_status="error", hash_computed_at__lt=timezone.now() - timedelta(hours=6))
    )


@transaction.atomic
def prepare_hashes(workgroup_id, task_id):
    """Claim the latest timer and reserve small batches for all missing hashes, without a group-size limit."""
    group = models.WorkGroup.objects.select_for_update().filter(pk=workgroup_id).first()
    schedule = models.DuplicateHashSchedule.objects.filter(workgroup_id=workgroup_id).first()
    if group is None or schedule is None or schedule.task_id != task_id:
        return {"status": "superseded"}
    if schedule.eta and schedule.eta > timezone.now():
        schedule.task_id = str(uuid.uuid4())
        schedule.save(update_fields=["task_id"])
        transaction.on_commit(partial(_publish_timer, workgroup_id, schedule.task_id, schedule.eta), robust=True)
        return {"status": "deferred"}
    schedule.task_id = schedule.eta = None
    if schedule.active_scan_id and schedule.active_scan.status == "queued":
        schedule.save(update_fields=["task_id", "eta"])
        return {"status": "active"}
    candidates = group_media(workgroup_id).filter(_needs_hash()).order_by("pk")
    scan = None
    pending = []
    size = max(1, min(256, getattr(settings, "DUPLICATE_HASH_BATCH_SIZE", 64)))
    for media in candidates.iterator(chunk_size=size):
        if scan is None:
            scan = models.DuplicateScan.objects.create(
                workgroup=group,
                purpose="exact",
                include_crops=False,
                algorithm=HASH_VERSION,
            )
        pending.append(hash_input(media))
        scan.total += 1
        if len(pending) == size:
            models.DuplicateHashBatch.objects.create(scan=scan, items=pending)
            pending = []
    if scan is not None:
        if pending:
            models.DuplicateHashBatch.objects.create(scan=scan, items=pending)
        scan.save(update_fields=["total"])
        logger.info("Prepared exact hash run group=%s scan=%s files=%s", workgroup_id, scan.pk, scan.total)
        schedule.active_scan = scan
        transaction.on_commit(partial(dispatch_next_batch, scan.pk), robust=True)
    else:
        scan = models.DuplicateScan.objects.filter(workgroup=group, purpose="exact").order_by("-pk").first()
        if scan is None:
            scan = models.DuplicateScan.objects.create(
                workgroup=group,
                purpose="exact",
                include_crops=False,
                status="succeeded",
                algorithm=HASH_VERSION,
                finished_at=timezone.now(),
            )
        refresh_catalogue(group.pk, scan)
    schedule.dirty = False
    schedule.message = ""
    schedule.save()
    return {"status": "queued" if schedule.active_scan_id else "current", "scan_uuid": str(scan.uuid)}


@transaction.atomic
def dispatch_next_batch(scan_id):
    """Publish at most one batch per group, preserving durable pending work on failure."""
    scan = models.DuplicateScan.objects.get(pk=scan_id)
    models.WorkGroup.objects.select_for_update().get(pk=scan.workgroup_id)
    if scan.status != "queued" or scan.hash_batches.filter(status="queued").exists():
        return
    batch = scan.hash_batches.filter(status="pending").order_by("pk").first()
    if batch is None:
        return
    batch.status = "queued"
    batch.dispatched_at = timezone.now()
    batch.save(update_fields=["status", "dispatched_at"])
    transaction.on_commit(partial(_dispatch_batch, batch.pk), robust=True)


def _dispatch_batch(batch_id):
    batch = models.DuplicateHashBatch.objects.get(pk=batch_id)
    try:
        logger.info("Publishing exact hash batch=%s task=%s files=%s", batch.pk, batch.uuid, len(batch.items))
        directory = batch_directory(batch)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "manifest.tmp"
        temporary.write_text(json.dumps({"items": batch.items}), encoding="utf-8")
        temporary.replace(directory / "manifest.json")
        signature("duplicates.hash_files", args=[str(batch.uuid)], queue="duplicate_worker").apply_async(
            task_id=str(batch.uuid),
            link=signature("caidapp.tasks.finish_duplicate_hashes", args=[batch.pk], queue="celery"),
            link_error=signature(
                "caidapp.tasks.fail_duplicate_hashes", args=[batch.pk], immutable=True, queue="celery"
            ),
        )
        logger.info("Published exact hash batch=%s task=%s", batch.pk, batch.uuid)
    except Exception:
        logger.exception("Could not publish hash batch %s", batch.pk)
        fail_batch(batch.pk, "Could not contact the duplicate worker")


def _stat_matches(record, media):
    for path_key, stat_key in [("source", "source_stat"), ("reference", "visual_stat")]:
        if path_key == "reference" and media.media_type != "video":
            continue
        if not record.get(stat_key):
            continue
        try:
            stat = review.media_path(record[path_key]).stat()
            if [stat.st_size, stat.st_mtime_ns] != record[stat_key]:
                return False
        except (OSError, ValueError):
            return False
    return True


def _finish_run(scan, schedule):
    if scan.hash_batches.filter(status__in=["pending", "queued"]).exists():
        transaction.on_commit(partial(_queue_next_batch, scan.pk), robust=True)
        return
    scan.status = "succeeded"
    scan.finished_at = timezone.now()
    scan.message = f"{scan.processed} processed, {scan.failed_count} failed, {scan.skipped} changed inputs skipped."
    scan.save()
    # Release batch member locks before locking complete duplicate groups, so
    # concurrent forced recalculation and manual merges use a consistent order.
    transaction.on_commit(partial(_refresh_finished_catalogue, scan.pk), robust=True)
    rerun = schedule.dirty
    schedule.active_scan = None
    schedule.save(update_fields=["active_scan"])
    if rerun:
        transaction.on_commit(partial(schedule_hashes, scan.workgroup_id, immediate=True), robust=True)


def _queue_next_batch(scan_id):
    try:
        signature("caidapp.tasks.dispatch_duplicate_hash_batch", args=[scan_id], queue="celery").apply_async()
    except Exception:
        logger.exception("Could not continue exact hash scan %s; maintenance will retry", scan_id)


def _refresh_finished_catalogue(scan_id):
    scan = models.DuplicateScan.objects.get(pk=scan_id)
    try:
        refresh_catalogue(scan.workgroup_id, scan)
    except Exception:
        logger.exception("Could not refresh duplicate catalogue for scan %s", scan_id)
        models.DuplicateHashSchedule.objects.filter(workgroup_id=scan.workgroup_id).update(
            dirty=True,
            message="Fingerprints saved. Suggestion refresh will be retried automatically.",
        )


@transaction.atomic
def finish_batch(batch_id):
    """Apply each result once, accepting only unchanged source generations still in the same group."""
    batch = models.DuplicateHashBatch.objects.select_related("scan").get(pk=batch_id)
    models.WorkGroup.objects.select_for_update().get(pk=batch.scan.workgroup_id)
    batch.refresh_from_db()
    if batch.status != "queued" or batch.scan.status != "queued":
        return
    result = json.loads((batch_directory(batch) / "result.json").read_text(encoding="utf-8"))
    originals = {item["id"]: item for item in batch.items}
    records = {item["id"]: item for item in result["items"]}
    if result.get("version") != HASH_VERSION or set(records) != set(originals) or len(records) != len(result["items"]):
        raise ValueError("Hash result does not match the batch manifest")
    scan = models.DuplicateScan.objects.get(pk=batch.scan_id)
    schedule = models.DuplicateHashSchedule.objects.get(workgroup_id=scan.workgroup_id)
    mediafiles = {
        media.pk: media
        for media in group_media(scan.workgroup_id)
        .filter(pk__in=records)
        .select_for_update(of=("self",))
        .order_by("pk")
    }
    for key, record in records.items():
        media = mediafiles.get(key)
        if (
            media is None
            or not _same_input(originals[key], media)
            or not _same_input(record, media)
            or not _stat_matches(record, media)
        ):
            scan.skipped += 1
            schedule.dirty = True
            continue
        error = record.get("error", "")
        if not error and not record.get("file_sha256"):
            raise ValueError("Missing file hash in a successful worker result")
        for field in ("file_sha256", "pixel_sha256", "reference_sha256"):
            value = record.get(field, "")
            if value and (len(value) != 64 or any(char not in "0123456789abcdef" for char in value)):
                raise ValueError("Invalid SHA-256 result")
        models.MediaFile.objects.filter(pk=key, hash_revision=record["revision"]).update(
            file_sha256=record["file_sha256"],
            pixel_sha256=record["pixel_sha256"],
            reference_sha256=record["reference_sha256"],
            hash_version=HASH_VERSION,
            hash_status="error" if error else "ready",
            hash_details=record,
            hash_message=error or record.get("visual_error", ""),
            hash_computed_at=timezone.now(),
        )
        scan.failed_count += bool(error)
    scan.processed += len(records)
    scan.save()
    schedule.save(update_fields=["dirty"])
    batch.status = "succeeded"
    batch.save(update_fields=["status"])
    logger.info(
        "Stored exact hash batch=%s scan=%s processed=%s/%s failed=%s skipped=%s",
        batch.pk,
        scan.pk,
        scan.processed,
        scan.total,
        scan.failed_count,
        scan.skipped,
    )
    _finish_run(scan, schedule)


@transaction.atomic
def fail_batch(batch_id, message="The duplicate worker failed. Failed inputs will be retried later."):
    """Release a failed batch, continue other batches, and bound retries for unavailable inputs."""
    batch = models.DuplicateHashBatch.objects.select_related("scan").get(pk=batch_id)
    models.WorkGroup.objects.select_for_update().get(pk=batch.scan.workgroup_id)
    batch.refresh_from_db()
    if batch.status != "queued" or batch.scan.status != "queued":
        return
    scan = models.DuplicateScan.objects.get(pk=batch.scan_id)
    for item in batch.items:
        group_media(scan.workgroup_id).filter(pk=item["id"], hash_revision=item["revision"]).update(
            hash_status="error",
            hash_version=HASH_VERSION,
            hash_computed_at=timezone.now(),
            hash_message=message,
        )
    batch.status, batch.message = "failed", message
    batch.save(update_fields=["status", "message"])
    scan.processed += len(batch.items)
    scan.failed_count += len(batch.items)
    scan.save()
    _finish_run(scan, models.DuplicateHashSchedule.objects.get(workgroup_id=scan.workgroup_id))


def batch_progress(batch):
    """Read an atomic worker heartbeat; absence means delivery has not been acknowledged."""
    try:
        progress = json.loads((batch_directory(batch) / "progress.json").read_text(encoding="utf-8"))
        updated = float(progress["updated_at"])
        completed = int(progress["completed"])
        if not math.isfinite(updated) or not 0 <= completed <= len(batch.items):
            return None
        active = timezone.now().timestamp() - updated < getattr(settings, "DUPLICATE_HASH_DELIVERY_RETRY_SECONDS", 120)
        return {"updated_at": updated, "completed": completed, "total": len(batch.items), "active": active}
    except (OSError, ValueError, KeyError, TypeError):
        return None


@transaction.atomic
def retry_batch_delivery(batch_id):
    """Recover a published message lost with the broker without restarting the scan or its counters."""
    batch = models.DuplicateHashBatch.objects.select_related("scan").get(pk=batch_id)
    models.WorkGroup.objects.select_for_update().get(pk=batch.scan.workgroup_id)
    batch.refresh_from_db()
    if batch.status != "queued" or batch.scan.status != "queued":
        return False
    if (batch_directory(batch) / "result.json").is_file():
        return False
    progress = batch_progress(batch)
    last_activity = batch.dispatched_at.timestamp() if batch.dispatched_at else 0
    if progress:
        last_activity = max(last_activity, progress["updated_at"])
    timeout = getattr(settings, "DUPLICATE_HASH_DELIVERY_RETRY_SECONDS", 120)
    if timezone.now().timestamp() - last_activity < timeout:
        return False
    batch.dispatched_at = timezone.now()
    batch.message = "Retrying delivery: no recent acknowledgement from the duplicate worker."
    batch.save(update_fields=["dispatched_at", "message"])
    logger.warning(
        "Retrying exact hash delivery group=%s batch=%s task=%s; no recent worker heartbeat",
        batch.scan.workgroup_id,
        batch.pk,
        batch.uuid,
    )
    transaction.on_commit(partial(_dispatch_batch, batch.pk), robust=True)
    return True


def recover_group(workgroup_id):
    """Recover lost callbacks/publish attempts; expire only the currently dispatched batch."""
    schedule = models.DuplicateHashSchedule.objects.filter(workgroup_id=workgroup_id).first()
    if schedule and schedule.task_id and schedule.eta and schedule.eta < timezone.now() - timedelta(minutes=5):
        with transaction.atomic():
            models.WorkGroup.objects.select_for_update().get(pk=workgroup_id)
            schedule.refresh_from_db()
            if schedule.task_id and schedule.eta and schedule.eta < timezone.now() - timedelta(minutes=5):
                schedule.task_id = schedule.eta = None
                schedule.save(update_fields=["task_id", "eta"])
                schedule_hashes(workgroup_id, immediate=True)
    if not schedule or not schedule.active_scan_id or schedule.active_scan.status != "queued":
        return
    for batch in schedule.active_scan.hash_batches.filter(status="queued"):
        try:
            if (batch_directory(batch) / "result.json").is_file():
                finish_batch(batch.pk)
            elif batch.dispatched_at < timezone.now() - timedelta(hours=2):
                fail_batch(batch.pk, "Duplicate hashing timed out")
            else:
                retry_batch_delivery(batch.pk)
        except (OSError, ValueError, TypeError, KeyError):
            logger.exception("Invalid hash result for batch %s", batch.pk)
            fail_batch(batch.pk, "Could not read the duplicate worker result")
    dispatch_next_batch(schedule.active_scan_id)


def maintain_hashes():
    """Automatically backfill existing groups and recover after outages, without any ReID prerequisite."""
    groups = models.WorkGroup.objects.filter(caiduser__uploadedarchive__import_finished=True).distinct()
    for group_id in groups.values_list("pk", flat=True).iterator():
        try:
            recover_group(group_id)
            schedule_hashes(group_id)
        except Exception:
            logger.exception("Duplicate maintenance failed for group %s", group_id)


def _snapshot(media):
    return {
        "id": media.pk,
        "name": media.original_filename,
        "source": media.mediafile.name,
        "sha256": media.file_sha256,
        "pixel_hash": media.pixel_sha256,
        "size": media.hash_details.get("size"),
        "media_type": media.media_type,
    }


def _store_proposal(scan, kind, value, members):
    ids = sorted(media.pk for media in members)
    fingerprint = review.digest([scan.workgroup_id, kind, value, ids])
    proposal, created = models.DuplicateProposal.objects.get_or_create(
        fingerprint=fingerprint,
        defaults={"scan": scan, "kind": kind, "evidence": {"hash": value, "version": HASH_VERSION}},
    )
    if not created and proposal.status in {"dismissed", "resolved"}:
        return fingerprint
    if proposal.status != "pending":
        proposal.status = "pending"
        proposal.save(update_fields=["status"])
    for index, media in enumerate(members):
        models.DuplicateProposalMember.objects.update_or_create(
            proposal=proposal,
            original_id=media.pk,
            defaults={"mediafile": media, "snapshot": _snapshot(media), "position": index},
        )
    return fingerprint


@transaction.atomic
def refresh_catalogue(workgroup_id, scan):
    """Compare cached hashes against the entire group, including files hashed in previous runs."""
    models.WorkGroup.objects.select_for_update().get(pk=workgroup_id)
    ready = group_media(workgroup_id).filter(hash_status="ready", hash_version=HASH_VERSION)
    signatures = set()
    for field, kind in [("pixel_sha256", "pixel_exact"), ("file_sha256", "byte_exact")]:
        buckets = ready.exclude(**{field: ""}).values(field).annotate(count=Count("pk")).filter(count__gt=1)
        for bucket in buckets.iterator():
            members = list(ready.filter(**{field: bucket[field]}).select_for_update(of=("self",)).order_by("pk"))
            if field == "file_sha256" and all(media.pixel_sha256 for media in members):
                # A complete pixel group already represents this byte-identical subgroup.
                continue
            signatures.add(_store_proposal(scan, kind, bucket[field], members))
    reference_hashes = ready.exclude(reference_sha256="").values("reference_sha256").distinct()
    for bucket in reference_hashes.iterator():
        value = bucket["reference_sha256"]
        members = list(
            ready.filter(Q(reference_sha256=value) | Q(pixel_sha256=value))
            .select_for_update(of=("self",))
            .order_by("pk")
        )
        if len(members) > 1 and len({media.file_sha256 for media in members}) > 1:
            signatures.add(_store_proposal(scan, "reference_exact", value, members))
    models.DuplicateProposal.objects.filter(scan__workgroup_id=workgroup_id, status="pending").exclude(
        fingerprint__in=signatures,
    ).update(status="stale")
    refresh_identity_suggestions(workgroup_id)


def refresh_identity_suggestions(workgroup_id):
    """Suggest identities only for matching bboxes on different exact image copies, never merely co-occurrence."""
    ready = group_media(workgroup_id).filter(media_type="image", hash_status="ready", hash_version=HASH_VERSION)
    hashes = ready.exclude(pixel_sha256="").values("pixel_sha256").annotate(count=Count("pk")).filter(count__gt=1)
    observations = (
        models.AnimalObservation.objects.filter(
            mediafile__in=ready.filter(pixel_sha256__in=hashes.values("pixel_sha256")),
            identity__owner_workgroup_id=workgroup_id,
            is_no_detection_placeholder=False,
        )
        .values("pk", "identity_id", "mediafile_id", "mediafile__pixel_sha256", *review.BBOX_FIELDS)
        .order_by("pk")
    )
    locations = {}
    for observation in observations.iterator():
        box = [observation[field] for field in review.BBOX_FIELDS]
        if any(value is None or not math.isfinite(value) for value in box) or min(box[2:]) <= 0:
            continue
        cx, cy, width, height = box
        if cx - width / 2 < 0 or cy - height / 2 < 0 or cx + width / 2 > 1 or cy + height / 2 > 1:
            continue
        key = (observation["mediafile__pixel_sha256"], *(round(value, 6) for value in box))
        identities = locations.setdefault(key, {})
        representatives = identities.setdefault(observation["identity_id"], [])
        if len(representatives) < 2 and all(
            row["mediafile_id"] != observation["mediafile_id"] for row in representatives
        ):
            representatives.append(observation)
    excluded = models.MergeIdentitySuggestionExclusion.excluded_pair_ids(models.WorkGroup.objects.get(pk=workgroup_id))
    evidence = {}
    for (image_hash, *box), identities in locations.items():
        for first, second in itertools.combinations(sorted(identities), 2):
            if (first, second) in excluded:
                continue
            pair = next(
                (
                    (a, b)
                    for a in identities[first]
                    for b in identities[second]
                    if a["mediafile_id"] != b["mediafile_id"]
                ),
                None,
            )
            if pair is None:
                continue
            samples = evidence.setdefault((first, second), [])
            if len(samples) < 10:
                a, b = pair
                samples.append(
                    {
                        "image_hash": image_hash,
                        "bbox": box,
                        "observation_a": a["pk"],
                        "observation_b": b["pk"],
                        "mediafile_a": a["mediafile_id"],
                        "mediafile_b": b["mediafile_id"],
                    }
                )
    models.DuplicateIdentitySuggestion.objects.filter(workgroup_id=workgroup_id).update(active=False)
    for (first, second), samples in evidence.items():
        models.DuplicateIdentitySuggestion.objects.update_or_create(
            workgroup_id=workgroup_id,
            identity_a_id=first,
            identity_b_id=second,
            defaults={"active": True, "evidence": samples},
        )
