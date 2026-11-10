"""Duplicate scans and explicit, transactional whole-media deduplication."""

import hashlib
import json
import logging
import uuid
from datetime import timedelta
from functools import partial
from pathlib import Path

from celery import signature
from django.conf import settings
from django.core import signing
from django.core.serializers.json import DjangoJSONEncoder
from django.db import transaction
from django.db.models import Q
from django.utils import timezone

from . import models

logger = logging.getLogger(__name__)
MAX_FILES = 100
FILE_FIELDS = ("mediafile", "image_file", "preview", "thumbnail", "static_thumbnail")
BBOX_FIELDS = ("bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height")


def serializable(value):
    """Normalize database values to stable JSON for signed previews and audit records."""
    return json.loads(json.dumps(value, cls=DjangoJSONEncoder))


def digest(value):
    """Compute an unambiguous digest of a merge snapshot."""
    return hashlib.sha256(json.dumps(serializable(value), sort_keys=True).encode()).hexdigest()


def media_path(name):
    """Resolve only files below the media root, including protection against symlinks."""
    root = Path(settings.MEDIA_ROOT).resolve()
    path = (root / name).resolve()
    if not name or path == root or not path.is_relative_to(root):
        raise ValueError("Invalid media file path")
    return path


def owned_file_names(media):
    """Collect this source's exact files and conventional derivatives, never an entire upload directory."""
    names = {getattr(media, field).name for field in FILE_FIELDS if getattr(media, field).name}
    if not media.parent_id or not media.mediafile.name:
        return names
    root = Path(settings.MEDIA_ROOT).resolve()
    basename = Path(media.mediafile.name).name
    try:
        output = media_path(media.parent.outputdir)
        variants = [
            output / "previews" / Path(basename).with_suffix(".mp4" if media.media_type == "video" else ".webp"),
            output / "thumbnails" / Path(basename).with_suffix(".webp"),
            output / "static_thumbnails" / Path(basename).with_suffix(".webp"),
        ]
        detection = output / "detection_images"
        variants.append(detection / basename)
        if detection.is_dir():
            stem, suffix = Path(basename).stem, Path(basename).suffix
            for candidate in detection.iterdir():
                middle = candidate.name[len(stem) + 1 : -len(suffix)] if suffix else ""
                if candidate.name.startswith(stem + ".") and candidate.suffix == suffix and middle.isdigit():
                    variants.append(candidate)
        for candidate in variants:
            resolved = candidate.resolve()
            if resolved.is_relative_to(root) and resolved.is_file():
                names.add(resolved.relative_to(root).as_posix())
    except (OSError, ValueError):
        logger.warning("Could not enumerate conventional derivatives for media file %s", media.pk)
    return names


def _file_is_shared(name):
    query = Q()
    for field in FILE_FIELDS:
        query |= Q(**{field: name})
    if (
        models.MediaFile.objects.filter(query).exists()
        or models.UploadedArchive.objects.filter(thumbnail=name).exists()
    ):
        return True
    path = Path(name)
    if path.parent.name != "detection_images":
        return False
    basenames = {path.name}
    stem, _, index = path.stem.rpartition(".")
    if stem and index.isdigit():
        basenames.add(stem + path.suffix)
    candidates = Q()
    for basename in basenames:
        candidates |= Q(mediafile=basename) | Q(mediafile__endswith="/" + basename)
    return any(
        name in owned_file_names(media)
        for media in models.MediaFile.objects.filter(candidates).select_related("parent")
    )


def source_hash(name):
    """Fingerprint a source again before any destructive action."""
    result = hashlib.sha256()
    with media_path(name).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def scan_directory(scan):
    """Return a server-generated path, never a user-supplied filename."""
    root = getattr(settings, "DUPLICATE_SCAN_ROOT", Path(settings.SHARED_DATA_PATH) / "duplicate_scans")
    return Path(root) / str(scan.uuid)


@transaction.atomic
def create_scan(user, ids, include_crops=True, request_key=None):
    """Freeze an authorized, bounded selection and dispatch after commit."""
    models.CaIDUser.objects.select_for_update().get(pk=user.pk)
    request_key = uuid.UUID(str(request_key)) if request_key else uuid.uuid4()
    existing = models.DuplicateScan.objects.filter(requested_by=user, request_key=request_key).first()
    if existing:
        return existing
    ids = sorted(set(int(value) for value in ids))
    if not 2 <= len(ids) <= MAX_FILES:
        raise ValueError(f"Select between 2 and {MAX_FILES} media files")
    mediafiles = list(models.MediaFile.objects.for_user(user).filter(pk__in=ids).order_by("pk"))
    if len(mediafiles) != len(ids):
        raise ValueError("Some selected media files are unavailable")
    items = [
        {
            "id": media.pk,
            "name": media.original_filename,
            "source": media.mediafile.name,
            "media_type": media.media_type,
        }
        for media in mediafiles
    ]
    for item in items:
        if not media_path(item["source"]).is_file():
            raise ValueError("A selected source media file is unavailable")
    scan = models.DuplicateScan.objects.create(
        requested_by=user,
        request_key=request_key,
        items=items,
        include_crops=include_crops,
        task_id=str(uuid.uuid4()),
    )
    transaction.on_commit(partial(dispatch_scan, scan.pk))
    return scan


def dispatch_scan(scan_id):
    """Write the manifest and publish to the dedicated queue; preserve dispatch failure."""
    scan = models.DuplicateScan.objects.get(pk=scan_id)
    try:
        directory = scan_directory(scan)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "manifest.tmp"
        temporary.write_text(json.dumps({"items": scan.items, "include_crops": scan.include_crops}), encoding="utf-8")
        temporary.replace(directory / "manifest.json")
        signature("duplicates.scan", args=[str(scan.uuid)], queue="duplicate_worker").apply_async(
            task_id=scan.task_id,
            link=signature("caidapp.tasks.finish_duplicate_scan", args=[scan.pk], queue="celery"),
            link_error=signature("caidapp.tasks.fail_duplicate_scan", args=[scan.pk], immutable=True, queue="celery"),
        )
    except Exception:
        logger.exception("Failed to dispatch duplicate scan %s", scan_id)
        fail_scan(scan_id, "Could not start the duplicate worker. Please try again.")


def fail_scan(scan_id, message="Duplicate scan failed or timed out. No media files were changed."):
    """Reject late results after a failed or expired scan."""
    models.DuplicateScan.objects.filter(pk=scan_id, status="queued").update(
        status="failed",
        message=message,
        finished_at=timezone.now(),
    )


@transaction.atomic
def finish_scan(scan_id):
    """Import a result once, preserving snapshots even if a source has disappeared."""
    scan = models.DuplicateScan.objects.select_for_update().get(pk=scan_id)
    if scan.status != "queued":
        return
    result = json.loads((scan_directory(scan) / "result.json").read_text(encoding="utf-8"))
    if result.get("version") != "duplicates-v1":
        raise ValueError("Unsupported duplicate algorithm version")
    originals = {item["id"]: item for item in scan.items}
    records = {item["id"]: item for item in result["items"]}
    if set(records) != set(originals):
        raise ValueError("Duplicate result does not match its manifest")
    for key, item in records.items():
        if "error" not in item and item.get("source") != originals[key]["source"]:
            raise ValueError("Duplicate source changed")
    available = set(
        models.MediaFile.objects.for_user(scan.requested_by).filter(pk__in=originals).values_list("pk", flat=True)
    )
    for proposal in result["proposals"]:
        ids = proposal["members"]
        if (
            proposal["kind"] not in {"byte_exact", "pixel_exact", "crop"}
            or len(set(ids)) != len(ids)
            or len(ids) < 2
            or not set(ids).issubset(records)
        ):
            raise ValueError("Invalid duplicate proposal")
        row = models.DuplicateProposal.objects.create(
            scan=scan,
            kind=proposal["kind"],
            evidence=proposal["evidence"],
            status="pending" if set(ids).issubset(available) else "stale",
        )
        models.DuplicateProposalMember.objects.bulk_create(
            [
                models.DuplicateProposalMember(
                    proposal=row,
                    mediafile_id=key if key in available else None,
                    original_id=key,
                    snapshot=records[key],
                    position=index,
                )
                for index, key in enumerate(ids)
            ]
        )
    scan.items = result["items"]
    scan.algorithm = result["version"]
    scan.status = "succeeded"
    scan.finished_at = timezone.now()
    scan.save()


def refresh_scan(scan):
    """Recover a missed callback from the durable result; bound a lost worker's lifetime."""
    if scan.status == "queued":
        if (scan_directory(scan) / "result.json").is_file():
            try:
                finish_scan(scan.pk)
            except (OSError, ValueError, KeyError, TypeError):
                logger.exception("Invalid duplicate scan result %s", scan.pk)
                fail_scan(scan.pk, "Could not read duplicate scan results.")
        elif scan.created_at < timezone.now() - timedelta(hours=2):
            fail_scan(scan.pk)
        scan.refresh_from_db()


def invalidate_deleted_media(mediafile_id):
    """Close affected proposals without inventing a new crop relationship."""
    proposals = models.DuplicateProposal.objects.filter(
        status="pending",
        members__original_id=mediafile_id,
    ).distinct()
    for proposal in proposals:
        if proposal.kind == "crop" or proposal.members.filter(mediafile__isnull=False).count() < 2:
            proposal.status = "stale"
            proposal.save(update_fields=["status"])


def proposal_media(proposal, user, lock=False):
    """Authorize all live members, not just the user who initiated the scan."""
    ids = list(proposal.members.filter(mediafile__isnull=False).values_list("mediafile_id", flat=True))
    queryset = models.MediaFile.objects.for_user(user).filter(pk__in=ids).order_by("pk")
    if proposal.scan.workgroup_id:
        if proposal.scan.workgroup_id != user.workgroup_id:
            raise ValueError("The duplicate proposal belongs to another workgroup")
        queryset = queryset.filter(parent__owner__workgroup_id=proposal.scan.workgroup_id, parent__import_finished=True)
    if lock:
        queryset = queryset.select_for_update(of=("self",))
    rows = list(queryset)
    if len(rows) != len(ids):
        raise ValueError("Some proposal members are no longer accessible")
    return rows


def _difference_warnings(rows, observations):
    """Compare meaningful file facts and assigned identities, without reconciling anonymous observations."""
    warnings = []
    real = [obs for obs in observations if not obs.is_no_detection_placeholder]
    identities = {
        media.pk: sorted({str(obs.identity) for obs in real if obs.mediafile_id == media.pk and obs.identity_id})
        for media in rows
    }
    taxa = {
        media.pk: sorted({str(obs.taxon) for obs in real if obs.mediafile_id == media.pk and obs.taxon_id})
        for media in rows
    }

    def compare(key, label, values):
        if len({digest(value) for value in values.values()}) > 1:
            warnings.append(
                {
                    "field": key,
                    "label": label,
                    "values": [
                        {"id": media.pk, "name": media.original_filename, "value": values[media.pk] or "—"}
                        for media in rows
                    ],
                }
            )

    # Compare IDs, not just names: two different identities may share a display name.
    identity_ids = {
        media.pk: sorted({obs.identity_id for obs in real if obs.mediafile_id == media.pk and obs.identity_id})
        for media in rows
    }
    if len({digest(value) for value in identity_ids.values()}) > 1:
        warnings.append(
            {
                "field": "identities",
                "label": "Assigned identities differ",
                "values": [
                    {
                        "id": media.pk,
                        "name": media.original_filename,
                        "value": ", ".join(identities[media.pk]) or "No assigned identity",
                    }
                    for media in rows
                ],
            }
        )
    taxon_ids = {
        media.pk: sorted({obs.taxon_id for obs in real if obs.mediafile_id == media.pk and obs.taxon_id})
        for media in rows
    }
    if len({digest(value) for value in taxon_ids.values()}) > 1:
        warnings.append(
            {
                "field": "taxa",
                "label": "Taxa differ",
                "values": [
                    {"id": media.pk, "name": media.original_filename, "value": ", ".join(taxa[media.pk]) or "No taxon"}
                    for media in rows
                ],
            }
        )
    compare("captured_at", "Capture dates differ", {media.pk: serializable(media.captured_at) for media in rows})
    for field, label in [
        ("locality", "Localities differ"),
        ("location", "Locations differ"),
        ("note", "Notes differ"),
    ]:
        values = {media.pk: str(getattr(media, field)) if getattr(media, field) else None for media in rows}
        compare(field, label, values)
    return warnings


def transfer_options(rows, observations, warnings):
    """Offer only reviewed sources and unambiguous observation-level replacements."""
    kept = rows[0]
    real = {
        media.pk: [o for o in observations if o.mediafile_id == media.pk and not o.is_no_detection_placeholder]
        for media in rows
    }
    fields = {warning["field"] for warning in warnings}
    result = []
    for source in rows[1:]:
        for field in sorted(fields & {"captured_at", "locality", "location", "identities", "taxa", "note"}):
            observation_field = {"identities": "identity", "taxa": "taxon"}.get(field)
            if observation_field:
                if len(real[kept.pk]) != 1 or len(real[source.pk]) != 1:
                    continue
                origin, target = real[source.pk][0], real[kept.pk][0]
                value = getattr(origin, observation_field)
                if value is None or getattr(origin, observation_field + "_id") == getattr(
                    target, observation_field + "_id"
                ):
                    continue
                if field == "identities" and value.owner_workgroup_id != kept.parent.owner.workgroup_id:
                    continue
            else:
                value = getattr(source, field)
                if value == getattr(kept, field) or (field != "locality" and (value is None or value == "")):
                    continue
            result.append(
                {
                    "field": field,
                    "source_id": source.pk,
                    "value": str(value) if value is not None and value != "" else "—",
                    "source_name": source.original_filename,
                }
            )
    return result


def build_plan(proposal, user, keep_id, remove_ids=None, lock=False):
    """Preview whole-file deletion and the explicit metadata replacements available to the reviewer."""
    if proposal.status != "pending":
        raise ValueError("This proposal is no longer pending")
    if proposal.kind == "reference_exact":
        raise ValueError(
            "Matching reference frames do not establish that the videos are duplicates. No removal is available."
        )
    all_media = proposal_media(proposal, user, lock=lock)
    by_id = {media.pk: media for media in all_media}
    if keep_id not in by_id:
        raise ValueError("Select a live proposal member to keep")
    remove_ids = sorted(set(remove_ids if remove_ids is not None else set(by_id) - {keep_id}))
    if not remove_ids or keep_id in remove_ids or not set(remove_ids).issubset(by_id):
        raise ValueError("Select other live members to remove")
    ids = [keep_id] + remove_ids
    rows = [by_id[key] for key in ids]
    snapshots = {member.original_id: member.snapshot for member in proposal.members.all()}
    observed = models.AnimalObservation.objects.filter(mediafile_id__in=ids).order_by("pk")
    if lock:
        observed = observed.select_for_update(of=("self",))
    observations = list(observed.select_related("identity", "taxon"))
    state = {
        "members": sorted(by_id),
        "media": [],
        "observations": serializable(
            list(
                models.AnimalObservation.objects.filter(pk__in=[item.pk for item in observations])
                .order_by("pk")
                .values()
            )
        ),
        "albums": list(models.Album.mediafiles.through.objects.filter(mediafile_id__in=ids).order_by("pk").values()),
    }
    for media in rows:
        snapshot = snapshots[media.pk]
        if proposal.scan.purpose == "exact":
            from .duplicate_hashes import HASH_VERSION

            field = "pixel_sha256" if proposal.kind == "pixel_exact" else "file_sha256"
            if (
                media.hash_status != "ready"
                or media.hash_version != HASH_VERSION
                or getattr(media, field) != proposal.evidence.get("hash")
            ):
                raise ValueError("Duplicate evidence changed. Wait for the group's updated suggestions.")
        if media.mediafile.name != snapshot["source"] or source_hash(media.mediafile.name) != snapshot["sha256"]:
            raise ValueError("A source media file changed. Run duplicate detection again.")
        state["media"].append(serializable(models.MediaFile.objects.filter(pk=media.pk).values().get()))
    warnings = _difference_warnings(rows, observations)
    if proposal.kind == "crop":
        warnings.append(
            {
                "field": "crop",
                "label": (
                    "These are overlapping images, not exact copies. "
                    "Deleting the marked file also removes content outside the crop."
                ),
                "values": [],
            }
        )
    return {
        "keep_id": keep_id,
        "remove_ids": remove_ids,
        "warnings": warnings,
        "transfer_options": transfer_options(rows, observations, warnings),
        "snapshot": digest(state),
        "audit": state,
        "members": all_media,
    }


def preview_token(proposal, user, plan):
    """Bind a short-lived preview to the user, exact action, and current database state."""
    return signing.dumps(
        {
            "operation": "retain-whole-media-v2",
            "proposal": str(proposal.uuid),
            "user": user.pk,
            "keep": plan["keep_id"],
            "remove": plan["remove_ids"],
            "snapshot": plan["snapshot"],
        },
        salt="duplicate-delete-v2",
    )


@transaction.atomic
def dismiss_proposal(proposal_id, user):
    """Dismiss a remaining suggestion without overwriting previous partial-merge audit."""
    proposal = models.DuplicateProposal.objects.get(pk=proposal_id)
    proposal_media(proposal, user, lock=True)
    proposal = models.DuplicateProposal.objects.select_for_update().get(pk=proposal_id)
    if proposal.status != "pending":
        raise ValueError("This proposal is no longer pending")
    decision = dict(proposal.decision)
    decision["dismissed_by"] = user.pk
    proposal.decision = decision
    proposal.status = "dismissed"
    proposal.reviewed_at = timezone.now()
    proposal.save(update_fields=["decision", "status", "reviewed_at"])


@transaction.atomic
def apply_merge(proposal_id, user, token, edits=None):
    """Apply opt-in edits and delete the reviewed counterparts atomically, preserving survivor geometry."""
    signed = signing.loads(token, salt="duplicate-delete-v2", max_age=1800)
    proposal = models.DuplicateProposal.objects.get(pk=proposal_id)
    # Acquire media locks before proposal locks: deleting media also invalidates
    # other proposals and must not deadlock with a concurrent overlapping merge.
    proposal_media(proposal, user, lock=True)
    proposal = models.DuplicateProposal.objects.select_for_update().get(pk=proposal_id)
    if (
        signed.get("operation") != "retain-whole-media-v2"
        or signed["user"] != user.pk
        or signed["proposal"] != str(proposal.uuid)
    ):
        raise ValueError("Invalid merge preview")
    # All merges acquire media locks in primary-key order, including overlapping scans.
    plan = build_plan(proposal, user, signed["keep"], signed["remove"], lock=True)
    if plan["snapshot"] != signed["snapshot"]:
        raise ValueError("Metadata or proposal members changed. Review the refreshed preview.")
    kept = next(media for media in plan["members"] if media.pk == plan["keep_id"])
    removed = [media for media in plan["members"] if media.pk in plan["remove_ids"]]
    edits = {} if edits is None else edits
    if not isinstance(edits, dict) or set(edits) - {"note", "take"}:
        raise ValueError("Invalid retained-file edits")
    take = edits.get("take", {})
    if not isinstance(take, dict):
        raise ValueError("Invalid metadata replacements")
    available = {(option["field"], option["source_id"]) for option in plan["transfer_options"]}
    if any((field, source) not in available for field, source in take.items()):
        raise ValueError("This replacement is unavailable. Review the retained media file in its detail.")
    if "note" in edits and not isinstance(edits["note"], str):
        raise ValueError("The note must be text")
    sources = {media.pk: media for media in removed}
    changed_fields = []
    if "note" in edits and edits["note"] != kept.note:
        kept.note = edits["note"]
        changed_fields.append("note")
    observation_changes = {}
    for field, source_id in take.items():
        source = sources[source_id]
        if field in {"identities", "taxa"}:
            target_field = "identity" if field == "identities" else "taxon"
            source_observation = source.observations.get(is_no_detection_placeholder=False)
            observation_changes[target_field] = getattr(source_observation, target_field)
        else:
            setattr(kept, field, getattr(source, field))
            changed_fields.append(field)
    if observation_changes:
        observation = kept.observations.get(is_no_detection_placeholder=False)
        if kept.used_for_init_identification or observation.identity_is_representative:
            from .tasks import schedule_init_identification_for_workgroup

            group = kept.parent.owner.workgroup
            models.WorkGroup.objects.filter(pk=group.pk).update(identification_initialized_model=None)
            transaction.on_commit(
                partial(schedule_init_identification_for_workgroup, group, delay_minutes=10), robust=True
            )
        for field, value in observation_changes.items():
            setattr(observation, field, value)
        fields = list(observation_changes) + ["updated_by", "updated_at"]
        if "identity" in observation_changes:
            observation.identity_is_representative = False
            fields.append("identity_is_representative")
        if "taxon" in observation_changes:
            observation.taxon_verified = True
            observation.taxon_verified_at = timezone.now()
            fields += ["taxon_verified", "taxon_verified_at"]
        observation.updated_by, observation.updated_at = user, timezone.now()
        observation.save(update_fields=fields)
        models.MediafileIdentificationSuggestion.objects.filter(mediafile=kept).delete()
    if changed_fields:
        kept.save(update_fields=changed_fields)
    from .bbox_services import _invalidate_reid

    # Never move observations, replace survivor previews, or invalidate its geometry embeddings.
    for media in removed:
        _invalidate_reid(media, list(media.observations.all()))
    decision = dict(proposal.decision)
    decision.setdefault("merges", []).append(
        serializable(
            {
                "kept": kept.pk,
                "removed": plan["remove_ids"],
                "by": user.pk,
                "at": timezone.now(),
                "operation": "retain-whole-media-v2",
                "warnings": plan["warnings"],
                "edits": edits,
                "before": plan["audit"],
                "retained": serializable(models.MediaFile.objects.filter(pk=kept.pk).values().get()),
            }
        )
    )
    proposal.decision = decision
    proposal.reviewed_at = timezone.now()
    remaining = len(plan["members"]) - len(removed)
    proposal.status = "resolved" if remaining < 2 else "pending"
    proposal.save()
    names = set().union(*(owned_file_names(media) for media in removed))
    cleanup_ids = [models.DuplicateFileCleanup.objects.create(storage_name=name).pk for name in sorted(names)]
    models.MediaFile.objects.filter(pk__in=plan["remove_ids"]).delete()
    from .tasks import cleanup_duplicate_files

    transaction.on_commit(partial(cleanup_duplicate_files.delay, cleanup_ids), robust=True)
    return kept.pk


def cleanup_files(ids):
    """Retryable deletion after commit; never delete a path still referenced by another media file."""
    for intent in models.DuplicateFileCleanup.objects.filter(pk__in=ids, status="pending"):
        try:
            if _file_is_shared(intent.storage_name):
                intent.status = "shared"
            else:
                path = media_path(intent.storage_name)
                path.unlink(missing_ok=True)
                intent.status = "deleted"
            intent.message = ""
        except (OSError, ValueError):
            intent.message = "File cleanup needs retry; media-file merge is complete."
            logger.exception("Duplicate cleanup failed %s", intent.pk)
        intent.save(update_fields=["status", "message"])
