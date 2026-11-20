"""Restore and export original media independently of inference variants."""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from zipfile import ZIP_STORED, ZipFile

from caidapp.zip_media_times import read_zip_file_times, zip_info_with_times
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import transaction

logger = logging.getLogger(__name__)


def original_relative_path(media):
    """Resolve the stored member name without guessing from duplicate basenames."""
    value = (media.metadata_json or {}).get("original_path") or media.original_filename
    return safe_member_path(value)


def safe_member_path(value):
    """Accept only relative archive member paths."""
    if not isinstance(value, str):
        raise ValidationError("Invalid original media path.")
    value = value.replace("\\", "/")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or ":" in value or "\x00" in value:
        raise ValidationError("Invalid original media path.")
    return path.as_posix()


def archive_entries(archive):
    """Index all members so duplicate paths can be rejected as ambiguous."""
    entries = defaultdict(list)
    for info in archive.infolist():
        if not info.is_dir():
            try:
                name = safe_member_path(info.filename)
            except ValidationError:
                continue
            entries[name].append(info)
    return entries


def _sha256_file(path):
    with path.open("rb") as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def _storage_path(name):
    root = Path(settings.MEDIA_ROOT).resolve()
    path = (root / name).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValidationError("Original media path is outside the media directory.")
    return path


def _private_original_path(name):
    media_root = Path(settings.MEDIA_ROOT).resolve()
    root = (media_root.parent / "media_originals").resolve()
    path = (root / name).resolve()
    if root.is_relative_to(media_root) or not path.is_relative_to(root) or path == root:
        raise ValidationError("Invalid private original path.")
    return path


def restore_media_original(media, stream, archive, entries, *, preserve_zip_times=True):
    """Verify and save the original bytes and timestamps for one registered media file."""
    member = original_relative_path(media)
    matches = entries.get(member, [])
    if len(matches) != 1:
        raise ValidationError(f"ZIP entry is missing or ambiguous: {member}")
    info = matches[0]
    times = read_zip_file_times(stream, info) if preserve_zip_times else {}
    filename = sha256(member.encode()).hexdigest() + PurePosixPath(member).suffix.lower()
    relative_destination = media.original_file or f"{media.parent_id}/{filename}"
    destination = _private_original_path(relative_destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with archive.open(info) as source, NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
            temp_path = Path(temp.name)
            digest = sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                temp.write(chunk)
        checksum = digest.hexdigest()
        if _sha256_file(temp_path) != checksum:
            raise ValidationError(f"Verification failed for {member}.")

        candidates = [(destination, False)]
        if media.mediafile:
            candidates.append((_storage_path(media.mediafile.name), True))
        for existing, uses_mediafile in candidates:
            if existing.is_file() and existing.stat().st_size == info.file_size and _sha256_file(existing) == checksum:
                break
        else:
            if destination.exists():
                raise ValidationError(f"An existing original differs from the ZIP entry: {member}")
            modified = times.get("modified")
            if modified:
                dt = datetime.fromisoformat(modified["value"])
                if dt.tzinfo is not None:
                    try:
                        os.utime(temp_path, (temp_path.stat().st_atime, dt.timestamp()))
                    except (OSError, OverflowError):
                        logger.warning("Could not restore filesystem modification time for %s", member, exc_info=True)
            os.replace(temp_path, destination)
            uses_mediafile = False
        media.original_file = "" if uses_mediafile else relative_destination
        media.metadata_json = {
            **(media.metadata_json or {}),
            "original_path": member,
            "original_file_times": times,
            "original_file_sha256": checksum,
            "original_file_is_mediafile": uses_mediafile,
        }
        media.save(update_fields=["original_file", "metadata_json"])
        return times
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def capture_time_from_zip(times):
    """Prefer creation to modification as a last-resort date, preserving its provenance."""
    for field in ("created", "modified"):
        if field in times:
            value = times[field]["value"]
            source = f"ZIP file {field} ({times[field]['source']})"
            if datetime.fromisoformat(value).tzinfo is None:
                source += f"; assumed {settings.TIME_ZONE}"
            return value, source
    return None, None


def has_source_zip_times(upload):
    """Distinguish supplied archive timestamps from times generated by upload packaging."""
    return upload.import_mapping.get("path_source") not in {"filename", "relative_path"}


def register_existing_added_original(media):
    """Link legacy additions whose original bytes were already stored directly."""
    source = _storage_path(media.mediafile.name)
    if not source.is_file():
        raise ValidationError("The original file of the added media is missing.")
    metadata = media.metadata_json or {}
    checksum = _sha256_file(source)
    if metadata.get("original_file_sha256", checksum) != checksum:
        raise ValidationError("Original checksum does not match for the added media.")
    media.metadata_json = {
        **metadata,
        "original_file_sha256": checksum,
        "original_file_is_mediafile": True,
    }
    media.save(update_fields=["metadata_json"])


def delete_unreferenced_original(name, upload_id, using):
    """Release a private original after the owning media deletion commits."""
    from caidapp.models import MediaFile, UploadedArchive

    with transaction.atomic(using=using):
        UploadedArchive.objects.using(using).select_for_update().filter(pk=upload_id).first()
        if not MediaFile.objects.using(using).filter(original_file=name).exists():
            _private_original_path(name).unlink(missing_ok=True)


def write_original_export(mediafiles, output):
    """Export verified originals with preserved paths and an explicit timestamp manifest."""
    seen = set()
    manifest = []
    try:
        with ZipFile(output, "w", allowZip64=True) as archive:
            for media in mediafiles:
                member = original_relative_path(media)
                metadata = media.metadata_json or {}
                uses_mediafile = metadata.get("original_file_is_mediafile") is True
                if member in seen or member == "_original_media.json" or not (media.original_file or uses_mediafile):
                    raise ValidationError(f"Original unavailable or export path ambiguous: {member}")
                seen.add(member)
                times = metadata.get("original_file_times", {})
                info = zip_info_with_times(member, times)
                info.compress_type = ZIP_STORED
                path = (
                    _private_original_path(media.original_file)
                    if media.original_file
                    else _storage_path(media.mediafile.name)
                )
                digest = sha256()
                with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                        target.write(chunk)
                checksum = digest.hexdigest()
                if metadata.get("original_file_sha256", checksum) != checksum:
                    raise ValidationError(f"Original checksum does not match: {member}")
                manifest.append({"path": member, "file_times": times, "sha256": checksum})
            if not manifest:
                raise ValidationError("The upload has no original media to export.")
            archive.writestr("_original_media.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    except Exception:
        Path(output).unlink(missing_ok=True)
        raise
