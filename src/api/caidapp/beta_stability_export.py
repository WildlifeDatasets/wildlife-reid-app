"""Build portable, observation-level beta-stability datasets from versioned vectors."""

import hashlib
import json
import pickle
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from django.core.exceptions import ValidationError
from django.utils import timezone
from PIL import Image

from .models import ObservationEmbedding
from .observation_embeddings import snapshot


class BetaExportError(ValidationError):
    """An export that cannot satisfy its explicit input/profile selection."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report
        # Celery reconstructs exceptions from their args when reporting failed exports.
        self.args = (message, report)


def _json(archive, name, data):
    archive.writestr(name, json.dumps(data, indent=2, allow_nan=False))


def _bbox(observation, width, height):
    if observation.bbox_x_center is None:
        return None
    left, top, right, bottom = observation.get_bbox_xyxy(width, height)
    left, right = max(0, min(width, left)), max(0, min(width, right))
    top, bottom = max(0, min(height, top)), max(0, min(height, bottom))
    if right <= left or bottom <= top:
        raise ValidationError("Observation crop is outside the source image.")
    return [left, top, right - left, bottom - top]


def export_beta_stability(observations, profiles, output_path, *, include_images=False, allow_partial=False):
    """Export a permission-scoped queryset; callers must authorize that scope first."""
    profiles = list(profiles)
    if not profiles or len({profile.pk for profile in profiles}) != len(profiles):
        raise ValidationError("Choose at least one distinct embedding profile.")
    observations = list(observations.select_related("mediafile", "mediafile__parent").order_by("pk").distinct())
    vectors = {
        (row.observation_id, row.profile_id, row.input_fingerprint): row
        for row in ObservationEmbedding.objects.filter(
            observation_id__in=[observation.pk for observation in observations], profile__in=profiles
        )
    }
    skipped, selected = [], []
    missing = 0
    for observation in observations:
        reason, detail = None, {}
        if observation.is_no_detection_placeholder:
            reason = "no_detection_placeholder"
        elif observation.mediafile.media_type == "video":
            reason = "video_frame_not_located"
        elif observation.mediafile.parent is None or not observation.mediafile.parent.import_finished:
            reason = "unfinished_upload"
        else:
            try:
                current = snapshot(observation)
                with Image.open(current["image_path"]) as image:
                    width, height = image.size
                    image.verify()
                bbox = _bbox(observation, width, height)
            except (OSError, ValidationError, ValueError):
                reason = "unavailable_or_invalid_input"
            else:
                rows = [vectors.get((observation.pk, profile.pk, current["input_fingerprint"])) for profile in profiles]
                absent = [str(profile.uuid) for profile, row in zip(profiles, rows) if row is None]
                if absent:
                    reason, detail = "missing_current_embedding", {"profile_uuids": absent}
                    missing += 1
                else:
                    for profile, row in zip(profiles, rows):
                        vector = np.asarray(row.vector, dtype=np.float32)
                        if vector.shape != (profile.dimension,) or not np.isfinite(vector).all() or not np.any(vector):
                            raise ValidationError(f"Invalid stored embedding {row.pk}.")
                    selected.append((observation, current, width, height, bbox, rows))
        if reason:
            skipped.append({"observation_uuid": str(observation.uuid), "reason": reason, **detail})
    report = {"requested": len(observations), "exported": len(selected), "skipped": skipped}
    if missing and not allow_partial:
        raise BetaExportError(
            f"{missing} observations lack current embeddings in the chosen profiles. "
            "Wait for computation or explicitly allow a partial export.",
            report,
        )
    if not selected:
        raise BetaExportError("No observations have eligible inputs and all requested embedding profiles.", report)

    uuids = [str(row[0].uuid) for row in selected]
    annotations, images, profile_records = [], {}, []
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as temporary:
        temporary_zip = Path(temporary) / "export.zip"
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for observation, current, width, height, bbox, rows in selected:
                image_uuid = f"media-{observation.mediafile_id}"
                source = Path(current["image_path"])
                if image_uuid not in images:
                    suffix = source.suffix.lower()
                    if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
                        suffix = ".img"
                    file_name = f"images/{image_uuid}{suffix}"
                    images[image_uuid] = {"uuid": image_uuid, "file_name": file_name, "width": width, "height": height}
                    if include_images:
                        # Hash the bytes actually written, rather than trusting an earlier source check.
                        digest = hashlib.sha256()
                        with source.open("rb") as stream, archive.open(file_name, "w", force_zip64=True) as target:
                            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                                digest.update(chunk)
                                target.write(chunk)
                        if digest.hexdigest() != current["input_specification"]["source_sha256"]:
                            raise ValidationError("A source media file changed during export. Please retry.")
                annotations.append(
                    {
                        "uuid": str(observation.uuid),
                        "image_uuid": image_uuid,
                        "bbox": bbox,
                        "orientation": observation.orientation,
                        "input_fingerprint": current["input_fingerprint"],
                    }
                )
            for index, profile in enumerate(profiles):
                name = f"embedding_{index + 1}"
                file_name = f"embeddings/{profile.uuid}.pickle"
                matrix = np.asarray([row[5][index].vector for row in selected], dtype=np.float32)
                with archive.open(file_name, "w", force_zip64=True) as stream:
                    pickle.dump((matrix, uuids), stream, protocol=4)
                profile_records.append(
                    {
                        "name": name,
                        "uuid": str(profile.uuid),
                        "fingerprint": profile.fingerprint,
                        "specification": profile.specification,
                        "file": file_name,
                    }
                )
            # Recheck database inputs after reading vectors and packaging source files.
            for observation, current, *_ in selected:
                observation.refresh_from_db()
                observation.mediafile.refresh_from_db()
                if snapshot(observation)["input_fingerprint"] != current["input_fingerprint"]:
                    raise ValidationError("An observation changed during export. Please retry.")
            manifest = {
                "format": "wildlife-reid-beta-stability",
                "version": 1,
                "created_at": timezone.now().isoformat(),
                "images_included": include_images,
                "allow_partial": allow_partial,
                "profiles": profile_records,
                "observation_uuids": uuids,
                "report": report,
                "inputs": [
                    {
                        "observation_uuid": str(row[0].uuid),
                        "input_fingerprint": row[1]["input_fingerprint"],
                        "input_specification": row[1]["input_specification"],
                        "embedding_ids": [embedding.pk for embedding in row[5]],
                    }
                    for row in selected
                ],
            }
            first = profile_records[0]["name"]
            config = {
                "algorithm_type": "stability",
                "seed": 42,
                "data": {
                    "annotation_file": "annotations.json",
                    "format": "drone",
                    "id_key": "uuid",
                    "name_keys": None,
                    "output_path": "results",
                    "images_dir": "." if include_images else None,
                    "embeddings": {profile["name"]: {"file": profile["file"]} for profile in profile_records},
                },
                "algorithm": {"initial_topk": "auto"},
                "stability": {"verifier_name": first, "max_human_reviews": 0},
                "edge_weights": {"verifier_names": f"{first} no_human", "classifier_thresholds": {first: "auto(0.2)"}},
            }
            _json(archive, "annotations.json", {"images": list(images.values()), "annotations": annotations})
            _json(archive, "manifest.json", manifest)
            # JSON is also valid YAML and avoids a second configuration serialization format.
            _json(archive, "config.yaml", config)
            archive.writestr(
                "README.txt",
                (
                    "Extract this ZIP and run from the extracted directory in a beta-stability environment:\n"
                    "  python -m beta_stability --config config.yaml\n\n"
                    "The example runs automatically with the first selected embedding profile.\n"
                    "All exported profiles cover the exact same ordered observation UUIDs.\n"
                    "No identity labels are exported or required; review integration comes separately.\n"
                    "When images are omitted, image paths are placeholders for internal review.\n"
                    "Importing clustering results into Wildlife ReID is not implemented yet.\n"
                ),
            )
        if output_path.exists():
            raise ValidationError("Output file already exists; choose a new path.")
        shutil.move(str(temporary_zip), str(output_path))
    return manifest
