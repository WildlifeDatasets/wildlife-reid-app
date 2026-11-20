"""Exact, observation-scoped identity retrieval from versioned embeddings."""

import hashlib
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from django.core.exceptions import ValidationError

from ..models import EmbeddingProfile, ObservationEmbedding, TaskRun
from ..observation_embeddings import fingerprint


@lru_cache(maxsize=16384)
def _source_digest(path, size, modified, changed):
    """Keep bounded process-local hashes keyed by the current file metadata."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    after = path.stat()
    if (size, modified, changed) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValidationError("The source media file changed. Try again.")
    return hasher.hexdigest()


def current_input(observation, digests=None):
    """Check crop metadata and hash each source once, reusing stat-versioned hashes."""
    mediafile = observation.mediafile
    if observation.is_no_detection_placeholder or mediafile.media_type != "image":
        raise ValidationError("Only real observations in image media files can be identified.")
    source = mediafile.image_file or mediafile.mediafile
    if not source:
        raise ValidationError("The source media file is unavailable.")
    path = Path(source.path).resolve()
    stat = path.stat()
    key = (path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    digests = digests if digests is not None else {}
    digest = digests.get(key)
    if digest is None:
        digest = _source_digest(*key)
    digests[key] = digest
    specification = {
        "mediafile_id": mediafile.pk,
        "source_sha256": digest,
        "bbox_cxcywh": [
            observation.bbox_x_center,
            observation.bbox_y_center,
            observation.bbox_width,
            observation.bbox_height,
        ],
        "orientation": observation.orientation,
    }
    bbox = specification["bbox_cxcywh"]
    if any(value is not None for value in bbox):
        if any(value is None or not np.isfinite(value) for value in bbox) or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValidationError("Save a complete, valid bounding box first.")
    return fingerprint(specification)


def active_profile(workgroup):
    """Use the actual profile reported by the latest run of the selected model."""
    model = workgroup.identification_model
    if model is None:
        raise ValidationError("Select an identification model first.")
    runs = TaskRun.objects.filter(
        kind="embedding",
        workgroup=workgroup,
        status__in=["succeeded", "partial"],
        model_snapshot__model_id=model.pk,
        model_snapshot__model_source=model.get_runtime_model_source(),
        model_snapshot__weights_path=model.get_runtime_checkpoint_path(),
    ).order_by("-finished_at", "-pk")
    for run in runs.iterator(chunk_size=32):
        profile_uuid = (run.result or {}).get("profile_uuid")
        if profile_uuid:
            return EmbeddingProfile.objects.filter(uuid=profile_uuid).first()
    # Untracked historical encodings cannot establish which checkpoint is active.
    return None


def normalized(vector, dimension):
    """Reject malformed stored vectors and normalize for cosine similarity."""
    if not isinstance(vector, list) or len(vector) != dimension:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector):
        return None
    array = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(array)
    if not np.isfinite(array).all() or not np.isfinite(norm) or norm == 0:
        return None
    return array / norm


def search(observation, workgroup, references, limit=5):
    """Rank distinct identities by their best current reference observation."""
    started = time.perf_counter()
    digests = {}
    input_key = current_input(observation, digests)
    profile = active_profile(workgroup)
    query = None
    if profile:
        embedding = observation.embeddings.filter(profile=profile, input_fingerprint=input_key).first()
        if embedding:
            query = normalized(embedding.vector, profile.dimension)
    payload = {
        "input_fingerprint": input_key,
        "profile_uuid": str(profile.uuid) if profile else None,
        "candidates": [],
        "status": "embedding_required",
        "reference_count": 0,
        "eligible_reference_count": 0,
    }
    if query is None:
        return payload
    references = references.filter(
        identity__isnull=False,
        identity__owner_workgroup=workgroup,
        is_no_detection_placeholder=False,
        mediafile__media_type="image",
        mediafile__parent__import_finished=True,
    ).exclude(pk=observation.pk)
    if observation.taxon_id:
        references = references.filter(taxon_id=observation.taxon_id)
    payload["eligible_reference_count"] = references.count()
    embeddings = (
        ObservationEmbedding.objects.filter(profile=profile, observation__in=references)
        .select_related("observation__mediafile", "observation__identity")
        .order_by("observation_id", "-created_at", "-pk")
    )
    best = {}
    seen = set()
    for embedding in embeddings.iterator(chunk_size=256):
        reference = embedding.observation
        if reference.pk in seen:
            continue
        try:
            if current_input(reference, digests) != embedding.input_fingerprint:
                continue
        except (OSError, ValidationError, ValueError):
            continue
        vector = normalized(embedding.vector, profile.dimension)
        if vector is None:
            continue
        seen.add(reference.pk)
        score = float(np.clip(np.dot(query, vector), -1, 1))
        previous = best.get(reference.identity_id)
        if previous is None or score > previous["score"]:
            best[reference.identity_id] = {
                "identity_id": reference.identity_id,
                "label": str(reference.identity),
                "score": score,
                "reference": reference,
            }
    payload.update(
        status="complete",
        reference_count=len(seen),
        candidates=sorted(best.values(), key=lambda row: (-row["score"], row["identity_id"]))[:limit],
        elapsed_ms=round((time.perf_counter() - started) * 1000),
    )
    return payload
