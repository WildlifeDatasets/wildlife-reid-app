"""Prepare persisted observation features for the existing identification matcher."""

import csv
import hashlib
import json
import math
import uuid
from pathlib import Path

from django.conf import settings
from django.core.exceptions import ValidationError

from ..models import AnimalObservation, EmbeddingProfile, ObservationEmbedding, WorkGroup
from ..observation_embeddings import embedding_generation, snapshot
from .identity_review import input_version


def feature_configuration(weights_path):
    """Identify compatible WildFusion features for this deployment."""
    digest = ""
    if weights_path:
        with open(weights_path, "rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "revision": settings.IDENTIFICATION_FEATURE_REVISION,
        "segmentation": settings.IDENTIFICATION_FEATURE_SEGMENTATION,
        "weights_sha256": digest,
    }


def prepare_feature_manifest(manifest):
    """Add stable feature profile and cache keys to an embedding request."""
    config = feature_configuration(manifest["weights_path"])
    manifest["feature_config"] = config
    profile = EmbeddingProfile.objects.filter(
        specification__feature_config=config, specification__model_source=manifest["model_source"]
    ).order_by("-created_at").first()
    manifest["cached"] = {}
    if profile:
        inputs = {row["observation_id"]: row["input_fingerprint"] for row in manifest["observations"]}
        cached = ObservationEmbedding.objects.filter(
            profile=profile,
            observation_id__in=inputs,
            local_features__isnull=False,
        ).values_list("observation_id", "input_fingerprint", "feature_image")
        root = Path(settings.MEDIA_ROOT)
        manifest["cached"][profile.fingerprint] = [
            f"{observation_id}:{fingerprint}"
            for observation_id, fingerprint, image in cached
            if inputs[observation_id] == fingerprint and image and (root / image).is_file()
        ]
        manifest["cached_profile"] = profile.fingerprint
        manifest["cached_dimension"] = profile.dimension
    for row in manifest["observations"]:
        directory = Path(settings.MEDIA_ROOT) / "identification_feature_images" / row["input_fingerprint"][:2]
        row["feature_image"] = str(directory / f"{row['observation_id']}-{uuid.uuid4().hex}.png")


def validated_local_features(row, submitted):
    """Check the local descriptor structure and bind its image to the submitted path."""
    local = row.get("local_features")
    if not isinstance(local, dict) or set(local) != {"keypoints", "descriptors", "image_size"}:
        raise ValidationError("The encoder did not return ALIKED features.")
    points, descriptors, size = (local[key] for key in ("keypoints", "descriptors", "image_size"))

    def valid_vector(values, width):
        return isinstance(values, list) and len(values) == width and all(
            type(value) in (int, float) and math.isfinite(value) for value in values
        )

    if (
        not valid_vector(size, 2)
        or min(size) <= 0
        or not isinstance(points, list)
        or not isinstance(descriptors, list)
        or len(points) != len(descriptors)
        or any(not valid_vector(point, 2) for point in points)
        or (descriptors and (not isinstance(descriptors[0], list) or not descriptors[0]))
        or any(not valid_vector(descriptor, len(descriptors[0])) for descriptor in descriptors)
    ):
        raise ValidationError("The encoder returned malformed ALIKED features.")
    actual = row.get("feature_image")
    expected = submitted["feature_image"]
    if all(value is None for value in submitted["input_specification"]["bbox_cxcywh"]):
        expected = submitted["image_path"]
    if actual != expected:
        raise ValidationError("Unexpected image for the computed features.")
    try:
        image = Path(actual).resolve(strict=True).relative_to(Path(settings.MEDIA_ROOT).resolve())
    except (OSError, ValueError) as exc:
        raise ValidationError("The image for the computed features is unavailable.") from exc
    return {"local_features": local, "feature_image": image.as_posix()}


def reference_observations(workgroup, *, include_non_representative=False):
    """Select the current labeled image references in the workgroup."""
    references = AnimalObservation.objects.filter(
        mediafile__parent__owner__workgroup=workgroup,
        mediafile__parent__import_finished=True,
        mediafile__media_type="image",
        identity__owner_workgroup=workgroup,
        is_no_detection_placeholder=False,
    )
    if not include_non_representative:
        references = references.filter(identity_is_representative=True)
    if workgroup.check_taxon_before_identification and workgroup.default_taxon_for_identification_id:
        references = references.filter(taxon_id=workgroup.default_taxon_for_identification_id)
    return references


def prepare_identification(
    organization_id, metadata_path, model, operation, generation=None, *, include_non_representative=False
):
    """Build versioned inputs from the submitted queries and Django's reference selection."""
    workgroup = WorkGroup.objects.get(pk=organization_id)
    current_generation = embedding_generation(organization_id)
    if generation is not None and generation != current_generation:
        raise ValidationError("Embeddings were reset. Start a new identification run.")
    with open(metadata_path, newline="", encoding="utf-8") as stream:
        ids = list(dict.fromkeys(int(row["observation_id"]) for row in csv.DictReader(stream) if row["observation_id"]))

    observations = AnimalObservation.objects.filter(
        pk__in=ids,
        mediafile__parent__owner__workgroup=workgroup,
        mediafile__media_type="image",
        is_no_detection_placeholder=False,
    )
    if operation == "compare_unknown":
        query_ids = reference_ids = ids
        observations = observations.filter(identity__isnull=True)
    elif operation in {"identify", "review_known"}:
        query_ids = ids
        references = reference_observations(workgroup, include_non_representative=include_non_representative)
        reference_ids = list(references.order_by("pk").values_list("pk", flat=True))
        if operation == "identify":
            observations = observations.filter(identity__isnull=True) | references
        else:
            observations = observations.filter(identity__owner_workgroup=workgroup) | references
    else:
        query_ids = []
        reference_ids = ids
        observations = observations.filter(identity__owner_workgroup=workgroup)
        if operation == "init":
            observations = observations.filter(identity_is_representative=True)
    if not reference_ids or (operation in {"identify", "compare_unknown", "review_known"} and not query_ids):
        raise ValidationError("Identification needs eligible image observations and labeled references.")
    by_id = {row.pk: row for row in observations.select_related("mediafile")}
    if set(query_ids + reference_ids) - by_id.keys():
        raise ValidationError("An identification input changed. Start a new run.")
    manifest = {
        "workgroup_id": workgroup.pk,
        "generation": current_generation,
        "model_source": model["model_source"],
        "weights_path": model["weights_path"],
        "observations": [snapshot(by_id[pk]) for pk in sorted(set(query_ids + reference_ids))],
        "query_ids": query_ids,
        "reference_ids": reference_ids,
        "operation": operation,
        "include_non_representative": include_non_representative,
    }
    prepare_feature_manifest(manifest)
    directory = Path(metadata_path).parent / f"features-{uuid.uuid4().hex}"
    directory.mkdir()
    target = directory / "input.json"
    target.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")
    return str(target)


def export_match_inputs(manifest_path, profile_uuid):
    """Stream data selected by the API into stable query and reference inputs for the worker."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    inputs = {row["observation_id"]: row for row in manifest["observations"]}
    reference_ids = set(manifest["reference_ids"])
    query_ids = set(manifest["query_ids"])
    operation = manifest["operation"]
    only_representatives = operation in {"identify", "review_known"} and not manifest.get("include_non_representative")
    workgroup_id = manifest["workgroup_id"]
    selected = ObservationEmbedding.objects.filter(
        profile__uuid=profile_uuid,
        observation_id__in=inputs,
        observation__mediafile__parent__owner__workgroup_id=workgroup_id,
        local_features__isnull=False,
    ).select_related("observation__identity", "observation__mediafile__sequence").order_by("observation_id")
    global_only = operation in {"compare_unknown", "review_known"}
    if global_only:
        selected = selected.defer("local_features")
    exported = set()
    query_metadata = []
    with (
        path.with_name("references.jsonl").open("w", encoding="utf-8") as references,
        path.with_name("queries.jsonl").open("w", encoding="utf-8") as queries,
    ):
        for feature in selected.iterator(chunk_size=32):
            pk = feature.observation_id
            if feature.input_fingerprint != inputs[pk]["input_fingerprint"]:
                continue
            observation = feature.observation
            mediafile = observation.mediafile
            ref = pk in reference_ids
            expects_identity = operation != "compare_unknown" and (ref or operation == "review_known")
            if (
                observation.is_no_detection_placeholder
                or (only_representatives and ref and not observation.identity_is_representative)
                or (expects_identity and observation.identity_id is None)
                or (observation.identity_id is not None and (
                    not expects_identity or observation.identity.owner_workgroup_id != workgroup_id
                ))
            ):
                raise ValidationError("An observation changed during identification preparation. Start a new run.")
            row = {
                "observation_id": pk,
                "mediafile_id": mediafile.pk,
                "image_path": str(Path(settings.MEDIA_ROOT) / feature.feature_image),
                "original_path": inputs[pk]["image_path"],
                "class_id": observation.identity_id,
                "label": str(observation.identity) if observation.identity_id else "",
                "sequence_number": mediafile.sequence.local_id if mediafile.sequence_id else -1,
                "review_version": input_version(observation, mediafile),
            }
            local = None if global_only else feature.local_features
            line = json.dumps({**row, "embedding": [feature.vector, local]}, allow_nan=False) + "\n"
            if ref:
                references.write(line)
            if pk in query_ids:
                queries.write(line)
                query_metadata.append(row)
            exported.add(pk)
    if exported != inputs.keys():
        raise ValidationError("Compatible features are missing or stale. Start a new identification run.")
    path.with_name("query_metadata.json").write_text(json.dumps(query_metadata), encoding="utf-8")
