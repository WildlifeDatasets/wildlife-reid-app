"""Count workgroup observations covered by features and comparison results."""

from django.db.models import Count, F, Q

from ..models import AnimalObservation, MediafileIdentificationSuggestion, MediaFile, TaskRun
from ..observation_embeddings import backfill_observations, embedding_generation
from .identity_review import input_version, model_version


def workgroup_coverage(workgroup):
    """Count distinct observations and disregard stale comparison inputs."""
    observations = AnimalObservation.objects.filter(
        mediafile__parent__owner__workgroup=workgroup, is_no_detection_placeholder=False,
    )
    counts = observations.aggregate(
        observations=Count("pk"),
        identified=Count("pk", filter=Q(identity__isnull=False)),
    )
    counts["unidentified"] = counts["observations"] - counts["identified"]
    counts["images"] = MediaFile.objects.filter(parent__owner__workgroup=workgroup, media_type="image").count()
    counts["eligible"] = backfill_observations(workgroup.pk).count()
    counts["embedded"] = observations.filter(embeddings__isnull=False).distinct().count()

    model = model_version(workgroup.identification_model)
    generation = embedding_generation(workgroup.pk)
    comparisons = {}
    if model:
        for mode in ("compare_unknown", "review_known"):
            comparisons[mode] = TaskRun.objects.filter(
                workgroup=workgroup, kind=mode, status="succeeded",
                model_snapshot__contains={**model, "generation": generation},
            ).only("result", "uuid").order_by("-finished_at", "-pk").first()
    comparison_rows = {
        mode: run.result.get("rows", []) for mode, run in comparisons.items() if run is not None
    }
    pending = (
        MediafileIdentificationSuggestion.objects.filter(
            for_identification__observation__in=observations.filter(identity__isnull=True),
            for_identification__model_version=model,
            identity__owner_workgroup=workgroup,
            reference_observation__in=observations,
            reference_observation__identity_id=F("identity_id"),
        )
        .order_by()
        .values_list("for_identification__observation_id", "for_identification__input_version")
        .distinct()
    ) if model else []
    pending_versions = dict(pending)
    ids = set(pending_versions)
    for rows in comparison_rows.values():
        for row in rows:
            ids.add(row["query"]["id"])
            ids.update(candidate["id"] for candidate in row["candidates"])
    versions = {
        observation.pk: input_version(observation, observation.mediafile)
        for observation in observations.filter(pk__in=ids).select_related("mediafile").only(
            "uuid", "identity", "identity_is_representative", "updated_at", "taxon", "orientation",
            "bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height", "is_no_detection_placeholder",
            "mediafile__image_file", "mediafile__mediafile", "mediafile__media_type",
        ).iterator(chunk_size=2000)
    }
    pending_ids = {pk for pk, version in pending_versions.items() if versions.get(pk) == version}
    comparison_ids = {
        mode: {
            row["query"]["id"]
            for row in rows
            if versions.get(row["query"]["id"]) == row["query"]["version"]
            and any(versions.get(candidate["id"]) == candidate["version"] for candidate in row["candidates"])
        }
        for mode, rows in comparison_rows.items()
    }
    unknown = comparison_ids.get("compare_unknown", set())
    known = comparison_ids.get("review_known", set())
    return {
        **counts,
        "with_suggestions": len(pending_ids | unknown | known),
        "identity_suggestions": len(pending_ids),
        "unknown_comparisons": len(unknown),
        "known_comparisons": len(known),
        "unknown_run": comparisons.get("compare_unknown"),
        "known_run": comparisons.get("review_known"),
    }
