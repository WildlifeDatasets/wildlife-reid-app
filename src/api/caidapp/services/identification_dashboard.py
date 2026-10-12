from dataclasses import asdict
from datetime import datetime, timedelta

from django.db.models import Count, Q
from django.utils import timezone

from .. import models
from ..models import IndividualIdentity, MediafilesForIdentification, UploadedArchive
from .workgroup_next_steps import NextStepCandidate, build_next_steps

REFRESH_INTERVAL = timedelta(hours=3)


def refresh_identification_dashboard_snapshot(workgroup):
    caiduser = models.CaIDUser(workgroup=workgroup)
    identity_queue_count = (
        MediafilesForIdentification.objects.filter(mediafile__parent__owner__workgroup=workgroup)
        .values("mediafile_id")
        .distinct()
        .count()
    )

    finished_archives = list(
        UploadedArchive.objects.filter(
            owner__workgroup=workgroup, import_finished=True, contains_identities=False
        ).select_related("taxon_for_identification")
    )
    suggestion_candidate_mediafile_count = 0
    suggestion_candidate_archive_count = 0
    suggestion_observation_taxon = None
    if workgroup.check_taxon_before_identification and workgroup.default_taxon_for_identification:
        suggestion_observation_taxon = workgroup.default_taxon_for_identification

    for uploaded_archive in finished_archives:
        observation_taxon = uploaded_archive.taxon_for_identification or suggestion_observation_taxon
        require_observations = observation_taxon is not None
        candidate_count = workgroup.mediafiles_for_identification(
            uploaded_archive_ids=[uploaded_archive.id],
            require_import_finished=True,
            require_identity=False,
            observation_taxon=observation_taxon,
            require_observations=require_observations,
        ).count()
        if candidate_count > 0:
            suggestion_candidate_archive_count += 1
            suggestion_candidate_mediafile_count += candidate_count

    if suggestion_candidate_mediafile_count > 0:
        suggestion_run_info = (
            f"Suggestions can be generated for {suggestion_candidate_mediafile_count} media files "
            f"in {suggestion_candidate_archive_count} finished uploaded archives."
        )
    else:
        suggestion_run_info = (
            "No eligible media files found for suggestions. "
            "Check whether the upload import is finished and whether the archive has the expected taxon."
        )

    # Only the first identity and the total count are displayed on the dashboard.
    identities = (
        IndividualIdentity.objects.filter(owner_workgroup=caiduser.workgroup, name__ne="nan")
        .annotate(
            representative_mediafile_count=Count(
                "animalobservation__mediafile",
                filter=Q(animalobservation__identity_is_representative=True),
                distinct=True,
            ),
            non_representative_mediafile_count=Count(
                "animalobservation__mediafile",
                filter=Q(animalobservation__identity_is_representative=False),
                distinct=True,
            ),
        )
        .filter(non_representative_mediafile_count__gt=0)
        .order_by("representative_mediafile_count", "-non_representative_mediafile_count")
    )
    manual_identification_count = models.get_mediafiles_with_missing_identity(caiduser).count()
    next_step_candidates = build_next_steps(workgroup)

    first_identity = identities.values("id", "name", "representative_mediafile_count").first()
    steps = []
    for step in next_step_candidates:
        item = asdict(step)
        item["updated_at"] = step.updated_at.isoformat() if step.updated_at else None
        steps.append(item)
    payload = {
        "identity_queue_count": identity_queue_count,
        "manual_identification_count": manual_identification_count,
        "suggestion_candidate_mediafile_count": suggestion_candidate_mediafile_count,
        "suggestion_candidate_archive_count": suggestion_candidate_archive_count,
        "suggestion_run_info": suggestion_run_info,
        "identities_by_representative_mediafiles": [first_identity] if first_identity else [],
        "representative_identity_count": identities.count(),
        "next_step_candidates": steps,
        "primary_next_step": steps[0] if steps else None,
        "identification_stats": {
            "identities": caiduser.number_of_identities(),
            "mediafiles": caiduser.number_of_media_files_for_identification(),
            "known_identity": caiduser.number_of_media_files_with_known_identity(),
            "missing_identity": caiduser.number_of_media_files_with_missing_identity(),
            "representative": workgroup.number_of_representative_media_files(),
        },
    }
    snapshot, _ = models.IdentificationDashboardSnapshot.objects.update_or_create(
        workgroup=workgroup, defaults={"payload": payload}
    )
    return snapshot


def identification_dashboard_context(workgroup):
    snapshot = models.IdentificationDashboardSnapshot.objects.filter(workgroup=workgroup).first()
    # Bootstrap once if the background refresher has not reached this group yet.
    # Always serve an existing snapshot, even when stale or refresh fails.
    if snapshot is None:
        snapshot = refresh_identification_dashboard_snapshot(workgroup)
    steps = [
        NextStepCandidate(
            **{**item, "updated_at": datetime.fromisoformat(item["updated_at"]) if item["updated_at"] else None}
        )
        for item in snapshot.payload["next_step_candidates"]
    ]
    return {
        **snapshot.payload,
        "next_step_candidates": steps,
        "primary_next_step": steps[0] if steps else None,
        "identification_dashboard_updated_at": snapshot.updated_at,
        "identification_dashboard_stale": snapshot.updated_at < timezone.now() - REFRESH_INTERVAL,
    }
