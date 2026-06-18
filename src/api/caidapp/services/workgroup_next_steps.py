from dataclasses import dataclass
from datetime import datetime

from django.db.models import Count, Q
from django.urls import reverse

from .. import models


@dataclass(frozen=True)
class NextStepCandidate:
    code: str
    priority: int
    text: str
    url: str
    updated_at: datetime | None = None
    source: str = "live"


def build_next_steps(workgroup: models.WorkGroup | None) -> list[NextStepCandidate]:
    """Build a prioritized list of next-step suggestions for a workgroup."""
    if workgroup is None:
        return []

    candidates: list[NextStepCandidate] = []

    upload_count = models.UploadedArchive.objects.filter(owner__workgroup=workgroup).count()
    identities = models.IndividualIdentity.objects.filter(owner_workgroup=workgroup).exclude(name="nan")
    identity_count = identities.count()

    if upload_count == 0:
        candidates.append(
            NextStepCandidate(
                code="no_uploads",
                priority=10,
                text="Upload media files to start building the identification database.",
                url=reverse("caidapp:new_upload"),
            )
        )

    identification_mediafiles = models.MediaFile.objects.filter(
        parent__owner__workgroup=workgroup,
        parent__is_for_identification=True,
        parent__import_finished=True,
    )
    identification_mediafiles = models.filter_mediafiles_by_identification_taxon(
        identification_mediafiles, workgroup
    )
    has_assigned_identity = identification_mediafiles.filter(
        Q(identity__isnull=False) | Q(observations__identity__isnull=False)
    ).exists()
    if identification_mediafiles.exists() and not has_assigned_identity:
        candidates.append(
            NextStepCandidate(
                code="manual_identification",
                priority=15,
                text="Assign identities manually to the uploaded identification media files.",
                url=reverse("caidapp:manual_identification"),
            )
        )

    if identity_count == 0:
        candidates.append(
            NextStepCandidate(
                code="no_identities",
                priority=20,
                text="Create the first identity or upload identified media files.",
                url=reverse("caidapp:new_upload"),
            )
        )
        return sorted(candidates, key=lambda item: item.priority)

    identities_with_counts = identities.annotate(
        mediafile_count=Count("mediafile", distinct=True),
        representative_mediafile_count=Count(
            "mediafile",
            filter=Q(mediafile__identity_is_representative=True),
            distinct=True,
        ),
    )

    identities_without_mediafiles = identities_with_counts.filter(mediafile_count=0)
    if identities_without_mediafiles.exists():
        candidates.append(
            NextStepCandidate(
                code="identities_without_mediafiles",
                priority=30,
                text=(
                    f"{identities_without_mediafiles.count()} identities do not have any media files assigned yet."
                ),
                url=reverse("caidapp:individual_identities"),
            )
        )

    identities_without_representatives = identities_with_counts.filter(
        mediafile_count__gt=0,
        representative_mediafile_count=0,
    ).order_by("name")
    if identities_without_representatives.exists():
        first_identity = identities_without_representatives.first()
        candidates.append(
            NextStepCandidate(
                code="identities_without_representatives",
                priority=40,
                text=(
                    f"{identities_without_representatives.count()} identities do not have a representative media file."
                ),
                url=reverse("caidapp:individual_identity_mediafiles", args=[first_identity.id]),
            )
        )

    code_suggestion_count = 0
    for identity in identities.only("id", "name", "code", "owner_workgroup"):
        suggested_code = identity.suggested_code_from_name()
        if suggested_code and identity.code != suggested_code:
            code_suggestion_count += 1

    if code_suggestion_count > 0:
        candidates.append(
            NextStepCandidate(
                code="identity_code_suggestions_available",
                priority=50,
                text=f"{code_suggestion_count} identities contain a code that can be extracted from the name.",
                url=reverse("caidapp:show_identity_code_suggestions"),
            )
        )

    latest_merge_suggestions = (
        models.MergeIdentitySuggestionResult.objects.filter(workgroup=workgroup).order_by("-created_at").first()
    )
    if latest_merge_suggestions and latest_merge_suggestions.suggestions:
        candidates.append(
            NextStepCandidate(
                code="merge_suggestions_available",
                priority=60,
                text=f"{len(latest_merge_suggestions.suggestions)} identity merge suggestions are available.",
                url=reverse("caidapp:suggest_merge_identities"),
                updated_at=latest_merge_suggestions.created_at,
                source="async",
            )
        )

    return sorted(candidates, key=lambda item: item.priority)
