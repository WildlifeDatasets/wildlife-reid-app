"""Configure and queue beta-stability exports for the saved observation selection."""

import uuid
from pathlib import Path

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError
from django.db.models import Q
from django.http import JsonResponse
from django.shortcuts import render

from . import models

BETA_STABILITY_SELECTION_KEY = "beta_stability_export_observation_ids"


def authorized_observations(caiduser, observation_ids):
    """Recheck access when the download is requested and when the worker executes it."""
    return models.AnimalObservation.objects.filter(
        Q(mediafile__album__albumsharerole__user=caiduser)
        | Q(**models.user_has_access_filter_params(caiduser, "mediafile__parent__owner")),
        pk__in=observation_ids,
    ).distinct()


def selected_profiles(observations, profile_uuids):
    """Only expose profiles belonging to embedding records of accessible observations."""
    requested = [uuid.UUID(value) for value in profile_uuids]
    if not requested or len(set(requested)) != len(requested):
        raise ValidationError("Select at least one distinct embedding profile.")
    available = models.EmbeddingProfile.objects.filter(
        embeddings__observation__in=observations, uuid__in=requested
    ).distinct()
    by_uuid = {profile.uuid: profile for profile in available}
    if set(requested) != set(by_uuid):
        raise ValidationError("An embedding profile is unavailable for this selection.")
    return [by_uuid[value] for value in requested]


@login_required
def beta_stability_export(request):
    """Show profile/options selection or queue a ZIP using the existing download workflow."""
    ids = request.session.get(BETA_STABILITY_SELECTION_KEY, [])
    observations = authorized_observations(request.user.caiduser, ids)
    if request.GET.get("download") == "1":
        if not ids or set(observations.values_list("pk", flat=True)) != set(ids):
            return JsonResponse({"message": "The observation selection is empty or no longer accessible."}, status=400)
        try:
            profiles = selected_profiles(observations, request.GET.getlist("profile"))
        except (ValueError, ValidationError) as exc:
            return JsonResponse({"message": str(exc)}, status=400)
        from .tasks import clean_old_mediafile_zips, create_beta_stability_export

        caiduser = request.user.caiduser
        output = Path(settings.MEDIA_ROOT) / "users" / caiduser.hash / f"beta_stability.{uuid.uuid4()}.zip"
        task = create_beta_stability_export.delay(
            caiduser.pk,
            ids,
            [str(profile.uuid) for profile in profiles],
            str(output),
            include_images=request.GET.get("include_images") == "1",
            allow_partial=request.GET.get("allow_partial") == "1",
        )
        clean_old_mediafile_zips.delay(str(output.parent), glob_pattern="beta_stability.*.zip", max_age_days=7)
        return JsonResponse({"task_id": task.id})
    profiles = (
        models.EmbeddingProfile.objects.filter(embeddings__observation__in=observations)
        .distinct()
        .order_by("-created_at")
    )
    return render(
        request,
        "caidapp/beta_stability_export.html",
        {
            "profiles": profiles,
            "observation_count": observations.count(),
        },
    )
