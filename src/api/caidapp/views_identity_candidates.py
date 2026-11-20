"""Retrieve and explicitly accept candidates for exactly one observation."""

from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from .models import AnimalObservation, IndividualIdentity, MediaFile
from .services.identity_candidates import current_input, search
from .views_comparison import _item_for_observation


def _observations(request):
    """Keep retrieval and writes within accessible media files of the current group."""
    member = request.user.caiduser
    if not member.workgroup_id:
        return AnimalObservation.objects.none()
    mediafiles = MediaFile.objects.for_user(member).filter(parent__owner__workgroup_id=member.workgroup_id)
    return AnimalObservation.objects.filter(mediafile__in=mediafiles)


@login_required
@require_GET
def identity_candidates(request, observation_uuid):
    """Return exact cosine candidates with reference crops and gallery coverage."""
    observations = _observations(request)
    observation = get_object_or_404(observations.select_related("mediafile", "identity"), uuid=observation_uuid)
    try:
        payload = search(observation, request.user.caiduser.workgroup, observations)
    except (ValidationError, OSError, ValueError) as exc:
        message = (
            "; ".join(exc.messages) if isinstance(exc, ValidationError) else "The source media file is unavailable."
        )
        return JsonResponse({"message": message}, status=400)
    payload["observation"] = _item_for_observation(observation)
    payload["expected_identity_id"] = observation.identity_id
    for candidate in payload["candidates"]:
        candidate["reference"] = _item_for_observation(candidate["reference"])
    response = JsonResponse(payload)
    response["Cache-Control"] = "no-store"
    return response


@login_required
@require_POST
def accept_identity_candidate(request, observation_uuid):
    """Reject stale dialogs and update the selected observation, never its siblings."""
    with transaction.atomic():
        observation = get_object_or_404(_observations(request).select_for_update(), uuid=observation_uuid)
        try:
            identity_id = int(request.POST.get("identity_id", ""))
            expected = request.POST["expected_identity_id"]
            expected = int(expected) if expected else None
            input_key = current_input(observation)
        except (ValueError, KeyError, ValidationError, OSError):
            return JsonResponse({"message": "The observation is unavailable or the request is invalid."}, status=400)
        if input_key != request.POST.get("input_fingerprint") or observation.identity_id != expected:
            return JsonResponse({"message": "The observation changed. Reopen the suggestions."}, status=409)
        identity = get_object_or_404(
            IndividualIdentity, pk=identity_id, owner_workgroup_id=request.user.caiduser.workgroup_id
        )
        if observation.identity_id != identity.pk:
            observation.identity_is_representative = False
        observation.identity = identity
        observation.updated_by = request.user.caiduser
        observation.updated_at = timezone.now()
        observation.save(update_fields=["identity", "identity_is_representative", "updated_by", "updated_at"])
    return JsonResponse({"identity_id": identity.pk, "label": str(identity)})
