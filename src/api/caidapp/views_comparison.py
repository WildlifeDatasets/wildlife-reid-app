"""Read-only data endpoints for the two-panel comparison tool."""

import math

from django.contrib.auth.decorators import login_required
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.db.models import Q
from django.http import Http404, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse

from .models import Album, AnimalObservation, IndividualIdentity, MediaFile, user_has_access_filter_params
from .services.media_bboxes import mediafile_bbox_unions

PAGE_SIZE = 24
SOURCE_KINDS = {"observation", "mediafile", "identity", "album"}


def _workgroup(request):
    """The comparison UI is deliberately limited to the signed-in workgroup."""
    return request.user.caiduser.workgroup


def _mediafiles_for_request(request):
    caiduser = request.user.caiduser
    return MediaFile.objects.filter(
        Q(album__albumsharerole__user=caiduser) | Q(**user_has_access_filter_params(caiduser, "parent__owner"))
    ).distinct()


def _identities_for_request(request, mediafiles):
    """Return identities belonging to the workgroup, or personally visible ones.

    A user without a workgroup must not get access to every legacy identity
    whose ``owner_workgroup`` happens to be NULL.
    """
    workgroup = _workgroup(request)
    if workgroup is not None:
        return IndividualIdentity.objects.filter(owner_workgroup=workgroup)
    return IndividualIdentity.objects.filter(animalobservation__mediafile__in=mediafiles).distinct()


def _visible_albums(request):
    caiduser = request.user.caiduser
    return Album.objects.filter(Q(owner=caiduser) | Q(albumsharerole__user=caiduser)).distinct()


def _item_for_observation(observation, editable=False):
    mediafile = observation.mediafile
    image = mediafile.image_file or mediafile.mediafile
    thumbnail = mediafile.card_image
    mediafile_url = reverse("caidapp:media_file_update", args=[mediafile.id])
    observation_url = f"{mediafile_url}#observation-{observation.id}"
    return {
        "id": observation.id,
        "observation_id": observation.id,
        "mediafile_id": mediafile.id,
        "label": f"Observation {observation.id} — {mediafile.original_filename or f'Media file {mediafile.id}'}",
        "media_type": mediafile.media_type,
        "image_url": image.url if image and image.name else "",
        "media_url": mediafile.mediafile.url if mediafile.mediafile and mediafile.mediafile.name else "",
        "thumbnail_url": thumbnail.url if thumbnail and thumbnail.name else "",
        "bbox": (
            [
                observation.bbox_x_center,
                observation.bbox_y_center,
                observation.bbox_width,
                observation.bbox_height,
            ]
            if not observation.is_no_detection_placeholder
            and mediafile.media_type != "video"
            and all(
                value is not None and math.isfinite(value)
                for value in (
                    observation.bbox_x_center,
                    observation.bbox_y_center,
                    observation.bbox_width,
                    observation.bbox_height,
                )
            )
            and observation.bbox_width > 0
            and observation.bbox_height > 0
            else None
        ),
        "is_placeholder": observation.is_no_detection_placeholder,
        "identity_label": observation.identity.name if observation.identity else "",
        "detail_url": observation_url,
        "observation_edit_url": observation_url if editable else "",
        "mediafile_url": mediafile_url,
        "mediafile_edit_url": mediafile_url if editable else "",
        "identity_url": (
            reverse("caidapp:individual_identity_update", args=[observation.identity_id])
            if editable and observation.identity_id
            else ""
        ),
        "locality_url": (
            reverse("caidapp:update_locality", args=[mediafile.locality_id])
            if editable and mediafile.locality_id
            else ""
        ),
    }


def _item_for_mediafile(mediafile, editable=False):
    image = mediafile.image_file or mediafile.mediafile
    thumbnail = mediafile.card_image
    mediafile_url = reverse("caidapp:media_file_update", args=[mediafile.id])
    return {
        "id": mediafile.id,
        "observation_id": None,
        "mediafile_id": mediafile.id,
        "label": mediafile.original_filename or f"Media file {mediafile.id}",
        "media_type": mediafile.media_type,
        "image_url": image.url if image and image.name else "",
        "media_url": mediafile.mediafile.url if mediafile.mediafile and mediafile.mediafile.name else "",
        "thumbnail_url": thumbnail.url if thumbnail and thumbnail.name else "",
        "bbox": None,
        "is_placeholder": False,
        "identity_label": "",
        "detail_url": mediafile_url,
        "observation_edit_url": "",
        "mediafile_url": mediafile_url,
        "mediafile_edit_url": mediafile_url if editable else "",
        "identity_url": "",
        "locality_url": (
            reverse("caidapp:update_locality", args=[mediafile.locality_id])
            if editable and mediafile.locality_id
            else ""
        ),
    }


def _source_payload(kind, source):
    if kind == "observation":
        return {
            "kind": kind,
            "id": source.id,
            "label": f"Observation {source.id}",
            "detail_url": f"{reverse('caidapp:media_file_update', args=[source.mediafile_id])}#observation-{source.id}",
        }
    if kind == "mediafile":
        return {
            "kind": kind,
            "id": source.id,
            "label": source.original_filename or f"Media file {source.id}",
            "detail_url": reverse("caidapp:media_file_update", args=[source.id]),
        }
    if kind == "identity":
        return {
            "kind": kind,
            "id": source.id,
            "label": source.name,
            "description": source.code or "",
            "detail_url": reverse("caidapp:individual_identity_update", args=[source.id]),
        }
    return {
        "kind": kind,
        "id": source.hash,
        "label": source.name,
        "description": source.description,
        "detail_url": reverse("caidapp:album", args=[source.hash]),
    }


def _related_sources(request, mediafiles, mediafile_ids):
    """Collect visible identities and albums for each displayed media file in two queries."""
    related = {pk: {"identities": [], "albums": []} for pk in mediafile_ids}
    identities = (
        AnimalObservation.objects.filter(
            mediafile_id__in=mediafile_ids,
            identity__in=_identities_for_request(request, mediafiles),
            is_no_detection_placeholder=False,
        )
        .values("mediafile_id", "identity_id", "identity__name")
        .order_by("identity__name", "identity_id")
        .distinct()
    )
    for row in identities:
        related[row["mediafile_id"]]["identities"].append(
            {
                "kind": "identity",
                "id": row["identity_id"],
                "label": row["identity__name"],
                "detail_url": reverse("caidapp:individual_identity_update", args=[row["identity_id"]]),
            }
        )
    albums = (
        _visible_albums(request)
        .filter(mediafiles__id__in=mediafile_ids)
        .values("mediafiles__id", "hash", "name")
        .order_by("name", "id")
        .distinct()
    )
    for row in albums:
        related[row["mediafiles__id"]]["albums"].append(
            {
                "kind": "album",
                "id": row["hash"],
                "label": row["name"],
                "detail_url": reverse("caidapp:album", args=[row["hash"]]),
            }
        )
    return related


def _parse_page(request):
    try:
        page = int(request.GET.get("page", 1))
        if page < 1:
            raise ValueError
        return page
    except (TypeError, ValueError):
        raise Http404("Invalid page")


def _numeric_id_or_error(source_id):
    try:
        return int(source_id)
    except (TypeError, ValueError):
        return None


@login_required
def comparison(request):
    """Render the comparison workspace."""
    caiduser = request.user.caiduser
    return render(
        request,
        "caidapp/comparison.html",
        {
            "can_exclude_pair": caiduser.workgroup_id is not None and caiduser.can_edit("identity"),
        },
    )


@login_required
def comparison_source(request):
    """Return a paginated comparison source restricted to visible media files and albums."""
    kind = request.GET.get("kind")
    source_id = request.GET.get("id")
    mode = request.GET.get("mode")
    if kind not in SOURCE_KINDS or not source_id:
        return JsonResponse({"error": "kind and id are required"}, status=400)
    if mode and mode not in {"mediafiles", "observations"}:
        return JsonResponse({"error": "invalid mode"}, status=400)

    mediafiles = _mediafiles_for_request(request)
    if kind != "album":
        source_id = _numeric_id_or_error(source_id)
        if source_id is None:
            return JsonResponse({"error": "id must be numeric"}, status=400)
    if kind == "observation":
        source = get_object_or_404(
            AnimalObservation.objects.select_related("mediafile__locality", "identity"),
            pk=source_id,
            mediafile__in=mediafiles,
        )
        item_queryset = [source]
        serialize_item = _item_for_observation
    elif kind == "mediafile":
        source = get_object_or_404(mediafiles.select_related("locality"), pk=source_id)
        item_queryset = [source]
        serialize_item = _item_for_mediafile
    elif kind == "identity":
        source = get_object_or_404(_identities_for_request(request, mediafiles), pk=source_id)
        item_queryset = (
            AnimalObservation.objects.filter(
                mediafile__in=mediafiles, identity=source, is_no_detection_placeholder=False
            )
            .select_related("mediafile__locality", "identity")
            .order_by("-identity_is_representative", "-mediafile__captured_at", "-id")
        )
        serialize_item = _item_for_observation
    else:
        if mode is None:
            mode = "mediafiles"
        source = get_object_or_404(_visible_albums(request), hash=source_id)
        album_mediafiles = mediafiles.filter(album=source).distinct()
        if mode == "observations":
            item_queryset = (
                AnimalObservation.objects.filter(mediafile__in=album_mediafiles, is_no_detection_placeholder=False)
                .select_related("mediafile__locality", "identity")
                .order_by("mediafile__captured_at", "mediafile_id", "id")
            )
            serialize_item = _item_for_observation
        else:
            item_queryset = album_mediafiles.select_related("locality").order_by("captured_at", "id")
            serialize_item = _item_for_mediafile

    paginator = Paginator(item_queryset, PAGE_SIZE)
    try:
        page = paginator.page(_parse_page(request))
    except (EmptyPage, PageNotAnInteger):
        raise Http404("Page not found")
    page_objects = list(page.object_list)
    mediafile_ids = {item.mediafile_id if isinstance(item, AnimalObservation) else item.id for item in page_objects}
    editable_mediafile_ids = set(
        MediaFile.objects.for_user(request.user.caiduser).filter(id__in=mediafile_ids).values_list("id", flat=True)
    )
    related = _related_sources(request, mediafiles, mediafile_ids)
    bbox_unions = mediafile_bbox_unions([item.pk for item in page_objects if isinstance(item, MediaFile)])
    items = []
    for item in page_objects:
        mediafile_id = item.mediafile_id if isinstance(item, AnimalObservation) else item.pk
        data = serialize_item(item, mediafile_id in editable_mediafile_ids)
        if isinstance(item, MediaFile):
            data["bbox"] = bbox_unions.get(item.pk)
        data["related_sources"] = related[mediafile_id]
        items.append(data)
    source_payload = _source_payload(kind, source)
    if kind == "identity":
        source_payload["mediafile_count"] = item_queryset.values("mediafile_id").order_by().distinct().count()
    elif kind == "album":
        source_payload["mediafile_count"] = album_mediafiles.count()
    return JsonResponse(
        {
            "source": source_payload,
            "items": items,
            "page": page.number,
            "num_pages": paginator.num_pages,
            "count": paginator.count,
        }
    )


@login_required
def comparison_search(request):
    """Search visible comparison sources and return at most 30 matching entries."""
    kind = request.GET.get("kind")
    query = (request.GET.get("q") or "").strip()
    if kind not in SOURCE_KINDS:
        return JsonResponse({"error": "invalid kind"}, status=400)
    if not query:
        return JsonResponse({"results": []})
    mediafiles = _mediafiles_for_request(request)
    numeric_query = _numeric_id_or_error(query)
    if kind == "observation":
        filters = Q(mediafile__original_filename__icontains=query)
        if numeric_query is not None:
            filters |= Q(pk=numeric_query)
        results = (
            AnimalObservation.objects.filter(mediafile__in=mediafiles)
            .filter(filters)
            .select_related("mediafile")
            .order_by("-id")[:30]
        )
        data = [
            {"kind": kind, "id": item.id, "label": f"Observation {item.id} — {item.mediafile.original_filename}"}
            for item in results
        ]
    elif kind == "mediafile":
        filters = Q(original_filename__icontains=query)
        if numeric_query is not None:
            filters |= Q(pk=numeric_query)
        results = mediafiles.filter(filters).order_by("-id")[:30]
        data = [
            {"kind": kind, "id": item.id, "label": item.original_filename or f"Media file {item.id}"}
            for item in results
        ]
    elif kind == "identity":
        results = (
            _identities_for_request(request, mediafiles)
            .filter(Q(name__icontains=query) | Q(code__icontains=query))
            .order_by("name", "id")[:30]
        )
        data = [{"kind": kind, "id": item.id, "label": item.name} for item in results]
    else:
        results = _visible_albums(request).filter(name__icontains=query).order_by("name", "id")[:30]
        data = [{"kind": kind, "id": item.hash, "label": item.name} for item in results]
    return JsonResponse({"results": data})
