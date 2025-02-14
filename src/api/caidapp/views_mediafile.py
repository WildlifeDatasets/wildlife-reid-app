import os
import random
from typing import Optional

import django
from django.http import Http404, HttpResponseNotAllowed, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse_lazy
from django.utils import timezone
from django.contrib.auth.decorators import login_required, user_passes_test

from . import model_extra, models
from .forms import MediaFileForm
from .models import MediaFile
from .views import logger, media_files_update, message_view


@login_required
def stream_video(request, mediafile_id):
    """Stream video file."""
    mediafile = get_object_or_404(MediaFile, id=mediafile_id)
    if mediafile.media_type != "video":
        raise Http404("Not a video file")

    if (mediafile.preview is not None) and os.path.exists(mediafile.preview.path):
        video_path = mediafile.preview.path
    else:
        video_path = mediafile.mediafile.path
        logger.warning(f"Preview does not exist for mediafile {mediafile.id=}")
    if not os.path.exists(video_path):
        raise Http404()

    def file_iterator(file_name, chunk_size=8192):
        with open(file_name, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    # response = StreamingHttpResponse(file_iterator(video_path), content_type='/video/mp4')
    response = StreamingHttpResponse(file_iterator(video_path), content_type="video/x-m4v")
    response["Content-Length"] = os.path.getsize(video_path)
    response["Accept-Ranges"] = "bytes"

    return response


@login_required
def missing_taxon_annotation(request, uploaded_archive_id: Optional[int] = None, prev_mediafile_id: Optional[int] = None):
    """List of uploads."""
    # get uploadeda archive or None
    if uploaded_archive_id is not None:
        uploadedarchive = get_object_or_404(
            models.UploadedArchive,
            id=uploaded_archive_id,
            # **get_content_owner_filter_params(request.user.caiduser, "owner"),
        )
    else:
        uploadedarchive = None

    # pick random non-classified media file
    mediafiles = models.get_mediafiles_with_missing_taxon(
        request.user.caiduser, uploadedarchive=uploadedarchive
    )
    missing_count = mediafiles.count()
    if prev_mediafile_id is not None and mediafiles.count() > 1:
        mediafiles = mediafiles.exclude(id=prev_mediafile_id)
    # not_classified_taxon = models.Taxon.objects.get(name="Not Classified")
    # animalia_taxon = models.Taxon.objects.get(name="Animalia")

    # mediafiles = (
    #     MediaFile.objects.filter(
    #         Q(category=None) | Q(category=not_classified_taxon) |
    #         (Q(category=animalia_taxon) & Q(taxon_verified=False)),
    #         **get_content_owner_filter_params(request.user.caiduser, "parent__owner"),
    #         parent=uploadedarchive,
    #         parent__contains_single_taxon=False,
    #     ))

    # order by parent u
    #     ploaded_at and then by mediafile captured_at, then take first 10

    # Order by parent uploaded_at and then by mediafile captured_at, then take last 10
    last_ten_mediafiles = list(mediafiles.order_by("-parent__uploaded_at", "-captured_at")[:100])

    # Select a random media file from the last 10
    if last_ten_mediafiles:
        mediafile = random.choice(last_ten_mediafiles)
    else:
        mediafile = None  # Handle the case when there are no media files

    # .order_by("?")
    # .first()

    if uploadedarchive is not None:
        kwargs = {"uploaded_archive_id": uploadedarchive.id}
        if mediafile:
            kwargs["prev_mediafile_id"] = mediafile.id
        next_url = reverse_lazy( "caidapp:missing_taxon_annotation", kwargs=kwargs )
        if missing_count > 1:
            skip_url = reverse_lazy( "caidapp:missing_taxon_annotation", kwargs=kwargs )
        else:
            skip_url = None
        cancel_url = reverse_lazy(
            "caidapp:uploadedarchive_mediafiles", kwargs={"uploadedarchive_id": uploadedarchive.id}
        )
    else:
        kwargs = {"prev_mediafile_id": prev_mediafile_id}
        next_url = reverse_lazy( "caidapp:missing_taxon_annotation", kwargs=kwargs )
        skip_url = next_url
        cancel_url = reverse_lazy("caidapp:taxon_processing")

    if mediafile is None:
        if uploadedarchive is not None:
            message = f"All taxa known for {uploadedarchive.name}"
        else:
            message = "All taxa known"
        return message_view(request, message, link=cancel_url)
    return media_file_update(
        request,
        mediafile.id,
        next_text="Save and show next",
        next_url=next_url,
        skip_url=skip_url,
        cancel_url=cancel_url,
    )


@login_required
def verify_taxa_view(request, uploaded_archive_id: Optional[int] = None):
    """See media files for verification."""
    return media_files_update(
        request,
        show_overview_button=True,
        taxon_verified=False,
        uploadedarchive_id=uploaded_archive_id,
        order_by="category__name",
        parent__contains_single_taxon=False,
    )
    # views.


@login_required
def taxons_on_page_are_verified(request):
    """Mark taxons on page as verified."""
    # get 'mediafiles_ids_page' from session
    mediafile_ids = request.session.get("mediafile_ids_page", [])
    mediafiles = MediaFile.objects.filter(id__in=mediafile_ids)
    for mediafile in mediafiles:
        mediafile.taxon_verified = True
        mediafile.save()

    # get next page
    next_url = request.GET.get("next", reverse_lazy("caidapp:verify_taxa"))

    return redirect(next_url)


@login_required
def set_mediafiles_order_by(request, order_by: str):
    """Set order by for media files."""
    request.session["mediafiles_order_by"] = order_by
    # go back to the same page
    return redirect(request.META.get("HTTP_REFERER", "/"))

@login_required
def set_mediafiles_records_per_page(request, records_per_page: int):
    """Set records per page for media files."""
    request.session["mediafiles_records_per_page"] = records_per_page

    return redirect(request.META.get("HTTP_REFERER", "/"))


@login_required
def confirm_prediction(request, mediafile_id: int) -> JsonResponse:
    """Confirm prediction for media file with low confidence."""
    try:
        mediafile = get_object_or_404(MediaFile, id=mediafile_id)
        # user has rw access
        if model_extra.user_has_rw_access_to_mediafile(
            request.user.caiduser, mediafile, accept_none=True
        ):
            # Update the MediaFile instance
            mediafile.category = mediafile.predicted_taxon
            mediafile.updated_at = timezone.now()
            mediafile.updated_by = request.user.caiduser
            # mediafile.taxon_verified = True
            mediafile.save()

            return JsonResponse({"success": True, "message": "Prediction confirmed."})
        return JsonResponse({"success": False, "message": "No read/write access to the file"})
    except Exception:
        return JsonResponse({"success": False, "message": "Invalid request."})


@login_required
def media_file_update(
    request, media_file_id, next_text="Save", next_url=None, skip_url=None, cancel_url=None
):
    """Show and update media file."""
    # | Q(parent__owner=request.user.caiduser)
    # | Q(parent__owner__workgroup=request.user.caiduser.workgroup)
    mediafile = get_object_or_404(MediaFile, pk=media_file_id)
    if (mediafile.parent.owner.id != request.user.id) and (
        mediafile.parent.owner.workgroup != request.user.caiduser.workgroup
    ):
        return HttpResponseNotAllowed("Not allowed to see this media file.")

    if request.method == "POST":

        logger.debug(f"{next_url=}")
        if next_url:
            pass
            # return HttpResponseRedirect(next_url)
        else:
            next_url = request.GET.get("next")

            if next_url is None:
                # next url is where i am comming from
                next_url = request.META.get("HTTP_REFERER", "/")
                # stay here
                # next_url = reverse_lazy("caidapp:media_file_update", kwargs={"media_file_id": media_file_id})
                # next_url = reverse_lazy(
                #     "caidapp:uploadedarchive_mediafiles",
                #     kwargs={"uploadedarchive_id": mediafile.parent.id},
                # )
        # if "confirmTaxonSubmit" in request.POST:
        #     confirm_prediction(request, media_file_id)
        #     # decode_json
        #     # stay on the same page
        #     actual_url = request.META.get("HTTP_REFERER", "/")
        #
        #     return redirect(actual_url)

        form = MediaFileForm(request.POST, instance=mediafile)
        logger.debug(f"Form in POST: {mediafile=}")
        if form.is_valid():
            logger.debug(f"Form is valid")

            mediafile.updated_by = request.user.caiduser
            mediafile.updated_at = django.utils.timezone.now()
            # get uploaded archive
            mediafile = form.save()
            logger.debug(f"{mediafile.category=}")
            logger.debug(f"{mediafile.mediafile.path=}")
            logger.debug(f"{mediafile.updated_at=}")
            logger.debug(f"{next_url=}")
            # if (mediafile.category is not None) and (mediafile.category.name != "Not Classified"):
            #     # mediafile.taxon_verified = True
            #     # mediafile.taxon_verified_at = django.utils.timezone.now()
            #     mediafile.save()
            #     logger.debug(f"{mediafile.taxon_verified=}")
            #
            #     return redirect(next_url)
            return redirect(next_url)
        else:
            logger.error("Form is not valid.")
            from django.contrib import messages
            messages.error(request, "Form is not valid.")

    else:
        logger.debug(f"{next_url=}")
        if next_url:
            pass
            # return HttpResponseRedirect(next_url)
        else:
            next_url = request.GET.get("next")

            if next_url is None:
                # next url is where i am comming from
                next_url = request.META.get("HTTP_REFERER", "/")
        cancel_url = next_url
        logger.debug(f"{next_url=}")

        form = MediaFileForm(instance=mediafile)
    return render(
        request,
        "caidapp/media_file_update.html",
        {
            "form": form,
            "headline": "Media File",
            "button": next_text,
            "mediafile": mediafile,
            "skip_url": skip_url,
            "cancel_url": cancel_url,
        },
    )
