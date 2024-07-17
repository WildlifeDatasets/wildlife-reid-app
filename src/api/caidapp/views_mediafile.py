from django.shortcuts import Http404, HttpResponse, get_object_or_404, redirect, render
from django.urls import reverse_lazy
from django.http import StreamingHttpResponse, Http404
from django.shortcuts import get_object_or_404
from .models import MediaFile, get_content_owner_filter_params
import os
import random
from typing import Optional

from .views import message, media_file_update


def stream_video(request, mediafile_id):
    mediafile = get_object_or_404(MediaFile, id=mediafile_id)
    if mediafile.media_type != "video":
        raise Http404("Not a video file")

    video_path = mediafile.mediafile.path
    if not os.path.exists(video_path):
        raise Http404()

    def file_iterator(file_name, chunk_size=8192):
        with open(file_name, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    response = StreamingHttpResponse(file_iterator(video_path), content_type='video/mp4')
    response['Content-Length'] = os.path.getsize(video_path)
    response['Accept-Ranges'] = 'bytes'

    return response


def manual_taxon_classification_on_non_classified(request):
    """List of uploads."""
    # pick random non-classified media file
    mediafiles = (
        MediaFile.objects.filter(
            **get_content_owner_filter_params(request.user.caiduser, "parent__owner"),
            # parent__owner__workgroup=request.user.caiduser.workgroup, # but this would work too
            category__name="Not Classified",
            parent__contains_single_taxon=False,
        ))
    # order by parent u
    #     ploaded_at and then by mediafile captured_at, then take first 10

    # Order by parent uploaded_at and then by mediafile captured_at, then take last 10
    last_ten_mediafiles = list(mediafiles.order_by("-parent__uploaded_at", "-captured_at")[:10])

    # Select a random media file from the last 10
    if last_ten_mediafiles:
        mediafile = random.choice(last_ten_mediafiles)
    else:
        mediafile = None  # Handle the case when there are no media files

    # .order_by("?")
    # .first()
    if mediafile is None:
        return message(request, "No non-classified media files.")
    return media_file_update(
        request,
        mediafile.id,
        next_text="Save",
        next_url=reverse_lazy("caidapp:manual_taxon_classification_on_non_classified"),
        skip_url=reverse_lazy("caidapp:manual_taxon_classification_on_non_classified"),
    )

def overview_taxons(request, uploaded_archive_id:Optional[int]=None):
    from .views import media_files_update

    return media_files_update(
        request, show_overview_button=True, taxon_overviewed=False, uploadedarchive_id=uploaded_archive_id,
        order_by="category__name"
        )
    # views.

def taxons_on_page_are_overviewed(request):
    # get 'mediafiles_ids_page' from session
    mediafile_ids = request.session.get("mediafile_ids_page", [])
    mediafiles = MediaFile.objects.filter(id__in=mediafile_ids)
    for mediafile in mediafiles:
        mediafile.taxon_overviewed = True
        mediafile.save()

    # get next page
    next_url = request.GET.get('next', reverse_lazy("caidapp:overview_taxons")  )

    return redirect(next_url)

def set_mediafiles_order_by(request, order_by:str):
    request.session["mediafiles_order_by"] = order_by
    # go back to the same page
    return redirect(request.META.get("HTTP_REFERER", "/"))

def set_mediafiles_records_per_page(request, records_per_page:int):
    request.session["mediafiles_records_per_page"] = records_per_page

    return redirect(request.META.get("HTTP_REFERER", "/"))
