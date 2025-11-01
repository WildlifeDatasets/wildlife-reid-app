import logging

from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.db.models import Q
from django.shortcuts import redirect
from tqdm import tqdm

from . import models
from .models import MediaFile

logger = logging.getLogger(__name__)


@staff_member_required
def do_admin_stuff(request, process_name: str):
    """Do admin stuff."""

    logger.debug(f"{process_name=}")
    if process_name == "clean_mediafiles_with_no_file_attached":
        return clean_mediafiles_with_no_file_attached_view(request)
    elif process_name == "refresh_area":
        return refresh_area(request)
    elif process_name == "refresh_thumbnails":
        refresh_thumbnails(request)
        return redirect(request.META.get("HTTP_REFERER", "/"))
    elif process_name == "force_refresh_thumbnails":
        refresh_thumbnails(request, force=True)
        return redirect(request.META.get("HTTP_REFERER", "/"))
    else:
        messages.error(request, f"Process name '{process_name}' not recognized.")
        return redirect(request.META.get("HTTP_REFERER", "/"))


@staff_member_required
def clean_mediafiles_with_no_file_attached_view(request):
    """View for cleaning mediafiles with no file attached."""
    # get all mediafiles
    mediafiles = MediaFile.objects.all()
    # get all mediafiles with no file attached
    # n_mediafiles = len(mediafiles)

    mediafiles_no_file = mediafiles.filter(Q(mediafile__isnull=True) | Q(mediafile=""))

    # mediafiles_no_file = mediafiles.filter(mediafile__isnull=True)
    n_empty_mediafiles = len(mediafiles_no_file)
    # delete mediafiles with no file attached
    mediafiles_no_file.delete()
    # get all mediafiles again
    mediafiles = MediaFile.objects.all()
    messages.info(request, f"Number of removed empty mediafiles: {n_empty_mediafiles}")
    messages.info(request, f"Actual number of mediafiles: {len(mediafiles)}")
    # go back to referer
    return redirect(request.META.get("HTTP_REFERER", "/"))


@staff_member_required
def refresh_area(request):

    for locality in tqdm(models.Locality.objects.all()):
        locality.set_closest_area()

    return redirect(request.META.get("HTTP_REFERER", "/"))


@staff_member_required
def refresh_thumbnails(request, force: bool = False):
    """Refresh all thumbnails."""
    logger.debug("Refreshing all thumbnails")
    for mf in tqdm(MediaFile.objects.all()):
        mf.make_thumbnail_for_mediafile_if_necessary(force=force)
    logger.debug("Refreshed all thumbnails")
