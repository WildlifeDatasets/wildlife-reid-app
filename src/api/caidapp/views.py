import json
import logging
import os
from pathlib import Path

import django
from celery import signature
from django.conf import settings
from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
from django.core.paginator import Paginator
from django.forms import modelformset_factory
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse_lazy

from .forms import (
    MediaFileBulkForm,
    MediaFileForm,
    MediaFileSelectionForm,
    MediaFileSetQueryForm,
    UploadedArchiveForm,
)
from .models import Location, MediaFile, UploadedArchive
from .tasks import predict_on_error, predict_on_success

logger = logging.getLogger("app")


def media_files(request):
    """List of uploads."""
    mediafiles = (
        MediaFile.objects.filter(parent__owner=request.user.ciduser)
        .all()
        .order_by("-parent__uploaded_at")
    )

    records_per_page = 10000
    paginator = Paginator(mediafiles, per_page=records_per_page)

    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)
    qs_data = {}
    for e in mediafiles:
        qs_data[e.id] = str(e.category) + " " + str(e.location)
        # qs_data.append(e.id)
    logger.debug(qs_data)
    qs_json = json.dumps(qs_data)
    return render(
        request,
        "caidapp/media_files.html",
        {
            "page_obj": page_obj,
            "page_title": "Media files",
            "qs_json": qs_json,
            "user_is_staff": request.user.is_staff,
        },
    )


def manage_locations(request):
    """Add new location or update names of locations."""
    LocationFormSet = modelformset_factory(
        Location, fields=("name",), can_delete=False, can_order=False
    )
    if request.method == "POST":
        form = LocationFormSet(request.POST)
        if form.is_valid():
            form.save()
    else:
        form = LocationFormSet()

    return render(
        request,
        "caidapp/manage_locations.html",
        {
            "page_obj": form,
        },
    )


def uploadedarchive_detail(request, uploadedarchive_id):
    """List of uploads."""
    uploadedarchive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    mediafile_set = uploadedarchive.mediafile_set.all()

    records_per_page = 12
    paginator = Paginator(mediafile_set, per_page=records_per_page)

    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)
    return render(
        request,
        "caidapp/uploadedarchive_detail.html",
        {"page_obj": page_obj, "page_title": uploadedarchive},
    )


def uploads(request):
    """List of uploads."""
    uploadedarchives = (
        UploadedArchive.objects.filter(
            owner=request.user.ciduser,
        )
        .all()
        .order_by("-uploaded_at")
    )

    records_per_page = 12
    paginator = Paginator(uploadedarchives, per_page=records_per_page)

    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)
    return render(request, "caidapp/uploads.html", {"page_obj": page_obj})


def logout_view(request):
    """Logout from the application."""
    logout(request)
    # Redirect to a success page.
    return redirect("/caidapp/login")


def media_file_update(request, media_file_id):
    """Show and update media file."""
    mediafile = get_object_or_404(MediaFile, pk=media_file_id)
    if request.method == "POST":
        form = MediaFileForm(request.POST, instance=mediafile)
        if form.is_valid():

            # get uploaded archive
            mediafile = form.save()
            return redirect("caidapp:uploadedarchive_detail", mediafile.parent.id)
    else:
        form = MediaFileForm(instance=mediafile)
    return render(
        request,
        "caidapp/media_file_update.html",
        {"form": form, "headline": "Media File", "button": "Save", "mediafile": mediafile},
    )


def _run_processing(uploaded_archive: UploadedArchive):
    # update record in the database
    output_dir = Path(settings.MEDIA_ROOT) / uploaded_archive.outputdir
    uploaded_archive.started_at = django.utils.timezone.now()
    output_archive_file = output_dir / "images.zip"
    output_metadata_file = output_dir / "metadata.csv"
    uploaded_archive.status = "Processing"
    uploaded_archive.save()

    # send celery message to the data worker
    logger.info("Sending request to inference worker.")
    sig = signature(
        "predict",
        kwargs={
            "input_archive_file": str(
                Path(settings.MEDIA_ROOT) / uploaded_archive.archivefile.name
            ),
            "output_dir": str(output_dir),
            "output_archive_file": str(output_archive_file),
            "output_metadata_file": str(output_metadata_file),
        },
    )
    task = sig.apply_async(
        link=predict_on_success.s(
            uploaded_archive_id=uploaded_archive.id,
            zip_file=os.path.relpath(str(output_archive_file), settings.MEDIA_ROOT),
            csv_file=os.path.relpath(str(output_metadata_file), settings.MEDIA_ROOT),
        ),
        link_error=predict_on_error.s(uploaded_archive_id=uploaded_archive.id),
    )
    logger.info(f"Created worker task with id '{task.task_id}'.")


@staff_member_required
def run_processing(request, uploadedarchive_id):
    """Run processing of uploaded archive."""
    uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    _run_processing(uploaded_archive)
    return redirect("/caidapp/uploads")


def model_form_upload(request):
    """Process the uploaded zip file."""
    if request.method == "POST":
        form = UploadedArchiveForm(
            request.POST,
            request.FILES,
        )
        if form.is_valid():

            # get uploaded archive
            uploaded_archive = form.save()
            uploaded_archive_suffix = Path(uploaded_archive.archivefile.name).suffix.lower()
            if uploaded_archive_suffix not in (".tar", ".tar.gz", ".zip"):
                logger.warning(
                    f"Uploaded file with extension '{uploaded_archive_suffix}' is not an archive."
                )
                messages.warning(
                    f"Uploaded file with extension '{uploaded_archive_suffix}' is not an archive."
                )

            uploaded_archive.owner = request.user.ciduser
            uploaded_archive.save()
            _run_processing(uploaded_archive)

            return JsonResponse({"data": "Data uploaded"})
        else:
            return JsonResponse({"data": "Someting went wrong"})

    else:
        form = UploadedArchiveForm()
    return render(
        request,
        "caidapp/model_form_upload.html",
        {"form": form, "headline": "Upload", "button": "Upload"},
    )


@login_required
def delete_upload(request, uploadedarchive_id):
    """Delete uploaded file."""
    uploadedarchive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    if uploadedarchive.owner.id == request.user.id:
        uploadedarchive.delete()
    else:
        messages.error(request, "Only the owner can delete the file")
    return redirect("/caidapp/uploads")


@login_required
def delete_mediafile(request, mediafile_id):
    """Delete uploaded file."""
    obj = get_object_or_404(MediaFile, pk=mediafile_id)
    parent_id = obj.parent_id
    if obj.parent.owner.id == request.user.id:
        obj.delete()
    else:
        messages.error(request, "Only the owner can delete the file.")
    return redirect("caidapp:uploadedarchive_detail", uploadedarchive_id=parent_id)


class MyLoginView(LoginView):
    redirect_authenticated_user = True

    def get_success_url(self):
        """Return url of next page."""
        return reverse_lazy("caidapp:uploads")

    def form_invalid(self, form):
        """Return error message if wrong username or password is given."""
        messages.error(self.request, "Invalid username or password")
        return self.render_to_response(self.get_context_data(form=form))


def _mediafiles_query(request, query: str):
    """Prepare list of mediafiles based on query search in category and location."""
    # from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
    # vector = SearchVector("category", "location")
    # query = SearchQuery("Vulpes")
    # mediafiles = MediaFile.objects.annotate(rank=SearchRank(vector, query)).order_by("-rank")
    mediafiles = (
        MediaFile.objects.filter(parent__owner=request.user.ciduser)
        .all()
        .order_by("-parent__uploaded_at")
    )

    if len(query) == 0:
        return mediafiles
    else:
        words = [query]

        queryset_combination = None
        for word in words:
            if queryset_combination is None:
                queryset_combination = mediafiles.filter(category__name__icontains=word).all()
            else:
                queryset_combination |= mediafiles.filter(category__name__icontains=word).all()

            queryset_combination |= mediafiles.filter(location__name__icontains=word).all()

        # queryset_combination.all().order_by("-parent_uploaded_at")
        mediafiles = queryset_combination.all().order_by("-parent__uploaded_at")
        return mediafiles


def media_files_update(request, records_per_page=120):
    """List of mediafiles based on query with bulk update of category."""
    # create list of mediafiles
    if request.method == "POST":
        # and ("querySubmit" in request.POST):
        queryform = MediaFileSetQueryForm(request.POST)
        logger.debug(queryform)
        query = queryform.cleaned_data["query"]
    else:
        queryform = MediaFileSetQueryForm()
        query = ""

    logger.debug(f"{query=}")
    full_mediafiles = _mediafiles_query(request, query)

    paginator = Paginator(full_mediafiles, per_page=records_per_page)

    page_number = request.GET.get("page")
    page_mediafiles = paginator.get_page(page_number)

    MediaFileFormSet = modelformset_factory(MediaFile, form=MediaFileSelectionForm, extra=0)
    if (request.method == "POST") and ("btnBulkProcessing" in request.POST):
        form_bulk_processing = MediaFileBulkForm(request.POST)

        form = MediaFileFormSet(request.POST)
        if form.is_valid():
            for mediafileform in form:
                if mediafileform.is_valid():
                    if mediafileform.cleaned_data["selected"]:
                        mediafileform.cleaned_data["selected"] = False
                        mediafileform.selected = False
                        instance = mediafileform.save(commit=False)
                        instance.category = form_bulk_processing.cleaned_data["category"]
                        instance.save()
                    mediafileform.save()
            form.save()
        queryform = MediaFileSetQueryForm(request.POST)
        form_bulk_processing = MediaFileBulkForm()
        page_query = full_mediafiles.filter(id__in=[object.id for object in page_mediafiles])
        form = MediaFileFormSet(queryset=page_query)
    else:
        form_bulk_processing = MediaFileBulkForm()
        page_query = full_mediafiles.filter(id__in=[object.id for object in page_mediafiles])
        form = MediaFileFormSet(queryset=page_query)

    return render(
        request,
        "caidapp/media_files_update.html",
        {
            "page_obj": page_mediafiles,
            "form_objects": form,
            "page_title": "Media files",
            "user_is_staff": request.user.is_staff,
            "form_bulk_processing": form_bulk_processing,
            "form_query": queryform,
        },
    )
