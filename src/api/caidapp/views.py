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
from django.db.models import Q
from django.forms import modelformset_factory
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse_lazy

from .forms import (
    AlbumForm,
    IndividualIdentityForm,
    MediaFileBulkForm,
    MediaFileForm,
    MediaFileSelectionForm,
    MediaFileSetQueryForm,
    UploadedArchiveForm,
    WorkgroupUsersForm,
)
from .models import (
    Album,
    IndividualIdentity,
    Location,
    MediaFile,
    MediafilesForIdentification,
    UploadedArchive,
    WorkGroup,
)
from .tasks import predict_on_error, predict_on_success

logger = logging.getLogger("app")


def login(request):
    """Login page."""
    if request.user.is_authenticated:
        return redirect("caidapp:uploads")
    else:
        return render(
            request,
            "caidapp/login.html",
        )


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

    records_per_page = 80
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


def individual_identities(request):
    """List of individual identities."""
    individual_identities = (
        IndividualIdentity.objects.filter(
            owner_workgroup=request.user.ciduser.workgroup,
        )
        .all()
        .order_by("-uploaded_at")
    )

    records_per_page = 24
    paginator = Paginator(individual_identities, per_page=records_per_page)

    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)
    return render(request, "caidapp/individual_identities.html", {"page_obj": page_obj})


def update_individual_identity(request, individual_identity_id):
    """Show and update media file."""
    individual_identity = get_object_or_404(
        IndividualIdentity,
        pk=individual_identity_id,
        owner_workgroup=request.user.ciduser.workgroup,
    )
    if request.method == "POST":
        form = IndividualIdentityForm(request.POST, instance=individual_identity)
        if form.is_valid():

            # get uploaded archive
            individual_identity = form.save()
            return redirect("caidapp:individual_identities")
    else:
        form = IndividualIdentityForm(instance=individual_identity)
    return render(
        request,
        "caidapp/individual_identity_update.html",
        {
            "form": form,
            "headline": "Individual Identity",
            "button": "Save",
            "individual_identity": individual_identity,
        },
    )


def get_individual_identity(request):
    """Show and update media file."""
    foridentification = MediafilesForIdentification.objects.order_by("?").first()
    return render(
        request, "caidapp/individual_identity.html", {"foridentification": foridentification}
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


def new_album(request):
    """Create new album."""
    if request.method == "POST":
        form = AlbumForm(request.POST)
        if form.is_valid():
            album = form.save(commit=False)
            album.owner = request.user.ciduser
            album.save()
            return redirect("caidapp:album_detail", album.id)
    else:
        form = AlbumForm()
    return render(
        request,
        "caidapp/model_form_upload.html",
        {"form": form, "headline": "New Album", "button": "Create"},
    )


def album_update(request, album_hash):
    """Show and update media file."""
    album = get_object_or_404(Album, hash=album_hash)
    if request.method == "POST":
        form = AlbumForm(request.POST, instance=album)
        if form.is_valid():

            # get uploaded archive
            album = form.save()
            return redirect("caidapp:albums")
    else:
        form = AlbumForm(instance=album)
    return render(
        request,
        "caidapp/album_update.html",
        {"form": form, "headline": "Album", "button": "Save", "mediafile": album},
    )


def delete_album(request, album_hash):
    """Delete album if it belongs to the user."""
    album = get_object_or_404(Album, hash=album_hash)
    if album.owner == request.user.ciduser:
        album.delete()
    return redirect("caidapp:albums")


# def album_update2(request, album_hash):
#     """Show album detail."""
#     album = get_object_or_404(Album, hash=album_hash)
#     mediafile_set = album.mediafile_set.all()
#
#     records_per_page = 80
#     paginator = Paginator(mediafile_set, per_page=records_per_page)
#
#     page_number = request.GET.get("page")
#     page_obj = paginator.get_page(page_number)
#     return render(
#         request,
#         "caidapp/album_detail.html",
#         {"page_obj": page_obj, "page_title": album},
#     )


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


@login_required
def albums(request):
    """Show all albums."""
    albums = (
        Album.objects.filter(
            Q(albumsharerole__user=request.user.ciduser) | Q(owner=request.user.ciduser)
        )
        .distinct()
        .all()
        .order_by("created_at")
    )
    return render(request, "caidapp/albums.html", {"albums": albums})


class MyLoginView(LoginView):
    redirect_authenticated_user = True

    def get_success_url(self):
        """Return url of next page."""
        return reverse_lazy("caidapp:uploads")

    def form_invalid(self, form):
        """Return error message if wrong username or password is given."""
        messages.error(self.request, "Invalid username or password")
        return self.render_to_response(self.get_context_data(form=form))


def _mediafiles_query(request, query: str, album_hash=None):
    """Prepare list of mediafiles based on query search in category and location."""
    # from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
    # vector = SearchVector("category", "location")
    # query = SearchQuery("Vulpes")
    # mediafiles = MediaFile.objects.annotate(rank=SearchRank(vector, query)).order_by("-rank")
    # mediafiles = (
    #     MediaFile.objects.filter(parent__owner=request.user.ciduser)
    #     .all()
    #     .order_by("-parent__uploaded_at")
    # )

    mediafiles = (
        MediaFile.objects.filter(
            Q(album__albumsharerole__user=request.user.ciduser)
            | Q(parent__owner=request.user.ciduser)
            | Q(parent__owner__workgroup=request.user.ciduser.workgroup)
        )
        .distinct()
        .order_by("-parent__uploaded_at")
    )
    if album_hash is not None:
        album = get_object_or_404(Album, hash=album_hash)
        mediafiles = (
            mediafiles.filter(album=album).all().distinct().order_by("-parent__uploaded_at")
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


def _page_number(request, queryform):
    """Prepare page number into queryform."""
    page_number = queryform.cleaned_data["pagenumber"]
    logger.debug(f"{page_number=}")
    if "nextPage" in request.POST:
        logger.debug("nextPage")
        if queryform.is_valid():
            queryform.cleaned_data["pagenumber"] += 1
            page_number = queryform.cleaned_data["pagenumber"]
            if queryform.is_valid():
                pass
                # queryform = MediaFileSetQueryForm(initial=queryform.cleaned_data)
    if "lastPage" in request.POST:
        logger.debug("nextPage")
        if queryform.is_valid():
            queryform.cleaned_data["pagenumber"] = -1
            page_number = queryform.cleaned_data["pagenumber"]
            if queryform.is_valid():
                pass
                # queryform = MediaFileSetQueryForm(initial=queryform.cleaned_data)
    if "prevPage" in request.POST:
        logger.debug("prevPage")
        if queryform.is_valid():
            queryform.cleaned_data["pagenumber"] -= 1
            page_number = queryform.cleaned_data["pagenumber"]
            if queryform.is_valid():
                pass
                # queryform = MediaFileSetQueryForm(initial=queryform.cleaned_data)
    if "firstPage" in request.POST:
        if queryform.is_valid():
            queryform.cleaned_data["pagenumber"] = 1
            page_number = queryform.cleaned_data["pagenumber"]
            if queryform.is_valid():
                pass
                # queryform = MediaFileSetQueryForm(initial=queryform.cleaned_data)
    return queryform.cleaned_data, page_number


def media_files_update(request, records_per_page=80, album_hash=None):
    """List of mediafiles based on query with bulk update of category."""
    # create list of mediafiles
    if request.method == "POST":
        queryform = MediaFileSetQueryForm(request.POST)
        logger.debug(queryform)
        query = queryform.cleaned_data["query"]
        queryform_cleaned_data, page_number = _page_number(request, queryform)
        queryform = MediaFileSetQueryForm(initial=queryform_cleaned_data)
    else:
        queryform = MediaFileSetQueryForm()
        page_number = 1
        query = ""
    albums_available = (
        Album.objects.filter(
            Q(albumsharerole__user=request.user.ciduser) | Q(owner=request.user.ciduser)
        )
        .distinct()
        .order_by("created_at")
    )
    logger.debug(f"{albums_available=}")

    logger.debug(f"{query=}")
    full_mediafiles = _mediafiles_query(request, query, album_hash=album_hash)
    number_of_mediafiles = len(full_mediafiles)

    paginator = Paginator(full_mediafiles, per_page=records_per_page)

    page_mediafiles = paginator.get_page(page_number)

    MediaFileFormSet = modelformset_factory(MediaFile, form=MediaFileSelectionForm, extra=0)
    if (request.method == "POST") and (
        ("btnBulkProcessing" in request.POST) or ("btnBulkProcessingAlbum" in request.POST)
    ):
        logger.debug("btnBulkProcessing")
        form_bulk_processing = MediaFileBulkForm(request.POST)
        if form_bulk_processing.is_valid():
            form_bulk_processing.save()

        form = MediaFileFormSet(request.POST)
        logger.debug("form")
        logger.debug(request.POST)
        if form.is_valid():
            logger.debug("form is valid")
            # if 'newsletter_sub' in .data:
            #     # do subscribe
            #     elif 'newsletter_unsub' in self.data:
            selected_album_hash = form.data["selectAlbum"]
            for mediafileform in form:
                if mediafileform.is_valid():
                    if mediafileform.cleaned_data["selected"]:
                        logger.debug("mediafileform is valid")
                        # reset selected field for refreshed view
                        mediafileform.cleaned_data["selected"] = False
                        mediafileform.selected = False
                        if "btnBulkProcessingAlbum" in form.data:
                            logger.debug("Select Album :" + form.data["selectAlbum"])
                            if selected_album_hash == "new":
                                logger.debug("Creating new album")
                                instance: MediaFile = mediafileform.save(commit=False)
                                album = create_new_album(request)
                                album.cover = instance
                                album.save()
                                instance.album_set.add(album)
                                instance.save()
                                selected_album_hash = album.hash
                            else:
                                logger.debug("selectAlbum")
                                instance = mediafileform.save(commit=False)
                                logger.debug(f"{selected_album_hash=}")
                                album = get_object_or_404(Album, hash=selected_album_hash)

                                # check if file is not already in album
                                if instance.album_set.filter(pk=album.pk).count() == 0:
                                    # add file to album
                                    instance.album_set.add(album)
                                    instance.save()
                        elif "btnBulkProcessing" in form.data:
                            instance = mediafileform.save(commit=False)
                            instance.category = form_bulk_processing.cleaned_data["category"]
                            instance.save()
                    # mediafileform.save()
            # form.save()
        else:
            logger.debug("form is not valid")
            logger.debug(form.errors)
        # queryform = MediaFileSetQueryForm(request.POST)
        form_bulk_processing = MediaFileBulkForm()
        page_query = full_mediafiles.filter(id__in=[object.id for object in page_mediafiles])
        form = MediaFileFormSet(queryset=page_query)
    else:

        logger.debug("initial form processing")
        form_bulk_processing = MediaFileBulkForm()
        page_query = full_mediafiles.filter(id__in=[object.id for object in page_mediafiles])
        form = MediaFileFormSet(queryset=page_query)

    logger.debug("ready to render page")
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
            "albums_available": albums_available,
            "number_of_mediafiles": number_of_mediafiles,
        },
    )


def create_new_album(request, name="New Album"):
    """Create new album."""
    album = Album()
    album.name = name
    album.owner = request.user.ciduser
    album.save()
    return album


def workgroup_update(request, workgroup_hash: str):
    """Update workgroup."""
    workgroup = get_object_or_404(WorkGroup, hash=workgroup_hash)
    if request.method == "POST":
        form = WorkgroupUsersForm(request.POST)
        logger.debug(request.POST)
        logger.debug(form)
        if form.is_valid():
            logger.debug(form.cleaned_data)
            workgroup_users_all = workgroup.ciduser_set.all()
            logger.debug(f"Former all users {workgroup_users_all}")
            workgroup.ciduser_set.set(form.cleaned_data["workgroup_users"])

            pass
            # logger
            # form.save()
            # return redirect("workgroup_list")
    else:

        workgroup_users = workgroup.ciduser_set.all()
        data = {
            # 'id': dog_request_id,
            # 'color': dog_color,
            "workgroup_users": workgroup_users,
        }
        form = WorkgroupUsersForm(data)
        # form = WorkgroupUsersForm(instance=workgroup.)
    return render(
        request,
        "caidapp/update_form.html",
        {
            "form": form,
            "headline": "Update workgroup",
            "button": "Save",
            # "user_is_staff": request.user.is_staff,
        },
    )
    return render(request, "caidapp/update_form.html", {"form": workgroup_hash})
