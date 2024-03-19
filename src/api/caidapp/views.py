import datetime
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import django
import pandas as pd
import plotly.graph_objects as go
import pytz
from celery import signature
from django.conf import settings
from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
from django.core.paginator import Paginator
from django.db.models import Q, QuerySet
from django.forms import modelformset_factory
from django.http import HttpResponseBadRequest, HttpResponseNotAllowed, JsonResponse
from django.shortcuts import Http404, HttpResponse, get_object_or_404, redirect, render
from django.urls import reverse_lazy

from .forms import (
    AlbumForm,
    IndividualIdentityForm,
    LocationForm,
    MediaFileBulkForm,
    MediaFileForm,
    MediaFileSelectionForm,
    MediaFileSetQueryForm,
    UploadedArchiveForm,
    WorkgroupUsersForm,
)
from .models import (
    Album,
    ArchiveCollection,
    CaIDUser,
    IndividualIdentity,
    Location,
    MediaFile,
    MediafilesForIdentification,
    Taxon,
    UploadedArchive,
    WorkGroup,
)
from .tasks import (
    _prepare_dataframe_for_identification,
    identify_on_success,
    init_identification_on_error,
    init_identification_on_success,
    on_error_in_upload_processing,
    run_species_prediction_async,
    update_metadata_csv_by_uploaded_archive,
)

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
        MediaFile.objects.filter(
            **_user_content_filter_params(request.user.caiduser, "parent__owner")
            # parent__owner=request.user.caiduser
        )
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


def manual_taxon_classification_on_non_classified(request):
    """List of uploads."""
    # pick random non-classified media file
    mediafile = (
        MediaFile.objects.filter(
            **_user_content_filter_params(request.user.caiduser, "parent__owner"),
            # parent__owner__workgroup=request.user.caiduser.workgroup, # but this would work too
            category__name="Not Classified",
            parent__contains_single_taxon=False,
        )
        .order_by("?")
        .first()
    )
    if mediafile is None:
        return message(request, "No non-classified media files.")
    return media_file_update(
        request,
        mediafile.id,
        next_text="Save",
        next_url=reverse_lazy("caidapp:manual_taxon_classification_on_non_classified"),
        skip_url=reverse_lazy("caidapp:manual_taxon_classification_on_non_classified"),
    )


def message(request, message):
    """Show message."""
    return render(
        request,
        "caidapp/message.html",
        {"message": message},
    )


def _round_location(location: Location, order: int = 3):
    """Round location for anonymization."""
    lat, lon = str(location.location).split(",")
    lat = round(float(lat), order)
    lon = round(float(lon), order)
    location.location = f"{lat},{lon}"
    location.save()
    return f"{lat},{lon}"


def update_location(request, location_id):
    """Show and update location."""
    location = get_object_or_404(
        Location,
        pk=location_id,
        **_user_content_filter_params(request.user.caiduser, "owner"),
    )
    # if location.owner:
    #     if request.user.caiduser.workgroup != location.owner__workgroup:
    #         return HttpResponseNotAllowed("Not allowed to see this location.")
    if request.method == "POST":
        form = LocationForm(request.POST, instance=location)
        if form.is_valid():

            # get uploaded archive
            location = form.save()
            _round_location(location, order=3)
            return redirect("caidapp:manage_locations")
    else:
        form = LocationForm(instance=location)
    return render(
        request,
        "caidapp/update_form.html",
        {"form": form, "headline": "Location", "button": "Save", "location": location},
    )


def manage_locations(request):
    """Add new location or update names of locations."""
    LocationFormSet = modelformset_factory(
        Location, fields=("name",), can_delete=False, can_order=False
    )
    params = _user_content_filter_params(request.user.caiduser, "owner")
    formset = LocationFormSet(queryset=Location.objects.filter(**params))

    if request.method == "POST":
        form = LocationFormSet(request.POST)
        if form.is_valid():
            form.save()
    else:
        form = formset

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


def uploads_identities(request):
    """List of uploads."""
    uploadedarchives = (
        UploadedArchive.objects.filter(
            owner__workgroup=request.user.caiduser.workgroup,
            contains_single_taxon=True,
        )
        .all()
        .order_by("-uploaded_at")
    )

    records_per_page = 12
    paginator = Paginator(uploadedarchives, per_page=records_per_page)

    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)
    return render(
        request,
        "caidapp/uploads_identities.html",
        context={
            "page_obj": page_obj,
            "btn_styles": _single_species_button_style(request),
        },
    )


@staff_member_required
def show_log(request):
    """List of uploads."""
    logfile = Path("/data/logging.log")
    # read lines from logfile
    with open(logfile, "r") as f:
        log = f.readlines()

    return render(request, "caidapp/show_log.html", {"log": log})


def show_taxons(request):
    """List of taxons."""
    all_taxons = Taxon.objects.all().order_by("name")
    taxons = []
    taxons_mediafiles = []
    # todo use here new function
    if request.user.caiduser.workgroup:
        filter_params = dict(parent__owner__workgroup=request.user.caiduser.workgroup)
    else:
        filter_params = dict(parent__owner=request.user.caiduser)

    for taxon in all_taxons:
        mediafiles_of_taxon = taxon.mediafile_set.filter(**filter_params).all()
        if len(mediafiles_of_taxon) > 0:
            taxons.append(taxon)
            taxons_mediafiles.append(mediafiles_of_taxon)

    return render(
        request,
        "caidapp/show_taxons.html",
        {"taxons": taxons, "taxons_with_mediafiles": zip(taxons, taxons_mediafiles)},
    )


def uploads_species(request):
    """List of uploads."""
    uploadedarchives = (
        UploadedArchive.objects.filter(
            owner__workgroup=request.user.caiduser.workgroup,
            contains_single_taxon=False,
        )
        .all()
        .order_by("-uploaded_at")
    )
    # uploadedarchives = (
    #     UploadedArchive.objects.filter(
    #         owner=request.user.caiduser,
    #     )
    #     .all()
    #     .order_by("-uploaded_at")
    # )

    records_per_page = 12
    paginator = Paginator(uploadedarchives, per_page=records_per_page)

    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    # btn_styles = _multiple_species_button_style_and_tooltips(request)
    #
    # btn_tooltips = _mutliple_species_button_tooltips(request)

    btn_styles, btn_tooltips = _multiple_species_button_style_and_tooltips(request)
    return render(
        request,
        "caidapp/uploads_species.html",
        {"page_obj": page_obj, "btn_styles": btn_styles, "btn_tooltips": btn_tooltips},
    )


def _multiple_species_button_style_and_tooltips(request) -> dict:
    n_non_classified_taxons = len(
        MediaFile.objects.filter(
            parent__owner__workgroup=request.user.caiduser.workgroup,
            category__name="Not Classified",
            parent__contains_single_taxon=False,
        ).all()
    )
    btn_tooltips = {
        "classify_non_classified": f"Classify {n_non_classified_taxons} "
        + "non-classified media files.",
    }
    if n_non_classified_taxons == 0:
        btn_styles = {
            "upload_species": "btn-primary",
            "classify_non_classified": "btn-secondary",
        }
    else:
        btn_styles = {
            "upload_species": "btn-secondary",
            "classify_non_classified": "btn-primary",
        }
    return btn_styles, btn_tooltips


def sample_data(request):
    """Sample data."""
    sample_data_collection = get_object_or_404(ArchiveCollection, name="sample_data")
    return render(
        request,
        "caidapp/sample_data.html",
        {"sample_data_collection": sample_data_collection},
    )


def logout_view(request):
    """Logout from the application."""
    logout(request)
    # Redirect to a success page.
    return redirect("caidapp:index")


def media_file_update(request, media_file_id, next_text="Save", next_url=None, skip_url=None):
    """Show and update media file."""
    # | Q(parent__owner=request.user.caiduser)
    # | Q(parent__owner__workgroup=request.user.caiduser.workgroup)
    mediafile = get_object_or_404(MediaFile, pk=media_file_id)
    if (mediafile.parent.owner.id != request.user.id) and (
        mediafile.parent.owner.workgroup != request.user.caiduser.workgroup
    ):
        return HttpResponseNotAllowed("Not allowed to see this media file.")

    if request.method == "POST":
        form = MediaFileForm(request.POST, instance=mediafile)
        if form.is_valid():

            mediafile.updated_by = request.user.caiduser
            mediafile.updated_at = django.utils.timezone.now()
            # get uploaded archive
            mediafile = form.save()
            if next_url is None:
                next_url = reverse_lazy(
                    "caidapp:uploadedarchive_detail",
                    kwargs={"uploadedarchive_id": mediafile.parent.id},
                )
            return redirect(next_url)

    else:
        form = MediaFileForm(instance=mediafile)
    return render(
        request,
        "caidapp/media_file_update.html",
        {
            "form": form,
            "headline": "Media File",
            "button": next_text,
            "mediafile": mediafile,
            skip_url: skip_url,
        },
    )


def individual_identities(request):
    """List of individual identities."""
    individual_identities = (
        IndividualIdentity.objects.filter(
            Q(owner_workgroup=request.user.caiduser.workgroup) and ~Q(name="nan")
        )
        .all()
        .order_by("-name")
    )

    records_per_page = 24
    paginator = Paginator(individual_identities, per_page=records_per_page)

    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)
    return render(
        request,
        "caidapp/individual_identities.html",
        {"page_obj": page_obj, "workgroup": request.user.caiduser.workgroup},
    )


def new_individual_identity(request):
    """Create new individual_identity."""
    if request.method == "POST":
        form = IndividualIdentityForm(request.POST)
        if form.is_valid():
            individual_identity = form.save(commit=False)
            individual_identity.owner_workgroup = request.user.caiduser.workgroup
            individual_identity.updated_by = request.user.caiduser
            individual_identity.save()
            return redirect("caidapp:individual_identities")
    else:
        form = IndividualIdentityForm()
    return render(
        request,
        "caidapp/update_form.html",
        {"form": form, "headline": "New Individual Identity", "button": "Create"},
    )


def update_individual_identity(request, individual_identity_id):
    """Show and update media file."""
    individual_identity = get_object_or_404(
        IndividualIdentity,
        pk=individual_identity_id,
        owner_workgroup=request.user.caiduser.workgroup,
    )

    if request.method == "POST":
        form = IndividualIdentityForm(request.POST, instance=individual_identity)
        if form.is_valid():

            # get uploaded archive
            individual_identity = form.save()
            individual_identity.updated_by = request.user.caiduser
            individual_identity.save()
            return redirect("caidapp:individual_identities")
    else:
        form = IndividualIdentityForm(instance=individual_identity)
    return render(
        request,
        "caidapp/update_form.html",
        {
            "form": form,
            "headline": "Individual Identity",
            "button": "Save",
            "individual_identity": individual_identity,
            "delete_button_url": reverse_lazy(
                "caidapp:delete_individual_identity",
                kwargs={"individual_identity_id": individual_identity_id},
            ),
        },
    )


def delete_individual_identity(request, individual_identity_id):
    """Delete individual identity if it belongs to the user."""
    individual_identity = get_object_or_404(
        IndividualIdentity,
        pk=individual_identity_id,
        owner_workgroup=request.user.caiduser.workgroup,
    )
    individual_identity.delete()
    return redirect("caidapp:individual_identities")


def get_individual_identity_zoomed(request, foridentification_id: int, top_id: int):
    """Show and update media file."""
    foridentifications = MediafilesForIdentification.objects.filter(
        mediafile__parent__owner__workgroup=request.user.caiduser.workgroup
    ).order_by("?")
    foridentification = MediafilesForIdentification.objects.get(id=foridentification_id)
    if foridentification.mediafile.parent.owner.workgroup != request.user.caiduser.workgroup:
        return HttpResponseNotAllowed("Not allowed to work with this media file.")

    # modulo
    if top_id == 0:
        top_id = 3
    elif top_id == 4:
        top_id = 1

    if top_id == 1:
        top_mediafile = foridentification.top1mediafile
        top_score = foridentification.top1score
        top_name = foridentification.top1name
    elif top_id == 2:
        top_mediafile = foridentification.top2mediafile
        top_score = foridentification.top2score
        top_name = foridentification.top2name
    elif top_id == 3:
        top_mediafile = foridentification.top3mediafile
        top_score = foridentification.top3score
        top_name = foridentification.top3name
    else:
        HttpResponseBadRequest("Wrong top_id.")
    return render(
        request,
        "caidapp/get_individual_identity_zoomed.html",
        {
            "foridentification": foridentification,
            "foridentifications": foridentifications,
            "top_id": top_id,
            "top_mediafile": top_mediafile,
            "top_score": top_score,
            "top_name": top_name,
        },
    )


def not_identified_mediafiles(request):
    """View for mediafiles with individualities that are not identified."""
    foridentification_set = MediafilesForIdentification.objects.filter(
        mediafile__parent__owner__workgroup=request.user.caiduser.workgroup
    )
    # mediafile_set = uploadedarchive.mediafile_set.all()

    records_per_page = 80
    paginator = Paginator(foridentification_set, per_page=records_per_page)

    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)
    return render(
        request,
        "caidapp/not_identified_mediafiles.html",
        {"page_obj": page_obj, "page_title": "Not Identified"},
    )


def get_individual_identity_from_foridentification(
    request, foridentification_id: Optional[int] = None
):
    """Show and update media file."""
    foridentifications = MediafilesForIdentification.objects.filter(
        mediafile__parent__owner__workgroup=request.user.caiduser.workgroup
    ).order_by("?")
    if foridentification_id is None:
        foridentification = foridentifications.first()
    else:
        foridentification = MediafilesForIdentification.objects.get(id=foridentification_id)
    return render(
        request,
        "caidapp/get_individual_identity.html",
        {
            "foridentification": foridentification,
            "foridentifications": foridentifications,
        },
    )


def remove_foridentification(request, foridentification_id: int):
    """Remove mediafile from list for identification."""
    foridentification = get_object_or_404(
        MediafilesForIdentification,
        id=foridentification_id,
        mediafile__parent__owner__workgroup=request.user.caiduser.workgroup,
    )
    foridentification.delete()
    return redirect("caidapp:get_individual_identity")


def set_individual_identity(
    request, mediafiles_for_identification_id: int, individual_identity_id: int
):
    """Set identity for mediafile."""
    mediafiles_for_identification = get_object_or_404(
        MediafilesForIdentification, id=mediafiles_for_identification_id
    )
    individual_identity = get_object_or_404(IndividualIdentity, id=individual_identity_id)

    # if request.user.caiduser.workgroup != mediafile.parent.owner.workgroup:
    #     return HttpResponseNotAllowed("Not allowed to work with this media file.")
    if request.user.caiduser.workgroup != individual_identity.owner_workgroup:
        return HttpResponseNotAllowed("Not allowed to work with this media file.")
    if (
        request.user.caiduser.workgroup
        != mediafiles_for_identification.mediafile.parent.owner.workgroup
    ):
        return HttpResponseNotAllowed("Not allowed to work with this media file.")

    mediafiles_for_identification.mediafile.identity = individual_identity
    mediafiles_for_identification.mediafile.identity_is_representative = True
    mediafiles_for_identification.mediafile.updated_by = request.user.caiduser
    mediafiles_for_identification.mediafile.save()
    mediafiles_for_identification.delete()

    return redirect("caidapp:get_individual_identity")


@staff_member_required
def run_taxon_classification(request, uploadedarchive_id):
    """Run processing of uploaded archive."""
    uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    next_page = request.GET.get("next", "/caidapp/uploads")
    run_species_prediction_async(uploaded_archive)
    return redirect(next_page)


# def init_identification(request, taxon_str:str="Lynx Lynx"):
#     return redirect("/caidapp/uploads")


def init_identification(request, taxon_str: str = "Lynx lynx"):
    """Run processing of uploaded archive."""
    # check if user is workgroup admin
    if not request.user.caiduser.workgroup_admin:
        return HttpResponseNotAllowed("Identification init is for workgroup admins only.")
    mediafiles = MediaFile.objects.filter(
        # category__name=taxon_str,
        identity__isnull=False,
        parent__owner__workgroup=request.user.caiduser.workgroup,
        identity_is_representative=True,
    ).all()

    logger.debug("Generating CSV for init_identification...")

    output_dir = Path(settings.MEDIA_ROOT) / request.user.caiduser.workgroup.name
    output_dir.mkdir(exist_ok=True, parents=True)

    csv_data = _prepare_dataframe_for_identification(mediafiles)
    logger.debug(f"{len(csv_data)=}")

    identity_metadata_file = output_dir / "init_identification.csv"
    pd.DataFrame(csv_data).to_csv(identity_metadata_file, index=False)
    logger.debug(f"{identity_metadata_file=}")
    workgroup = request.user.caiduser.workgroup
    workgroup.identification_init_at = django.utils.timezone.now()
    workgroup.identification_init_status = "Processing"
    workgroup.identification_init_message = (
        f"Using {len(csv_data)} images for identification initialization."
    )
    workgroup.save()

    logger.debug("Calling init_identification...")
    sig = signature(
        "init_identification",
        kwargs={
            # csv file should contain image_path, class_id, label
            "input_metadata_file": str(identity_metadata_file),
            "organization_id": request.user.caiduser.workgroup.id,
        },
    )
    # task =
    sig.apply_async(
        link=init_identification_on_success.s(
            workgroup_id=request.user.caiduser.workgroup.id,
            # uploaded_archive_id=uploaded_archive.id,
            # zip_file=os.path.relpath(str(output_archive_file), settings.MEDIA_ROOT),
            # csv_file=os.path.relpath(str(output_metadata_file), settings.MEDIA_ROOT),
        ),
        link_error=init_identification_on_error.s(
            # uploaded_archive_id=uploaded_archive.id
        ),
    )
    # return redirect("caidapp:individual_identities")
    return redirect("caidapp:uploads_identities")


def _single_species_button_style(request) -> dict:

    is_initiated = request.user.caiduser.workgroup.identification_init_at is not None

    exists_representative = (
        len(
            MediaFile.objects.filter(
                parent__owner__workgroup=request.user.caiduser.workgroup,
                identity_is_representative=True,
                parent__contains_single_taxon=True,
            )
        )
        > 0
    )

    exists_unidentified = (
        len(
            UploadedArchive.objects.filter(
                owner__workgroup=request.user.caiduser.workgroup,
                contains_identities=False,
                contains_single_taxon=True,
                status="Species Finished",
            )
        )
        > 0
    )

    exists_for_confirmation = (
        len(
            MediafilesForIdentification.objects.filter(
                mediafile__parent__owner__workgroup=request.user.caiduser.workgroup
            )
        )
        > 0
    )

    btn_styles = {}

    btn_styles["upload_identified"] = (
        "btn-primary" if (not is_initiated) and (not exists_representative) else "btn-secondary"
    )
    btn_styles["init_identification"] = (
        "btn-primary" if (not is_initiated) and exists_representative else "btn-secondary"
    )
    btn_styles["upload_unidentified"] = (
        "btn-primary"
        if is_initiated and (not exists_unidentified) and (not exists_for_confirmation)
        else "btn-secondary"
    )
    btn_styles["run_identification"] = (
        "btn-primary"
        if is_initiated and exists_unidentified and (not exists_for_confirmation)
        else "btn-secondary"
    )
    btn_styles["confirm_identification"] = (
        "btn-primary" if exists_for_confirmation else "btn-secondary"
    )

    btn_styles["init_identification"] += " disabled" if not exists_representative else ""
    btn_styles["run_identification"] += " disabled" if not is_initiated else ""

    return btn_styles


def run_identification_on_unidentified(request):
    """Run identification in all uploaded archives."""
    uploaded_archives = UploadedArchive.objects.filter(
        owner__workgroup=request.user.caiduser.workgroup,
        status="Species Finished",
        contains_single_taxon=True,
        contains_identities=False,
    ).all()
    for uploaded_archive in uploaded_archives:
        _run_identification(uploaded_archive)
    next_page = request.GET.get("next", "caidapp:uploads_identities")
    return redirect(next_page)


def run_identification(request, uploadedarchive_id):
    """Run identification of uploaded archive."""
    uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    # check if user is owner member of the workgroup
    if uploaded_archive.owner.workgroup != request.user.caiduser.workgroup:
        return HttpResponseNotAllowed("Identification is for workgroup members only.")
    _run_identification(uploaded_archive)
    next_page = request.GET.get("next", "caidapp:uploads_identities")
    return redirect(next_page)


def _run_identification(uploaded_archive: UploadedArchive, taxon_str="Lynx lynx"):
    logger.debug("Generating CSV for run_identification...")
    mediafiles = uploaded_archive.mediafile_set.filter(category__name=taxon_str).all()
    logger.debug(f"Generating CSV for init_identification with {len(mediafiles)} records...")
    # csv_len = len(mediafiles)
    # csv_data = {"image_path": [None] * csv_len, "mediafile_id": [None] * csv_len}

    csv_data = _prepare_dataframe_for_identification(mediafiles)
    media_root = Path(settings.MEDIA_ROOT)
    # output_dir = Path(settings.MEDIA_ROOT) / request.user.caiduser.workgroup.name
    # output_dir.mkdir(exist_ok=True, parents=True)

    # logger.debug(f"number of records={len(mediafiles)}")
    #
    # for i, mediafile in enumerate(mediafiles):
    #     # if mediafile.identity is not None:
    #     csv_data["image_path"][i] = str(media_root / mediafile.mediafile.name)
    #     csv_data["mediafile_id"][i] = mediafile.id

    identity_metadata_file = media_root / uploaded_archive.outputdir / "identification_metadata.csv"
    # cropped_identity_metadata_file = (
    #     media_root / uploaded_archive.outputdir / "cropped_identification_metadata.csv"
    # )
    pd.DataFrame(csv_data).to_csv(identity_metadata_file, index=False)
    output_json_file = media_root / uploaded_archive.outputdir / "identification_result.json"

    from celery import current_app

    tasks = current_app.tasks.keys()
    logger.debug(f"tasks={tasks}")

    logger.debug("Calling run_detection and run_identification ...")

    # uploaded_archive = UploadedArchive.objects.get(id=uploaded_archive_id)
    uploaded_archive.status = "Identification started"
    uploaded_archive.save()

    identify_signature = signature(
        "identify",
        kwargs=dict(
            input_metadata_file_path=str(identity_metadata_file),
            organization_id=uploaded_archive.owner.workgroup.id,
            output_json_file_path=str(output_json_file),
            top_k=3,
            uploaded_archive_id=uploaded_archive.id,
        ),
    )
    identify_task = identify_signature.apply_async(
        link=identify_on_success.s(
            uploaded_archive_id=uploaded_archive.id,
        ),
        link_error=on_error_in_upload_processing.s(),
    )
    logger.debug(f"{identify_task=}")
    return redirect("caidapp:uploads_identities")


def new_album(request):
    """Create new album."""
    if request.method == "POST":
        form = AlbumForm(request.POST)
        if form.is_valid():
            album = form.save(commit=False)
            album.owner = request.user.caiduser
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
    if album.owner == request.user.caiduser:
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


def upload_archive(
    request,
    contains_single_taxon=False,
    contains_identities=False,
):
    """Process the uploaded zip file."""
    text_note = ""
    next = "caidapp:uploads"
    if contains_single_taxon:
        text_note = "The archive contains images of a single taxon."
        next = "caidapp:upload_archive_contains_single_taxon"
    if contains_identities:
        text_note = (
            "The archive contains identities (of single taxon). "
            + "Each identity is in individual folder"
        )
        next = "caidapp:upload_archive_contains_identities"

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

            uploaded_archive.owner = request.user.caiduser
            logger.debug(f"{uploaded_archive.contains_identities=}, {contains_identities=}")
            logger.debug(f"{uploaded_archive.contains_single_taxon=}, {contains_single_taxon=}")
            # log actual url
            logger.debug(f"{request.build_absolute_uri()=}")
            uploaded_archive.contains_identities = contains_identities
            uploaded_archive.contains_single_taxon = contains_single_taxon
            uploaded_archive.save()
            run_species_prediction_async(uploaded_archive)

            return JsonResponse({"data": "Data uploaded"})
        else:
            return JsonResponse({"data": "Someting went wrong"})

    else:
        form = UploadedArchiveForm(
            initial={
                "contains_identities": contains_identities,
                "contains_single_taxon": contains_single_taxon,
            }
        )

    return render(
        request,
        "caidapp/model_form_upload.html",
        {
            "form": form,
            "headline": "Upload",
            "button": "Upload",
            "text_note": text_note,
            "next": next,
            "locations": _get_all_user_locations(request),
        },
    )


def _user_has_rw_access_to_mediafile(ciduser: CaIDUser, mediafile: MediaFile) -> bool:
    """Check if user has access to mediafile."""
    return (mediafile.parent.owner.id == ciduser.id) or (
        mediafile.parent.owner.workgroup == ciduser.workgroup
    )


def _user_has_rw_acces_to_uploadedarchive(
    ciduser: CaIDUser, uploadedarchive: UploadedArchive
) -> bool:
    """Check if user has access to uploadedarchive."""
    return (uploadedarchive.owner.id == ciduser.id) or (
        uploadedarchive.owner.workgroup == ciduser.workgroup
    )


def _user_content_filter_params(ciduser: CaIDUser, prefix: str) -> dict:
    """Parameters for filtering user content based on existence of workgroup.

    Parameters
    ----------
    request : HttpRequest
        Request object.
    prefix : str
        Prefix for filtering with ciduser.
        If the filter will be used in MediaFile, the prefix should be "parent__owner".
        If the filter will be used in Location, the prefix should be "owner".
    """
    if ciduser.workgroup:
        # filter_params = dict(parent__owner__workgroup=request.user.caiduser.workgroup)
        filter_params = {f"{prefix}__workgroup": ciduser.workgroup}
    else:
        filter_params = {f"{prefix}": ciduser}
    return filter_params


def _get_all_user_locations(request):
    """Get all users locations."""
    params = _user_content_filter_params(request.user.caiduser, "owner")
    logger.debug(f"{params=}")
    locations = Location.objects.filter(**params).order_by("name")
    return locations


@login_required
def delete_upload(request, uploadedarchive_id):
    """Delete uploaded file."""
    uploadedarchive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)

    if _user_has_rw_acces_to_uploadedarchive(request.user.caiduser, uploadedarchive):
        uploadedarchive.delete()
    else:
        messages.error(request, "Not allowed to delete this uploaded archive.")
    return redirect("/caidapp/uploads")


@login_required
def delete_mediafile(request, mediafile_id):
    """Delete uploaded file."""
    mediafile = get_object_or_404(MediaFile, pk=mediafile_id)
    if _user_has_rw_access_to_mediafile(request.user.caiduser, mediafile):
        parent_id = mediafile.parent_id
        uploaded_archive = mediafile.parent
        uploaded_archive.output_updated_at = None
        uploaded_archive.save()

        mediafile.delete()
        return redirect("caidapp:uploadedarchive_detail", uploadedarchive_id=parent_id)
    else:
        return HttpResponseNotAllowed("Not allowed to delete this media file.")


@login_required
def albums(request):
    """Show all albums."""
    albums = (
        Album.objects.filter(
            Q(albumsharerole__user=request.user.caiduser) | Q(owner=request.user.caiduser)
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


def _mediafiles_query(
    request, query: str, album_hash=None, individual_identity_id=None, taxon_id=None
):
    """Prepare list of mediafiles based on query search in category and location."""
    mediafiles = (
        MediaFile.objects.filter(
            Q(album__albumsharerole__user=request.user.caiduser)
            | Q(parent__owner=request.user.caiduser)
            | Q(parent__owner__workgroup=request.user.caiduser.workgroup)
        )
        .distinct()
        .order_by("-parent__uploaded_at")
    )
    if album_hash is not None:
        album = get_object_or_404(Album, hash=album_hash)
        mediafiles = (
            mediafiles.filter(album=album).all().distinct().order_by("-parent__uploaded_at")
        )
    if individual_identity_id is not None:
        individual_identity = get_object_or_404(IndividualIdentity, pk=individual_identity_id)
        mediafiles = (
            mediafiles.filter(identity=individual_identity)
            .all()
            .distinct()
            .order_by("-parent__uploaded_at")
        )
    if taxon_id is not None:
        taxon = get_object_or_404(Taxon, pk=taxon_id)
        mediafiles = (
            mediafiles.filter(category=taxon).all().distinct().order_by("-parent__uploaded_at")
        )

    if len(query) == 0:
        return mediafiles
    else:
        from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector

        vector = SearchVector("category__name", "location__name")
        query = SearchQuery(query)
        logger.debug(str(query))
        mediafiles = (
            MediaFile.objects.filter(
                Q(album__albumsharerole__user=request.user.caiduser)
                | Q(parent__owner=request.user.caiduser)
                | Q(parent__owner__workgroup=request.user.caiduser.workgroup)
                # parent__owner=request.user.caiduser
            )
            .annotate(rank=SearchRank(vector, query))
            .filter(rank__gt=0)
            .order_by("-rank")
        )
        # words = [query]
        #
        # queryset_combination = None
        # for word in words:
        #     if queryset_combination is None:
        #         queryset_combination = mediafiles.filter(category__name__icontains=word).all()
        #     else:
        #         queryset_combination |= mediafiles.filter(category__name__icontains=word).all()
        #
        #     queryset_combination |= mediafiles.filter(location__name__icontains=word).all()
        #
        # # queryset_combination.all().order_by("-parent_uploaded_at")
        # mediafiles = queryset_combination.all().order_by("-parent__uploaded_at")
        return mediafiles


def _page_number(request, page_number: int) -> int:
    """Prepare page number into queryform."""
    if "nextPage" in request.POST:
        page_number += 1
    if "lastPage" in request.POST:
        page_number = -1
    if "prevPage" in request.POST:
        page_number -= 1
    if "firstPage" in request.POST:
        page_number = 1
    return page_number


def update_mediafile_is_representative(request, mediafile_hash: str, is_representative: bool):
    """Update mediafile is_representative."""
    mediafile = get_object_or_404(MediaFile, hash=mediafile_hash)
    if (mediafile.parent.owner.id != request.user.id) | (
        mediafile.parent.owner.workgroup != request.user.caiduser.workgroup
    ):
        return HttpResponseNotAllowed("Not allowed to work with this media file.")
    mediafile.is_representative = is_representative
    mediafile.save()
    return JsonResponse({"data": "Data uploaded"})


def _create_map_from_mediafiles(mediafiles: Union[QuerySet, List[MediaFile]]):
    """Create dataframe from mediafiles."""
    # create dataframe

    queryset_list = list(mediafiles.values("id", "location__name", "location__location"))
    df = pd.DataFrame.from_records(queryset_list)
    logger.debug(f"{list(df.keys())}")
    data = []
    for mediafile in mediafiles:
        if (
            mediafile.location
            and mediafile.location.location
            and mediafile.location.location.count(",") == 1
        ):
            row = {
                "id": mediafile.id,
                "category": mediafile.category.name if mediafile.category else None,
                "category_id": mediafile.category.id if mediafile.category else None,
                "location": mediafile.location.name if mediafile.location else None,
                "location__location": mediafile.location.location
                if mediafile.location.location
                else None,
            }
            data.append(row)

    df2 = pd.DataFrame.from_records(data)
    if "location__location" not in df2.keys():
        return None
    df2[["lat", "lon"]] = df2["location__location"].str.split(",", expand=True)
    df2["lat"] = df2["lat"].astype(float)
    df2["lon"] = df2["lon"].astype(float)
    logger.debug(f"{list(df2.keys())}")
    # if len(df2) > 10:
    #     logger.debug(f"{df2.sample(10).to_dict()=}")
    # else:
    #     logger.debug(f"{df2.to_dict()=}")

    # Calculate the range of your data to set the zoom level
    lat_range = df2["lat"].max() - df2["lat"].min()
    lon_range = df2["lon"].max() - df2["lon"].min()

    # Set an appropriate zoom level based on the maximum range
    max_range = max(lat_range, lon_range)
    zoom = 0  # Set a default zoom level
    if max_range < 10:
        zoom = 6
    elif max_range < 30:
        zoom = 5
    elif max_range < 60:
        zoom = 4
    else:
        zoom = 3  # For larger ranges, set a smaller zoom level

    fig = go.Figure(go.Densitymapbox(lat=df2.lat, lon=df2.lon, radius=10))
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center_lon=df2.lon.unique().mean(),
        mapbox_center_lat=df2.lat.unique().mean(),
        mapbox_zoom=zoom,
    )

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    map_html = fig.to_html()
    return map_html


def media_files_update(
    request,
    records_per_page=80,
    album_hash=None,
    individual_identity_id=None,
    taxon_id=None,
) -> Union[QuerySet, List[MediaFile]]:
    """List of mediafiles based on query with bulk update of category."""
    # create list of mediafiles
    if request.method == "POST":
        queryform = MediaFileSetQueryForm(request.POST)
        if queryform.is_valid():
            query = queryform.cleaned_data["query"]
            # logger.debug(f"{queryform.cleaned_data=}")
            page_number = _page_number(request, page_number=queryform.cleaned_data["pagenumber"])
            if "querySubmit" in request.POST:
                logger.debug("querySubmit")
                page_number = 1
            queryform.cleaned_data["pagenumber"] = page_number
            queryform = MediaFileSetQueryForm(initial=queryform.cleaned_data)
        else:
            logger.error("queryform is not valid")
            logger.error(queryform.errors)
            for error in queryform.non_field_errors():
                logger.error(error)
    else:
        page_number = 1
        queryform = MediaFileSetQueryForm(dict(query="", pagenumber=page_number))
        query = ""
    albums_available = (
        Album.objects.filter(
            Q(albumsharerole__user=request.user.caiduser) | Q(owner=request.user.caiduser)
        )
        .distinct()
        .order_by("created_at")
    )
    # logger.debug(f"{albums_available=}")
    # logger.debug(f"{query=}")
    # logger.debug(f"{queryform}")
    full_mediafiles = _mediafiles_query(
        request,
        query,
        album_hash=album_hash,
        individual_identity_id=individual_identity_id,
        taxon_id=taxon_id,
    )
    number_of_mediafiles = len(full_mediafiles)
    map_html = _create_map_from_mediafiles(full_mediafiles)

    paginator = Paginator(full_mediafiles, per_page=records_per_page)

    page_mediafiles = paginator.get_page(page_number)

    MediaFileFormSet = modelformset_factory(MediaFile, form=MediaFileSelectionForm, extra=0)
    if (request.method == "POST") and (
        any([(type(key) == str) and (key.startswith("btnBulkProcessing")) for key in request.POST])
        # ("btnBulkProcessing" in request.POST) or ("btnBulkProcessingAlbum" in request.POST)
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
                        elif "btnBulkProcessing_id_category" in form.data:
                            instance = mediafileform.save(commit=False)
                            instance.category = form_bulk_processing.cleaned_data["category"]
                            instance.updated_by = request.user.caiduser
                            instance.updated_at = django.utils.timezone.now()
                            instance.save()
                        elif "btnBulkProcessing_id_identity" in form.data:
                            instance = mediafileform.save(commit=False)
                            instance.identity = form_bulk_processing.cleaned_data["identity"]
                            instance.identity_is_representative = False
                            instance.updated_by = request.user.caiduser
                            instance.updated_at = django.utils.timezone.now()
                            instance.save()
                        elif "btnBulkProcessing_id_identity_is_representative" in form.data:
                            instance = mediafileform.save(commit=False)
                            instance.identity_is_representative = form_bulk_processing.cleaned_data[
                                "identity_is_representative"
                            ]
                            instance.updated_by = request.user.caiduser
                            instance.updated_at = django.utils.timezone.now()
                            instance.save()
                        elif "btnBulkProcessingDelete" in form.data:
                            instance = mediafileform.save(commit=False)
                            instance.delete()
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
            "map_html": map_html,
        },
    )


def create_new_album(request, name="New Album"):
    """Create new album."""
    album = Album()
    album.name = name
    album.owner = request.user.caiduser
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


def _update_csv_by_uploadedarchive(request, uploadedarchive_id: int):
    uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)

    if uploaded_archive.owner.workgroup == request.user.caiduser.workgroup:
        updated_at = uploaded_archive.output_updated_at
        logger.debug(f"{updated_at=}")
        if updated_at is None:
            # set updated_at to old date
            updated_at = datetime.datetime(
                2000, 1, 1, 0, 0, 0, 0, tzinfo=pytz.timezone(settings.TIME_ZONE)
            )
        # check if mediafiles are updated later than updated_at

        mediafiles = MediaFile.objects.filter(parent=uploaded_archive)
        logger.debug(f"  1  {mediafiles=}")
        logger.debug(f"  1  {mediafiles.first().updated_at=}")
        mediafiles = mediafiles.filter(updated_at__gt=updated_at).all()
        logger.debug(f"  2  {mediafiles=}")
        # mediafiles = MediaFile.objects.filter(
        #     Q(parent=uploaded_archive) & Q(updated_at__gt=updated_at)
        # ).all()
        # logger.debug(f"{mediafiles=}")
        if len(mediafiles) > 0:
            logger.debug("  sync mediafiles with csv")
            logger.debug(f"  {uploaded_archive.csv_file=}")
            update_metadata_csv_by_uploaded_archive(uploaded_archive, create_missing=False)
            return True

    return False


# def _generate_csv(request, uploadedarchive_id: int):
#     uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
#     output_dir = Path(settings.MEDIA_ROOT) / uploaded_archive.outputdir
#     output_metadata_file = output_dir / "metadata.csv"
#     output_archive_file = output_dir / "images.zip"
#     mediafiles = MediaFile.objects.filter(parent=uploaded_archive).all()
#     csv_len = len(mediafiles)
#     csv_data = {
#         "image_path": [None] * csv_len,
#         "class_id": [None] * csv_len,
#         "label": [None] * csv_len,
#     }
#
#     media_root = Path(settings.MEDIA_ROOT)
#     logger.debug(f"number of records={len(mediafiles)}")
#     for i, mediafile in enumerate(mediafiles):
#
#         # if mediafile.identity is not None:
#         csv_data["image_path"][i] = str(media_root / mediafile.mediafile.name)
#         csv_data["class_id"][i] = int(mediafile.identity.id)
#         csv_data["label"][i] = str(mediafile.identity.name)
#
#     pd.DataFrame(csv_data).to_csv(output_metadata_file, index=False)
#
#     return output_metadata_file, output_archive_file


def download_uploadedarchive_images(request, uploadedarchive_id: int):
    """Download uploaded file."""
    uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)

    if uploaded_archive.owner.workgroup == request.user.caiduser.workgroup:
        _update_csv_by_uploadedarchive(request, uploadedarchive_id)
        # file_path = Path(settings.MEDIA_ROOT) / uploaded_file.archivefile.name
        file_path = Path(settings.MEDIA_ROOT) / uploaded_archive.zip_file.name
        logger.debug(f"{file_path=}")

        if file_path.exists():
            with open(file_path, "rb") as fh:
                response = HttpResponse(fh.read(), content_type="application/zip")
                response["Content-Disposition"] = "inline; filename=" + os.path.basename(file_path)
                return response
        raise Http404
    else:
        messages.error(request, "Only the owner can download the file")
        return redirect("/caidapp/uploads")


def download_uploadedarchive_csv(request, uploadedarchive_id: int):
    """Download uploaded file."""
    uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)

    if uploaded_archive.owner.workgroup == request.user.caiduser.workgroup:
        _update_csv_by_uploadedarchive(request, uploadedarchive_id)
        # file_path = Path(settings.MEDIA_ROOT) / uploaded_file.archivefile.name
        file_path = Path(settings.MEDIA_ROOT) / uploaded_archive.csv_file.name
        logger.debug(f"{file_path=}")
        if file_path.exists():
            with open(file_path, "rb") as fh:
                response = HttpResponse(fh.read(), content_type="application/zip")
                response["Content-Disposition"] = "inline; filename=" + os.path.basename(file_path)
                return response
        raise Http404
    else:
        messages.error(request, "Only the owner can download the file")
        return redirect("/caidapp/uploads")
