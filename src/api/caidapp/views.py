import datetime
from io import BytesIO
import json
import logging
import os
import traceback
from pathlib import Path
from typing import List, Optional, Union, Tuple

import django
import plotly.express as px
import plotly.graph_objects as go
import pytz
from celery import signature
from django.conf import settings
from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import logout, get_user_model
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
from django.core.paginator import Paginator, Page
from django.db.models import Count, Q, QuerySet
from django.forms import modelformset_factory
from django.http import HttpResponseBadRequest, HttpResponseNotAllowed, JsonResponse, HttpResponseRedirect
from django.shortcuts import Http404, HttpResponse, get_object_or_404, redirect, render
from django.urls import reverse_lazy
from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
import shutil
import pandas as pd

from django.contrib.auth import login as auth_login
from django.contrib.auth.decorators import user_passes_test
from django.shortcuts import redirect, render
from .forms import UserSelectForm

from . import tasks
from . import models, model_tools

from .forms import (
    AlbumForm,
    IndividualIdentityForm,
    MediaFileBulkForm,
    MediaFileForm,
    MediaFileSelectionForm,
    MediaFileSetQueryForm,
    UploadedArchiveForm,
    UploadedArchiveFormWithTaxon,
    UploadedArchiveUpdateForm,
    WorkgroupUsersForm,
    UploadedArchiveSelectTaxonForIdentificationForm,
)
from .model_extra import _user_has_rw_access_to_mediafile, _user_has_rw_acces_to_uploadedarchive
from .models import (
    Album,
    ArchiveCollection,
    IndividualIdentity,
    Location,
    MediaFile,
    MediafilesForIdentification,
    Taxon,
    UploadedArchive,
    WorkGroup,
    get_content_owner_filter_params,
)
from .tasks import (
    _prepare_dataframe_for_identification,
    get_location,
    identify_on_success,
    init_identification_on_error,
    init_identification_on_success,
    on_error_in_upload_processing,
    run_species_prediction_async,
    update_metadata_csv_by_uploaded_archive, _iterate_over_location_checks,
)
from .views_location import _get_all_user_locations, _set_location_to_mediafiles_of_uploadedarchive

logger = logging.getLogger("app")
User = get_user_model()



@user_passes_test(lambda u: u.is_superuser)
def impersonate_user(request):
    if request.method == 'POST':
        form = UserSelectForm(request.POST)
        if form.is_valid():
            user = form.cleaned_data['user']
            request.session['original_user_id'] = request.user.id
            request.session['impersonate_user_id'] = user.id
            return redirect('caidapp:uploads')
    else:
        form = UserSelectForm()

    return render(request, 'caidapp/impersonate_user.html', {'form': form})


# @user_passes_test(lambda u: u.is_superuser)
# def stop_impersonation(request):
#     logger.debug("Stopping Impersonation ...")
#     if 'impersonate_user_id' in request.session:
#         del request.session['impersonate_user_id']
#         logger.debug("Impersonation stopped.")
#
#     logger.debug("Redirecting to uploads ...")
#     return redirect('caidapp:uploads')

@login_required
def stop_impersonation(request):
    if 'impersonate_user_id' in request.session:
        del request.session['impersonate_user_id']
    if 'original_user_id' in request.session:
        original_user = User.objects.get(id=request.session['original_user_id'])
        auth_login(request, original_user)
        del request.session['original_user_id']
    return redirect('caidapp:uploads')

def login(request):
    """Login page."""
    if request.user.is_authenticated:
        return redirect("caidapp:uploads")
    else:
        return render(
            request,
            "caidapp/login.html",
        )


# def media_files(request):
#     """List of uploads."""
#     mediafiles = (
#         MediaFile.objects.filter(
#             **get_content_owner_filter_params(request.user.caiduser, "parent__owner")
#             # parent__owner=request.user.caiduser
#         )
#         .all()
#         .order_by("-parent__uploaded_at")
#     )
#
#     records_per_page = 10000
#     # page_number = request.GET.get("page")
#     paginator = Paginator(mediafiles, per_page=records_per_page)
#     page_obj, elided_page_range, page_context = _prepare_page(paginator, request=request)
#
#     qs_data = {}
#     for e in mediafiles:
#         qs_data[e.id] = str(e.category) + " " + str(e.location)
#         # qs_data.append(e.id)
#     logger.debug(qs_data)
#     qs_json = json.dumps(qs_data)
#     return render(
#         request,
#         "caidapp/media_files.html",
#         {
#             **page_context,
#             "page_title": "Media files",
#             "qs_json": qs_json,
#             "user_is_staff": request.user.is_staff,
#         },
#     )


def message(request, message):
    """Show message."""
    return render(
        request,
        "caidapp/message.html",
        {"message": message},
    )


def uploadedarchive_detail(request, uploadedarchive_id):
    """List of uploads."""
    uploadedarchive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    mediafile_set = uploadedarchive.mediafile_set.all()

    records_per_page = 80
    paginator = Paginator(mediafile_set, per_page=records_per_page)
    _,_, page_context = _prepare_page(paginator, request=request)

    return render(
        request,
        "caidapp/uploadedarchive_detail.html",
        {
            **page_context,
            "page_title": uploadedarchive,
        },
    )


def _prepare_page(paginator: Paginator, request:Optional = None, page_number: Optional[int] = None) -> Tuple[Page, List, dict]:
    if page_number is None:
        page_number = request.GET.get("page")
    elided_page_range = paginator.get_elided_page_range(page_number, on_each_side=3, on_ends=2)
    page_obj = paginator.get_page(page_number)

    context = {
        "page_obj": page_obj,
        "elided_page_range": elided_page_range,
    }

    return page_obj, elided_page_range, context

def uploads_identities(request):
    """List of uploads."""
    order_by = uploaded_archive_get_order_by(request)

    uploadedarchives = (
        UploadedArchive.objects.filter(
            **get_content_owner_filter_params(request.user.caiduser, "owner"),
            # contains_single_taxon=True,
            taxon_for_identification__isnull=False,
        )
        .all()
        .order_by(order_by)
    )

    records_per_page = get_item_number_uploaded_archives(request)
    paginator = Paginator(uploadedarchives, per_page=records_per_page)
    _,_, page_context = _prepare_page(paginator, request=request)

    return render(
        request,
        "caidapp/uploads_identities.html",
        context={
            **page_context,
            # "page_obj": page_obj,
            # "elided_page_range": elided_page_range,
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
    order_by = uploaded_archive_get_order_by(request)
    uploadedarchives = (
        UploadedArchive.objects.filter(
            **get_content_owner_filter_params(request.user.caiduser, "owner"),
            contains_single_taxon=False,
            # parent__owner=request.user.caiduser
        )
        .all()
        .order_by(order_by)
    )

    records_per_page = get_item_number_uploaded_archives(request)
    paginator = Paginator(uploadedarchives, per_page=records_per_page)
    _,_, page_context = _prepare_page(paginator, request=request)

    btn_styles, btn_tooltips = _multiple_species_button_style_and_tooltips(request)
    return render(
        request,
        "caidapp/uploads_species.html",
        {
            **page_context,
            "btn_styles": btn_styles, "btn_tooltips": btn_tooltips},
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
    btn_styles["overview_taxons"] = "btn-secondary"
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
            logger.debug(f"{mediafile.category=}")
            if (mediafile.category is not None) and (mediafile.category.name != "Not Classified"):
                mediafile.taxon_overviewed = True
                mediafile.taxon_overviewed_at = django.utils.timezone.now()
                mediafile.save()

            if next_url:
                return HttpResponseRedirect(next_url)
            else:
                next_url = request.GET.get('next')

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
    _,_, page_context = _prepare_page(paginator, request=request)

    return render(
        request,
        "caidapp/individual_identities.html",
        {
            **page_context,
            "workgroup": request.user.caiduser.workgroup
        },
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
    media_file = MediaFile.objects.filter(identity=individual_identity, identity_is_representative=True).first()

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
            "mediafile": media_file,
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


def get_individual_identity_zoomed_paired_points(request, foridentification_id: int, top_id: int):
    """Show detail with paired points."""
    return get_individual_identity_zoomed(request, foridentification_id, top_id, points=True)


def get_individual_identity_zoomed(request, foridentification_id: int, top_id: int, points=False):
    """Show and update media file."""
    foridentifications = MediafilesForIdentification.objects.filter(
        mediafile__parent__owner__workgroup=request.user.caiduser.workgroup
    ).order_by("?")
    foridentification = MediafilesForIdentification.objects.get(id=foridentification_id)
    if foridentification.mediafile.parent.owner.workgroup != request.user.caiduser.workgroup:
        return HttpResponseNotAllowed("Not allowed to work with this media file.")

    (
        top_id,
        top_mediafile,
        top_name,
        top_score,
        paired_points,
    ) = _select_pair_for_detail_identification(foridentification, top_id)

    if points:
        from . import gui_tools

        logger.debug(f"{top_mediafile.mediafile.name}")
        # read image with PIL
        import numpy as np
        from PIL import Image

        paired_pts0 = np.asarray(paired_points[0])
        paired_pts1 = np.asarray(paired_points[1])

        pth0 = Path(settings.MEDIA_ROOT) / str(foridentification.mediafile.mediafile.name).replace(
            "/images/", "/masked_images/"
        )
        pth1 = Path(settings.MEDIA_ROOT) / str(top_mediafile.mediafile.name).replace(
            "/images/", "/masked_images/"
        )
        pil_img0 = Image.open(pth0)
        pil_img1 = Image.open(pth1)
        img0 = np.array(pil_img0)
        img1 = np.array(pil_img1)

        # Compensate points coordinates because points are calculated on resized images 512x512
        #    We have two options resize the images, or recalculate the points coordinates

        # Option 1) Resize images with PIL
        scale0 = 512.0 / img0.shape[0]
        scale1 = 512.0 / img1.shape[0]
        pil_img0 = pil_img0.resize((512, int(img0.shape[1] * scale0)))
        pil_img1 = pil_img1.resize((512, int(img1.shape[1] * scale1)))
        img0 = np.array(pil_img0)
        img1 = np.array(pil_img1)

        # Option 2) Compensate points coordinates because points are calculated on resized
        #    images 512x512
        # logger.debug(f"{foridentification.paired_points=}")
        # logger.debug(f"{paired_points=}")
        # paired_pts0 = (paired_pts0 / 512.0) * img0.shape[:2]
        # paired_pts1 = (paired_pts1 / 512.0) * img1.shape[:2]

        html_img_src = gui_tools.create_match_img_src(
            paired_pts0.tolist(), paired_pts1.tolist(), img0, img1, top_name, top_name
        )
        template = "caidapp/get_individual_identity_zoomed_paired_points.html"
        btn_link = reverse_lazy(
            "caidapp:get_individual_identity_zoomed",
            kwargs={"foridentification_id": foridentification_id, "top_id": top_id},
        )
        btn_icon_style = "fa fa-eye"
    else:
        template = "caidapp/get_individual_identity_zoomed.html"
        html_img_src = None
        btn_link = reverse_lazy(
            "caidapp:get_individual_identity_zoomed_paired_points",
            kwargs={"foridentification_id": foridentification_id, "top_id": top_id},
        )
        btn_icon_style = "fa-solid fa-arrows-to-dot"

    return render(
        request,
        template,
        {
            "foridentification": foridentification,
            "foridentifications": foridentifications,
            "top_id": top_id,
            "top_mediafile": top_mediafile,
            "top_score": top_score,
            "top_name": top_name,
            "html_img_src": html_img_src,
            "btn_link": btn_link,
            "btn_icon_style": btn_icon_style,
        },
    )


def _select_pair_for_detail_identification(foridentification, top_id):

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

    if (top_id - 1) < len(foridentification.paired_points):
        paired_points = foridentification.paired_points[top_id - 1]
    else:
        paired_points = [[], []]
    return top_id, top_mediafile, top_name, top_score, paired_points


def not_identified_mediafiles(request):
    """View for mediafiles with individualities that are not identified."""
    foridentification_set = MediafilesForIdentification.objects.filter(
        mediafile__parent__owner__workgroup=request.user.caiduser.workgroup
    )
    # mediafile_set = uploadedarchive.mediafile_set.all()

    records_per_page = 80
    paginator = Paginator(foridentification_set, per_page=records_per_page)
    _,_, page_context = _prepare_page(paginator, request=request)

    return render(
        request,
        "caidapp/not_identified_mediafiles.html",
        {
            **page_context,
            "page_title": "Not Identified"},
    )


def get_individual_identity_from_foridentification(
    request, foridentification_id: Optional[int] = None,
        media_file_id: Optional[int] = None
):
    """Show and update media file."""
    foridentifications = MediafilesForIdentification.objects.filter(
        mediafile__parent__owner__workgroup=request.user.caiduser.workgroup
    ).order_by("?")
    if media_file_id:
        foridentification = foridentifications.get(mediafile__id=media_file_id)
    else:
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
def run_taxon_classification_force_init(request, uploadedarchive_id):
    """Run processing of uploaded archive with removal of previous outputs."""
    return run_taxon_classification(request, uploadedarchive_id=uploadedarchive_id, force_init=True)


@staff_member_required
def run_taxon_classification(request, uploadedarchive_id, force_init=False):
    """Run processing of uploaded archive."""
    uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    run_species_prediction_async(uploaded_archive, force_init=force_init)
    # next_page = request.GET.get("next", "/caidapp/uploads")
    # return redirect(next_page)
    return redirect(request.META.get('HTTP_REFERER', '/'))


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

    identity_metadata_file = output_dir / "init_identification.csv"
    pd.DataFrame(csv_data).to_csv(identity_metadata_file, index=False)
    logger.debug(f"{identity_metadata_file=}")
    workgroup = request.user.caiduser.workgroup
    workgroup.identification_init_at = django.utils.timezone.now()
    workgroup.identification_init_status = "Processing"
    workgroup.identification_init_message = (
        f"Using {len(csv_data['image_path'])}"
        + "representative images for identification initialization."
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


# TODO rename to identification button style
def _single_species_button_style(request) -> dict:

    is_initiated = request.user.caiduser.workgroup.identification_init_at is not None

    exists_representative = (
        len(
            MediaFile.objects.filter(
                parent__owner__workgroup=request.user.caiduser.workgroup,
                identity_is_representative=True,
                # parent__contains_single_taxon=True,
                parent__taxon_for_identification__isnull=False,
            )
        )
        > 0
    )

    exists_unidentified = (
        len(
            UploadedArchive.objects.filter(
                owner__workgroup=request.user.caiduser.workgroup,
                contains_identities=False,
                # contains_single_taxon=True,
                taxon_for_identification__isnull=False,
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
        status="Taxon classification finished",
        # contains_single_taxon=True,
        taxon_for_identification__isnull=False,
        contains_identities=False,
    ).all()
    for uploaded_archive in uploaded_archives:
        _run_identification(uploaded_archive)
    # next_page = request.GET.get("next", "caidapp:uploads_identities")
    # return redirect(next_page)
    return redirect(request.META.get('HTTP_REFERER', '/'))


def run_identification(request, uploadedarchive_id):
    """Run identification of uploaded archive."""
    uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    # check if user is owner member of the workgroup
    if uploaded_archive.owner.workgroup != request.user.caiduser.workgroup:
        return HttpResponseNotAllowed("Identification is for workgroup members only.")
    _run_identification(uploaded_archive)
    # next_page = request.GET.get("next", "caidapp:uploads_identities")
    # return redirect(next_page)
    return redirect(request.META.get('HTTP_REFERER', '/'))


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
        logger.debug(f"{request.POST=}")
        if contains_single_taxon:
            form = UploadedArchiveFormWithTaxon(
                request.POST,
                request.FILES,
            )
        else:
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
                    request, f"Uploaded file with extension '{uploaded_archive_suffix}' is not an archive."
                )

            uploaded_archive.owner = request.user.caiduser
            logger.debug(f"{uploaded_archive.contains_identities=}, {contains_identities=}")
            logger.debug(f"{uploaded_archive.contains_single_taxon=}, {contains_single_taxon=}")
            # log actual url
            logger.debug(f"{request.build_absolute_uri()=}")
            uploaded_archive.contains_identities = contains_identities
            uploaded_archive.contains_single_taxon = contains_single_taxon
            uploaded_archive.save()
            uploaded_archive.extract_location_check_at_from_filename(commit=True)
            run_species_prediction_async(uploaded_archive, extract_identites=contains_identities)

            return JsonResponse({"data": "Data uploaded"})
        else:
            return JsonResponse({"data": "Something went wrong"})

    else:

        initial_data = {
            "contains_identities": contains_identities,
            "contains_single_taxon": contains_single_taxon,
            # "taxon_for_identification": models.get_taxon("Lynx lynx") if contains_single_taxon else None,
        }

        if contains_single_taxon:
            initial_data["taxon_for_identification"] = models.get_taxon("Lynx lynx")
            logger.debug(f"{initial_data=}")
            form = UploadedArchiveFormWithTaxon(initial=initial_data)
        else:
            form = UploadedArchiveForm(initial=initial_data)

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


@login_required
def cloud_import_preview_view(request):
    """Check the content of the import directory and analyze if it is ready for import."""
    if len(request.user.caiduser.import_dir) == 0:
        return HttpResponseNotAllowed("No import directory specified. Ask admin to set it up.")

    # get list of available localities

    path = Path(request.user.caiduser.import_dir)
    # paths_of_locality_check = path.glob("*")
    # paths_of_locality_check = Path("/caid_import").glob("*")
    caiduser = request.user.caiduser

    list_of_location_checks = []
    text = str(path) + ""

    for yield_dict in _iterate_over_location_checks(path, caiduser):

        if yield_dict.parent_dir_to_be_deleted:
            continue

        if yield_dict.is_already_processed:
            continue

        list_of_location_checks.append(yield_dict.__dict__)
        # text += str(path_of_location_check.relative_to(path)) + "<br>"

    return render(
        request,
        "caidapp/cloud_import_checks_preview.html",
        {
            "page_obj": list_of_location_checks,
            "text": text,
            # "form_objects": form,
            # "page_title": "Media files",
            # "user_is_staff": request.user.is_staff,
            # "form_bulk_processing": form_bulk_processing,
            # "form_query": queryform,
            # "albums_available": albums_available,
            # "number_of_mediafiles": number_of_mediafiles,
            # "map_html": map_html,
            # "taxon_stats_html": taxon_stats_html,
        },
    )


def make_zipfile(output_filename: Path, source_dir: Path):
    """Make archive (zip, tar.gz) from a folder.

    Parameters
    ----------
    output_filename: Path of output file
    source_dir: Path to input directory
    """
    import shutil

    output_filename = Path(output_filename)
    source_dir = Path(source_dir)
    archive_type = "zip"

    shutil.make_archive(
        output_filename.parent / output_filename.stem, archive_type, root_dir=source_dir
    )


@login_required
def do_cloud_import_view(request):
    """Bulk import from one dir and prepare zip file for every check.

    Make zip file from every check. The information encoded in path is code of lynx season (i.e.
    LY2019), locality (Prachatice), date of check (2019-07-01). In the leaf directory are media
    files (images and videos). For every check there will be zip file. The name of the zip file
    will be composed of locality and date of check.

    Example of path structure:
    NETRIDENA/LY2019/PRACHATICE/2019-07-01/2019-07-01_12-00-00_0001.jpg
    """
    if len(request.user.caiduser.import_dir) == 0:
        return HttpResponseNotAllowed("No import directory specified. Ask admin to set it up.")

    # get list of available localities
    caiduser = request.user.caiduser

    tasks.do_cloud_import_for_user(caiduser)
    return redirect("caidapp:cloud_import_preview")

@login_required
def break_cloud_import_view(request):
    caiduser = request.user.caiduser
    caiduser.dir_import_status="Interrupted"
    caiduser.save()
    return redirect("caidapp:cloud_import_preview")


def locations_view(request):
    """List of locations."""
    locations = _get_all_user_locations(request)
    logger.debug(f"{len(locations)=}")
    return render(request, "caidapp/locations.html", {"locations": locations})

def update_uploadedarchive(request, uploadedarchive_id):
    """Show and update uploaded archive."""
    uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    if not _user_has_rw_acces_to_uploadedarchive(request.user.caiduser, uploaded_archive):
        return HttpResponseNotAllowed("Not allowed to see this uploaded archive.")
    uploaded_archive_location_at_upload = uploaded_archive.location_at_upload

    if request.method == "POST":
        form = UploadedArchiveUpdateForm(request.POST, instance=uploaded_archive)
        if form.is_valid():
            cleaned_location_at_upload = form.cleaned_data["location_at_upload"]
            uploaded_archive = form.save()
            logger.debug(f"{uploaded_archive.location_at_upload=}, {cleaned_location_at_upload=}")
            if uploaded_archive_location_at_upload != cleaned_location_at_upload:
                logger.debug("Location has been changed.")
                # location_str = form.cleaned_data["location_at_upload"]
                location = get_location(request.user.caiduser, cleaned_location_at_upload)
                _set_location_to_mediafiles_of_uploadedarchive(request, uploaded_archive, location)
                uploaded_archive.location_at_upload_object = location
                uploaded_archive.save()

            return redirect("caidapp:uploads")
    else:
        form = UploadedArchiveUpdateForm(instance=uploaded_archive)
    return render(
        request,
        "caidapp/update_form.html",
        # "caidapp/update_form.html",
        {
            "form": form,
            "headline": "Uploaded Archive",
            "button": "Save",
            "uploadedarchive": uploaded_archive,
            "locations": _get_all_user_locations(request),
        },
    )


@login_required
def delete_upload(request, uploadedarchive_id, next_page="caidapp:uploads"):
    """Delete uploaded file."""
    uploadedarchive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)

    if _user_has_rw_acces_to_uploadedarchive(request.user.caiduser, uploadedarchive):
        uploadedarchive.delete()
    else:
        messages.error(request, "Not allowed to delete this uploaded archive.")
    return redirect(next_page)


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
    request,
    query: str,
    album_hash=None,
    individual_identity_id=None,
    taxon_id=None,
    uploadedarchive_id=None,
    identity_is_representative=None,
    location_hash=None,
    order_by:Optional[str]=None,
    taxon_overviewed:Optional[bool]=None,
):
    """Prepare list of mediafiles based on query search in category and location."""

    if order_by is None:
        order_by = request.session.get("mediafiles_order_by", "-parent__uploaded_at")
    mediafiles = (
        MediaFile.objects.filter(
            Q(album__albumsharerole__user=request.user.caiduser)
            | Q(parent__owner=request.user.caiduser)
            | Q(parent__owner__workgroup=request.user.caiduser.workgroup)
        )
        .distinct()
        .order_by(order_by)
    )
    if taxon_overviewed is not None:
        mediafiles = mediafiles.filter(taxon_overviewed=taxon_overviewed).all()
    if album_hash is not None:
        album = get_object_or_404(Album, hash=album_hash)
        mediafiles = (
            mediafiles.filter(album=album).all().distinct().order_by(order_by)
        )
    if individual_identity_id is not None:
        individual_identity = get_object_or_404(IndividualIdentity, pk=individual_identity_id)
        mediafiles = (
            mediafiles.filter(identity=individual_identity)
            .all()
            .distinct()
            .order_by(order_by)
        )
    if taxon_id is not None:
        taxon = get_object_or_404(Taxon, pk=taxon_id)
        mediafiles = (
            mediafiles.filter(category=taxon).all().distinct().order_by(order_by)
        )
    if uploadedarchive_id is not None:
        uploadedarchive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
        mediafiles = (
            mediafiles.filter(parent=uploadedarchive)
            .all()
            .distinct()
            .order_by(order_by)
        )
    if identity_is_representative is not None:
        mediafiles = (
            mediafiles.filter(identity_is_representative=identity_is_representative)
            .all()
            .distinct()
            .order_by(order_by)
        )
    if location_hash is not None:
        location = get_object_or_404(Location, hash=location_hash)
        mediafiles = (
            mediafiles.filter(location=location).all().distinct().order_by(order_by)
        )

    if len(query) == 0:
        return mediafiles
    else:

        vector = SearchVector("category__name", "location__name")
        query = SearchQuery(query)
        logger.debug(str(query))
        mediafiles = (
            mediafiles.annotate(rank=SearchRank(vector, query)).filter(rank__gt=0).order_by("-rank")
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
    if "goToPage" in request.POST:
        logger.debug(f"{request.POST['goToPage']=}")
        logger.debug(f"{request.GET=}")
        logger.debug(f"{request.POST=}")
        page_number = int(request.POST["goToPage"])
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

    fig = go.Figure(go.Densitymapbox(lat=df2.lat, lon=df2.lon, radius=10, showscale=False))
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center_lon=df2.lon.unique().mean(),
        mapbox_center_lat=df2.lat.unique().mean(),
        mapbox_zoom=zoom,
    )

    fig.update_layout(margin={"r": 0, "t": 10, "l": 0, "b": 0}, height=300)
    map_html = fig.to_html()
    return map_html


def _taxon_stats_for_mediafiles(mediafiles: Union[QuerySet, List[MediaFile]]) -> str:
    """Create taxon stats for mediafiles."""
    taxon_stats = ""
    if len(mediafiles) > 0:
        taxon_stats = (
            mediafiles.values("category__name")
            .annotate(count=Count("category__name"))
            .order_by("-count")
        )
        logger.debug(f"{taxon_stats=}")
        df = pd.DataFrame.from_records(taxon_stats)
        df.rename(columns={"category__name": "Taxon", "count": "Count"}, inplace=True)
        fig = px.bar(df, x="Taxon", y="Count", height=300)
        taxon_stats_html = fig.to_html()
    else:
        taxon_stats_html = None
    return taxon_stats_html


def media_files_update(
    request,
    records_per_page:Optional[int]=None,
    album_hash=None,
    individual_identity_id=None,
    taxon_id=None,
    uploadedarchive_id=None,
    identity_is_representative=None,
    location_hash=None,
    show_overview_button=False,
    order_by=None,
    taxon_overviewed:Optional[bool]=None,
) -> Union[QuerySet, List[MediaFile]]:
    """List of mediafiles based on query with bulk update of category."""
    # create list of mediafiles

    page_number = 1
    if records_per_page is None:
        records_per_page = request.session.get("mediafiles_records_per_page", 20)


    if request.method == "POST":
        queryform = MediaFileSetQueryForm(request.POST)
        if queryform.is_valid():
            query = queryform.cleaned_data["query"]
            # logger.debug(f"{queryform.cleaned_data=}")
            # page_number = int(request.GET.get('page'))
            # logger.debug(f"{page_number=}")

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
        uploadedarchive_id=uploadedarchive_id,
        identity_is_representative=identity_is_representative,
        location_hash=location_hash,
        order_by=order_by,
        taxon_overviewed=taxon_overviewed,
    )
    number_of_mediafiles = len(full_mediafiles)

    request.session['mediafile_ids'] = list(full_mediafiles.values_list('id', flat=True))
    paginator = Paginator(full_mediafiles, per_page=records_per_page)
    page_with_mediafiles, _, page_context = _prepare_page(paginator, page_number=page_number)

    page_ids = [obj.id for obj in page_with_mediafiles.object_list]
    request.session['mediafile_ids_page'] = page_ids

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
        page_query = full_mediafiles.filter(id__in=[object.id for object in page_with_mediafiles])
        form = MediaFileFormSet(queryset=page_query)
    else:

        logger.debug("initial form processing")
        form_bulk_processing = MediaFileBulkForm()
        page_query = full_mediafiles.filter(id__in=[object.id for object in page_with_mediafiles])
        form = MediaFileFormSet(queryset=page_query)

    logger.debug("ready to render page")
    return render(
        request,
        "caidapp/media_files_update.html",
        {
            # "page_obj": page_with_mediafiles,
            # "elided_page_range": elided_page_range,
            **page_context,
            "form_objects": form,
            "page_title": "Media files",
            "user_is_staff": request.user.is_staff,
            "form_bulk_processing": form_bulk_processing,
            "form_query": queryform,
            "albums_available": albums_available,
            "number_of_mediafiles": number_of_mediafiles,
            "show_overview_button": show_overview_button,
            # "map_html": map_html,
            # "taxon_stats_html": taxon_stats_html,
        },
    )

def mediafiles_stats_view(request):

    mediafile_ids = request.session.get('mediafile_ids', [])
    mediafiles = MediaFile.objects.filter(id__in=mediafile_ids)

    map_html = _create_map_from_mediafiles(mediafiles)
    taxon_stats_html = _taxon_stats_for_mediafiles(mediafiles)
    return render(
        request,
        "caidapp/media_files_stats.html",
        # "caidapp/media_files_update.html",
        {
            "map_html": map_html,
            "taxon_stats_html": taxon_stats_html,
        }
    )

@login_required
def select_taxon_for_identification(request, uploadedarchive_id: int):
    """Select taxon for identification."""
    uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    if not _user_has_rw_acces_to_uploadedarchive(request.user.caiduser, uploaded_archive):
        return HttpResponseNotAllowed("Not allowed to edit this uploaded archive.")
    if request.method == "POST":
        form = UploadedArchiveSelectTaxonForIdentificationForm(request.POST)
        if form.is_valid():
            taxon = form.cleaned_data["taxon_for_identification"]
            uploaded_archive.taxon_for_identification = taxon
            uploaded_archive.save()
            return redirect("caidapp:uploads_identities")
    else:
        form = UploadedArchiveSelectTaxonForIdentificationForm()
    return render(
        request,
        "caidapp/update_form.html",
        {
            "form": form,
            "headline": "Select taxon",
            "button": "Select",
            "text_note": "Select taxon for identification",
            "next": "caidapp:uploads_identities",
            "mediafile": uploaded_archive.mediafile_set.all().first(),
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

        workgroup_users = workgroup.caiduser_set.all()
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

def download_csv_for_mediafiles_view(request):
    mediafile_ids = request.session.get('mediafile_ids', [])
    mediafiles = MediaFile.objects.filter(id__in=mediafile_ids)

    try:
        df = tasks.create_dataframe_from_mediafiles(mediafiles)
        if df.empty:
            return HttpResponse("No data available to export.", content_type="text/plain")
    except Exception as e:
        logger.error(traceback.format_exc())
        return HttpResponse("Error during export.", content_type="text/plain")
    # df = tasks.create_dataframe_from_mediafiles(mediafiles)
    response = HttpResponse(df.to_csv(), content_type="text/csv")
    response["Content-Disposition"] = "attachment; filename=metadata.csv"
    return response


def download_xlsx_for_mediafiles_view(request):
    mediafile_ids = request.session.get('mediafile_ids', [])
    mediafiles = MediaFile.objects.filter(id__in=mediafile_ids)

    try:
        df = tasks.create_dataframe_from_mediafiles(mediafiles)
        if df.empty:
            return HttpResponse("No data available to export.", content_type="text/plain")
    except Exception as e:
        logger.error(traceback.format_exc())
        return HttpResponse("Error during export.", content_type="text/plain")
    # df = tasks.create_dataframe_from_mediafiles(mediafiles)

    # convert timezone-aware datetime to naive datetime
    df = model_tools.convert_datetime_to_naive(df)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Locations')

    # Rewind the buffer
    output.seek(0)

    response = HttpResponse(output,
                            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=metadata.xlsx'
    return response

def download_zip_for_mediafiles_view(request):
    request.user.caiduser.hash
    mediafile_ids = request.session.get('mediafile_ids', [])
    mediafiles = MediaFile.objects.filter(id__in=mediafile_ids)
    # number_of_mediafiles = len(mediafiles)

    abs_zip_path = Path(settings.MEDIA_ROOT) / "users" / request.user.caiduser.hash / f"mediafiles.zip"
    abs_zip_path.parent.mkdir(parents=True, exist_ok=True)
    # get temp dir
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        mediafiles_dir = Path(tmpdirname) / "images"
        mediafiles_dir.mkdir()
        for mediafile in mediafiles:
            src = Path(settings.MEDIA_ROOT) / mediafile.mediafile.name
            dst = mediafiles_dir / Path(mediafile.mediafile.name).name
            logger.debug(f"{src=}, {dst=}")
            assert src.exists()
            shutil.copy(src, dst)
        make_zipfile(abs_zip_path, mediafiles_dir)
    with open(abs_zip_path, "rb") as fh:
        response = HttpResponse(fh.read(), content_type="application/zip")
        response["Content-Disposition"] = "inline; filename=mediafiles.zip"
        return response

    return Http404


def _generate_new_hash_for_locations():
    for location in Location.objects.all():
        location.hash = models.get_hash8()
        location.save()

def refresh_data(request):
    """Update new calculations for formerly uploaded archives."""
    uploaded_archives = UploadedArchive.objects.all()
    for uploaded_archive in uploaded_archives:
        uploaded_archive.update_earliest_and_latest_captured_at()

        if uploaded_archive.contains_single_taxon and uploaded_archive.taxon_for_identification is None:
            # this fixes the compatibility with the old version before 2024-05
            uploaded_archive.taxon_for_identification = models.get_taxon("Lynx lynx")
            uploaded_archive.save()

    # this was used to fix same hashes generated by wrong function
    # _generate_new_hash_for_locations()

    _refresh_media_file_original_name(request)

    # get taxon (and create it if it does not exist
    models.get_taxon("Unclassifiable")

    return redirect("caidapp:uploads")

def _refresh_media_file_original_name(request):
    for mediafile in MediaFile.objects.all():
        mediafile.extract_original_filename()


def shared_individual_identity_view(request, identity_hash:str):
    identity = get_object_or_404(IndividualIdentity, hash=identity_hash)
    mediafiles = MediaFile.objects.filter(identity=identity, identity_is_representative=True).all()

    return render(
        request,
        "caidapp/identity_detail_public.html",
        {
            "identity": identity,
            "mediafiles": mediafiles,
        }
    )


def set_sort_uploaded_archives_by(request, sort_by: str):
    """Sort uploaded archives by."""
    request.session['sort_uploaded_archives_by'] = sort_by

    # go back to previous page
    return redirect(request.META.get('HTTP_REFERER', '/'))


def uploaded_archive_get_order_by(request):
    """Get order by for uploaded archives."""
    sort_by = request.session.get('sort_uploaded_archives_by', '-uploaded_at')
    return sort_by


def set_item_number_uploaded_archives(request, item_number:int):
    """Sort uploaded archives by."""
    request.session['item_number_uploaded_archives'] = item_number

    # go back to previous page
    return redirect(request.META.get('HTTP_REFERER', '/'))

def get_item_number_uploaded_archives(request):
    """Get order by for uploaded archives."""
    item_number = request.session.get('item_number_uploaded_archives', 12)
    return item_number


def switch_private_mode(request):
    """Switch private mode."""

    actual_mode = request.session.get('private_mode', False)
    request.session['private_mode'] = not actual_mode

    return redirect(request.META.get('HTTP_REFERER', '/'))

