import datetime
import logging
import os
import shutil
import traceback
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import django
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
from celery import signature
from django.conf import settings
from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import get_user_model
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.views import LoginView
from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
from django.core.paginator import Page, Paginator
from django.db.models import Count, F, Min, Q, QuerySet
from django.forms import modelformset_factory
from django.http import HttpResponseBadRequest, HttpResponseNotAllowed, JsonResponse
from django.shortcuts import Http404, HttpResponse, get_object_or_404, redirect, render
from django.template.loader import render_to_string
from django.urls import reverse_lazy
from django.views import View
from django.db.models import OuterRef, Subquery
from django.http import JsonResponse
from celery.result import AsyncResult
from django.shortcuts import get_object_or_404, render, redirect
from django.views import View
from django.urls import reverse_lazy
from .models import IndividualIdentity, MediaFile
from .forms import IndividualIdentityForm
from functools import wraps
from django.shortcuts import redirect
from django.core.exceptions import PermissionDenied



from . import forms, model_tools, models, tasks, views_uploads
from .forms import (
    AlbumForm,
    IndividualIdentityForm,
    MediaFileBulkForm,
    MediaFileSelectionForm,
    MediaFileSetQueryForm,
    UploadedArchiveForm,
    UploadedArchiveFormWithTaxon,
    UploadedArchiveSelectTaxonForIdentificationForm,
    UploadedArchiveUpdateForm,
    UserSelectForm,
    WorkgroupUsersForm,
)
from .model_extra import user_has_rw_acces_to_uploadedarchive, user_has_rw_access_to_mediafile
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
    _iterate_over_location_checks,
    _prepare_dataframe_for_identification,
    get_location,
    identify_on_success,
    init_identification_on_error,
    init_identification_on_success,
    on_error_in_upload_processing,
    run_species_prediction_async,
    update_metadata_csv_by_uploaded_archive,
)
from .views_location import _get_all_user_locations, _set_location_to_mediafiles_of_uploadedarchive

logger = logging.getLogger("app")
User = get_user_model()


@user_passes_test(lambda u: u.is_superuser)
def impersonate_user(request):
    """Impersonate user."""
    if request.method == "POST":
        form = UserSelectForm(request.POST)
        if form.is_valid():
            user = form.cleaned_data["user"]
            request.session["original_user_id"] = request.user.id
            request.session["impersonate_user_id"] = user.id
            return redirect("caidapp:uploads")
    else:
        form = UserSelectForm()

    return render(request, "caidapp/impersonate_user.html", {"form": form})


@login_required
def stop_impersonation(request):
    """Stop impersonation."""
    if "impersonate_user_id" in request.session:
        del request.session["impersonate_user_id"]
    if "original_user_id" in request.session:
        original_user = User.objects.get(id=request.session["original_user_id"])
        auth_login(request, original_user)

        del request.session["original_user_id"]
    return redirect("caidapp:uploads")

def is_impersonating(request):
    """Check if user is impersonating."""
    return "impersonate_user_id" in request.session


def staff_or_impersonated_staff_required(view_func):
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        # Check if the user is staff
        if request.user.is_staff:
            return view_func(request, *args, **kwargs)

        # Check if the user is impersonating a staff member
        if "original_user_id" in request.session:
            original_user_id = request.session.get("original_user_id")
            # logger.debug(f"{original_user_id=}")
            try:
                impersonated_user = User.objects.get(id=original_user_id)
                if impersonated_user.is_staff:
                    return view_func(request, *args, **kwargs)
            except User.DoesNotExist:
                pass  # If impersonated user does not exist, continue to deny access

        # If neither condition is met, deny access
        raise PermissionDenied("You do not have permission to access this page.")

    return _wrapped_view


def login(request):
    """Login page."""
    if request.user.is_authenticated:
        return redirect("caidapp:uploads")
    else:
        return render(
            request,
            "caidapp/login.html",
        )


def message_view(request, message, headline=None, link=None, button_label="Ok"):
    """Show message."""
    return render(
        request,
        "caidapp/message.html",
        {
            "message": message,
            "headline": headline,
            "link": link,
            "button_label": button_label,
         },
    )


def uploadedarchive_mediafiles(request, uploadedarchive_id):
    """List of uploads."""
    uploadedarchive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    mediafile_set = uploadedarchive.mediafile_set.all()

    records_per_page = 80
    paginator = Paginator(mediafile_set, per_page=records_per_page)
    _, _, page_context = _prepare_page(paginator, request=request)

    return render(
        request,
        "caidapp/uploadedarchive_mediafiles.html",
        {
            **page_context,
            "page_title": uploadedarchive,
        },
    )


def _prepare_page(
    paginator: Paginator, request: Optional = None, page_number: Optional[int] = None
) -> Tuple[Page, List, dict]:
    if page_number is None:
        page_number = request.GET.get("page", 1)
    logger.debug(f"{page_number=}")
    elided_page_range = paginator.get_elided_page_range(page_number, on_each_side=3, on_ends=2)
    page_obj = paginator.get_page(page_number)

    context = {
        "page_obj": page_obj,
        "elided_page_range": elided_page_range,
    }

    return page_obj, elided_page_range, context


@staff_or_impersonated_staff_required
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


def uploads_identities(request) -> HttpResponse:
    """List of uploads."""
    page_context = _uploads_general(request, taxon_for_identification__isnull=False)

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


def uploads_species(request) -> HttpResponse:
    """List of uploads."""
    page_context = _uploads_general(request, contains_single_taxon=False)

    dates = views_uploads._get_check_dates(
        request, contains_single_taxon=False, taxon_for_identification__isnull=None
    )
    sorted_grouped_dates = views_uploads._get_grouped_dates(dates)

    # get list of years
    years = list(sorted_grouped_dates.keys())

    btn_styles, btn_tooltips = _multiple_species_button_style_and_tooltips(request)
    return render(
        request,
        "caidapp/uploads_species.html",
        {
            **page_context,
            "btn_styles": btn_styles,
            "btn_tooltips": btn_tooltips,
            "years": years,
        },
    )


def _uploads_general_order_annotation():
    # for UploadedArchive.objects.annotate()
    return dict(
        mediafile_count=Count("mediafile"),  # Count of all related MediaFiles
        mediafile_count_with_taxon=Count(
            "mediafile", filter=Q(mediafile__category=F("taxon_for_identification"))
        ),  # Count of MediaFiles with a specific taxon
        earliest_mediafile_captured_at=Min("mediafile__captured_at"),  # Earliest capture date
    )


def update_taxon(request, taxon_id: Optional[int] = None):
    """Update species form. Create taxon if taxon_id is None."""
    if taxon_id is not None:
        taxon = get_object_or_404(Taxon, pk=taxon_id)
        headline = "Update taxon"
        button_text = "Update"
    else:
        taxon = Taxon()
        headline = "New taxon"
        button_text = "Create"

    if request.method == "POST":
        form = forms.TaxonForm(request.POST, instance=taxon)
        if form.is_valid():
            taxon = form.save(commit=False)
            taxon.updated_by = request.user.caiduser
            taxon.save()
            return redirect("caidapp:show_taxons")
    else:
        form = forms.TaxonForm(instance=taxon)
    return render(
        request,
        "caidapp/update_form.html",
        {
            "form": form,
            "headline": headline,
            "button": button_text,
        },
    )

    # """Update species form."""
    # if request.method == "POST":
    #
    #     form = MediaFileBulkForm(request.POST)
    #     if form.is_valid():
    #         mediafiles = form.cleaned_data["mediafiles"]
    #         taxon = form.cleaned_data["taxon"]
    #         for mediafile in mediafiles:
    #             mediafile.category = taxon
    #             mediafile.save()
    #         return redirect("caidapp:uploads_species")
    # else:
    #     form = MediaFileBulkForm()
    # return render(
    #     request,
    #     "caidapp/update_form.html",
    #     {
    #         "form": form,
    #         "headline": "Update species",
    #         "button": "Update",
    #     },
    # )


def _uploads_general(
    request,
    contains_single_taxon: Optional[bool] = None,
    taxon_for_identification__isnull: Optional[bool] = None,
    **filter_params,
):
    """List of uploads."""
    order_by = uploaded_archive_get_order_by(request)
    uploadedarchives = UploadedArchive.objects.annotate(**_uploads_general_order_annotation())
    if filter_params is None:
        filter_params = {}

    if contains_single_taxon is not None:
        filter_params.update(dict(contains_single_taxon=contains_single_taxon))
    elif taxon_for_identification__isnull is not None:
        filter_params.update(
            dict(taxon_for_identification__isnull=taxon_for_identification__isnull)
        )

    uploadedarchives = (
        uploadedarchives.all()
        .filter(
            **get_content_owner_filter_params(request.user.caiduser, "owner"),
            **filter_params
            # contains_single_taxon=contains_single_taxon,
            # taxon_for_identification__isnull=taxon_for_identification__isnull,
            # parent__owner=request.user.caiduser
        )
        .all()
        .order_by(order_by)
    )
    logger.debug(f"{uploadedarchives.count()=}")

    records_per_page = get_item_number_uploaded_archives(request)
    paginator = Paginator(uploadedarchives, per_page=records_per_page)
    _, _, page_context = _prepare_page(paginator, request=request)

    # go over all items in the paginator
    for uploadedarchive in page_context["page_obj"]:
        uploadedarchive.update_status()

    return page_context

def select_reid_model(request):
    """Select reid model."""
    form = forms.UserIdentificationModelForm()
    if request.method == "POST":
        form = forms.UserIdentificationModelForm(request.POST)
        if form.is_valid():
            request.user.caiduser.identification_model = form.cleaned_data["identification_model"]
            request.user.caiduser.save()
            return redirect("caidapp:uploads_identities")

    else:

        initial = {"identification_model": request.user.caiduser.identification_model}
        form = forms.UserIdentificationModelForm(initial=initial)
    return render(
        request,
        "caidapp/update_form.html",
        {
            "form": form,
            "headline": "Select identification model",
            "button": "Save",
        },
    )


def _multiple_species_button_style_and_tooltips(request) -> dict:
    models.get_content_owner_filter_params(request.user.caiduser, "owner")
    n_non_classified_taxons = len(models.get_mediafiles_with_missing_taxon(request.user.caiduser))
    n_missing_verifications = len(
        models.get_mediafiles_with_missing_verification(request.user.caiduser)
    )

    some_missing_taxons = n_non_classified_taxons > 0
    some_missing_verifications = n_missing_verifications > 0

    btn_tooltips = {
        "annotate_missing_taxa": f"Annotate {n_non_classified_taxons} media files "
        + "with missing taxon.",
        "verify_taxa": f"Go to verification of {n_missing_verifications} media files.",
    }
    btn_styles = {
        "upload_species": "btn-secondary",
        "annotate_missing_taxa": "btn-secondary",
        "verify_taxa": "btn-secondary",
    }
    if not some_missing_taxons and not some_missing_verifications:
        btn_styles["upload_species"] = "btn-primary"
    elif some_missing_taxons:
        btn_styles["annotate_missing_taxa"] = "btn-primary"
    elif some_missing_verifications:
        btn_styles["verify_taxa"] = "btn-primary"

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


def individual_identities(request):
    """List of individual identities."""
    individual_identities = (
        IndividualIdentity.objects.filter(
            Q(owner_workgroup=request.user.caiduser.workgroup) & ~Q(name="nan")
        )
        .all()
        .order_by("-name")
    )

    records_per_page = 24
    paginator = Paginator(individual_identities, per_page=records_per_page)
    _, _, page_context = _prepare_page(paginator, request=request)

    return render(
        request,
        "caidapp/individual_identities.html",
        {**page_context, "workgroup": request.user.caiduser.workgroup},
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
    media_file = MediaFile.objects.filter(
        identity=individual_identity, identity_is_representative=True
    ).first()

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

def get_individual_identity_zoomed_by_identity(request, foridentification_id: int, identity_id: int):
    foridentifications = MediafilesForIdentification.objects.filter(
        mediafile__parent__owner__workgroup=request.user.caiduser.workgroup
    ).order_by("?")
    foridentification = MediafilesForIdentification.objects.get(id=foridentification_id)
    if foridentification.mediafile.parent.owner.workgroup != request.user.caiduser.workgroup:
        return HttpResponseNotAllowed("Not allowed to work with this media file.")

    identity = IndividualIdentity.objects.get(id=identity_id)
    top_mediafile = MediaFile.objects.filter(identity=identity, identity_is_representative=True).first()
    top_name = identity.name

    btn_link = reverse_lazy(
        "caidapp:get_individual_identity_zoomed_by_identity",
        kwargs={"foridentification_id": foridentification_id, "identity_id": identity_id},
    )
    btn_icon_style = "fa-solid fa-arrows-to-dot"

    template = "caidapp/get_individual_identity_zoomed.html"
    return render(
        request,
        template,
        {
            "foridentification": foridentification,
            "foridentifications": foridentifications,
            # "reid_suggestion_id": reid_suggestion_id,
            "top_mediafile": top_mediafile,
            # "top_score": top_score,
            "top_name": top_name,
            # "html_img_src": html_img_src,
            "btn_link": btn_link,
            "btn_icon_style": btn_icon_style,
        },
    )

def get_individual_identity_zoomed_paired_points(request, foridentification_id: int, reid_suggestion_id: int):
    """Show detail with paired points."""
    return get_individual_identity_zoomed(request, foridentification_id, reid_suggestion_id, points=True)


def get_individual_identity_zoomed(request, foridentification_id: int, reid_suggestion_id: int, points=False):
    """Show and update media file."""
    foridentifications = MediafilesForIdentification.objects.filter(
        mediafile__parent__owner__workgroup=request.user.caiduser.workgroup
    ).order_by("?")
    foridentification = MediafilesForIdentification.objects.get(id=foridentification_id)
    if foridentification.mediafile.parent.owner.workgroup != request.user.caiduser.workgroup:
        return HttpResponseNotAllowed("Not allowed to work with this media file.")


    reid_suggestion = models.MediafileIdentificationSuggestion.objects.get(id=reid_suggestion_id)
    top_mediafile = reid_suggestion.mediafile
    top_name = reid_suggestion.name
    top_score = reid_suggestion.score
    paired_points = reid_suggestion.paired_points
    # ) = _select_pair_for_detail_identification(foridentification, reid_suggestion_id)

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
            kwargs={"foridentification_id": foridentification_id, "reid_suggestion_id": reid_suggestion_id},
        )
        btn_icon_style = "fa fa-eye"
    else:
        template = "caidapp/get_individual_identity_zoomed.html"
        html_img_src = None
        btn_link = reverse_lazy(
            "caidapp:get_individual_identity_zoomed_paired_points",
            kwargs={"foridentification_id": foridentification_id, "reid_suggestion_id": reid_suggestion_id},
        )
        btn_icon_style = "fa-solid fa-arrows-to-dot"

    return render(
        request,
        template,
        {
            "foridentification": foridentification,
            "foridentifications": foridentifications,
            "reid_suggestion_id": reid_suggestion_id,
            "top_mediafile": top_mediafile,
            "top_score": top_score,
            "top_name": top_name,
            "html_img_src": html_img_src,
            "btn_link": btn_link,
            "btn_icon_style": btn_icon_style,
        },
    )


# def _select_pair_for_detail_identification(foridentification:models.MediafilesForIdentification, reid_suggestion_id):
#     # get identification suggestion by reid_suggestion_id
#     reid_suggestion = foridentification.
#     # get order of the suggestion in r
#
#     # modulo
#     if reid_suggestion_id == 0:
#         reid_suggestion_id = 3
#     elif reid_suggestion_id == 4:
#         reid_suggestion_id = 1
#     if reid_suggestion_id == 1:
#         top_mediafile = foridentification.top1mediafile
#         top_score = foridentification.top1score
#         top_name = foridentification.top1name
#     elif reid_suggestion_id == 2:
#         top_mediafile = foridentification.top2mediafile
#         top_score = foridentification.top2score
#         top_name = foridentification.top2name
#     elif reid_suggestion_id == 3:
#         top_mediafile = foridentification.top3mediafile
#         top_score = foridentification.top3score
#         top_name = foridentification.top3name
#     else:
#         HttpResponseBadRequest("Wrong reid_suggestion_id.")
#
#     if (reid_suggestion_id - 1) < len(foridentification.paired_points):
#         paired_points = foridentification.paired_points[reid_suggestion_id - 1]
#     else:
#         paired_points = [[], []]
#     return reid_suggestion_id, top_mediafile, top_name, top_score, paired_points


def not_identified_mediafiles(request):
    """View for mediafiles with individualities that are not identified."""
    foridentification_set = MediafilesForIdentification.objects.filter(
        mediafile__parent__owner__workgroup=request.user.caiduser.workgroup
    )
    # mediafile_set = uploadedarchive.mediafile_set.all()

    records_per_page = 80
    paginator = Paginator(foridentification_set, per_page=records_per_page)
    _, _, page_context = _prepare_page(paginator, request=request)

    return render(
        request,
        "caidapp/not_identified_mediafiles.html",
        {**page_context, "page_title": "Not Identified"},
    )


def get_individual_identity_from_foridentification(
    request, foridentification_id: Optional[int] = None, media_file_id: Optional[int] = None
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

    if foridentification is not None:
        # give me all identities in foridentification.top_mediafile_set.mediafile.identity
        identity_ids= foridentification.top_mediafiles.values_list("mediafile__identity", flat=True)

        # for identity in identities:
        #     # select representative mediafile for each identity
        #     identity.representative_mediafiles = identity.mediafile_set.filter(identity_is_representative=True)

        # Fetch the identities
        # related_identities = IndividualIdentity.objects.filter(
        #     id__in=identity_ids,
        #     owner_workgroup=request.user.caiduser.workgroup
        # ).order_by("name")


        remaining_identities = (
            IndividualIdentity.objects.filter(
                Q(owner_workgroup=request.user.caiduser.workgroup) & ~Q(name="nan") &
                ~Q(id__in=identity_ids)
            )
            .all()
            .order_by("name")
        )

        # Add `representative_mediafiles` to related identities
        # for identity in related_identities:
        #     identity.representative_mediafiles = identity.mediafile_set.filter(identity_is_representative=True)

        reid_suggestions = list(foridentification.top_mediafiles.all())
        for reid_suggestion in reid_suggestions:
            representative_mediafiles = reid_suggestion.identity.mediafile_set.filter(identity_is_representative=True)
            # insert as first element the reid_suggestion mediafile
            representative_mediafiles = [reid_suggestion.mediafile] + [mf for mf in list(representative_mediafiles) if mf != reid_suggestion.mediafile]
            reid_suggestion.representative_mediafiles = representative_mediafiles


        for identity in remaining_identities:
            identity.representative_mediafiles = identity.mediafile_set.filter(identity_is_representative=True)



        # for identity in identities:
        #     identity.representative_mediafiles = identity.mediafile_set.filter(identity_is_representative=True)

        logger.debug(f"remaining:  {len(remaining_identities)=}")
        if len(remaining_identities) > 10:
            # print first 10 remaining identities
            logger.debug(f"{remaining_identities[:10]=}")
    else:
        return message_view(request, "No mediafiles for identification.")

    return render(
        request,
        "caidapp/get_individual_identity.html",
        {
            "foridentification": foridentification,
            "foridentifications": foridentifications,
            "remaining_identities": remaining_identities,
            "reid_suggestions": reid_suggestions,
            # "related_identities": identity_ids,
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


@staff_or_impersonated_staff_required
def run_taxon_classification_force_init(request, uploadedarchive_id):
    """Run processing of uploaded archive with removal of previous outputs."""
    return run_taxon_classification(request, uploadedarchive_id=uploadedarchive_id, force_init=True)


@staff_or_impersonated_staff_required
def run_taxon_classification(request, uploadedarchive_id, force_init=False):
    """Run processing of uploaded archive."""
    uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    if uploaded_archive.mediafiles_at_upload == 0:
        uploaded_archive.number_of_media_files_in_archive()

    run_species_prediction_async(uploaded_archive, force_init=force_init, extract_identites=uploaded_archive.contains_identities)
    # next_page = request.GET.get("next", "/caidapp/uploads")
    # return redirect(next_page)
    return redirect(request.META.get("HTTP_REFERER", "/"))


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
    caiduser = request.user.caiduser
    sig = signature(
        "init_identification",
        kwargs={
            # csv file should contain image_path, class_id, label
            "input_metadata_file": str(identity_metadata_file),
            "organization_id": request.user.caiduser.workgroup.id,
            "identification_model": {
                "name": caiduser.identification_model.name,
                "path": caiduser.identification_model.model_path,
            }
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
                # status="Species Finished",
                Q(taxon_status="TKN") | Q(taxon_status="TV"),
                owner__workgroup=request.user.caiduser.workgroup,
                contains_identities=False,
                # contains_single_taxon=True,
                taxon_for_identification__isnull=False,
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
        identification_status="IR",  # Ready for identification
        # contains_single_taxon=True,
        taxon_for_identification__isnull=False,
        contains_identities=False,
    ).all()
    for uploaded_archive in uploaded_archives:
        _run_identification(
            uploaded_archive,
            caiduser = request.user.caiduser
        )
    # next_page = request.GET.get("next", "caidapp:uploads_identities")
    # return redirect(next_page)
    return redirect(request.META.get("HTTP_REFERER", "/"))


def run_identification(request, uploadedarchive_id):
    """Run identification of uploaded archive."""
    uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    # check if user is owner member of the workgroup
    if uploaded_archive.owner.workgroup != request.user.caiduser.workgroup:
        return HttpResponseNotAllowed("Identification is for workgroup members only.")
    _run_identification(
        uploaded_archive,
        caiduser=request.user.caiduser,
    )
    # next_page = request.GET.get("next", "caidapp:uploads_identities")
    # return redirect(next_page)
    return redirect(request.META.get("HTTP_REFERER", "/"))


def _run_identification(
        uploaded_archive: UploadedArchive,
        caiduser: models.CaIDUser,
        taxon_str="Lynx lynx",
                        ):
    logger.debug("Generating CSV for run_identification...")
    mediafiles = uploaded_archive.mediafile_set.filter(category__name=taxon_str).all()
    logger.debug(f"Generating CSV for init_identification with {len(mediafiles)} records...")
    # csv_len = len(mediafiles)
    # csv_data = {"image_path": [None] * csv_len, "mediafile_id": [None] * csv_len}
    uploaded_archive.identification_status = "IAIP"

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
    uploaded_archive.identification_status = "IAIP"
    uploaded_archive.save()

    identify_signature = signature(
        "identify",
        kwargs=dict(
            input_metadata_file_path=str(identity_metadata_file),
            organization_id=uploaded_archive.owner.workgroup.id,
            output_json_file_path=str(output_json_file),
            top_k=3,
            uploaded_archive_id=uploaded_archive.id,
            identification_model={
                "name": caiduser.identification_model.name,
                "path": caiduser.identification_model.model_path,
            }
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
                    request,
                    f"Uploaded file with extension '{uploaded_archive_suffix}' is not an archive.",
                )


            if contains_single_taxon:
                next_url = reverse_lazy("caidapp:uploads_identities")
            else:
                next_url = reverse_lazy("caidapp:uploads")
            counts = uploaded_archive.number_of_media_files_in_archive()


            uploaded_archive.owner = request.user.caiduser
            logger.debug(f"{uploaded_archive.contains_identities=}, {contains_identities=}")
            logger.debug(f"{uploaded_archive.contains_single_taxon=}, {contains_single_taxon=}")
            # log actual url
            logger.debug(f"{request.build_absolute_uri()=}")
            uploaded_archive.contains_identities = contains_identities
            uploaded_archive.contains_single_taxon = contains_single_taxon
            uploaded_archive.name = Path(uploaded_archive.archivefile.name).stem
            # done in number_of_media_files_in_archive
            # uploaded_archive.videos_at_upload = counts["video_count"]
            # uploaded_archive.images_at_upload = counts["image_count"]
            # uploaded_archive.files_at_upload = counts["file_count"]
            # uploaded_archive.mediafiles_at_upload = counts["video_count"] + counts["image_count"]
            uploaded_archive.save()
            uploaded_archive.extract_location_check_at_from_filename(commit=True)
            run_species_prediction_async(uploaded_archive, extract_identites=contains_identities)

            # return JsonResponse({"data": "Data uploaded"})
            context = dict(
                headline="Upload finished",
                text=f"Uploaded {counts['file_count']} files ("
                + f"{counts['image_count']} images and {counts['video_count']} videos).",
                next=next_url,
                next_text="Back to uploads",
            )




            html = render_to_string(
                "caidapp/partial_message.html", context=context, request=request
            )
            return JsonResponse({"html": html})
        else:
            # Error
            if contains_single_taxon:
                next_url = reverse_lazy("caidapp:uploads_identities")
            else:
                next_url = reverse_lazy("caidapp:uploads")
            context = dict(
                headline="Upload failed",
                text="Upload failed. Try it again.",
                next=next_url,
                next_text="Back to uploads",
            )
            html = render_to_string(
                "caidapp/partial_message.html", context=context, request=request
            )
            return JsonResponse({"html": html})

    else:

        initial_data = {
            "contains_identities": contains_identities,
            "contains_single_taxon": contains_single_taxon,
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




@login_required
def do_cloud_import_view(request):
    """Bulk import from one dir and prepare zip file for every check.

    Make zip file from every check. The information encoded in path is code of lynx season (i.e.
    LY2019), locality (Prachatice), date of check (2019-07-01). In the leaf directory are media
    files (images and /dvideos). For every check there will be zip file. The name of the zip file
    will be composed of locality and date of check.

    Example of path structure:
    NETRIDENA/LY2019/PRACHATICE/2019-07-01/2019-07-01_12-00-00_0001.jpg
    """
    if len(request.user.caiduser.import_dir) == 0:
        return HttpResponseNotAllowed("No import directory specified. Ask admin to set it up.")

    # get list of available localities
    caiduser = request.user.caiduser

    tasks.do_cloud_import_for_user_async(caiduser)
    # tasks.do_cloud_import_for_user(caiduser)

    return redirect("caidapp:cloud_import_preview")


@login_required
def break_cloud_import_view(request):
    """View for interrupted import from the cloud."""
    caiduser = request.user.caiduser
    caiduser.dir_import_status = "Interrupted"
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
    if not user_has_rw_acces_to_uploadedarchive(request.user.caiduser, uploaded_archive):
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

    if user_has_rw_acces_to_uploadedarchive(
        request.user.caiduser, uploadedarchive, accept_none=True
    ):
        uploadedarchive.delete()
    else:
        messages.error(request, "Not allowed to delete this uploaded archive.")
    return redirect(next_page)


@login_required
def delete_mediafile(request, mediafile_id):
    """Delete uploaded file."""
    mediafile = get_object_or_404(MediaFile, pk=mediafile_id)
    if user_has_rw_access_to_mediafile(request.user.caiduser, mediafile, accept_none=True):
        parent_id = mediafile.parent_id
        uploaded_archive = mediafile.parent
        if uploaded_archive is not None:
            uploaded_archive.output_updated_at = None
            uploaded_archive.save()
        mediafile.delete()
        if uploaded_archive is None:
            return redirect("caidapp:uploads")
        else:
            return redirect("caidapp:uploadedarchive_mediafiles", uploadedarchive_id=parent_id)
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


def _mediafiles_annotate() -> dict:

    return dict()


def _mediafiles_query(
    request,
    query: str,
    album_hash=None,
    individual_identity_id=None,
    taxon_id=None,
    uploadedarchive_id=None,
    identity_is_representative=None,
    location_hash=None,
    order_by: Optional[str] = None,
    taxon_verified: Optional[bool] = None,
    filter_kwargs: Optional[dict] = None,
    exclude_filter_kwargs: Optional[dict] = None,
):
    """Prepare list of mediafiles based on query search in category and location."""
    if filter_kwargs is None:
        filter_kwargs = {}

    if exclude_filter_kwargs is None:
        exclude_filter_kwargs = {}
    if order_by is None:
        order_by = request.session.get("mediafiles_order_by", "-parent__uploaded_at")

    mediafiles = MediaFile.objects.annotate(**_mediafiles_annotate())
    # mediafiles = (
    #     mediafiles.filter(
    #         Q(album__albumsharerole__user=request.user.caiduser)
    #         | Q(parent__owner=request.user.caiduser)
    #         | Q(parent__owner__workgroup=request.user.caiduser.workgroup)
    #     )
    #     .distinct()
    #     .order_by(order_by)
    # )
    if taxon_verified is not None:
        filter_kwargs.update(dict(taxon_verified=taxon_verified))
        # mediafiles = mediafiles.filter(taxon_verified=taxon_verified).all()
    if album_hash is not None:
        album = get_object_or_404(Album, hash=album_hash)
        filter_kwargs.update(dict(album=album))
        # mediafiles = (
        #     mediafiles.filter(album=album).all().distinct().order_by(order_by)
        # )
    if individual_identity_id is not None:
        individual_identity = get_object_or_404(IndividualIdentity, pk=individual_identity_id)
        filter_kwargs.update(dict(identity=individual_identity))
        # mediafiles = (
        #     mediafiles.filter(identity=individual_identity)
        #     .all()
        #     .distinct()
        #     .order_by(order_by)
        # )
    if taxon_id is not None:
        taxon = get_object_or_404(Taxon, pk=taxon_id)
        filter_kwargs.update(dict(category=taxon))
        # mediafiles = (
        #     mediafiles.filter(category=taxon).all().distinct().order_by(order_by)
        # )
    if uploadedarchive_id is not None:
        uploadedarchive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
        filter_kwargs.update(dict(parent=uploadedarchive))
        # mediafiles = (
        #     mediafiles.filter(parent=uploadedarchive)
        #     .all()
        #     .distinct()
        #     .order_by(order_by)
        # )
    if identity_is_representative is not None:
        filter_kwargs.update(dict(identity_is_representative=identity_is_representative))
        # mediafiles = (
        #     mediafiles.filter(identity_is_representative=identity_is_representative)
        #     .all()
        #     .distinct()
        #     .order_by(order_by)
        # )
    if location_hash is not None:
        location = get_object_or_404(Location, hash=location_hash)
        filter_kwargs.update(dict(location=location))
        # mediafiles = (
        #     mediafiles.filter(location=location).all().distinct().order_by(order_by)
        # )
    logger.debug(f"{filter_kwargs=}")
    # order by mediafile__sequence__mediafile_set order by
    order_by_safe = order_by if order_by[0] != "-" else order_by[1:]
    first_image_order_by = (
        mediafiles.filter(sequence=OuterRef('sequence'))
        .order_by(order_by)
        .values(order_by_safe)[:1]
    )

    # Build the base query with the conditions that are always applied
    mediafiles = mediafiles.filter(
        Q(album__albumsharerole__user=request.user.caiduser)
        | Q(parent__owner=request.user.caiduser),
        **filter_kwargs,
    )

    # Add workgroup filtering only if `request.user.caiduser.workgroup` is not None
    if request.user.caiduser.workgroup is not None:
        mediafiles = mediafiles.filter(
            Q(parent__owner__workgroup=request.user.caiduser.workgroup)
        )

    # Apply the exclusion, annotations, and ordering
    mediafiles = (
        mediafiles
        .exclude(**exclude_filter_kwargs)
        .distinct()
        .annotate(first_image_order_by=Subquery(first_image_order_by))
        .order_by('first_image_order_by', 'sequence', 'captured_at')
    )

    # mediafiles = (
    #     mediafiles.filter(
    #         Q(album__albumsharerole__user=request.user.caiduser)
    #         | Q(parent__owner=request.user.caiduser)
    #         | Q(parent__owner__workgroup=request.user.caiduser.workgroup),
    #         **filter_kwargs,
    #     )
    #     .exclude(**exclude_filter_kwargs)
    #     .distinct()
    #     .annotate(first_image_order_by=Subquery(first_image_order_by))
    #     .order_by('first_image_order_by', 'sequence', 'captured_at')
    #     # .order_by(order_by)
    # )


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

    # fig = go.Figure(go.Densitymapbox(lat=df2.lat, lon=df2.lon, radius=10, showscale=False))
    # fig.update_layout(
    #     mapbox_style="open-street-map",
    #     mapbox_center_lon=df2.lon.unique().mean(),
    #     mapbox_center_lat=df2.lat.unique().mean(),
    #     mapbox_zoom=zoom,
    # )
    #
    # fig.update_layout(margin={"r": 0, "t": 10, "l": 0, "b": 0}, height=300)

    # Create the base map
    fig = go.Figure()

    # Add a density map for points
    fig.add_trace(
        go.Densitymapbox(
            lat=df2.lat,
            lon=df2.lon,
            radius=10,
            showscale=False,
            name = 'Density Map',
        )
    )

    # Add a Scattermapbox trace to connect points with lines
    fig.add_trace(
        go.Scattermapbox(
            lat=df2.lat,
            lon=df2.lon,
            mode='lines+markers',  # Shows both lines and markers
            marker=dict(size=7),  # Adjust marker size
            line=dict(width=2, color='blue'),  # Adjust line style
            name = 'Path (Lines and Markers)',  # Legend label
            visible = True  # Default visibility
        )
    )

    # Update layout with map style and center
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center_lon=df2.lon.mean(),
        mapbox_center_lat=df2.lat.mean(),
        mapbox_zoom=zoom,
        margin={"r": 0, "t": 10, "l": 0, "b": 30},
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="top",  # Align to the top of the legend
            y=-0.1,  # Place below the map (negative value for outside the map area)
            xanchor="center",  # Center align horizontally
            x=0.5  # Position at the center
        )

    )

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


def _merge_form_filter_kwargs_with_filter_kwargs(
    filter_kwargs: dict, exclude_filter_kwargs: dict, form_filter_kwargs: dict
) -> (dict, dict):

    # filter parameters for MediaFiles

    ffk = form_filter_kwargs
    if ffk.get("filter_show_videos", None) and ffk.get("filter_show_images", None):
        pass
    elif ffk.get("filter_show_videos", None):
        filter_kwargs["media_type"] = "video"
    elif ffk.get("filter_show_images", None):
        filter_kwargs["media_type"] = "image"

    if ffk.get("filter_hide_empty", None):
        # MediaFile.category.name is not "Empty"
        exclude_filter_kwargs.update(dict(category__name="Nothing"))

    return filter_kwargs, exclude_filter_kwargs


def media_files_update(
    request,
    records_per_page: Optional[int] = None,
    album_hash=None,
    individual_identity_id=None,
    taxon_id=None,
    uploadedarchive_id=None,
    identity_is_representative=None,
    location_hash=None,
    show_overview_button=False,
    order_by=None,
    taxon_verified: Optional[bool] = None,
    **filter_kwargs,
) -> Union[QuerySet, List[MediaFile]]:
    """List of mediafiles based on query with bulk update of category."""
    # create list of mediafiles

    page_number = 1
    exclude_filter_kwargs = {}
    form_filter_kwargs = {}
    if records_per_page is None:
        records_per_page = request.session.get("mediafiles_records_per_page", 20)

    if request.method == "POST":
        queryform = MediaFileSetQueryForm(request.POST)
        if queryform.is_valid():
            query = queryform.cleaned_data["query"]
            # logger.debug(f"{queryform.cleaned_data=}")
            # page_number = int(request.GET.get('page'))
            # logger.debug(f"{page_number=}")
            # pick all parameters from queryform with key startin with "filter_"
            for key in queryform.cleaned_data.keys():
                if key.startswith("filter_"):
                    form_filter_kwargs[key] = queryform.cleaned_data[key]
            logger.debug(f"{form_filter_kwargs=}")

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
        # request.method == "GET"
        # logger.debug("GET")
        page_number = 1
        initial_data = dict(
            query="",
            pagenumber=page_number,
            filter_show_videos=True,
            filter_show_images=True,
            filter_hide_empty=not show_overview_button,
        )

        queryform = MediaFileSetQueryForm(initial=initial_data)
        query = ""
    albums_available = (
        Album.objects.filter(
            Q(albumsharerole__user=request.user.caiduser) | Q(owner=request.user.caiduser)
        )
        .distinct()
        .order_by("created_at")
    )

    filter_kwargs, exclude_filter_kwargs = _merge_form_filter_kwargs_with_filter_kwargs(
        filter_kwargs, exclude_filter_kwargs, form_filter_kwargs
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
        taxon_verified=taxon_verified,
        filter_kwargs=filter_kwargs,
        exclude_filter_kwargs=exclude_filter_kwargs,
    )
    if uploadedarchive_id is not None:
        uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
        # datetime format YYYY-MM-DD HH:MM:SS
        if uploaded_archive.location_check_at is not None:
            location_check_at = " - " + uploaded_archive.location_check_at.strftime("%Y-%m-%d %H:%M:%S")
        else:
            location_check_at = ""
        page_title = f"Media files - {uploaded_archive.location_at_upload}{location_check_at}"

    elif album_hash is not None:
        album = get_object_or_404(Album, hash=album_hash)
        page_title = f"Media files - {album.name}"
    elif individual_identity_id is not None:
        individual_identity = get_object_or_404(IndividualIdentity, pk=individual_identity_id)
        page_title = f"Media files - {individual_identity.name}"
    elif taxon_id is not None:
        taxon = get_object_or_404(Taxon, pk=taxon_id)
        page_title = f"Media files - {taxon.name}"
    elif location_hash is not None:
        location = get_object_or_404(Location, hash=location_hash)
        page_title = f"Media files - {location.name}"
    else:
        page_title = "Media files"

    number_of_mediafiles = len(full_mediafiles)

    request.session["mediafile_ids"] = list(full_mediafiles.values_list("id", flat=True))
    paginator = Paginator(full_mediafiles, per_page=records_per_page)
    page_with_mediafiles, _, page_context = _prepare_page(paginator, page_number=page_number)

    page_ids = [obj.id for obj in page_with_mediafiles.object_list]
    request.session["mediafile_ids_page"] = page_ids

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

            select_all = True if form.data.get("select_all", '') == "on" else False
            logger.debug(f"{select_all=}")
            for mediafileform in form:
                if mediafileform.is_valid():
                    if mediafileform.cleaned_data["selected"] or select_all:
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
                        elif "btnBulkProcessing_id_taxon_verified" in form.data:
                            instance = mediafileform.save(commit=False)
                            instance.taxon_verified = form_bulk_processing.cleaned_data[
                                "taxon_verified"
                            ]
                            instance.updated_by = request.user.caiduser
                            instance.updated_at = django.utils.timezone.now()
                            instance.save()

                        elif "btnBulkProcessing_set_taxon_verified" in form.data:
                            instance = mediafileform.save(commit=False)
                            instance.taxon_verified = True
                            instance.updated_by = request.user.caiduser
                            instance.updated_at = django.utils.timezone.now()
                            instance.save()

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
            "page_title": page_title,
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


from dateutil.relativedelta import relativedelta  # Import relativedelta


def change_mediafiles_datetime(request):
    """Change time of media files."""
    next_url = request.GET.get("next_url", None)
    mediafile_ids = request.session.get("mediafile_ids", [])
    mediafiles = MediaFile.objects.filter(id__in=mediafile_ids)
    if request.method == "POST":
        form = forms.ChangeMediaFilesTimeForm(request.POST)
        if form.is_valid():
            change_by_hours = form.cleaned_data["change_by_hours"]
            change_by_days = form.cleaned_data["change_by_days"]
            change_by_years = form.cleaned_data["change_by_years"]

            for mediafile in mediafiles:
                mediafile.captured_at = mediafile.captured_at + relativedelta(
                    hours=change_by_hours if change_by_hours else 0,
                    days=change_by_days if change_by_days else 0,
                    years=change_by_years if change_by_years else 0,
                )
                mediafile.save()
            # go to previous url

            logger.debug("Going back")
            if next_url is None:
                next_url = reverse_lazy("caidapp:uploads")
            return redirect(next_url)

        else:
            text_note = "Change time of media files. Use negative values to subtract time."
            return render(
                request,
                "caidapp/update_form.html",
                {
                    "form": form,
                    "headline": "Change time",
                    "button": "Change",
                    "text_note": text_note,
                    "next": "caidapp:uploads",
                },
            )

    else:
        form = forms.ChangeMediaFilesTimeForm()

    return render(
        request,
        "caidapp/update_form.html",
        {
            "form": form,
            "headline": "Change time",
            "button": "Change",
            "text_note": "Change time of media files. Use negative values to subtract time.",
            "next": "caidapp:uploads",
        },
    )


def mediafiles_stats_view(request):
    """Show mediafiles stats."""
    mediafile_ids = request.session.get("mediafile_ids", [])
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
        },
    )


@login_required
def select_taxon_for_identification(request, uploadedarchive_id: int):
    """Select taxon for identification."""
    uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    if not user_has_rw_acces_to_uploadedarchive(request.user.caiduser, uploaded_archive):
        return HttpResponseNotAllowed("Not allowed to edit this uploaded archive.")
    if request.method == "POST":
        form = UploadedArchiveSelectTaxonForIdentificationForm(request.POST)
        if form.is_valid():
            taxon = form.cleaned_data["taxon_for_identification"]
            uploaded_archive.taxon_for_identification = taxon
            uploaded_archive.identification_status = "IR"  # Ready for identification
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
    # get mediaifles_ids based on uplodedarchive id
    uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)

    # full_mediafiles = MediaFile.objects.filter(parent=uploaded_archive)
    # mediafile_ids = list(full_mediafiles.values_list("id", flat=True))
    if (
        uploaded_archive.ownder == request.user.caiduser
        or uploaded_archive.owner.workgroup == request.user.caiduser.workgroup
    ):
        _update_csv_by_uploadedarchive(request, uploadedarchive_id)
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


def _get_mediafiles(request, uploadedarchive_id: Optional[int]) -> Tuple[QuerySet, Optional[str]]:
    """Get mediafiles based on uploadedarchive_id or session."""
    name_suggestion = None
    if uploadedarchive_id is not None:
        uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
        if (
            uploaded_archive.owner == request.user.caiduser
            or uploaded_archive.owner.workgroup == request.user.caiduser.workgroup
        ):
            mediafiles = MediaFile.objects.filter(parent=uploaded_archive)
            name_suggestion = uploaded_archive.name
            logger.debug(f"{name_suggestion=}")
        else:
            messages.error(request, "Only the owner or work group member can access the data.")
    else:
        mediafile_ids = request.session.get("mediafile_ids", [])
        mediafiles = MediaFile.objects.filter(id__in=mediafile_ids)

    return mediafiles, name_suggestion


def download_csv_for_mediafiles_view(request, uploadedarchive_id: Optional[int] = None):
    """Download csv for media files."""
    mediafiles, name_suggestion = _get_mediafiles(request, uploadedarchive_id)
    fn = ("metadata_" + name_suggestion) if name_suggestion is not None else "metadata"

    try:
        df = tasks.create_dataframe_from_mediafiles(mediafiles)
        if df.empty:
            return HttpResponse("No data available to export.", content_type="text/plain")
    except Exception:
        logger.error(traceback.format_exc())
        return HttpResponse("Error during export.", content_type="text/plain")
    # df = tasks.create_dataframe_from_mediafiles(mediafiles)
    response = HttpResponse(df.to_csv(), content_type="text/csv")
    response["Content-Disposition"] = f"attachment; filename={fn}.csv"
    return response


def download_xlsx_for_mediafiles_view(request, uploadedarchive_id: Optional[int] = None):
    """Download xlsx for media files."""
    mediafiles, name_suggestion = _get_mediafiles(request, uploadedarchive_id)
    fn = ("metadata_" + name_suggestion) if name_suggestion is not None else "metadata"

    try:
        df = tasks.create_dataframe_from_mediafiles(mediafiles)
        if df.empty:
            return HttpResponse("No data available to export.", content_type="text/plain")
    except Exception:
        logger.error(traceback.format_exc())
        return HttpResponse("Error during export.", content_type="text/plain")

    # convert timezone-aware datetime to naive datetime
    df = model_tools.convert_datetime_to_naive(df)

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Locations")

    # Rewind the buffer
    output.seek(0)

    response = HttpResponse(
        output, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    response["Content-Disposition"] = f"attachment; filename={fn}.xlsx"
    return response


def download_zip_for_mediafiles_view(request, uploadedarchive_id: Optional[int] = None) -> JsonResponse:
    """Download zip for media files."""
    mediafiles, name_suggestion = _get_mediafiles(request, uploadedarchive_id)
    fn = ("mediafiles_" + name_suggestion) if name_suggestion is not None else "mediafiles"
    # number_of_mediafiles = len(mediafiles)
    logger.debug(f"{len(mediafiles)=}")

    user_hash = request.user.caiduser.hash
    abs_zip_path = (
            Path(settings.MEDIA_ROOT) / "users" / request.user.caiduser.hash / "mediafiles.zip"
    )

    # Prepare the mediafiles list for serialization (e.g., paths and output names)
    mediafiles_data = [
        {"path": mf.mediafile.name, "output_name": _make_output_name(mf)} for mf in mediafiles
    ]

    # Start the Celery task
    task = tasks.create_mediafiles_zip.delay(user_hash, mediafiles_data, str(abs_zip_path))

    # Return the task ID so the frontend can poll for completion
    return JsonResponse({"task_id": task.id})


# def download_zip_for_mediafiles_view_old(request, uploadedarchive_id: Optional[int] = None):
#     """Download zip for media files."""
#     mediafiles, name_suggestion = _get_mediafiles(request, uploadedarchive_id)
#     fn = ("mediafiles_" + name_suggestion) if name_suggestion is not None else "mediafiles"
#     # number_of_mediafiles = len(mediafiles)
#
#     abs_zip_path = (
#         Path(settings.MEDIA_ROOT) / "users" / request.user.caiduser.hash / "mediafiles.zip"
#     )
#     abs_zip_path.parent.mkdir(parents=True, exist_ok=True)
#     # get temp dir
#     import tempfile
#
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         mediafiles_dir = Path(tmpdirname) / "images"
#         mediafiles_dir.mkdir()
#         for mediafile in mediafiles:
#             # output_name = Path(mediafile.mediafile.name).name
#             output_name = _make_output_name(mediafile)
#             # get mediafile name with suffix
#             src = Path(settings.MEDIA_ROOT) / mediafile.mediafile.name
#             dst = mediafiles_dir / output_name
#             logger.debug(f"{src=}, {dst=}")
#             assert src.exists()
#             shutil.copy(src, dst)
#         tasks.make_zipfile(abs_zip_path, mediafiles_dir)
#     with open(abs_zip_path, "rb") as fh:
#         response = HttpResponse(fh.read(), content_type="application/zip")
#         response["Content-Disposition"] = f"inline; filename={fn}.zip"
#         return response
#
#     return Http404

def check_zip_status_view(request, task_id):
    """Check the status of the zip creation task."""

    # find task based on task_id
    task = AsyncResult(task_id)
    logger.debug(f"{task.state=}")
    if task.state == "SUCCESS":
        # Task is complete, return the download link
        # download_url = f"/media/users/{request.user.caiduser.hash}/mediafiles.zip"
        download_url = f"{settings.MEDIA_URL}users/{request.user.caiduser.hash}/mediafiles.zip"
        logger.debug(f"{download_url=}")
        # download_url = request.build_absolute_uri(download_url)
        # logger.debug(f"{download_url=}")

        return JsonResponse({"status": "ready", "download_url": download_url})

    elif task.state == "PENDING":
        return JsonResponse({"status": "pending"})
    elif task.state == "FAILURE":
        return JsonResponse({"status": "error", "message": str(task.result)})
    return JsonResponse({"status": task.state})



def _make_output_name(mediafile: models.MediaFile):
    """Create output name.

    pattern: {locality}_{date}_{original_name}_{taxon}_{identity}
    """
    locality = mediafile.location.name if mediafile.location else "no_locality"
    date = mediafile.captured_at.strftime("%Y-%m-%d") if mediafile.captured_at else "no_date"
    original_name = (
        mediafile.original_filename if mediafile.original_filename else "no_original_name"
    )
    # remove extension
    original_name = Path(original_name).stem
    taxon = mediafile.category.name if mediafile.category else "no_taxon"
    identity = mediafile.identity.name if mediafile.identity else "no_identity"
    suffix = Path(mediafile.mediafile.name).suffix
    output_name = f"{locality}_{date}_{original_name}_{taxon}_{identity}.{suffix}"
    return output_name


def _generate_new_hash_for_locations():
    for location in Location.objects.all():
        location.hash = models.get_hash8()
        location.save()


def refresh_data(request):
    """Update new calculations for formerly uploaded archives."""
    uploaded_archives = UploadedArchive.objects.all()
    for uploaded_archive in uploaded_archives:
        uploaded_archive.update_earliest_and_latest_captured_at()
        uploaded_archive.make_sequences()

        if (
            uploaded_archive.contains_single_taxon
            and uploaded_archive.taxon_for_identification is None
        ):
            # this fixes the compatibility with the old version before 2024-05
            uploaded_archive.taxon_for_identification = models.get_taxon("Lynx lynx")
            uploaded_archive.save()

        # uploaded_archive.refresh_status_after_migration(request)

    # this was used to fix same hashes generated by wrong function
    # _generate_new_hash_for_locations()

    # _refresh_media_file_original_name(request)
    tasks.refresh_thumbnails()

    # get taxon (and create it if it does not exist
    models.get_taxon("Unclassifiable")

    return redirect("caidapp:uploads")


def _refresh_media_file_original_name(request):
    for mediafile in MediaFile.objects.all():
        mediafile.extract_original_filename()


def shared_individual_identity_view(request, identity_hash: str):
    """Show shared individual identity to any user."""
    identity = get_object_or_404(IndividualIdentity, hash=identity_hash)
    mediafiles = MediaFile.objects.filter(identity=identity, identity_is_representative=True).all()

    return render(
        request,
        "caidapp/identity_detail_public.html",
        {
            "identity": identity,
            "mediafiles": mediafiles,
        },
    )


def set_sort_uploaded_archives_by(request, sort_by: str):
    """Sort uploaded archives by."""
    request.session["sort_uploaded_archives_by"] = sort_by

    # go back to previous page
    return redirect(request.META.get("HTTP_REFERER", "/"))


def uploaded_archive_get_order_by(request):
    """Get order by for uploaded archives."""
    sort_by = request.session.get("sort_uploaded_archives_by", "-uploaded_at")
    return sort_by


def set_item_number_uploaded_archives(request, item_number: int):
    """Sort uploaded archives by."""
    request.session["item_number_uploaded_archives"] = item_number

    # go back to previous page
    return redirect(request.META.get("HTTP_REFERER", "/"))


def get_item_number_uploaded_archives(request):
    """Get order by for uploaded archives."""
    item_number = request.session.get("item_number_uploaded_archives", 12)
    return item_number


def switch_private_mode(request):
    """Switch private mode."""
    actual_mode = request.session.get("private_mode", False)
    request.session["private_mode"] = not actual_mode

    return redirect(request.META.get("HTTP_REFERER", "/"))


class ImageUploadGraphView(View):
    def get(self, request):
        """Render the image upload graph."""
        # Fetch data from MediaFile model
        mediafiles = MediaFile.objects.all().values(
            "parent__uploaded_at", "parent__owner__user__username"
        )

        # Convert to DataFrame
        df = pd.DataFrame(mediafiles)
        df["parent__uploaded_at"] = pd.to_datetime(df["parent__uploaded_at"])
        df["date"] = df["parent__uploaded_at"].dt.date

        # Create Plotly histogram
        fig = px.histogram(
            df,
            x="date",
            color="parent__owner__user__username",
            title="Media Files Uploaded Over Time by User",
            labels={
                "date": "Upload Date",
                "count": "Number of Uploaded Files",
                "parent__owner__user__username": "User",
            },
        )

        # Customize x-axis to show dates properly
        fig.update_xaxes(type="category", title_text="Upload Date")
        fig.update_yaxes(title_text="Number of Uploads")

        # Convert Plotly figure to HTML
        graph = fig.to_html(full_html=False)

        return render(request, "caidapp/image_upload_graph.html", {"graph": graph})



class MergeIdentities(View):

    def get_individuals(self, request, id1, id2):
        """Fetch the individual identities."""
        individual_identity1 = get_object_or_404(
            IndividualIdentity,
            pk=id1,
            owner_workgroup=request.user.caiduser.workgroup,
        )
        individual_identity2 = get_object_or_404(
            IndividualIdentity,
            pk=id2,
            owner_workgroup=request.user.caiduser.workgroup,
        )
        return individual_identity1, individual_identity2

    def generate_differences(self, individual1, individual2):
        """Generate differences between two identities."""
        differences = {}
        fields_to_compare = ["sex", "coat_type", "birth_date", "death_date"]
        for field in fields_to_compare:
            value1 = getattr(individual1, field)
            value2 = getattr(individual2, field)
            if value1 != value2:
                differences[field] = f"{value1} , {value2}"
        return differences

    def get(self, request, individual_identity1_id, individual_identity2_id):
        """Render the merge form."""
        individual1, individual2 = self.get_individuals(request, individual_identity1_id, individual_identity2_id)

        # Suggestion based on merging logic
        suggestion = IndividualIdentity(
            name=f"{individual1.name} + {individual2.name}",
            sex=individual1.sex if individual1.sex != "U" else individual2.sex,
            coat_type=individual1.coat_type if individual1.coat_type != "U" else individual2.coat_type,
            birth_date=individual1.birth_date or individual2.birth_date,
            death_date=individual1.death_date or individual2.death_date,
            note=f"{individual1.note}\n{individual2.note}",
            code=f"{individual1.code} {individual2.code}",
            juv_code=f"{individual1.juv_code} {individual2.juv_code}",
        )

        # Differences for the right column
        differences = self.generate_differences(individual1, individual2)
        differences_html = "<h3>Differences</h3><ul>" + "".join(f"<li>{key}: {value}</li>" for key, value in differences.items()) + "</ul>"

        form = IndividualIdentityForm(instance=suggestion)
        media_file = MediaFile.objects.filter(identity=individual1, identity_is_representative=True).first()

        return render(
            request,
            "caidapp/update_form.html",
            {
                "form": form,
                "headline": "Merge Individual Identity",
                "button": "Save",
                "individual_identity": individual1,
                "mediafile": media_file,
                "delete_button_url": reverse_lazy(
                    "caidapp:delete_individual_identity",
                    kwargs={"individual_identity_id": individual_identity1_id},
                ),
                "right_col_raw_html": differences_html,
            },
        )

    def post(self, request, individual_identity1_id, individual_identity2_id):
        """Handle form submission."""
        individual1, individual2 = self.get_individuals(request, individual_identity1_id, individual_identity2_id)

        form = IndividualIdentityForm(request.POST, instance=individual1)
        if form.is_valid():
            individual_identity = form.save(commit=False)
            individual_identity.updated_by = request.user.caiduser
            individual_identity.save()

            # mediafiles of identity2 are reassigned to identity1
            individual2.mediafile_set.update(identity=individual1)
            # remove old identity
            individual2.delete()

            return redirect("caidapp:individual_identities")

        # On failure, re-render the form with errors
        return self.get(request, individual_identity1_id, individual_identity2_id)



class UpdateUploadedArchiveBySpreadsheetFile(View):

    def __init__(self):
        self.prev_url = None


    def post(self, request, uploaded_archive_id):
        """Handle the form submission."""
        uploaded_archive = get_object_or_404(UploadedArchive, pk=uploaded_archive_id)
        output_dir = Path(settings.MEDIA_ROOT) / uploaded_archive.outputdir

        form = forms.UploadedArchiveUpdateBySpreadsheetForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the form data to the output directory
            spreadsheet_file = request.FILES["spreadsheet_file"]
            file_path = output_dir / spreadsheet_file.name
            logger.debug(f"{file_path=}")
            with open(file_path, "wb+") as destination:
                for chunk in spreadsheet_file.chunks():
                    destination.write(chunk)

            logger.debug(f"{file_path.exists()=}")


            if file_path.suffix == ".csv":
                df = pd.read_csv(file_path, index_col=0)
            elif file_path.suffix == ".xlsx":
                df = pd.read_excel(file_path)
            else:
                return messages.error(request, "Only CSV and XLSX files are supported.")
                df = None

            # load metadata

            logger.debug(f"{uploaded_archive.csv_file.name=}")
            # metadata = pd.read_csv(Path(settings.MEDIA_ROOT) / uploaded_archive.csv_file.name, index_col=0)

            # metadata = merge_update_spreadsheet_with_metadata_spreadsheet(df, metadata)
            logger.debug("deleting uploaded file")
            Path(file_path).unlink()
            logger.debug(f"{df.columns=}")


            # metadata.to_csv(uploaded_archive.csv_file.name, encoding="utf-8-sig")
            df = df.rename({
                "original path": "original_path",
                "taxon": "category",
                "unique name": "unique_name",
                "location_name": "location name",
                "lat": "latitude",
                "lon": "longitude",
                "datetime": "datetime",
            })

            counter0 = 0
            counter1 = 0
            counter_file_in_spreadsheet_does_not_exist = 0
            self.prev_url = request.META.get("HTTP_REFERER", "/")
            if "original_path" not in df.columns:
                return message_view(
                    request,
                    "The 'original_path' column is required in the uploaded spreadsheet.",
                    headline="Update metadata",
                    link=self.prev_url,
                    button="Ok",
                )
            for i, row in df.iterrows():
                original_path = row['original_path']

                # get or None
                mf = MediaFile.objects.filter(parent=uploaded_archive, original_filename=original_path).first()
                if mf:
                    logger.debug(f"{mf=}")
                    counter0 += 1
                    # mf.category = row['category']
                    if "predicted_category" in row:
                        mf.category = models.get_taxon(row["predicted_category"])  # remove this
                        counter1 += 1

                    if "unique_name" in row:
                        mf.identity = models.get_unique_name(
                            row["unique_name"], workgroup=uploaded_archive.owner.workgroup
                        )
                        counter1 += 1
                    if "location name" in row:
                        mf.location = models.get_location(
                            caiduser=request.user.caiduser,
                            name=row["location name"])
                        if ("latitude" in row) and ("longitude" in row):
                            mf.location.set_location(float(row["latitude"]), float(row["longitude"]))
                            counter1 += 1
                        counter1 += 1
                    if "datetime" in row:
                        mf.captured_at = row["datetime"]
                        counter1 += 1

                    mf.save()
                else:
                    counter_file_in_spreadsheet_does_not_exist += 1
            return message_view(
                request,
                "Updated metadata for " + str(counter0) + " mediafiles. " + str(counter1) + " fields updated. " +\
                str(counter_file_in_spreadsheet_does_not_exist) + " files in spreadsheet do not exist.",
                headline="Update metadata",
                link=self.prev_url,
            )
            # return redirect(self.prev_url)
        else:
            return render(
                request,
                "caidapp/update_form.html",
                {
                    "form": form,
                    "headline": "Upload XLSX or CSV with column 'original_path'...",
                    "button": "Save",
                    "errors": form.errors,
                    "text_note": "The 'original_path' is required in the uploaded spreadsheet. " \
                        + "The 'predicted_category', 'unique_name', 'location name', 'latitude', " \
                        + "'longitude', 'datetime' are optional.",
                },
            )


    def get(self, request, uploaded_archive_id):
        """Render the form for updating the uploaded archive."""
        uploaded_archive = get_object_or_404(UploadedArchive, pk=uploaded_archive_id)
        output_dir = Path(settings.MEDIA_ROOT) / uploaded_archive.outputdir
        assert output_dir.exists()
        form = forms.UploadedArchiveUpdateBySpreadsheetForm()

        prev_url = request.META.get("HTTP_REFERER", "/")
        self.prev_url
        return render(request, "caidapp/update_form.html", {
            "form": form,
            "headline": "Upload XLSX or CSV",
            "button": "Save",
            "next": prev_url,
            "text_note": "The 'original_path' is required in the uploaded spreadsheet. " \
                         + "The 'predicted_category', 'unique_name', 'location name', 'latitude', " \
                         + "'longitude', 'datetime' are optional.",
        })


@login_required
def select_second_id_for_identification_merge(request, individual_identity1_id: int):
    """Select taxon for identification."""
    individual_identity1 = get_object_or_404(IndividualIdentity, pk=individual_identity1_id)
    identities = IndividualIdentity.objects.filter(owner_workgroup=request.user.caiduser.workgroup).exclude(pk=individual_identity1_id)
    if request.method == "POST":
        form = forms.IndividualIdentitySelectSecondForMergeForm(request.POST, identities=identities)
        logger.debug("we are in POST")
        if form.is_valid():
            logger.debug("form is valid")
            identity = form.cleaned_data["identity"]
            return redirect("caidapp:merge_identities", individual_identity1_id, identity.pk)
    else:
        form = forms.IndividualIdentitySelectSecondForMergeForm(identities=identities)
    return render(
        request,
        "caidapp/update_form.html",
        {
            "form": form,
            "headline": "Select identity for merge",
            "button": "Select",
            "text_note": "The selected identity will be merged into the first one and then deleted.",
            # "next": "caidapp:uploads_identities",
            "mediafile": individual_identity1.mediafile_set.all().first(),
        },
    )

