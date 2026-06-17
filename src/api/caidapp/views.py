import datetime
import io
import logging
import os
import random
import re
import time
import traceback
import urllib.parse
import zipfile
from functools import wraps
from io import BytesIO
from pathlib import Path
from string import Formatter
from typing import Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import django
import django.db
import django.utils.timezone
import numpy as np
import pandas as pd
import plotly.express as px
from celery import current_app, signature
from celery.result import AsyncResult
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin

# from django.contrib.auth.models import User
from django.contrib.auth.views import LoginView
from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
from django.core.exceptions import PermissionDenied
from django.core.files.base import ContentFile
from django.core.paginator import Page, Paginator
from django.db.models import CharField, Count, F, Func, Max, Min, OuterRef, Prefetch, Q, QuerySet, Subquery, Value
from django.db.models.functions import Cast, Coalesce
from django.forms import modelformset_factory
from django.forms.models import model_to_dict
from django.http import HttpRequest, HttpResponseNotAllowed, JsonResponse
from django.shortcuts import Http404, HttpResponse, get_object_or_404, redirect, render
from django.template.loader import render_to_string
from django.urls import reverse, reverse_lazy
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.utils.translation import gettext_lazy as _
from django.views import View
from django.views.decorators.http import require_POST
from django.views.generic import CreateView, DeleteView, DetailView, ListView, UpdateView
from djangoaddicts.pygwalker.views import PygWalkerView
from tqdm import tqdm

from . import filters, forms, model_tools, models, tasks, upload_services, views_general, views_locality, views_uploads
from .forms import (  # WorkgroupUsersForm,
    AlbumForm,
    IndividualIdentityForm,
    MediaFileBulkForm,
    MediaFileSelectionForm,
    UploadedArchiveForm,
    UploadedArchiveFormWithTaxon,
    UploadedArchiveSelectTaxonForIdentificationForm,
    UploadedArchiveUpdateForm,
    UserSelectForm,
)
from .model_extra import compute_identity_suggestions, user_has_rw_acces_to_uploadedarchive, user_has_rw_access_to_mediafile
from .model_tools import timesince_now
from .models import (
    Album,
    AnimalObservation,
    ArchiveCollection,
    IndividualIdentity,
    Locality,
    MediaFile,
    MediafilesForIdentification,
    Notification,
    Taxon,
    UploadedArchive,
    WorkGroup,
    get_all_relevant_localities,
    user_has_access_filter_params,
)
from .services.home_dashboard import render_home_dashboard_context
from .services.workgroup_migration import migrate_user_to_workgroup
from .services.workgroup_next_steps import build_next_steps
from .tasks import (
    _iterate_over_locality_checks,
    _prepare_dataframe_for_identification,
    get_locality,
    identify_on_success,
    init_identification_on_error,
    on_error_in_upload_processing,
    run_species_prediction_async,
    update_metadata_csv_by_uploaded_archive,
)
from .views_locality import _set_localities_to_mediafiles_of_uploadedarchive
from .views_tools import add_querystring_to_context

logger = logging.getLogger("app")
User = get_user_model()


MEDIAFILE_EXPORT_SCHEMAS = {
    "species_identity": "{species}/{identity}/{hash}_{species}_{identity}{dotext}",
    "identity_dirs": "{identity}/{hash}_{species}_{identity}{dotext}",
    "flat": "{hash}_{species}_{identity}{dotext}",
}

SEQUENCE_DOWNLOAD_SESSION_KEY = "sequence_download_mediafile_ids"
SEQUENCE_DOWNLOAD_RETURN_URL_SESSION_KEY = "sequence_download_return_url"

SEQUENCE_EXPORT_COLUMNS = [
    ("unique_name", "Identity"),
    ("code", "Identity code"),
    ("juv_code", "Juvenile code"),
    ("locality name", "Locality name"),
    ("locality coordinates", "Locality coordinates"),
    ("latitude", "Latitude"),
    ("longitude", "Longitude"),
    ("original_path", "Original path"),
    ("export_path", "Export path"),
    ("datetime", "Datetime"),
    ("uploaded_archive", "Uploaded archive"),
    ("locality_check_at", "Locality check at"),
    ("mediafile_id", "Media file ID"),
    ("sequence_id", "Sequence ID"),
    ("observation_id", "Observation ID"),
    ("predicted_category", "Taxon"),
    ("media_type", "Media type"),
    ("taxon_verified", "Taxon verified"),
    ("predicted_taxon", "Predicted taxon"),
    ("predicted_taxon_confidence", "Predicted taxon confidence"),
    ("identity_is_representative", "Identity is representative"),
    ("orientation", "Orientation"),
    ("bbox_x_center", "BBox x center"),
    ("bbox_y_center", "BBox y center"),
    ("bbox_width", "BBox width"),
    ("bbox_height", "BBox height"),
    ("note", "Note"),
]
SEQUENCE_EXPORT_DEFAULT_COLUMNS = [
    "unique_name",
    "code",
    "juv_code",
    "locality name",
    "locality coordinates",
    "latitude",
    "longitude",
    "original_path",
    "export_path",
    "datetime",
    "uploaded_archive",
    "locality_check_at",
    "mediafile_id",
    "sequence_id",
    "observation_id",
    "predicted_category",
]

PATH_REGEX_CHATGPT_PROMPT_PREFIX_LINES = [
    "Help me write a Python regular expression for parsing wildlife dataset file paths.",
    "Use named groups only from: taxon, locality, unique_name, code, juv_code, check_date, date.",
    "Use unique_name for the individual identity name. The legacy group name identity is accepted, but do not use it unless the user asks for it.",
    "If check_date or date is present, prefer YYYY-MM-DD with (?P<check_date>\\d{4}-\\d{2}-\\d{2}).",
    "Return the regex pattern only, without Python prefixes or quotes such as r\"...\".",
    "The regex should match these sample paths:",
]
PATH_REGEX_CHATGPT_PROMPT_SUFFIX = (
    "The following is a description of the individual path parts and what I want to extract from the path:"
)


@user_passes_test(lambda u: u.is_superuser)
def impersonate_user(request):
    """Impersonate user."""
    if request.method == "POST":
        form = UserSelectForm(request.POST)
        if form.is_valid():
            user = form.cleaned_data["user"]
            request.session["original_user_id"] = request.user.id
            request.session["impersonate_user_id"] = user.id
            return redirect("caidapp:home")
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

        if "original_user_id" in request.session:
            # remove original_user_id from session
            request.session.pop("original_user_id")
            # del request.session["original_user_id"]
    return redirect("caidapp:home")


def is_impersonating(request):
    """Check if user is impersonating."""
    return "impersonate_user_id" in request.session


def staff_or_impersonated_staff_required(view_func):
    """Decorator to check if the user is staff or impersonating a staff member."""

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


def home_view(request):
    """Render the home view."""
    context = {}
    if request.user.is_authenticated:
        context = render_home_dashboard_context(request.user.caiduser)
        context["home_next_step"] = _home_next_step(request.user.caiduser)
    return render(
        request,
        "caidapp/home.html",
        context,
    )


def login(request):
    """Login page."""
    if request.user.is_authenticated:
        return redirect("caidapp:home")
    else:
        # redirect to allauth login page
        return redirect("/accounts/login")


def message_view(
    request,
    message,
    headline=None,
    link=None,
    button_label="Ok",
    link_secondary=None,
    button_label_secondary=None,
):
    """Show message."""
    return render(
        request,
        "caidapp/message.html",
        {
            "message": message,
            "headline": headline,
            "link": link,
            "button_label": button_label,
            "link_secondary": link_secondary,
            "button_label_secondary": button_label_secondary,
        },
    )


@login_required
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
        page_number = int(request.GET.get("page", 1))
    page_number = int(page_number)
    # is page number in paginator range?
    if page_number > paginator.num_pages:
        messages.warning(request, "Page not found")
        page_number = 1

    # logger.debug(f"{page_number=}")
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
    # logger.debug(f"Found {len(all_taxons)} taxa in total.")
    taxons = []
    taxons_mediafiles = []
    if request.user.caiduser.workgroup:
        filter_params = dict(parent__owner__workgroup=request.user.caiduser.workgroup)
    else:
        filter_params = dict(parent__owner=request.user.caiduser)

    for taxon in all_taxons:
        mediafiles_of_taxon = MediaFile.objects.filter(observations__taxon=taxon, **filter_params).order_by(
            "-captured_at"
        )
        # logger.debug(f"Taxon '{taxon.name}' has {len(mediafiles_of_taxon)} media files.")
        if len(mediafiles_of_taxon) > 0:
            # taxon.image = mediafiles_of_taxon.first().image
            taxons.append(taxon)
            taxons_mediafiles.append(mediafiles_of_taxon)
            # logger.debug(f"{mediafiles_of_taxon=}")

    return render(
        request,
        "caidapp/show_taxons.html",
        {
            "taxons": taxons,
            "taxons_with_mediafiles": zip(taxons, taxons_mediafiles),
        },
    )


@login_required
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


@method_decorator(login_required, name="dispatch")
class WellcomeView(View):
    """Wellcome view for the CAID application."""

    template_name = "caidapp/wellcome.html"
    # template_name = "caidapp/update_form.html"

    def get(self, request):
        """Render the user settings page."""
        instance = request.user.caiduser
        instance.show_wellcome_message_on_next_login = False
        form = forms.WellcomeForm(instance=instance)
        return render(
            request,
            self.template_name,
            {
                "form": form,
                "headline": "User settings",
                "button": "Save",
            },
        )

    def post(self, request):
        """Handle the form submission for user settings."""
        form = forms.WellcomeForm(request.POST, instance=request.user.caiduser)
        if form.is_valid():
            form.save()
            messages.success(request, "Settings updated successfully.")

            return redirect("caidapp:home")
        else:
            messages.error(request, "Please correct the errors below.")
        return render(request, self.template_name, {"form": form})


@method_decorator(login_required, name="dispatch")
class CaIDUserSettingsView(View):
    template_name = "caidapp/update_form.html"

    def get(self, request):
        """Render the user settings page."""
        form = forms.CaIDUserSettingsForm(instance=request.user.caiduser)
        context = {
            "form": form,
            "headline": "User settings",
            "button": "Save",
        }
        context["nav_dict"] = {
            "Invitations": reverse("caidapp:workgroup_invitations_for_user"),
        }
        return render(request, self.template_name, context)

    def post(self, request):
        """Handle the form submission for user settings."""
        form = forms.CaIDUserSettingsForm(request.POST, instance=request.user.caiduser)
        if form.is_valid():
            form.save()
            messages.success(request, "Settings updated successfully.")
            url = request.META.get("HTTP_REFERER", "/")
            return redirect(url)
        else:
            messages.error(request, "Please correct the errors below.")
        context = {
            "form": form,
        }
        context["nav_dict"] = {
            "Invitations": reverse("caidapp:workgroup_invitations_for_user"),
        }
        return render(request, self.template_name, context)


def get_filtered_mediafiles(
    user,
    contains_single_taxon: Optional[bool] = None,
    taxon_for_identification__isnull: Optional[bool] = None,
    contains_identities: Optional[bool] = None,
    is_for_identification: Optional[bool] = None,
    **extra_filters,
):
    """Retrieve media files filtered by specific parameters."""
    filter_params = {}
    if contains_single_taxon is not None:
        filter_params["contains_single_taxon"] = contains_single_taxon
    if taxon_for_identification__isnull is not None:
        filter_params["taxon_for_identification__isnull"] = taxon_for_identification__isnull
    if contains_identities is not None:
        filter_params["contains_identities"] = contains_identities
    if is_for_identification is not None:
        filter_params["is_for_identification"] = is_for_identification

    filter_params.update(extra_filters)

    return UploadedArchive.objects.annotate(**_uploads_general_order_annotation()).filter(
        **user_has_access_filter_params(user.caiduser, "owner"), **filter_params
    )


@login_required
def uploads_species(request) -> HttpResponse:
    """View for mediafiles with contains_single_taxon=False and taxon_for_identification__isnull=True."""
    queryset = get_filtered_mediafiles(
        request.user,
        contains_single_taxon=False,
        taxon_for_identification__isnull=True,
    )
    page_context = paginate_queryset(queryset, request)

    dates = views_uploads._get_check_dates(request, contains_single_taxon=False, taxon_for_identification__isnull=None)
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


@login_required
def uploads_known_identities(request) -> HttpResponse:
    """View for mediafiles with contains_identities=True."""
    queryset = get_filtered_mediafiles(
        request.user,
        contains_identities=True,
        is_for_identification=True,
    )
    page_context = paginate_queryset(queryset, request)

    return render(
        request,
        "caidapp/uploads_known_identities.html",
        {
            **page_context,
            "btn_styles": _single_species_button_style(request),
        },
    )


@login_required
def uploads_identities(request) -> HttpResponse:
    """View for mediafiles not in other categories."""
    filter_kwargs = {
        "is_for_identification": True,
    }
    if not request.user.caiduser.show_base_between_regular_uploads:
        filter_kwargs["contains_identities"] = False
    queryset = get_filtered_mediafiles(request.user, **filter_kwargs)
    page_context = paginate_queryset(queryset, request)

    return render(
        request,
        "caidapp/uploads_identities.html",
        {
            **page_context,
            "btn_styles": _single_species_button_style(request),
        },
    )


@login_required
def dash_identities(request) -> HttpResponse:
    """View for mediafiles not in other categories."""
    # queryset = get_filtered_mediafiles(
    #     request.user,
    #     # contains_single_taxon=True,
    #     contains_identities=False,
    #     taxon_for_identification__isnull=False,
    # )
    # page_context = paginate_queryset(queryset, request)
    workgroup = request.user.caiduser.workgroup
    if workgroup.identification_model is None:
        messages.error(request, "No identification model for workgroup. Please set it before running identification.")

    identity_queue_count = (
        MediafilesForIdentification.objects.filter(mediafile__parent__owner__workgroup=workgroup)
        .values("mediafile_id")
        .distinct()
        .count()
    )

    finished_archives = list(
        UploadedArchive.objects.filter(owner__workgroup=workgroup, import_finished=True, contains_identities=False)
    )
    suggestion_candidate_mediafile_count = 0
    suggestion_candidate_archive_count = 0
    suggestion_observation_taxon = None
    if workgroup.check_taxon_before_identification and workgroup.default_taxon_for_identification:
        suggestion_observation_taxon = workgroup.default_taxon_for_identification

    for uploaded_archive in finished_archives:
        observation_taxon = uploaded_archive.taxon_for_identification or suggestion_observation_taxon
        require_observations = observation_taxon is not None
        candidate_count = workgroup.mediafiles_for_identification(
            uploaded_archive_ids=[uploaded_archive.id],
            require_import_finished=True,
            require_identity=False,
            observation_taxon=observation_taxon,
            require_observations=require_observations,
        ).count()
        if candidate_count > 0:
            suggestion_candidate_archive_count += 1
            suggestion_candidate_mediafile_count += candidate_count

    if suggestion_candidate_mediafile_count > 0:
        suggestion_run_info = (
            f"Suggestions can be generated for {suggestion_candidate_mediafile_count} media files "
            f"in {suggestion_candidate_archive_count} finished uploaded archives."
        )
    else:
        suggestion_run_info = (
            "No eligible media files found for suggestions. "
            "Check whether the upload import is finished and whether the archive has the expected taxon."
        )

    # find the identity with minimum number of representative mediafiles
    identities = (
        IndividualIdentity.objects.filter(owner_workgroup=request.user.caiduser.workgroup, name__ne="nan")
        .annotate(
            representative_mediafile_count=Count("mediafile", filter=Q(mediafile__identity_is_representative=True)),
            non_representative_mediafile_count=Count(
                "mediafile", filter=Q(mediafile__identity_is_representative=False)
            ),
        )
        .filter(non_representative_mediafile_count__gt=0)
        .order_by("representative_mediafile_count", "-non_representative_mediafile_count")
    )
    next_step_candidates = build_next_steps(workgroup)

    return render(
        request,
        "caidapp/dash_identities.html",
        dict(
            # **page_context,
            btn_styles=_single_species_button_style(request),
            identities_by_representative_mediafiles=identities,
            next_step_candidates=next_step_candidates,
            primary_next_step=next_step_candidates[0] if next_step_candidates else None,
            identity_queue_count=identity_queue_count,
            suggestion_candidate_mediafile_count=suggestion_candidate_mediafile_count,
            suggestion_candidate_archive_count=suggestion_candidate_archive_count,
            suggestion_run_info=suggestion_run_info,
        ),
    )


@login_required
@require_POST
def clear_identity_suggestions_view(request):
    """Delete the entire identity suggestion queue for the current workgroup."""
    caiduser = request.user.caiduser
    workgroup = caiduser.workgroup
    if workgroup is None:
        return HttpResponseNotAllowed("No workgroup assigned.")
    if not caiduser.workgroup_admin:
        return HttpResponseNotAllowed("Only workgroup admins can clear identity suggestions.")

    queue_qs = MediafilesForIdentification.objects.filter(mediafile__parent__owner__workgroup=workgroup)
    deleted_queue_count = queue_qs.count()
    deleted_suggestion_count = models.MediafileIdentificationSuggestion.objects.filter(
        for_identification__in=queue_qs
    ).count()
    queue_qs.delete()

    messages.success(
        request,
        f"Deleted identification queue for {deleted_queue_count} media files and "
        + f"{deleted_suggestion_count} suggestions in workgroup {workgroup.name}.",
    )
    logger.info(
        "Deleted identification queue for workgroup %s by user %s: mediafiles=%s suggestions=%s",
        workgroup.id,
        request.user.username,
        deleted_queue_count,
        deleted_suggestion_count,
    )
    return redirect("caidapp:dash_identities")


def paginate_queryset(queryset, request):
    """Paginate a queryset and return page context."""
    order_by = uploaded_archive_get_order_by(request)
    queryset = queryset.order_by(order_by)

    records_per_page = get_item_number_uploaded_archives(request)
    paginator = Paginator(queryset, per_page=records_per_page)
    _, _, page_context = _prepare_page(paginator, request=request)

    for uploadedarchive in page_context["page_obj"]:
        uploadedarchive.update_status()

    return page_context


def _uploads_general_order_annotation():
    # for UploadedArchive.objects.annotate()
    return dict(
        mediafile_count=Count("mediafile"),  # Count of all related MediaFiles
        mediafile_count_with_taxon=Count(
            "mediafile", filter=Q(mediafile__taxon=F("taxon_for_identification"))
        ),  # Count of MediaFiles with a specific taxon
        earliest_mediafile_captured_at=Min("mediafile__captured_at"),  # Earliest capture date
    )


def _multiple_species_button_style_and_tooltips(request) -> dict:
    taxon_state = _taxon_workflow_state(request.user.caiduser)

    btn_tooltips = {
        "annotate_missing_taxa": f"Annotate {taxon_state['missing_taxa_count']} media files with missing taxon.",
        "verify_taxa": f"Go to verification of {taxon_state['missing_verification_count']} media files.",
        "send_to_identification": f"Send {taxon_state['ready_for_identification_count']} uploads to identification.",
    }
    btn_styles = {
        "upload_species": "secondary",
        "annotate_missing_taxa": "secondary",
        "verify_taxa": "secondary",
        "send_to_identification": "secondary",
    }
    btn_styles[taxon_state["next_action"]] = "primary"

    return btn_styles, btn_tooltips


def _taxon_workflow_state(caiduser) -> dict:
    access_filter = models.user_has_access_filter_params(caiduser, "owner")
    missing_taxa_count = len(models.get_mediafiles_with_missing_taxon(caiduser))
    missing_verification_count = len(models.get_mediafiles_with_missing_verification(caiduser))
    ready_for_identification_count = models.UploadedArchive.objects.filter(
        **access_filter,
        contains_single_taxon=False,
        is_for_identification=False,
        taxon_status__in=["TV", "TKN"],
    ).count()

    if missing_taxa_count > 0:
        next_action = "annotate_missing_taxa"
    elif missing_verification_count > 0:
        next_action = "verify_taxa"
    elif ready_for_identification_count > 0:
        next_action = "send_to_identification"
    else:
        next_action = "upload_species"

    return {
        "missing_taxa_count": missing_taxa_count,
        "missing_verification_count": missing_verification_count,
        "ready_for_identification_count": ready_for_identification_count,
        "next_action": next_action,
    }


def _home_next_step(caiduser) -> dict:
    taxon_state = _taxon_workflow_state(caiduser)
    taxon_actions = {
        "annotate_missing_taxa": {
            "label": "Annotate missing taxa",
            "url": reverse("caidapp:missing_taxon_annotation"),
            "section": "taxon",
        },
        "verify_taxa": {
            "label": "Verify taxa",
            "url": reverse("caidapp:sequences") + "?show_overview_button=true&taxon_verified=false",
            "section": "taxon",
        },
        "send_to_identification": {
            "label": "Send to identification",
            "url": reverse("caidapp:uploads_ready_for_identification"),
            "section": "taxon",
        },
    }
    if taxon_state["next_action"] in taxon_actions:
        return taxon_actions[taxon_state["next_action"]]

    workgroup = caiduser.workgroup
    if workgroup:
        confirmation_count = MediafilesForIdentification.objects.filter(
            mediafile__parent__owner__workgroup=workgroup
        ).count()
        if confirmation_count > 0:
            return {
                "label": "Confirm identification",
                "url": reverse("caidapp:get_individual_identity"),
                "section": "identification",
            }

        pending_identification_count = tasks.get_uploaded_archives_pending_identification(workgroup).count()
        identification_is_initiated = workgroup.identification_init_at is not None
        representative_count = MediaFile.objects.filter(
            parent__owner__workgroup=workgroup,
            identity_is_representative=True,
            parent__taxon_for_identification__isnull=False,
        ).count()

        if pending_identification_count > 0 and identification_is_initiated:
            return {
                "label": "Run identification",
                "url": reverse("caidapp:pre_identify"),
                "section": "identification",
            }
        if pending_identification_count > 0 and representative_count > 0:
            return {
                "label": "Init identification",
                "url": reverse("caidapp:dash_identities"),
                "section": "identification",
            }
        if pending_identification_count > 0:
            return {
                "label": "Set up identification",
                "url": reverse("caidapp:dash_identities"),
                "section": "identification",
            }

    return {
        "label": "Upload media",
        "url": reverse("caidapp:new_upload"),
        "section": "upload",
    }


def sample_data(request):
    """Sample data."""
    # TODO do we need page with sample data from our database? Maybe a link to public dataset would be enough?
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


# TODO remove?
# @login_required
# def individual_identities(request):
#     """List of individual identities."""
#     individual_identities = (
#         IndividualIdentity.objects.filter(
#             Q(owner_workgroup=request.user.caiduser.workgroup) & ~Q(name="nan")
#         )
#         .all()
#         .order_by("-name")
#     )
#
#     records_per_page = 24
#     paginator = Paginator(individual_identities, per_page=records_per_page)
#     _, _, page_context = _prepare_page(paginator, request=request)
#
#     return render(
#         request,
#         "caidapp/individual_identities.html",
#         {**page_context, "workgroup": request.user.caiduser.workgroup},
#     )


class IdentityListView(LoginRequiredMixin, ListView):
    model = IndividualIdentity
    template_name = "caidapp/individual_identities.html"
    context_object_name = "individual_identities"
    paginate_by = 24
    title = "Identities"
    # order by
    ordering = ["-name"]

    def _get_selected_identities(self):
        """Return selected identities limited to the current user's workgroup."""
        selected_ids = []
        for raw_id in self.request.POST.getlist("selected_identity_ids"):
            try:
                selected_ids.append(int(raw_id))
            except (TypeError, ValueError):
                continue
        return (
            IndividualIdentity.objects.filter(
                pk__in=selected_ids,
                owner_workgroup=self.request.user.caiduser.workgroup,
            )
            .annotate(
                mediafile_count=Count("mediafile"),
                representative_mediafile_count=Count("mediafile", filter=Q(mediafile__identity_is_representative=True)),
                locality_count=Count("mediafile__locality", distinct=True),
                last_seen=Max("mediafile__captured_at"),
            )
            .order_by("name", "id")
        )

    def post(self, request, *args, **kwargs):
        """Handle bulk identity actions from list and card views."""
        action = request.POST.get("bulk_action")
        selected_identities = self._get_selected_identities()
        selected_ids = list(selected_identities.values_list("id", flat=True))
        if not selected_ids:
            messages.warning(request, "Select at least one identity.")
            return redirect(request.get_full_path())

        if action == "open_sequences":
            query_string = urllib.parse.urlencode({"individual_identity_ids": selected_ids}, doseq=True)
            return redirect(f"{reverse('caidapp:sequences')}?{query_string}")

        if action == "confirm_delete":
            return render(
                request,
                "caidapp/individual_identities_bulk_delete_confirm.html",
                {
                    "identities": selected_identities,
                    "selected_identity_ids": selected_ids,
                    "return_url": request.get_full_path(),
                },
            )

        if action == "delete_selected" and request.POST.get("confirm_delete") == "yes":
            deleted_count = selected_identities.count()
            selected_identities.delete()
            messages.success(request, f"Deleted {deleted_count} identities.")
            return redirect(request.POST.get("return_url") or reverse("caidapp:individual_identities"))

        messages.warning(request, "Choose a bulk action.")
        return redirect(request.get_full_path())

    def get_queryset(self):
        """Get queryset for the view."""
        class_prefix = "identities_" + self.request.GET.get("view", "cards")

        self.paginate_by = views_general.get_item_number_anything(self.request, class_prefix)
        qs = IndividualIdentity.objects.filter(Q(owner_workgroup=self.request.user.caiduser.workgroup) & ~Q(name="nan"))
        qs = qs.annotate(
            mediafile_count=Count("mediafile"),
            representative_mediafile_count=Count("mediafile", filter=Q(mediafile__identity_is_representative=True)),
            locality_count=Count("mediafile__locality", distinct=True),
            last_seen=Max("mediafile__captured_at"),
        )

        self.filterset = filters.IndividualIdentityFilter(self.request.GET, queryset=qs)
        qs = self.filterset.qs

        # class_prefix = self.__class__.__name__.lower() # maybe this is more general
        # class_prefix = 'identities'
        sort, direction = views_general.get_order_by_anything(self.request, class_prefix, IndividualIdentity)
        list_of_fields = [f.name for f in self.model._meta.fields] + [
            "mediafile_count",
            "representative_mediafile_count",
            "locality_count",
            "last_seen",
        ]

        if sort in list_of_fields:
            if direction == "desc":
                sort = f"-{sort}"
            logger.debug(f"Sorting by {sort}")
            qs = qs.order_by(sort)
        return qs

    def get_template_names(self):
        """Get template names based on view type."""
        view_type = self.request.GET.get("view", "cards")
        if view_type == "cards":
            return ["caidapp/individual_identities.html"]
        else:
            return ["caidapp/individual_identities_list.html"]

    def get_context_data(self, **kwargs):
        """Get context data for the template."""
        context = super().get_context_data(**kwargs)
        # context["filter_form"] = self.filterset.form
        context["filter"] = self.filterset
        context["list_display"] = []
        context = add_querystring_to_context(self.request, context)
        # query_params = self.request.GET.copy()
        # query_params.pop('page', None)
        # context['query_string'] = query_params.urlencode()
        return context


@login_required
def individual_identity_create(request, media_file_id: Optional[int] = None):
    """Create new individual_identity."""
    if request.method == "POST":
        form = IndividualIdentityForm(request.POST)
        if form.is_valid():
            individual_identity = form.save(commit=False)
            individual_identity.owner_workgroup = request.user.caiduser.workgroup
            individual_identity.updated_by = request.user.caiduser
            individual_identity.save()
            # go back to prev page
            if media_file_id:
                media_file = get_object_or_404(
                    MediaFile,
                    pk=media_file_id,
                    parent__owner__workgroup=request.user.caiduser.workgroup,
                )
                media_file.identity = individual_identity
                media_file.save()
                messages.success(request, "Individual identity created and linked to media file.")
            url = request.META.get("HTTP_REFERER", reverse("caidapp:individual_identities"))
            next_url = request.GET.get("next") or request.POST.get("next") or url
            return redirect(next_url)
    else:
        form = IndividualIdentityForm()
    return render(
        request,
        "caidapp/update_form.html",
        {"form": form, "headline": "New Individual Identity", "button": "Create"},
    )


class IndividualIdentityUpdateView(LoginRequiredMixin, UpdateView):
    """Update individual identity."""

    model = IndividualIdentity
    form_class = IndividualIdentityForm
    template_name = "caidapp/update_form.html"
    context_object_name = "individual_identity"

    def get_queryset(self):
        """Get queryset for the view."""
        return IndividualIdentity.objects.filter(
            owner_workgroup=self.request.user.caiduser.workgroup,
        )

    def get_context_data(self, **kwargs):
        """Get context data for the template."""
        context = super().get_context_data(**kwargs)
        individual_identity = self.get_object()
        media_files = MediaFile.objects.filter(identity=individual_identity, identity_is_representative=True)
        # media_file = media_files.first()

        nav_dict = {}
        if individual_identity:
            nav_dict["Media Files"] = reverse_lazy(
                "caidapp:individual_identity_mediafiles",
                kwargs={"individual_identity_id": individual_identity.id},
            )
            nav_dict["Sequences"] = f"{reverse('caidapp:sequences')}?individual_identity_id={individual_identity.id}"
        right_nav = {"Localities": None}
        for locality in individual_identity.localities():
            right_nav[locality.name] = reverse_lazy(
                "caidapp:update_locality",
                kwargs={"locality_id": locality.id},
            )

        context.update(
            {
                "headline": "Individual Identity",
                "button": "Save",
                # "mediafile": media_file,
                "mediafiles": media_files[:4],
                "mediafiles_url": reverse_lazy(
                    "caidapp:individual_identity_mediafiles",
                    kwargs={"individual_identity_id": individual_identity.id},
                ),
                "delete_button_url": reverse_lazy(
                    "caidapp:delete_individual_identity",
                    kwargs={"individual_identity_id": individual_identity.id},
                ),
                "nav_dict": nav_dict,
                "right_nav": right_nav,
            }
        )
        return context

    def get_success_url(self):
        """Return to individual identities list."""
        return reverse_lazy("caidapp:individual_identities")

    # validation
    def form_valid(self, form):
        """Set updated_by before saving."""
        individual_identity = form.save(commit=False)
        individual_identity.updated_by = self.request.user.caiduser
        individual_identity.save()
        return super().form_valid(form)


@login_required
def delete_individual_identity(request, individual_identity_id):
    """Delete individual identity if it belongs to the user."""
    individual_identity = get_object_or_404(
        IndividualIdentity,
        pk=individual_identity_id,
        owner_workgroup=request.user.caiduser.workgroup,
    )
    individual_identity.delete()
    return redirect("caidapp:individual_identities")


# from cruds_adminlte.crud import CRUDView
#
# class IndividualIdentityCRUDView(CRUDView):
#     model = IndividualIdentity
# form = IndividualIdentityForm
# template_name = "caidapp/update_form.html"
# list_template_name = "caidapp/individual_identities.html"
# list_context = {"workgroup": request.user.caiduser.workgroup}
# list_paginate_by = 24
# list_order_by = "-name"
# list_queryset = lambda self, request: IndividualIdentity.objects.filter(
#     Q(owner_workgroup=request.user.caiduser.workgroup) & ~Q(name="nan")
# ).annotate(
#     mediafile_count=Count("mediafile"),
#     representative_mediafile_count=Count("mediafile", filter=Q(mediafile__identity_is_representative=True)),
#     locality_count=Count("mediafile__locality", distinct=True),
# )
# list_order_by = "-name"


@login_required
def get_individual_identity_zoomed_by_identity(request, foridentification_id: int, identity_id: int):
    """Show detail by identity."""
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
            # reid_sugestion_index: None,
            "top_mediafile": top_mediafile,
            # "top_score": top_score,
            "top_name": top_name,
            # "html_img_src": html_img_src,
            "btn_link": btn_link,
            "btn_icon_style": btn_icon_style,
        },
    )


@login_required
def get_individual_identity_zoomed_paired_points(request, foridentification_id: int, reid_suggestion_id: int):
    """Show detail with paired points."""
    return get_individual_identity_zoomed(request, foridentification_id, reid_suggestion_id, points=True)


@login_required
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

    if points and paired_points:
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
        if foridentification.mediafile.media_type == "video":
            pth0 = pth0.with_suffix(".jpg")
        pth1 = Path(settings.MEDIA_ROOT) / str(top_mediafile.mediafile.name).replace("/images/", "/masked_images/")
        if top_mediafile.media_type == "video":
            pth1 = pth1.with_suffix(".jpg")
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
            kwargs={
                "foridentification_id": foridentification_id,
                "reid_suggestion_id": reid_suggestion_id,
            },
        )
        btn_icon_style = "fa fa-eye"
    else:
        template = "caidapp/get_individual_identity_zoomed.html"
        html_img_src = None
        btn_link = reverse_lazy(
            "caidapp:get_individual_identity_zoomed_paired_points",
            kwargs={
                "foridentification_id": foridentification_id,
                "reid_suggestion_id": reid_suggestion_id,
            },
        )
        btn_icon_style = "fa-solid fa-arrows-to-dot"
    # get order number of reid_suggestion
    reid_suggestions = list(foridentification.top_mediafiles.all())
    # which is the actual reid_suggestion
    try:
        reid_suggestion_index = reid_suggestions.index(reid_suggestion)
    except ValueError:
        reid_suggestion_index = None

    return render(
        request,
        template,
        {
            "foridentification": foridentification,
            "foridentifications": foridentifications,
            "reid_suggestion_id": reid_suggestion_id,
            "reid_suggestion_index": reid_suggestion_index,
            "top_mediafile": top_mediafile,
            "top_score": top_score,
            "top_name": top_name,
            "html_img_src": html_img_src,
            "btn_link": btn_link,
            "btn_icon_style": btn_icon_style,
        },
    )


@login_required
def not_identified_mediafiles(request):
    """View for mediafiles with individualities that are not identified."""
    foridentification_set = (
        MediafilesForIdentification.objects.filter(mediafile__parent__owner__workgroup=request.user.caiduser.workgroup)
        .annotate(max_score=Max("top_mediafiles__score"))
        .order_by("-max_score")
    )

    # sort by highest score
    # for foridentification in foridentification_set:
    #     reid_suggestion = models.MediafileIdentificationSuggestion.objects.get(id=reid_suggestion_id)
    #     suggestions = MediafilesIdentificationSuggestion
    # foridentification_set.annotate(
    #
    # )

    records_per_page = 80
    paginator = Paginator(foridentification_set, per_page=records_per_page)
    _, _, page_context = _prepare_page(paginator, request=request)

    return render(
        request,
        "caidapp/not_identified_mediafiles.html",
        {**page_context, "page_title": "Not Identified"},
    )


# delete
# def get_best_representative_mediafiles(identity, orientation=None, max_count=5) -> List[MediaFile]:
#     qs = identity.mediafile_set
#     mf = qs.filter(identity_is_representative=True, orientation=orientation)
#
#     if not mf.exists():
#         mf = qs.filter(identity_is_representative=True)
#
#     if not mf.exists():
#         mf = qs.all()
#
#     return list(mf.order_by("-captured_at")[:max_count])


def get_best_representative_mediafiles(identity, orientation=None, max_count=5) -> list[MediaFile]:
    """Get best representative mediafiles for identity."""
    # Pokud máme předem načtené reprezentativní mediafiles
    candidates = getattr(identity, "representative_mediafiles_candidates", None)
    if candidates is not None:
        if candidates:
            return candidates[:max_count]
        # fallback na reprezentativní bez orientace
        fallback = list(
            identity.mediafile_set.filter(identity_is_representative=True).order_by("-captured_at")[:max_count]
        )
        if fallback:
            return fallback
        return list(identity.mediafile_set.all().order_by("-captured_at")[:max_count])
    else:
        # fallback: přímé dotazy jako dřív
        qs = identity.mediafile_set
        mf = qs.filter(identity_is_representative=True, orientation=orientation)
        if not mf.exists():
            mf = qs.filter(identity_is_representative=True)
        if not mf.exists():
            mf = qs.all()
        return list(mf.order_by("-captured_at")[:max_count])


@login_required
def get_individual_identity_from_foridentification(
    request,
    foridentification_id: Optional[int] = None,
    media_file_id: Optional[int] = None,
    max_representative_mediafiles: int = 5,
):
    """Show and update media file."""
    t0 = time.time()
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
        identity_ids = foridentification.top_mediafiles.values_list("mediafile__identity", flat=True)
        logger.debug(f"{identity_ids=}")

        identity_ids = [i for i in identity_ids if i is not None]
        logger.debug(f"{identity_ids=}")

        orientation_of_unknown = foridentification.mediafile.orientation

        # remaining_identities = (
        #     IndividualIdentity.objects.filter(
        #         Q(owner_workgroup=request.user.caiduser.workgroup) & ~Q(name="nan") &
        #         ~Q(id__in=identity_ids)
        #     )
        #     .all()
        #     .order_by("name")
        # )

        # -------------------------------- vvvvv ------
        from django.db.models import Prefetch

        # připrav filtr: reprezentativní a orientované (pokud zadané)
        mediafile_filter = {"identity_is_representative": True}
        if orientation_of_unknown is not None:
            mediafile_filter["orientation"] = orientation_of_unknown

        # prefetch_candidates = Prefetch(
        #     "mediafile_set",
        #     queryset=MediaFile.objects.filter(**mediafile_filter).order_by("-captured_at"),
        #     to_attr="representative_mediafiles_candidates"
        # )
        # remaining_identities = (
        #     IndividualIdentity.objects.filter(
        #         Q(owner_workgroup=request.user.caiduser.workgroup),
        #         ~Q(name="nan"),
        #         ~Q(id__in=identity_ids)
        #     )
        #     .prefetch_related(prefetch_candidates)
        #     .order_by("name")
        # )
        # ----------------------
        prefetch_first_mediafile = Prefetch(
            "mediafile_set",
            queryset=MediaFile.objects.order_by("captured_at"),
            to_attr="all_mediafiles_ordered",
        )

        remaining_identities = (
            IndividualIdentity.objects.filter(
                Q(owner_workgroup=request.user.caiduser.workgroup),
                ~Q(name="nan"),
                ~Q(id__in=identity_ids),
            )
            .prefetch_related(prefetch_first_mediafile)
            .order_by("name")
        )
        # -------------------------------- ^^^^^ ------

        logger.debug(f"  1 {time.time() - t0=:.2f} [s]")

        # Add `representative_mediafiles` to related identities
        # for identity in related_identities:
        #     identity.representative_mediafiles = identity.mediafile_set.filter(identity_is_representative=True)

        reid_suggestions = list(foridentification.top_mediafiles.all().select_related("identity", "mediafile"))
        for reid_suggestion in reid_suggestions:
            if reid_suggestion.identity is None:
                # i.e. The identity was removed from the app
                # remove from foridentification.top_mediafiles
                try:
                    reid_suggestion.delete()
                    logger.warning(
                        "Missing identity for reid_suggestion. Removed one suggestion for "
                        + f"{foridentification.mediafile.mediafile.name=}"
                    )
                except Exception as e:
                    logger.debug(traceback.format_exc())
                    logger.error(f"Error removing reid_suggestion from foridentification.top_mediafiles: {e}")
            else:
                representative_mediafiles: list = get_best_representative_mediafiles(
                    reid_suggestion.identity, orientation=orientation_of_unknown
                )
                # insert as first element the reid_suggestion mediafile
                representative_mediafiles = [reid_suggestion.mediafile] + [
                    mf for mf in representative_mediafiles if mf != reid_suggestion.mediafile
                ]

                reid_suggestion.representative_mediafiles = representative_mediafiles[:max_representative_mediafiles]
                reid_suggestion.is_representative_dict = is_candidate_for_representative_mediafile(
                    reid_suggestion.mediafile, reid_suggestion.identity
                )

        logger.debug(f"  2 {time.time() - t0=:.2f} [s]")

        for identity in remaining_identities:
            mf = identity.all_mediafiles_ordered
            identity.representative_mediafiles = mf[:3] if mf else []

        logger.debug(f"  3 {time.time() - t0=:.2f} [s]")
        # for identity in identities:
        #     identity.representative_mediafiles = identity.mediafile_set.filter(identity_is_representative=True)

        logger.debug(f"{len(remaining_identities)=}")
        logger.debug(f"   {remaining_identities[:10]=}")

        # max_score for current foridentification
        current_max_score = foridentification.top_mediafiles.aggregate(max_score=Max("score"))["max_score"] or 0.0

        logger.debug(f"  4 {time.time() - t0=:.2f} [s]")
        # find the next foridentification with lower max_score
        from django.db.models.functions import Coalesce

        next_foridentification = (
            foridentifications.exclude(pk=foridentification.pk)
            .annotate(max_score=Coalesce(Max("top_mediafiles__score"), 0.0))
            .filter(max_score__lte=current_max_score)
            .order_by("-max_score")
            .first()
        )

    else:
        return message_view(request, "No mediafiles for identification.")

    logger.debug(f"  5 {time.time() - t0=:.2f} [s]")
    logger.debug(f"{remaining_identities[:5]}")
    return render(
        request,
        "caidapp/get_individual_identity.html",
        {
            "foridentification": foridentification,
            "foridentifications": foridentifications,
            "remaining_identities": remaining_identities,
            "reid_suggestions": reid_suggestions,
            "next_foridentification": next_foridentification,
            # "related_identities": identity_ids,
        },
    )


def get_individual_identity_remaining_card_content(
    request,
    foridentification_id: int,
    identity_id: int,
) -> HttpResponse:
    """Get remaining card content for individual identity."""
    identity = get_object_or_404(
        IndividualIdentity,
        id=identity_id,
        owner_workgroup=request.user.caiduser.workgroup,
    )
    foridentification_id = get_object_or_404(
        MediafilesForIdentification,
        id=foridentification_id,
        mediafile__parent__owner__workgroup=request.user.caiduser.workgroup,
    )
    mediafile = foridentification_id.mediafile
    is_representative_dict = is_candidate_for_representative_mediafile(mediafile, identity)

    identity.representative_mediafiles = identity.mediafile_set.filter(identity_is_representative=True)

    html = render_to_string(
        "caidapp/get_individual_identity_remaining_card_content.html",
        context=dict(
            individual_identity=identity,
            foridentification=foridentification_id,
            is_representative_dict=is_representative_dict,
        ),
    )
    return HttpResponse(html)


def is_candidate_for_representative_mediafile(
    mediafile: models.MediaFile,
    identity: models.IndividualIdentity,
    representative_count_coefficient: float = 5.0,
) -> dict:
    """Check if mediafile is candidate for representative mediafile.

    Args:
        # request: HttpRequest object.
        # mediafile_identification_suggestion_id: ID of the MediafileIdentificationSuggestion.
        representative_count_coefficient: The threshold is about 0.60 if this is thenumber of representative mediafiles.
    """
    # suggestion = get_object_or_404(
    #     models.MediafileIdentificationSuggestion,
    #     id=mediafile_identification_suggestion_id,
    #     mediafile__parent__owner__workgroup=request.user.caiduser.workgroup,
    # )

    orientation_score = 0.0
    animal_score = 0.0
    count_of_representative_mediafiles = identity.count_of_representative_mediafiles()

    # dr = "nic"
    debug_info = ""

    threshold = 1.0 - np.exp(-count_of_representative_mediafiles / representative_count_coefficient)  # i
    meta = mediafile.metadata_json
    logger.debug(f"{mediafile.metadata_json=}")
    if "detection_results" in meta and len(meta["detection_results"]) > 0:

        debug_info += " detection_results found"
        # dr = meta["detection_results"]
        detection_results = meta["detection_results"]
        debug_info += f" detection_results (json) ={detection_results}"
        if isinstance(detection_results, str):
            # it is probably not a json, but the python string representation of a list of dicts
            import ast

            detection_results = ast.literal_eval(detection_results)
            # import json
            # detection_results = json.loads(detecion_results_in_json)

        if len(detection_results) > 0:

            first_bbox = detection_results[0]
            logger.debug(f"{first_bbox=}")
            debug_info += f" bbox={first_bbox=}, "
            if "class" in first_bbox and "confidence" in first_bbox:
                orientation_score = float(first_bbox["confidence"])
                debug_info += f" {first_bbox['confidence']=}"
            if "orientation_score" in first_bbox:
                orientation_score = float(first_bbox["orientation_score"])
                debug_info += f" {first_bbox['orientation_score']=}"

    is_candidate = ((animal_score + orientation_score) / 2.0) > threshold
    message = (
        "This is "
        + ("not " if not is_candidate else "")
        + "a candidate for representative mediafile. "
        + "Actual count of representative media files for this individuality is: "
        + str(count_of_representative_mediafiles)
        + ". Orientation score: "
        + str(orientation_score)
        + ". Animal score: "
        + str(animal_score)
        + ". Threshold: "
        + str(threshold)
        + "."
    )

    return dict(
        is_candidate=is_candidate,
        orientation_score=orientation_score,
        animal_score=animal_score,
        threshold=threshold,
        count_of_representative_mediafiles=count_of_representative_mediafiles,
        message=message,
        debug_info=debug_info,
        # detection_results=dr,
        # meta=meta
    )

    return False


@login_required
def remove_foridentification(request, foridentification_id: int):
    """Remove mediafile from list for identification."""
    foridentification = get_object_or_404(
        MediafilesForIdentification,
        id=foridentification_id,
        mediafile__parent__owner__workgroup=request.user.caiduser.workgroup,
    )
    foridentification.delete()
    return redirect("caidapp:get_individual_identity")


@login_required
def set_individual_identity(request, mediafiles_for_identification_id: int, individual_identity_id: int):
    """Set identity for mediafile."""
    mediafiles_for_identification = get_object_or_404(MediafilesForIdentification, id=mediafiles_for_identification_id)
    representative = request.GET.get("representative") == "1"
    individual_identity = get_object_or_404(IndividualIdentity, id=individual_identity_id)

    # if request.user.caiduser.workgroup != mediafile.parent.owner.workgroup:
    #     return HttpResponseNotAllowed("Not allowed to work with this media file.")
    if request.user.caiduser.workgroup != individual_identity.owner_workgroup:
        return HttpResponseNotAllowed("Not allowed to work with this media file.")
    if request.user.caiduser.workgroup != mediafiles_for_identification.mediafile.parent.owner.workgroup:
        return HttpResponseNotAllowed("Not allowed to work with this media file.")

    mediafiles_for_identification.mediafile.identity = individual_identity
    mediafiles_for_identification.mediafile.identity_is_representative = representative
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

    run_species_prediction_async(
        uploaded_archive,
        force_init=force_init,
        extract_identites=uploaded_archive.contains_identities,
    )
    # next_page = request.GET.get("next", "/caidapp/uploads")
    # return redirect(next_page)
    return redirect(request.META.get("HTTP_REFERER", "/"))


# def init_identification(request, taxon_str:str="Lynx Lynx"):
#     return redirect("/caidapp/uploads")


# def _get_mediafiles_for_train_or_init_identification(
#     workgroup: models.WorkGroup,
#     # request,
#     # workgroup, taxon=None, identity_is_representative=True
# ):
#     """Get mediafiles for training or initialization of identification."""
#     # Nejprve resetuj všechny mediafiles v daném workgroupu
#     mediafiles_qs = MediaFile.objects.filter(
#         parent__owner__workgroup=workgroup,
#     )
#     mediafiles_qs.update(used_for_init_identification=False)
#
#     # Pokud má workgroup nastavený výchozí taxon pro identifikaci
#     if workgroup.check_taxon_before_identification and workgroup.default_taxon_for_identification:
#         # Najdi ID všech mediafiles, které mají aspoň jednu observaci s daným taxonem
#         mf_ids = (
#             AnimalObservation.objects.filter(
#                 taxon=workgroup.default_taxon_for_identification,
#                 mediafile__parent__owner__workgroup=workgroup,
#             )
#             .values_list("mediafile_id", flat=True)
#             .distinct()
#         )
#
#         # A těmto mediafiles nastav příznak
#         mediafiles_qs = MediaFile.objects.filter(
#             parent__owner__workgroup=workgroup,
#             id__in=mf_ids,
#             identity_is_representative=True,
#             identity__isnull=False,
#         )
#
#     else:
#         logger.warning(f"No default taxon for identification set in {workgroup=}. Nothing updated.")
#         mediafiles_qs = MediaFile.objects.filter(
#             parent__owner__workgroup=workgroup,
#             # id__in=mf_ids,
#             identity_is_representative=True,
#             identity__isnull=False,
#         )
#
#     logger.debug(f"Found {mediafiles_qs.count()} mediafiles for identification init.")
#     if mediafiles_qs.count() == 0:
#         logger.error("No mediafiles found for identification init.")
#
#     return mediafiles_qs


def str_bumpversion(version_str: str) -> str:
    """Add or increase version in the string.

    Keep the first part of the string. If the string ends with a number, increase it by 1.
    Otherwise, append ".1" to the string.

    """
    parts = version_str.rsplit(".", 1)
    if len(parts) == 2 and parts[1].isdigit():
        # If the last part is a number, increase it
        return f"{parts[0]}.{int(parts[1]) + 1}"
    else:
        # Otherwise, append ".1"
        return f"{version_str}.1"


@login_required
def train_identification(
    request,
    # taxon_str: str = "Lynx lynx"
):
    """Run processing of uploaded archive."""
    # check if user is workgroup admin

    if not request.user.caiduser.workgroup_admin:
        return HttpResponseNotAllowed("Identification init is for workgroup admins only.")
    if not request.user.caiduser.workgroup.identification_model:
        # go back to the page
        link = request.META.get("HTTP_REFERER", "/")
        return message_view(request, "No identification model set.", link=link)
    caiduser = request.user.caiduser
    mediafiles_qs = caiduser.workgroup.mediafiles_for_train_or_init_identification()

    logger.debug("Generating CSV for init_identification...")

    caiduser = request.user.caiduser
    # new_name = str_bumpversion("MegaDescriptor-T-224-v0") # caiduser.identification_model.name
    now_str = django.utils.timezone.now().strftime("%Y%m%d-%H%M%S")
    new_name = f"MegaDescriptor-T-224-v0.{now_str}"
    # caiduser.identification_model.name
    clean_new_name = re.sub(r"[^a-zA-Z0-9 _-]", "", new_name)

    group_dir = Path(settings.MEDIA_ROOT) / request.user.caiduser.workgroup.name
    output_dir = group_dir / "models" / clean_new_name
    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_path = output_dir / f"{clean_new_name}.pth"
    identity_metadata_file = group_dir / "train_identification.csv"

    csv_data = _prepare_dataframe_for_identification(mediafiles_qs)

    messages.info(
        request,
        f"Using {len(csv_data['image_path'])} representative images for identification initialization. ",
    )
    if len(csv_data) > 0:
        df = pd.DataFrame(csv_data)
        df.to_csv(identity_metadata_file, index=False)
        # counts = df["class_id"].value_counts()
        counts = df["label"].value_counts()
        logger.debug(f"Class counts:{str(counts)}")
        if len(counts) == 0:
            messages.error(
                request,
                f"No classes found in the data for taxon {caiduser.workgroup.default_taxon_for_identification}",
            )
            go_back = request.META.get("HTTP_REFERER", "/")
            return redirect(go_back)
        # nejpočetnější třída a kolik jich tam je
        most_common_class = counts.idxmax()
        most_common_count = counts.max()
        logger.debug(f"Most common class: {most_common_class} with {most_common_count} images.")
        # min class
        min_class = counts.idxmin()
        min_count = counts.min()
        logger.debug(f"Min class: {min_class} with {min_count} images.")

    messages.info(
        request,
        f"Most common class: {most_common_class} with {most_common_count} images. "
        + f"Min class: {min_class} with {min_count} images.",
    )

    # logger.debug(f"{identity_metadata_file=}")
    # workgroup = request.user.caiduser.workgroup
    # workgroup.identification_init_at = django.utils.timezone.now()
    # workgroup.identification_init_status = "Processing"
    # workgroup.identification_init_model_path = str(request.user.caiduser.identification_model.model_path)
    # workgroup.identification_init_message = (
    #         f"Using {len(csv_data['image_path'])}"
    #         + "representative images for identification initialization."
    # )
    # workgroup.save()

    logger.debug("Calling train_identification...")
    new_identification_model = models.IdentificationModel(
        name=new_name, model_path=output_model_path, workgroup=request.user.caiduser.workgroup
    )
    sig = signature(
        "train_identification",
        kwargs={
            # csv file should contain image_path, class_id, label
            "input_metadata_file": str(identity_metadata_file),
            "organization_id": request.user.caiduser.workgroup.id,
            "identification_model": {
                "name": new_identification_model.name,
                "init_path": "hf-hub:BVRA/MegaDescriptor-T-224",  # str(caiduser.identification_model.model_path),
                "path": str(new_identification_model.model_path),
            },
        },
    )
    # task =
    sig.apply_async(
        link=tasks.train_identification_on_success.s(
            workgroup_id=request.user.caiduser.workgroup.id,
            caiduser_id=request.user.caiduser.id,
            user_name=request.user.username,
            # uploaded_archive_id=uploaded_archive.id,
            # zip_file=os.path.relpath(str(output_archive_file), settings.MEDIA_ROOT),
            # csv_file=os.path.relpath(str(output_metadata_file), settings.MEDIA_ROOT),
        ),
        link_error=init_identification_on_error.s(
            caiduser_id=request.user.caiduser.id,
            user_name=request.user.username,
            # uploaded_archive_id=uploaded_archive.id
        ),
    )
    new_identification_model.save()
    # return redirect("caidapp:individual_identities")
    go_back = request.META.get("HTTP_REFERER", "/")
    return redirect(go_back)
    # return redirect("caidapp:uploads_known_identities")


@login_required
def init_identification_view(
    request,
    # taxon_str: str = "Lynx lynx"
):
    """Run processing of uploaded archive."""
    # check if user is workgroup admin
    caiduser = request.user.caiduser
    if not request.user.caiduser.workgroup_admin:
        return HttpResponseNotAllowed("Identification init is for workgroup admins only.")

    # reset mediafiles in workgroup
    mediafiles_qs = MediaFile.objects.filter(
        parent__owner__workgroup=caiduser.workgroup,
    )
    mediafiles_qs.update(used_for_init_identification=False)
    mediafiles_qs = caiduser.workgroup.mediafiles_for_train_or_init_identification()
    mf_count = mediafiles_qs.count()
    messages.info(
        request,
        f"Scheduling identification initialization for workgroup {caiduser.workgroup.name} "
        f"with {mf_count} media files.",
    )

    if not request.user.caiduser.workgroup.identification_model:

        caiduser.workgroup.identification_model = models.IdentificationModel.objects.filter(public=True).first()
        messages.warning(
            request, f"Setting default identification model: {caiduser.workgroup.identification_model.name}"
        )

    from .tasks import schedule_init_identification_for_workgroup

    schedule_init_identification_for_workgroup(caiduser.workgroup, delay_minutes=0)
    # return redirect("caidapp:individual_identities")
    return redirect("caidapp:dash_identities")


def stop_init_identification(request):
    """Stop identification initialization."""
    workgroup = request.user.caiduser.workgroup
    if workgroup.identification_init_status in {"Processing", "Scheduled"}:
        if workgroup.identification_scheduled_init_task_id:
            current_app.control.revoke(workgroup.identification_scheduled_init_task_id, terminate=True)
        workgroup.identification_init_status = "Not initiated"
        workgroup.identification_init_at = django.utils.timezone.now()
        workgroup.identification_init_message = "Initialization was stopped manually."
        workgroup.identification_scheduled_init_task_id = None
        workgroup.identification_scheduled_init_eta = None
        workgroup.save(
            update_fields=[
                "identification_init_status",
                "identification_init_at",
                "identification_init_message",
                "identification_scheduled_init_task_id",
                "identification_scheduled_init_eta",
            ]
        )
    elif workgroup.identification_reid_status in {"Processing", "Scheduled"}:
        if workgroup.identification_scheduled_run_task_id:
            current_app.control.revoke(workgroup.identification_scheduled_run_task_id, terminate=True)
        workgroup.identification_reid_status = "Not initiated"
        workgroup.identification_reid_at = django.utils.timezone.now()
        workgroup.identification_reid_message = "Identification run was stopped manually."
        workgroup.identification_scheduled_run_task_id = None
        workgroup.identification_scheduled_run_eta = None
        workgroup.save(
            update_fields=[
                "identification_reid_status",
                "identification_reid_at",
                "identification_reid_message",
                "identification_scheduled_run_task_id",
                "identification_scheduled_run_eta",
            ]
        )
    return redirect("caidapp:uploads_known_identities")


# TODO rename to identification button style
def _single_species_button_style(request) -> dict:
    workgroup = request.user.caiduser.workgroup

    is_initiated = request.user.caiduser.workgroup.identification_init_at is not None

    n_representative = len(
        MediaFile.objects.filter(
            parent__owner__workgroup=request.user.caiduser.workgroup,
            identity_is_representative=True,
            # parent__contains_single_taxon=True,
            parent__taxon_for_identification__isnull=False,
        )
    )
    exists_representative = n_representative > 0

    n_unidentified = len(
        UploadedArchive.objects.filter(
            # status="Species Finished",
            Q(taxon_status="TKN") | Q(taxon_status="TV"),
            owner__workgroup=request.user.caiduser.workgroup,
            contains_identities=False,
            # contains_single_taxon=True,
            taxon_for_identification__isnull=False,
        )
    )

    exists_unidentified = n_unidentified > 0

    n_for_confirmation = len(
        MediafilesForIdentification.objects.filter(mediafile__parent__owner__workgroup=request.user.caiduser.workgroup)
    )
    exists_for_confirmation = n_for_confirmation > 0

    # btn_tooltips = {}
    btn_styles = {}

    btn_styles["upload_identified"] = {
        "class": "primary" if (not is_initiated) and (not exists_representative) else "secondary"
    }
    btn_styles["init_identification"] = {
        "class": "primary" if (not is_initiated) and exists_representative else "secondary"
    }
    btn_styles["upload_unidentified"] = {
        "class": (
            "primary" if is_initiated and (not exists_unidentified) and (not exists_for_confirmation) else "secondary"
        )
    }
    btn_styles["run_identification"] = {
        "class": ("primary" if is_initiated and exists_unidentified and (not exists_for_confirmation) else "secondary")
    }
    btn_styles["confirm_identification"] = {"class": "primary" if exists_for_confirmation else "secondary"}

    init_disabled = (not exists_representative) or (workgroup.identification_reid_status == "Processing")
    logger.debug(f"{init_disabled=}, {workgroup.identification_reid_status=}, {exists_representative=}")
    btn_styles["init_identification"]["class"] += " disabled" if init_disabled else ""
    btn_styles["init_identification"][
        "tooltip"
    ] = f"Identification initialization with {n_representative} representative media files."
    btn_styles["init_identification"][
        "confirm"
    ] = f"Identification initialization with {n_representative} media files will take some time. Continue?"

    btn_styles["run_identification"]["class"] += (
        " disabled" if ((not is_initiated) or (workgroup.identification_init_status == "Processing")) else ""
    )
    btn_styles["run_identification"]["tooltip"] = f"Identification suggestion for {n_unidentified} archives."
    btn_styles["run_identification"][
        "confirm"
    ] = f"Identification of {n_unidentified} archives will take some time. Continue?"
    btn_styles["n_for_confirmation"] = n_for_confirmation
    btn_styles["n_unidentified"] = n_unidentified

    return btn_styles


@login_required
def assign_unidentified_to_identification_view(request):
    """Assign unidentified archive to identification."""
    # logger.debug("Generating CSV for run_identification...")
    caiduser = request.user.caiduser
    tasks.assign_unidentified_to_identification(caiduser)
    return redirect(request.META.get("HTTP_REFERER", "/"))


@login_required
def run_identification_on_unidentified(request):
    """Run identification suggestions for all finished uploaded archives."""
    workgroup = request.user.caiduser.workgroup
    tasks.run_identification_on_unidentified_for_workgroup_task.delay(workgroup.id)
    messages.info(request, "Regeneration of identification suggestions has started.")
    return redirect(request.META.get("HTTP_REFERER", "/"))


@login_required
def run_identification_view(request, uploadedarchive_id):
    """Run identification of uploaded archive."""
    uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    # check if user is owner member of the workgroup

    if uploaded_archive.owner.workgroup != request.user.caiduser.workgroup:
        return HttpResponseNotAllowed("Identification is for workgroup members only.")
    workgroup = request.user.caiduser.workgroup
    if workgroup.identification_model is None:
        messages.error(request, "No identification model for workgroup. Please set it before running identification.")
        return redirect(request.META.get("HTTP_REFERER", "/"))

    status_ok = run_identification(uploaded_archive, workgroup=request.user.caiduser.workgroup)
    if status_ok:
        messages.info(request, f"Identification started for {uploaded_archive.name}.")
    else:
        messages.error(request, "No records for identification with the expected taxon.")
    return redirect(request.META.get("HTTP_REFERER", "/"))


def run_identification_bulk(
    workgroup: models.WorkGroup,
    uploaded_archives=None,
    selection: dict | None = None,
) -> bool:
    """Run one identification job for all eligible uploads in the workgroup."""
    selection = selection or {}
    if uploaded_archives is None:
        uploaded_archives = tasks.get_uploaded_archives_pending_identification(workgroup)

    uploaded_archives = list(uploaded_archives)
    if not uploaded_archives:
        return False

    uploaded_archive_ids = [uploaded_archive.id for uploaded_archive in uploaded_archives]
    bulk_selection = {**selection, "uploaded_archive_ids": uploaded_archive_ids}
    mediafiles, _observation_taxon, _require_observations = tasks.resolve_identification_selection(
        workgroup,
        selection=bulk_selection,
    )
    logger.debug(f"Generating CSV for bulk identification with {len(mediafiles)} records...")

    csv_data = _prepare_dataframe_for_identification(mediafiles)
    df = pd.DataFrame(csv_data)
    if df.shape[0] == 0:
        logger.warning("No records found for bulk identification in workgroup %s.", workgroup.id)
        return False

    run_name = django.utils.timezone.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(settings.MEDIA_ROOT) / workgroup.name / "reid_runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_metadata_file = output_dir / "identification_metadata.csv"
    output_json_file = output_dir / "identification_result.json"
    df.to_csv(identity_metadata_file, index=False)

    for uploaded_archive in uploaded_archives:
        uploaded_archive.identification_status = "IAIP"
        uploaded_archive.save(update_fields=["identification_status"])

    if workgroup.identification_model is None:
        logger.error("No identification model for workgroup. Selecting default model.")
        model = models.IdentificationModel.objects.filter(public=True).first()
        if model is None:
            logger.error("No default identification model found. Using first available model.")
            model = models.IdentificationModel.objects.first()
            if model is None:
                logger.error("No identification model found. Cannot run identification.")
                return False
        workgroup.identification_model = model
        workgroup.save()

    identify_signature = signature(
        "identify",
        kwargs=dict(
            input_metadata_file_path=str(identity_metadata_file),
            organization_id=workgroup.id,
            output_json_file_path=str(output_json_file),
            top_k=3,
            identification_model={
                "name": workgroup.identification_model.name,
                "path": workgroup.identification_model.model_path,
            },
        ),
    )
    identify_task = identify_signature.apply_async(
        link=tasks.identify_bulk_on_success.s(
            workgroup_id=workgroup.id,
            uploaded_archive_ids=uploaded_archive_ids,
        ),
        link_error=tasks.identify_bulk_on_error.s(
            workgroup_id=workgroup.id,
            uploaded_archive_ids=uploaded_archive_ids,
        ),
    )
    workgroup.identification_reid_status = "Processing"
    workgroup.identification_reid_at = django.utils.timezone.now()
    workgroup.identification_reid_message = (
        f"Running identification for {len(uploaded_archive_ids)} uploads and {df.shape[0]} media files."
    )
    workgroup.identification_scheduled_run_task_id = identify_task.id
    workgroup.identification_scheduled_run_eta = None
    workgroup.save(
        update_fields=[
            "identification_reid_status",
            "identification_reid_at",
            "identification_reid_message",
            "identification_scheduled_run_task_id",
            "identification_scheduled_run_eta",
        ]
    )
    logger.debug(f"{identify_task=}")
    return True


def run_identification(
    uploaded_archive: UploadedArchive,
    workgroup: models.WorkGroup,
    selection: dict | None = None,
) -> bool:
    """Run identification of uploaded archive."""
    logger.debug("Generating CSV for run_identification...")
    mediafiles, _observation_taxon, _require_observations = tasks.resolve_identification_selection(
        workgroup,
        uploaded_archive=uploaded_archive,
        selection=selection,
    )
    logger.debug(f"Generating CSV for init_identification with {len(mediafiles)} records...")
    uploaded_archive.identification_status = "IAIP"

    csv_data = _prepare_dataframe_for_identification(mediafiles)
    media_root = Path(settings.MEDIA_ROOT)

    identity_metadata_file = media_root / uploaded_archive.outputdir / "identification_metadata.csv"
    df = pd.DataFrame(csv_data)
    df.to_csv(identity_metadata_file, index=False)
    output_json_file = media_root / uploaded_archive.outputdir / "identification_result.json"

    # if no records in df
    if df.shape[0] == 0:
        logger.warning("No records for identification with the expected taxon. ")

        expected_taxon_string = ""
        if workgroup.default_taxon_for_identification:
            expected_taxon_string = f"(with the expected taxon {workgroup.default_taxon_for_identification.name}) "

        models.Notification.create_for(
            message=f"No records for identification {expected_taxon_string} in {uploaded_archive=}. ",
            level=Notification.LevelChoices.WARNING,
            workgroups=[workgroup],
        )

        return False
        # return redirect(request.META.get("HTTP_REFERER", "/"))

    from celery import current_app

    available_tasks = current_app.tasks.keys()
    logger.debug(f"tasks={available_tasks}")

    logger.debug("Calling run_detection and run_identification ...")

    uploaded_archive.identification_status = "IAIP"
    uploaded_archive.save()
    if workgroup.identification_model is None:
        logger.error("No identification model for workgroup. Selecting default model.")
        model = models.IdentificationModel.objects.filter(public=True).first()
        if model is None:
            logger.error("No default identification model found. Using first available model.")
            model = models.IdentificationModel.objects.first()
            if model is None:
                logger.error("No identification model found. Cannot run identification.")
                return False
        workgroup.identification_model = model
        workgroup.save()

    identify_signature = signature(
        "identify",
        kwargs=dict(
            input_metadata_file_path=str(identity_metadata_file),
            organization_id=uploaded_archive.owner.workgroup.id,
            output_json_file_path=str(output_json_file),
            top_k=3,
            uploaded_archive_id=uploaded_archive.id,
            identification_model={
                "name": workgroup.identification_model.name,
                "path": workgroup.identification_model.model_path,
            },
        ),
    )
    identify_task = identify_signature.apply_async(
        link=identify_on_success.s(
            uploaded_archive_id=uploaded_archive.id,
        ),
        link_error=on_error_in_upload_processing.s(),
    )
    logger.debug(f"{identify_task=}")
    return True
    # return redirect("caidapp:uploads_identities")


@login_required
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


@login_required
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


@login_required
def delete_album(request, album_hash):
    """Delete album if it belongs to the user."""
    album = get_object_or_404(Album, hash=album_hash)
    if album.owner == request.user.caiduser:
        album.delete()
    return redirect("caidapp:albums")


def _one_zip_from_request_FILES(request: HttpRequest) -> HttpRequest:
    """Create one ZIP file from multiple uploaded files in request.FILES."""
    files = request.FILES.getlist("archivefile")

    if not files:
        return request
        # return JsonResponse({"error": "No files uploaded."}, status=400)

    # if there is just one file and it is an archive → keep the original logic
    if len(files) == 1 and files[0].name.lower().endswith((".zip", ".tar", ".tar.gz")):
        # dále zpracování běží přes form.save(), viz níže
        pass
    else:
        # 📦 Uživateli došlo více souborů → zabalíme je do ZIP sami

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for f in files:
                zipf.writestr(f.name, f.read())

        buffer.seek(0)
        now_str = django.utils.timezone.now().strftime("%Y%m%d-%H%M%S")
        # create pseudo file for the form
        zipped_file = ContentFile(buffer.read(), name=f"uploaded_multiple_files.{now_str}.zip")

        # substitute the original request.FILES with the zip file
        request.FILES.setlist("archivefile", [zipped_file])

    return request


def user_can_use_new_upload(user) -> bool:
    if not user.is_authenticated:
        return False
    caiduser = getattr(user, "caiduser", None)
    return bool(
        user.is_staff
        or (
            caiduser
            and (
                caiduser.workgroup_admin
                or caiduser.show_taxon_classification
                or caiduser.show_reid
            )
        )
    )


class NewUploadView(LoginRequiredMixin, UserPassesTestMixin, View):
    template_name = "caidapp/new_upload.html"

    def test_func(self):
        return user_can_use_new_upload(self.request.user)

    def _get_form_initial(self, request):
        initial = {}
        upload_target = request.GET.get("upload_target")
        if upload_target in {"taxon_processing", "identification"}:
            initial["upload_target"] = upload_target
        contains_identities = request.GET.get("contains_identities")
        if contains_identities in {"1", "true", "True", "on", "yes"}:
            initial["contains_identities"] = True
        return initial

    def get(self, request):
        form = forms.NewUploadForm(user=request.user, initial=self._get_form_initial(request))
        return render(
            request,
            self.template_name,
            {
                "form": form,
                "headline": "Upload",
                "localities": get_all_relevant_localities(request),
                "max_upload_files": settings.DATA_UPLOAD_MAX_NUMBER_FILES,
                "path_regex_chatgpt_prompt_prefix_lines": PATH_REGEX_CHATGPT_PROMPT_PREFIX_LINES,
                "path_regex_chatgpt_prompt_suffix": PATH_REGEX_CHATGPT_PROMPT_SUFFIX,
            },
        )

    def post(self, request):
        form = forms.NewUploadForm(request.POST, request.FILES, user=request.user)
        if not form.is_valid():
            html = render_to_string(
                "caidapp/partial_message.html",
                {
                    "headline": "Upload failed",
                    "text": "Upload form is not valid.",
                    "next": reverse_lazy("caidapp:new_upload"),
                    "next_text": "Back to upload",
                },
                request=request,
            )
            return JsonResponse({"ok": False, "html": html, "errors": form.errors.get_json_data()}, status=400)

        caiduser = request.user.caiduser
        if not caiduser.ml_consent_given and form.cleaned_data.get("ml_consent"):
            caiduser.ml_consent_given = True
            caiduser.ml_consent_given_date = timezone.now().astimezone(ZoneInfo(caiduser.timezone))
            caiduser.save()

        upload_files = request.FILES.getlist("upload_files")
        spreadsheet_file = form.cleaned_data.get("spreadsheet_file")
        relative_path_manifest = upload_services.load_relative_path_manifest(
            form.cleaned_data.get("upload_relative_paths", "")
        )
        directory_mapping = upload_services.parse_json_mapping(form.cleaned_data.get("directory_mapping", ""))
        spreadsheet_column_mapping = upload_services.parse_json_mapping(
            form.cleaned_data.get("spreadsheet_column_mapping", "")
        )
        spreadsheet_path_adjustment = upload_services.parse_json_mapping(
            form.cleaned_data.get("spreadsheet_path_adjustment", "")
        )

        try:
            zip_result = upload_services.build_upload_zip(
                upload_files,
                spreadsheet_file=spreadsheet_file,
                relative_path_manifest=relative_path_manifest,
                directory_structure=form.cleaned_data.get("directory_structure", ""),
                directory_mapping=directory_mapping,
                path_regex=form.cleaned_data.get("path_regex", ""),
                spreadsheet_column_mapping=spreadsheet_column_mapping,
                spreadsheet_path_adjustment=spreadsheet_path_adjustment,
            )
        except Exception as exc:
            logger.warning("New upload preparation failed: %s", exc)
            html = render_to_string(
                "caidapp/partial_message.html",
                {
                    "headline": "Upload preparation failed",
                    "text": str(exc),
                    "next": reverse_lazy("caidapp:new_upload"),
                    "next_text": "Back to upload",
                },
                request=request,
            )
            return JsonResponse({"ok": False, "html": html}, status=400)

        blocking_import_log = [
            line
            for line in zip_result.import_log.splitlines()
            if "too shallow" in line or "no relative paths" in line
        ]
        if blocking_import_log:
            html = render_to_string(
                "caidapp/partial_message.html",
                {
                    "headline": "Directory mapping needs attention",
                    "text": "\n".join(blocking_import_log),
                    "next": reverse_lazy("caidapp:new_upload"),
                    "next_text": "Back to upload",
                },
                request=request,
            )
            return JsonResponse({"ok": False, "html": html, "import_log": blocking_import_log}, status=400)

        uploaded_archive = UploadedArchive(
            owner=caiduser,
            locality_at_upload=form.cleaned_data.get("locality_at_upload", ""),
            locality_check_at=form.cleaned_data.get("locality_check_at"),
            contains_single_taxon=form.cleaned_data["taxon_mode"] == "single_taxon",
            contains_identities=form.cleaned_data.get("contains_identities", False),
            is_for_identification=form.cleaned_data.get("is_for_identification", False),
            taxon_for_identification=form.cleaned_data.get("taxon_for_identification"),
            import_log=zip_result.import_log,
            import_mapping=zip_result.import_mapping,
            path_structure_regex=zip_result.import_mapping.get("path_regex", ""),
        )
        uploaded_archive.archivefile.save(zip_result.filename, zip_result.file, save=False)
        uploaded_archive.name = zip_result.archive_name
        if uploaded_archive.locality_at_upload:
            uploaded_archive.locality_at_upload_object = models.get_locality(caiduser, uploaded_archive.locality_at_upload)
        uploaded_archive.save()
        counts = uploaded_archive.number_of_media_files_in_archive()

        run_species_prediction_async(uploaded_archive, extract_identites=uploaded_archive.contains_identities)

        next_url = reverse_lazy("caidapp:uploads")
        if uploaded_archive.contains_identities:
            next_url = reverse_lazy("caidapp:uploads_known_identities")
        elif uploaded_archive.contains_single_taxon:
            next_url = reverse_lazy("caidapp:uploads_identities")

        summary_lines = [
            f"Uploaded {counts['file_count']} files ({counts['image_count']} images and {counts['video_count']} videos).",
        ]
        if zip_result.spreadsheet_summary.filename:
            summary_lines.append(
                "Spreadsheet columns: " + ", ".join(zip_result.spreadsheet_summary.normalized_columns)
            )
        if zip_result.import_log:
            summary_lines.append("Warnings: " + zip_result.import_log.replace("\n", " "))

        html = render_to_string(
            "caidapp/partial_message.html",
            {
                "headline": "Upload finished",
                "text": "\n".join(summary_lines),
                "next": next_url,
                "next_text": "Back to uploads",
            },
            request=request,
        )
        return JsonResponse(
            {
                "ok": True,
                "html": html,
                "uploaded_archive_id": uploaded_archive.id,
                "counts": counts,
                "spreadsheet": zip_result.spreadsheet_summary.__dict__,
                "import_log": zip_result.import_log,
            }
        )


@login_required
def upload_archive(
    request,
    contains_single_taxon=False,
    contains_identities=False,
):
    """Process the uploaded zip file."""
    # TODO remove this legacy upload flow after the new upload has been stable in production for one month.
    text_note = ""
    next = "caidapp:uploads"
    next_url = reverse_lazy("caidapp:uploads")
    if contains_single_taxon:
        text_note = "The archive contains images of a single taxon."
        next = "caidapp:upload_archive_contains_single_taxon"
        next_url = reverse_lazy("caidapp:uploads_identities")
    if contains_identities:
        text_note = "The archive contains identities (of single taxon). " + "Each identity is in individual folder"
        next = "caidapp:upload_archive_contains_identities"
        next_url = reverse_lazy("caidapp:uploads_known_identities")

    if request.method == "POST":
        # logger.debug(f" before: {request.FILES}")
        request = _one_zip_from_request_FILES(request)
        # logger.debug(f"  after: {request.FILES}")

        if contains_single_taxon:
            form = UploadedArchiveFormWithTaxon(
                request.POST,
                request.FILES,
                user=request.user,
            )
        else:
            form = UploadedArchiveForm(
                request.POST,
                request.FILES,
                user=request.user,
            )
        if form.is_valid():
            caiduser = request.user.caiduser
            if not caiduser.ml_consent_given:
                if form.cleaned_data.get("ml_consent"):
                    caiduser.ml_consent_given = True
                    caiduser.ml_consent_given_date = timezone.now().astimezone(ZoneInfo(request.user.caiduser.timezone))
                    caiduser.save()
                else:
                    messages.error(
                        request,
                        "To upload data, you must agree to their use for training AI models.",
                    )
                    return JsonResponse(
                        {
                            "html": render_to_string(
                                "caidapp/partial_message.html",
                                context={
                                    "headline": "Consent required",
                                    "text": "Upload cancelled. Consent of AI training required.",
                                    "next": reverse_lazy("caidapp:uploads"),
                                    "next_text": "Back to uploads",
                                },
                                request=request,
                            )
                        }
                    )
            # get uploaded archive
            uploaded_archive = form.save(commit=False)

            # Získáme ZIP, který vytvořil _one_zip_from_request_FILES
            files = request.FILES.getlist("archivefile")
            if not files:
                raise ValueError("No uploaded file found")

            zip_file = files[0]

            # uložíme ho do modelu
            uploaded_archive.archivefile.save(zip_file.name, zip_file, save=False)

            # uploaded_archive.owner = request.user.caiduser
            # uploaded_archive.contains_identities = contains_identities
            # uploaded_archive.contains_single_taxon = contains_single_taxon
            uploaded_archive.save()
            uploaded_archive_suffix = Path(uploaded_archive.archivefile.name).suffix.lower()
            if uploaded_archive_suffix not in (".tar", ".tar.gz", ".zip"):
                logger.warning(f"Uploaded file with extension '{uploaded_archive_suffix}' is not an archive.")
                messages.warning(
                    request,
                    f"Uploaded file with extension '{uploaded_archive_suffix}' is not an archive.",
                )

            if contains_single_taxon:
                uploaded_archive.taxon_for_identification = form.cleaned_data["taxon_for_identification"]
                # next_url = reverse_lazy("caidapp:uploads_identities")
            else:
                # next_url = reverse_lazy("caidapp:uploads")
                pass
            counts = uploaded_archive.number_of_media_files_in_archive()

            uploaded_archive.owner = request.user.caiduser
            logger.debug(f"{uploaded_archive.contains_identities=}, {contains_identities=}")
            logger.debug(f"{uploaded_archive.contains_single_taxon=}, {contains_single_taxon=}")
            # log actual url
            logger.debug(f"{request.build_absolute_uri()=}")
            uploaded_archive.contains_identities = contains_identities
            uploaded_archive.contains_single_taxon = contains_single_taxon
            uploaded_archive.is_for_identification = contains_identities or contains_single_taxon
            uploaded_archive.name = Path(uploaded_archive.archivefile.name).stem
            # Done in number_of_media_files_in_archive
            # uploaded_archive.videos_at_upload = counts["video_count"]
            # uploaded_archive.images_at_upload = counts["image_count"]
            # uploaded_archive.files_at_upload = counts["file_count"]
            # uploaded_archive.mediafiles_at_upload = counts["video_count"] + counts["image_count"]

            uploaded_archive.save()
            uploaded_archive.extract_locality_check_at_from_filename(commit=True)

            run_species_prediction_async(uploaded_archive, extract_identites=contains_identities)

            context = dict(
                headline="Upload finished",
                text=f"Uploaded {counts['file_count']} files ("
                + f"{counts['image_count']} images and {counts['video_count']} videos).",
                next=next_url,
                next_text="Back to uploads",
            )

            html = render_to_string("caidapp/partial_message.html", context=context, request=request)
            return JsonResponse({"html": html})
        else:
            # Error
            context = dict(
                headline="Upload failed",
                text="Upload failed. Try it again.",
                next=next_url,
                next_text="Back to uploads",
            )
            html = render_to_string("caidapp/partial_message.html", context=context, request=request)
            return JsonResponse({"html": html})

    else:

        initial_data = {
            "contains_identities": contains_identities,
            "contains_single_taxon": contains_single_taxon,
        }

        if contains_single_taxon:
            if request.user.caiduser.default_taxon_for_identification:
                default_taxon = request.user.caiduser.default_taxon_for_identification
            else:
                default_taxon = models.get_taxon("Animalia")
            initial_data["taxon_for_identification"] = default_taxon
            logger.debug(f"{initial_data=}")
            form = UploadedArchiveFormWithTaxon(initial=initial_data, user=request.user)
        else:
            form = UploadedArchiveForm(initial=initial_data, user=request.user)

    return render(
        request,
        "caidapp/model_form_upload.html",
        {
            "form": form,
            "headline": "Upload",
            "button": "Upload",
            "text_note": text_note,
            "next": next,
            "localities": get_all_relevant_localities(request),
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

    list_of_locality_checks = []
    text = str(path) + ""

    for yield_dict in _iterate_over_locality_checks(path, caiduser):

        if yield_dict.parent_dir_to_be_deleted:
            continue

        if yield_dict.is_already_processed:
            continue

        list_of_locality_checks.append(yield_dict.__dict__)
        # text += str(path_of_locality_check.relative_to(path)) + "<br>"

    return render(
        request,
        "caidapp/cloud_import_checks_preview.html",
        {
            "page_obj": list_of_locality_checks,
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
def do_cloud_import_view_single_taxon(request):
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

    tasks.do_cloud_import_for_user_async(caiduser, contains_identities=False, contains_single_taxon=True)
    # tasks.do_cloud_import_for_user(caiduser)

    return redirect("caidapp:cloud_import_preview")


@login_required
def do_cloud_import_view_single_taxon_known_identities(request):
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

    tasks.do_cloud_import_for_user_async(caiduser, contains_identities=True, contains_single_taxon=True)
    # tasks.do_cloud_import_for_user(caiduser)

    return redirect("caidapp:cloud_import_preview")


@login_required
def break_cloud_import_view(request):
    """View for interrupted import from the cloud."""
    caiduser = request.user.caiduser
    caiduser.dir_import_status = "Interrupted"
    caiduser.save()
    return redirect("caidapp:cloud_import_preview")


@login_required
def update_uploadedarchive(request, uploadedarchive_id):
    """Show and update uploaded archive."""
    uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    if not user_has_rw_acces_to_uploadedarchive(request.user.caiduser, uploaded_archive):
        return HttpResponseNotAllowed("Not allowed to see this uploaded archive.")
    uploaded_archive_locality_at_upload = uploaded_archive.locality_at_upload

    if request.method == "POST":
        form = UploadedArchiveUpdateForm(request.POST, instance=uploaded_archive)
        if form.is_valid():
            cleaned_locality_at_upload = form.cleaned_data["locality_at_upload"]
            uploaded_archive = form.save()
            logger.debug(f"{uploaded_archive.locality_at_upload=}, {cleaned_locality_at_upload=}")
            if uploaded_archive_locality_at_upload != cleaned_locality_at_upload:
                logger.debug("Locality has been changed.")
                locality = get_locality(request.user.caiduser, cleaned_locality_at_upload)
                _set_localities_to_mediafiles_of_uploadedarchive(request, uploaded_archive, locality)
                uploaded_archive.locality_at_upload_object = locality
                uploaded_archive.save()

            if uploaded_archive.contains_identities:
                return redirect("caidapp:uploads_identities")
            elif uploaded_archive.contains_single_taxon:
                return redirect("caidapp:uploads_known_identities")
            else:
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
            "localities": get_all_relevant_localities(request),
        },
    )


@login_required
def delete_upload(request, uploadedarchive_id, next_page="caidapp:uploads"):
    """Delete uploaded file."""
    uploadedarchive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)

    if user_has_rw_acces_to_uploadedarchive(request.user.caiduser, uploadedarchive, accept_none=True):
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
            return redirect("caidapp:home")
        else:
            return redirect("caidapp:uploadedarchive_mediafiles", uploadedarchive_id=parent_id)
    else:
        return HttpResponseNotAllowed("Not allowed to delete this media file.")


@login_required
def albums(request):
    """Show all albums."""
    albums = (
        Album.objects.filter(Q(albumsharerole__user=request.user.caiduser) | Q(owner=request.user.caiduser))
        .distinct()
        .all()
        .order_by("created_at")
    )
    return render(request, "caidapp/albums.html", {"albums": albums})


class MyLoginView(LoginView):
    redirect_authenticated_user = True

    def get_success_url(self):
        """Return url of next page."""
        caid_user = self.request.user.caiduser
        if caid_user.show_wellcome_message_on_next_login:
            return reverse("caidapp:wellcome")
        else:
            return reverse_lazy("caidapp:home")

    def form_invalid(self, form):
        """Return error message if wrong username or password is given."""
        messages.error(self.request, "Invalid username or password")
        return self.render_to_response(self.get_context_data(form=form))


def _mediafiles_annotate() -> dict:
    """Prepare annotations for mediafiles."""
    return dict()


# TODO remove
def _mediafiles_query(
    request,
    query: str,
    album_hash=None,
    individual_identity_id=None,
    taxon_id=None,
    uploadedarchive_id=None,
    identity_is_representative=None,
    locality_hash=None,
    order_by: Optional[str] = None,
    taxon_verified: Optional[bool] = None,
    filter_kwargs: Optional[dict] = None,
    exclude_filter_kwargs: Optional[dict] = None,
):
    """Prepare list of mediafiles based on query search in category and locality."""
    if filter_kwargs is None:
        filter_kwargs = {}

    if exclude_filter_kwargs is None:
        exclude_filter_kwargs = {}
    if order_by is None:
        order_by = request.session.get("mediafiles_order_by", "-parent__uploaded_at")

    # logger.debug(f"{filter_kwargs=}, {exclude_filter_kwargs=}, {order_by=}")

    mediafiles = MediaFile.objects.annotate(**_mediafiles_annotate())
    if taxon_verified is not None:
        filter_kwargs.update(dict(taxon_verified=taxon_verified))
    if album_hash is not None:
        album = get_object_or_404(Album, hash=album_hash)
        filter_kwargs.update(dict(album=album))
    if individual_identity_id is not None:
        individual_identity = get_object_or_404(IndividualIdentity, pk=individual_identity_id)
        filter_kwargs.update(dict(identity=individual_identity))
    if taxon_id is not None:
        taxon = get_object_or_404(Taxon, pk=taxon_id)
        filter_kwargs.update(dict(taxon=taxon))
    if uploadedarchive_id is not None:
        uploadedarchive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
        filter_kwargs.update(dict(parent=uploadedarchive))
    if identity_is_representative is not None:
        filter_kwargs.update(dict(identity_is_representative=identity_is_representative))
    # logger.debug(f"{filter_kwargs=}, {exclude_filter_kwargs=}, {order_by=}")
    if locality_hash is not None:
        locality = get_object_or_404(Locality, hash=locality_hash)
        filter_kwargs.update(dict(locality=locality))
    # logger.debug(f"{filter_kwargs=}")
    # order by mediafile__sequence__mediafile_set order by
    order_by_safe = order_by if order_by[0] != "-" else order_by[1:]
    first_image_order_by = mediafiles.filter(sequence=OuterRef("sequence")).order_by(order_by).values(order_by_safe)[:1]

    # ownership filter params
    # Build the base query with the conditions that are always applied
    mediafiles = mediafiles.filter(
        Q(album__albumsharerole__user=request.user.caiduser)
        | Q(**models.user_has_access_filter_params(request.user.caiduser, "parent__owner")),
        **filter_kwargs,
    )
    # logger.debug(f"{len(mediafiles)=}")

    # Add workgroup filtering only if `request.user.caiduser.workgroup` is not None
    if request.user.caiduser.workgroup is not None:
        mediafiles = mediafiles.filter(Q(parent__owner__workgroup=request.user.caiduser.workgroup))

    # logger.debug(f"{len(mediafiles)=}")
    # Apply the exclusion, annotations, and ordering
    mediafiles = (
        mediafiles.exclude(**exclude_filter_kwargs)
        .distinct()
        .annotate(first_image_order_by=Subquery(first_image_order_by))
        .order_by("first_image_order_by", "sequence", "captured_at")
    )
    logger.debug(f"{len(mediafiles)=}")

    if len(query) == 0:
        pass
        # return mediafiles
    else:

        vector = SearchVector("taxon__name", "locality__name")
        query = SearchQuery(query)
        logger.debug(str(query))
        mediafiles = mediafiles.annotate(rank=SearchRank(vector, query)).filter(rank__gt=0).order_by("-rank")
        # return mediafiles
    mediafiles = mediafiles.select_related(
        "parent", "taxon", "predicted_taxon", "locality", "identity", "updated_by", "sequence"
    )

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


@login_required
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


def _taxon_stats_for_mediafiles(mediafiles: Union[QuerySet, List[MediaFile]]) -> str:
    """Create taxon stats for mediafiles."""
    if isinstance(mediafiles, QuerySet):
        mediafile_ids = mediafiles.values_list("id", flat=True)
    else:
        mediafile_ids = [mediafile.id for mediafile in mediafiles]

    if not mediafile_ids:
        return None

    taxon_stats = (
        AnimalObservation.objects.filter(mediafile_id__in=mediafile_ids, taxon__isnull=False)
        .values("taxon__name")
        .annotate(count=Count("mediafile_id", distinct=True))
        .order_by("-count", "taxon__name")
    )
    logger.debug(f"{taxon_stats=}")

    df = pd.DataFrame.from_records(taxon_stats)
    if df.empty:
        return None

    df.rename(columns={"taxon__name": "Taxon", "count": "Count"}, inplace=True)
    fig = px.bar(df, x="Taxon", y="Count", height=300)
    return fig.to_html()


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
    if ffk.get("filter_orientation", None):
        orientation = ffk.get("filter_orientation")
        if orientation != "All":
            filter_kwargs["orientation"] = orientation

    if ffk.get("filter_hide_empty", None):
        # MediaFile.category.name is not "Empty"
        exclude_filter_kwargs.update(dict(taxon__name="Nothing"))

    return filter_kwargs, exclude_filter_kwargs


def _parse_bool_query_param(value: Optional[str]) -> Optional[bool]:
    """Parse boolean-like query parameter values."""
    if value is None:
        return None

    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _parse_int_query_param_or_404(value: Optional[str], label: str) -> Optional[int]:
    """Parse integer query parameter or raise 404 for malformed values."""
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise Http404(f"Invalid {label}.") from exc


def _get_uploadedarchive_for_user_or_404(request: HttpRequest, uploadedarchive_id: int) -> UploadedArchive:
    """Return an uploaded archive only when it is visible to the current user."""
    return get_object_or_404(
        UploadedArchive,
        pk=uploadedarchive_id,
        **models.user_has_access_filter_params(request.user.caiduser, "owner"),
    )


def _get_active_uploadedarchive_from_request(
    request: HttpRequest,
    uploadedarchive_id: Optional[int] = None,
) -> Optional[UploadedArchive]:
    """Resolve active uploaded archive scope from explicit arg or query string."""
    if uploadedarchive_id is None:
        uploadedarchive_id = _parse_int_query_param_or_404(
            request.GET.get("uploadedarchive_id") or request.GET.get("uploadedarchive"),
            "uploaded archive id",
        )
    if uploadedarchive_id is None:
        return None
    return _get_uploadedarchive_for_user_or_404(request, uploadedarchive_id)


def _get_active_taxon_from_request(request: HttpRequest) -> Optional[Taxon]:
    """Resolve active taxon scope from query string."""
    taxon_id = _parse_int_query_param_or_404(request.GET.get("taxon"), "taxon id")
    if taxon_id is None:
        return None
    return get_object_or_404(Taxon, pk=taxon_id)


def _get_album_for_user_or_404(request: HttpRequest, album_hash: str) -> Album:
    """Return an album only when it is visible to the current user."""
    return get_object_or_404(
        Album.objects.filter(Q(albumsharerole__user=request.user.caiduser) | Q(owner=request.user.caiduser)).distinct(),
        hash=album_hash,
    )


def _get_active_album_from_request(request: HttpRequest, album_hash: Optional[str] = None) -> Optional[Album]:
    """Resolve active album scope from explicit arg or query string."""
    if album_hash is None:
        album_hash = request.GET.get("album_hash")
    if not album_hash:
        return None
    return _get_album_for_user_or_404(request, album_hash)


def _get_identity_for_user_or_404(request: HttpRequest, individual_identity_id: int) -> IndividualIdentity:
    """Return an identity only when it belongs to the current user's workgroup."""
    return get_object_or_404(
        IndividualIdentity,
        pk=individual_identity_id,
        owner_workgroup=request.user.caiduser.workgroup,
    )


def _get_active_identity_from_request(
    request: HttpRequest,
    individual_identity_id: Optional[int] = None,
) -> Optional[IndividualIdentity]:
    """Resolve active identity scope from explicit arg or query string."""
    if individual_identity_id is None:
        individual_identity_id = _parse_int_query_param_or_404(
            request.GET.get("individual_identity_id"),
            "individual identity id",
        )
    if individual_identity_id is None:
        return None
    return _get_identity_for_user_or_404(request, individual_identity_id)


def _parse_identity_ids_from_request(request: HttpRequest) -> List[int]:
    """Return identity ids from repeated or comma-separated query parameters."""
    raw_values = request.GET.getlist("individual_identity_ids")
    parsed_ids = []
    for raw_value in raw_values:
        for raw_id in str(raw_value).split(","):
            raw_id = raw_id.strip()
            if not raw_id:
                continue
            parsed_ids.append(_parse_int_query_param_or_404(raw_id, "individual identity id"))
    return parsed_ids


def _get_active_identities_from_request(request: HttpRequest) -> List[IndividualIdentity]:
    """Resolve a multiple-identity sequence scope for the current user's workgroup."""
    identity_ids = _parse_identity_ids_from_request(request)
    if not identity_ids:
        return []
    identities = list(
        IndividualIdentity.objects.filter(
            pk__in=identity_ids,
            owner_workgroup=request.user.caiduser.workgroup,
        ).order_by("name", "id")
    )
    if len({identity.id for identity in identities}) != len(set(identity_ids)):
        raise Http404("Individual identity was not found.")
    return identities


def _get_locality_for_user_or_404(request: HttpRequest, locality_hash: str) -> Locality:
    """Return a locality only when it is visible to the current user's workgroup."""
    return get_object_or_404(
        Locality,
        hash=locality_hash,
        **models.user_has_access_filter_params(request.user.caiduser, "owner"),
    )


def _get_active_locality_from_request(
    request: HttpRequest,
    locality_hash: Optional[str] = None,
) -> Optional[Locality]:
    """Resolve active locality scope from explicit arg or query string."""
    if locality_hash is None:
        locality_hash = request.GET.get("locality_hash")
    if not locality_hash:
        return None
    return _get_locality_for_user_or_404(request, locality_hash)


def _build_mediafiles_scope_query_string(
    request,
    uploadedarchive_id: Optional[int] = None,
    album_hash: Optional[str] = None,
    individual_identity_id: Optional[int] = None,
    identity_is_representative: Optional[bool] = None,
    locality_hash: Optional[str] = None,
    show_overview_button: bool = False,
    taxon_verified: Optional[bool] = None,
) -> str:
    """Build a query string preserving the active mediafiles scope."""
    query_params = request.GET.copy()

    if uploadedarchive_id is not None:
        query_params["uploadedarchive_id"] = str(uploadedarchive_id)
    if album_hash is not None:
        query_params["album_hash"] = album_hash
    if individual_identity_id is not None:
        query_params["individual_identity_id"] = str(individual_identity_id)
    if identity_is_representative is not None:
        query_params["identity_is_representative"] = str(identity_is_representative).lower()
    if locality_hash is not None:
        query_params["locality_hash"] = locality_hash
    if show_overview_button:
        query_params["show_overview_button"] = "true"
    if taxon_verified is not None:
        query_params["taxon_verified"] = str(taxon_verified).lower()

    return query_params.urlencode()


def _get_filtered_mediafiles_queryset(
    request,
    uploadedarchive_id: Optional[int] = None,
    album_hash: Optional[str] = None,
    individual_identity_id: Optional[int] = None,
    individual_identity_ids: Optional[List[int]] = None,
    identity_is_representative: Optional[bool] = None,
    locality_hash: Optional[str] = None,
    show_overview_button: bool = False,
    taxon_verified: Optional[bool] = None,
    extra_filter_kwargs: Optional[dict] = None,
) -> Tuple[QuerySet, filters.MediaFileFilter, str, Optional[str]]:
    """Build the filtered mediafiles queryset shared by multiple views."""
    filter_kwargs = dict(extra_filter_kwargs or {})

    if uploadedarchive_id is None and request.GET.get("uploadedarchive_id"):
        uploadedarchive_id = _parse_int_query_param_or_404(request.GET.get("uploadedarchive_id"), "uploaded archive id")
    if album_hash is None:
        album_hash = request.GET.get("album_hash")
    if individual_identity_id is None and request.GET.get("individual_identity_id"):
        individual_identity_id = _parse_int_query_param_or_404(
            request.GET.get("individual_identity_id"),
            "individual identity id",
        )
    if individual_identity_ids is None:
        individual_identity_ids = _parse_identity_ids_from_request(request)
    if identity_is_representative is None:
        identity_is_representative = _parse_bool_query_param(request.GET.get("identity_is_representative"))
    if locality_hash is None:
        locality_hash = request.GET.get("locality_hash")
    if not show_overview_button:
        show_overview_button = bool(_parse_bool_query_param(request.GET.get("show_overview_button")))
    if taxon_verified is None:
        taxon_verified = _parse_bool_query_param(request.GET.get("taxon_verified"))

    if request.GET.get("taxon"):
        taxon = _get_active_taxon_from_request(request)
        page_title = f"Media files - {taxon.name}"
    else:
        page_title = "Media files"

    sequence_id = request.GET.get("sequence")
    sequence = get_object_or_404(models.Sequence, pk=sequence_id) if sequence_id else None

    mediafiles_name_suggestion = None
    if taxon_verified is not None:
        filter_kwargs["taxon_verified"] = taxon_verified

    if show_overview_button:
        filter_kwargs["taxon_verified"] = False
        mediafiles_name_suggestion = "taxon_not_verified"

    if uploadedarchive_id is not None:
        uploaded_archive = _get_uploadedarchive_for_user_or_404(request, uploadedarchive_id)
        if uploaded_archive.locality_check_at is not None:
            locality_check_at = " - " + uploaded_archive.locality_check_at.strftime("%Y-%m-%d %H:%M:%S")
        else:
            locality_check_at = ""
        locality_label = uploaded_archive.localities_display or uploaded_archive.locality_at_upload
        page_title = f"Media files - {locality_label}{locality_check_at}"
        filter_kwargs["parent"] = uploaded_archive
        mediafiles_name_suggestion = f"uploaded_archive_{locality_label}{locality_check_at}"
    elif album_hash is not None:
        album = _get_album_for_user_or_404(request, album_hash)
        page_title = f"Media files - {album.name}"
        filter_kwargs["album"] = album
        mediafiles_name_suggestion = f"album_{album.name}"
    elif individual_identity_id is not None:
        individual_identity = _get_identity_for_user_or_404(request, individual_identity_id)
        page_title = f"Media files - {individual_identity.name}"
        filter_kwargs["identity"] = individual_identity
        mediafiles_name_suggestion = f"individual_identity_{individual_identity.name}"
    elif individual_identity_ids:
        active_identities = _get_active_identities_from_request(request)
        identity_names = ", ".join(str(identity) for identity in active_identities[:3])
        if len(active_identities) > 3:
            identity_names += f", +{len(active_identities) - 3}"
        page_title = f"Media files - {identity_names}"
        filter_kwargs["identity_id__in"] = [identity.id for identity in active_identities]
        mediafiles_name_suggestion = "individual_identities"
    elif locality_hash is not None:
        locality = _get_locality_for_user_or_404(request, locality_hash)
        page_title = f"Media files - {locality.name}"
        filter_kwargs["locality"] = locality
        mediafiles_name_suggestion = f"locality_{locality.name}"
    elif identity_is_representative is not None:
        page_title = "Media files - representative"
        filter_kwargs["identity_is_representative"] = identity_is_representative
        mediafiles_name_suggestion = f"representative_identity_{str(identity_is_representative)}"

    mediafiles = MediaFile.objects.filter(
        Q(album__albumsharerole__user=request.user.caiduser)
        | Q(**models.user_has_access_filter_params(request.user.caiduser, "parent__owner")),
        **filter_kwargs,
    )
    mediafile_filter = filters.MediaFileFilter(request.GET, queryset=mediafiles, request=request)
    full_mediafiles = mediafile_filter.qs.filter(sequence=sequence) if sequence else mediafile_filter.qs
    full_mediafiles = full_mediafiles.distinct()

    return full_mediafiles, mediafile_filter, page_title, mediafiles_name_suggestion


def _verification_taxon_group_label(mediafile: MediaFile) -> str:
    """Return a display label used for grouping verification cards."""
    taxons = mediafile.taxons
    if not taxons:
        return "No taxon"
    if len(taxons) == 1:
        return str(taxons[0])
    return "Mixed taxa: " + ", ".join(str(taxon) for taxon in taxons)


def _annotate_verification_taxon_groups(form_objects) -> None:
    """Mark form instances where a taxon group heading should be rendered."""
    previous_label = None
    for mediafile_form in form_objects:
        label = _verification_taxon_group_label(mediafile_form.instance)
        mediafile_form.instance.verification_taxon_group_label = label
        mediafile_form.instance.starts_verification_taxon_group = label != previous_label
        previous_label = label


def _verification_sequence_taxon_group_label(sequence: models.Sequence) -> str:
    """Return a taxon group label for a sequence based on displayed mediafiles."""
    taxons = []
    for mediafile in sequence.mediafile_set.all():
        mediafile_taxons = mediafile.taxons
        if mediafile_taxons:
            taxons.extend(mediafile_taxons)
    unique_taxons = models._unique_sorted_model_objects(taxons)
    if not unique_taxons:
        return "No taxon"
    if len(unique_taxons) == 1:
        return str(unique_taxons[0])
    return "Mixed taxa: " + ", ".join(str(taxon) for taxon in unique_taxons)


def _annotate_verification_sequence_taxon_groups(sequences: List[models.Sequence]) -> None:
    """Mark sequence objects where a taxon group heading should be rendered."""
    previous_label = None
    for sequence in sequences:
        label = _verification_sequence_taxon_group_label(sequence)
        sequence.verification_taxon_group_label = label
        sequence.starts_verification_taxon_group = label != previous_label
        previous_label = label


def _build_sequence_scope_query_string(
    request,
    uploadedarchive_id: Optional[int] = None,
    album_hash: Optional[str] = None,
    individual_identity_id: Optional[int] = None,
    identity_is_representative: Optional[bool] = None,
    locality_hash: Optional[str] = None,
    show_overview_button: bool = False,
    taxon_verified: Optional[bool] = None,
) -> str:
    """Build a query string preserving the active sequence scope."""
    return _build_mediafiles_scope_query_string(
        request,
        uploadedarchive_id=uploadedarchive_id,
        album_hash=album_hash,
        individual_identity_id=individual_identity_id,
        identity_is_representative=identity_is_representative,
        locality_hash=locality_hash,
        show_overview_button=show_overview_button,
        taxon_verified=taxon_verified,
    )


def _get_sequences_queryset_from_mediafiles(full_mediafiles: QuerySet) -> QuerySet:
    """Return sequences derived from a filtered mediafile queryset."""
    scoped_mediafiles = full_mediafiles.exclude(sequence__isnull=True)
    first_captured_at = scoped_mediafiles.filter(sequence=OuterRef("pk")).order_by("captured_at", "id").values("captured_at")[:1]
    first_mediafile_id = scoped_mediafiles.filter(sequence=OuterRef("pk")).order_by("captured_at", "id").values("id")[:1]
    sequence_ids = scoped_mediafiles.values("sequence_id")

    sequences = (
        models.Sequence.objects.filter(id__in=Subquery(sequence_ids))
        .annotate(
            first_captured_at=Subquery(first_captured_at),
            first_mediafile_id=Subquery(first_mediafile_id),
            mediafile_count=Count("mediafile"),
        )
        .order_by("first_captured_at", "first_mediafile_id", "pk")
    )
    return sequences


SEQUENCE_PER_PAGE_OPTIONS = [12, 24, 48, 96, 192]
SEQUENCE_SORT_OPTIONS = {
    "captured_desc": {"label": "Newest first", "order_by": ("-first_captured_at", "-first_mediafile_id", "-pk")},
    "captured_asc": {"label": "Oldest first", "order_by": ("first_captured_at", "first_mediafile_id", "pk")},
    "count_desc": {"label": "Most files first", "order_by": ("-mediafile_count", "-first_captured_at", "-pk")},
    "count_asc": {"label": "Fewest files first", "order_by": ("mediafile_count", "first_captured_at", "pk")},
    "sequence_asc": {"label": "Sequence number ascending", "order_by": ("local_id", "first_captured_at", "pk")},
    "sequence_desc": {"label": "Sequence number descending", "order_by": ("-local_id", "-first_captured_at", "-pk")},
}


def _get_sequence_records_per_page(request, explicit_value: Optional[int] = None) -> int:
    """Return validated per-page value for sequence view."""
    raw_value = explicit_value if explicit_value is not None else request.GET.get("per_page")
    try:
        value = int(raw_value) if raw_value is not None else request.session.get("mediafiles_records_per_page", 24)
    except (TypeError, ValueError):
        value = 24
    if value not in SEQUENCE_PER_PAGE_OPTIONS:
        value = 24
    return value


def _get_sequence_sort(request) -> str:
    """Return validated sort key for sequence view."""
    sort = request.GET.get("sort", "captured_desc")
    if sort not in SEQUENCE_SORT_OPTIONS:
        return "captured_desc"
    return sort


def _resolve_selected_mediafile_ids_from_post(request) -> List[int]:
    """Resolve bulk-selected mediafiles from sequence and mediafile checkboxes."""
    selected_sequence_ids = [int(v) for v in request.POST.getlist("selected_sequence_ids") if str(v).isdigit()]
    selected_mediafile_ids = {int(v) for v in request.POST.getlist("selected_mediafile_ids") if str(v).isdigit()}
    deselected_mediafile_ids = {int(v) for v in request.POST.getlist("deselected_mediafile_ids") if str(v).isdigit()}

    if selected_sequence_ids:
        sequence_mediafile_ids = MediaFile.objects.filter(sequence_id__in=selected_sequence_ids).values_list("id", flat=True)
        selected_mediafile_ids.update(sequence_mediafile_ids)

    selected_mediafile_ids.difference_update(deselected_mediafile_ids)
    return sorted(selected_mediafile_ids)


def _dissolve_mediafiles_into_singleton_sequences(caiduser, mediafile_ids: List[int]) -> int:
    """Move selected mediafiles into single-media sequences within their uploads."""
    selected_mediafiles = list(
        MediaFile.objects.for_user(caiduser)
        .filter(id__in=mediafile_ids)
        .select_related("parent", "sequence")
        .order_by("parent_id", "id")
    )
    if not selected_mediafiles:
        return 0

    affected_sequence_ids = {mediafile.sequence_id for mediafile in selected_mediafiles if mediafile.sequence_id}
    sequence_sizes = dict(
        MediaFile.objects.filter(sequence_id__in=affected_sequence_ids)
        .values("sequence_id")
        .annotate(sequence_mediafile_count=Count("id"))
        .values_list("sequence_id", "sequence_mediafile_count")
    )
    mediafiles_to_reassign = [
        mediafile
        for mediafile in selected_mediafiles
        if mediafile.sequence_id is None or sequence_sizes.get(mediafile.sequence_id, 0) > 1
    ]
    if not mediafiles_to_reassign:
        return 0

    next_local_id_by_archive = {}
    for archive_id in {mediafile.parent_id for mediafile in mediafiles_to_reassign if mediafile.parent_id}:
        current_max_local_id = (
            models.Sequence.objects.filter(uploaded_archive_id=archive_id).aggregate(max_local_id=Max("local_id"))[
                "max_local_id"
            ]
        )
        next_local_id_by_archive[archive_id] = 0 if current_max_local_id is None else current_max_local_id + 1

    for mediafile in mediafiles_to_reassign:
        archive_id = mediafile.parent_id
        local_id = next_local_id_by_archive[archive_id]
        next_local_id_by_archive[archive_id] += 1
        mediafile.sequence = models.Sequence.objects.create(uploaded_archive_id=archive_id, local_id=local_id)
        mediafile.save(update_fields=["sequence"])

    models.Sequence.objects.filter(id__in=affected_sequence_ids).annotate(mediafile_count=Count("mediafile")).filter(
        mediafile_count=0
    ).delete()
    return len(mediafiles_to_reassign)


def _parse_filename_metadata_date(value: str):
    """Parse date captured from a filename/path regex."""
    if not value:
        return None
    for date_format in ("%Y-%m-%d", "%Y%m%d"):
        try:
            parsed_date = datetime.datetime.strptime(str(value), date_format)
            return parsed_date.replace(tzinfo=ZoneInfo(settings.TIME_ZONE))
        except ValueError:
            continue
    return None


def _metadata_value_is_empty(value) -> bool:
    """Return whether a model field value should be treated as empty for filename metadata fill."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _can_apply_filename_metadata_value(current_value, force_rewrite_filled_data: bool) -> bool:
    return force_rewrite_filled_data or _metadata_value_is_empty(current_value)


def _get_observation_for_filename_metadata(mediafile: MediaFile) -> Tuple[Optional[AnimalObservation], bool]:
    """Return a single unambiguous observation and whether multiple observations exist."""
    observations = list(mediafile.observations.all())
    if len(observations) > 1:
        return None, True
    if observations:
        return observations[0], False
    return AnimalObservation.objects.create(mediafile=mediafile), False


def _apply_value_to_observation(
    observation: AnimalObservation,
    field_name: str,
    value,
    force_rewrite_filled_data: bool,
) -> bool:
    """Apply a filename-derived value to one observation where it is allowed."""
    value_id = getattr(value, "id", value)
    current_id = getattr(observation, f"{field_name}_id", None)
    if current_id != value_id and _can_apply_filename_metadata_value(current_id, force_rewrite_filled_data):
        setattr(observation, field_name, value)
        return True
    return False


def _apply_filename_metadata_to_mediafile(
    mediafile: MediaFile,
    regex,
    caiduser,
    apply_to_manually_updated: bool,
    force_rewrite_filled_data: bool,
) -> str:
    """Apply metadata captured from original_filename to a single mediafile."""
    if mediafile.updated_by_id and not apply_to_manually_updated:
        return "skipped_manual"

    source_path = (mediafile.original_filename or mediafile.mediafile.name or "").replace("\\", "/")
    match = regex.search(source_path)
    if not match:
        return "no_match"

    groups = {key: value.strip() for key, value in match.groupdict().items() if value and value.strip()}
    if not groups:
        return "no_groups"

    changed_fields = []
    observation, has_multiple_observations = _get_observation_for_filename_metadata(mediafile)
    skipped_observation_metadata = False
    taxon_name = groups.get("taxon")
    if taxon_name:
        if has_multiple_observations:
            skipped_observation_metadata = True
        else:
            taxon = models.get_taxon(taxon_name)
            observation_changed = _apply_value_to_observation(
                observation,
                "taxon",
                taxon,
                force_rewrite_filled_data,
            )
            if mediafile.taxon_id != taxon.id and _can_apply_filename_metadata_value(
                mediafile.taxon_id,
                force_rewrite_filled_data,
            ):
                mediafile.taxon = taxon
                changed_fields.append("taxon")
            elif observation_changed:
                changed_fields.append("taxon")

    locality_name = groups.get("locality")
    if locality_name:
        locality = models.get_locality(caiduser, locality_name)
        if locality and mediafile.locality_id != locality.id and _can_apply_filename_metadata_value(
            mediafile.locality_id,
            force_rewrite_filled_data,
        ):
            mediafile.locality = locality
            changed_fields.append("locality")

    code = groups.get("code")
    identity_name = groups.get("identity") or groups.get("unique_name")
    juv_code = groups.get("juv_code")
    identity = None
    identity_metadata_present = bool(code or identity_name or juv_code)
    if has_multiple_observations and identity_metadata_present:
        skipped_observation_metadata = True
    elif mediafile.identity_id and not force_rewrite_filled_data:
        identity = mediafile.identity
    elif code:
        identity = models.get_unique_code(code, workgroup=caiduser.workgroup)
    elif identity_name:
        identity = models.get_unique_name(identity_name, workgroup=caiduser.workgroup)
    if identity is not None:
        if has_multiple_observations:
            skipped_observation_metadata = True
        else:
            identity_changed = False
            if identity_name and identity.name != identity_name and _can_apply_filename_metadata_value(
                identity.name,
                force_rewrite_filled_data,
            ):
                identity.name = identity_name[:100]
                identity_changed = True
            if code and identity.code != code and _can_apply_filename_metadata_value(
                identity.code,
                force_rewrite_filled_data,
            ):
                identity.code = code[:50]
                identity_changed = True
            if juv_code and identity.juv_code != juv_code and _can_apply_filename_metadata_value(
                identity.juv_code,
                force_rewrite_filled_data,
            ):
                identity.juv_code = juv_code[:50]
                identity_changed = True
            if identity_changed:
                identity.save()
                changed_fields.append("identity_fields")
            if mediafile.identity_id != identity.id and _can_apply_filename_metadata_value(
                mediafile.identity_id,
                force_rewrite_filled_data,
            ):
                mediafile.identity = identity
                changed_fields.append("identity")
            if _apply_value_to_observation(
                observation,
                "identity",
                identity,
                force_rewrite_filled_data,
            ) and "identity" not in changed_fields:
                changed_fields.append("identity")

    captured_at = _parse_filename_metadata_date(groups.get("check_date") or groups.get("date"))
    if captured_at and mediafile.captured_at != captured_at and _can_apply_filename_metadata_value(
        mediafile.captured_at,
        force_rewrite_filled_data,
    ):
        mediafile.captured_at = captured_at
        changed_fields.append("captured_at")

    if not changed_fields:
        if skipped_observation_metadata:
            return "skipped_multiple_observations"
        return "unchanged"

    mediafile.save()
    if observation is not None:
        observation.save()
    if skipped_observation_metadata:
        return "updated_skipped_multiple_observations"
    return "updated"


def _build_filename_metadata_regex_prompt(sample_paths: List[str]) -> str:
    """Build a ChatGPT prompt for suggesting a filename metadata regex."""
    sample_lines = "\n".join(f"- {path}" for path in sample_paths) if sample_paths else "- taxon/unique_name/example.jpg"
    return "\n".join([*PATH_REGEX_CHATGPT_PROMPT_PREFIX_LINES, sample_lines, "", PATH_REGEX_CHATGPT_PROMPT_SUFFIX])


def _directory_parts_from_path(path: str) -> List[Tuple[int, str]]:
    """Return indexed directory parts from a normalized media path."""
    path_parts = [part for part in str(path or "").replace("\\", "/").split("/") if part]
    return list(enumerate(path_parts[:-1]))


def _start_filename_metadata_session(
    request: HttpRequest,
    mediafile_ids: List[int],
    return_url: str,
    source_label: str,
) -> HttpResponse:
    request.session["filename_metadata_mediafile_ids"] = mediafile_ids
    request.session["filename_metadata_return_url"] = return_url
    request.session["filename_metadata_source_label"] = source_label
    return redirect("caidapp:apply_filename_metadata_to_mediafiles")


@login_required
def apply_filename_metadata_to_uploadedarchive(request, uploadedarchive_id: int) -> HttpResponse:
    """Start filename/path metadata extraction for all media files in one upload."""
    uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    if not user_has_rw_acces_to_uploadedarchive(request.user.caiduser, uploaded_archive):
        return HttpResponseNotAllowed("Not allowed to work with this uploaded archive.")

    return_url = request.GET.get("next") or reverse_lazy("caidapp:uploadedarchive_mediafiles", args=[uploaded_archive.id])
    mediafile_ids = list(uploaded_archive.mediafile_set.values_list("id", flat=True))
    return _start_filename_metadata_session(
        request,
        mediafile_ids,
        return_url,
        f"Upload: {uploaded_archive}",
    )


@login_required
def apply_filename_metadata_to_mediafiles(request) -> HttpResponse:
    """Configure and apply filename/path metadata extraction to a selected mediafile set."""
    caiduser = request.user.caiduser
    mediafile_ids = request.session.get("filename_metadata_mediafile_ids", [])
    return_url = request.session.get("filename_metadata_return_url") or reverse_lazy("caidapp:sequences")
    source_label = request.session.get("filename_metadata_source_label") or "Sequences"
    mediafiles = MediaFile.objects.for_user(caiduser).filter(id__in=mediafile_ids).select_related("updated_by", "parent")
    mediafile_count = mediafiles.count()

    if mediafile_count == 0:
        return message_view(
            request,
            "No media files were selected.",
            headline="Extract metadata from filenames",
            link=return_url,
            button_label="Back",
        )

    if request.method == "POST":
        form = forms.MediaFileFilenameMetadataForm(request.POST)
        if form.is_valid():
            path_regex = form.cleaned_data["path_regex"]
            if not path_regex:
                directory_mapping = upload_services.parse_json_mapping(form.cleaned_data.get("directory_mapping", ""))
                path_regex = upload_services.build_path_regex_from_directory_mapping(directory_mapping)
            if not path_regex:
                form.add_error("path_regex", "Choose path parts to extract, or enter an advanced regex.")
                regex = None
            else:
                regex = None
            try:
                if path_regex:
                    regex = re.compile(path_regex)
            except re.error as exc:
                form.add_error("path_regex", f"Invalid regex: {exc}")
            if regex is not None:
                status_counts = {
                    "updated": 0,
                    "updated_skipped_multiple_observations": 0,
                    "unchanged": 0,
                    "skipped_manual": 0,
                    "skipped_multiple_observations": 0,
                    "no_match": 0,
                    "no_groups": 0,
                }
                for mediafile in mediafiles:
                    status = _apply_filename_metadata_to_mediafile(
                        mediafile,
                        regex,
                        caiduser,
                        form.cleaned_data["apply_to_manually_updated"],
                        form.cleaned_data["force_rewrite_filled_data"],
                    )
                    status_counts[status] = status_counts.get(status, 0) + 1

                request.session.pop("filename_metadata_mediafile_ids", None)
                request.session.pop("filename_metadata_return_url", None)
                request.session.pop("filename_metadata_source_label", None)
                updated_count = status_counts["updated"] + status_counts["updated_skipped_multiple_observations"]
                skipped_multiple_observations = (
                    status_counts["skipped_multiple_observations"]
                    + status_counts["updated_skipped_multiple_observations"]
                )
                messages.success(
                    request,
                    (
                        f"Filename metadata applied to {updated_count} media files. "
                        f"Skipped observation metadata for multiple-observation media files: "
                        f"{skipped_multiple_observations}. "
                        f"Skipped manually updated: {status_counts['skipped_manual']}. "
                        f"No regex match: {status_counts['no_match']}."
                    ),
                )
                return redirect(return_url)
    else:
        form = forms.MediaFileFilenameMetadataForm(
            initial={
                "path_regex": r"^(?:.*/)?(?P<locality>[^/]+)/(?P<unique_name>[^/]+)/[^/]+$",
            }
        )

    sample_mediafiles = list(mediafiles.order_by("id")[:10])
    sample_paths = [
        (mediafile.original_filename or mediafile.mediafile.name or "").replace("\\", "/")
        for mediafile in sample_mediafiles[:5]
    ]
    regex_chatgpt_prompt = _build_filename_metadata_regex_prompt(sample_paths)
    example_path = sample_paths[0] if sample_paths else ""
    example_filename = str(example_path or "").replace("\\", "/").split("/")[-1] if example_path else ""

    return render(
        request,
        "caidapp/mediafiles_filename_metadata.html",
        {
            "form": form,
            "mediafile_count": mediafile_count,
            "manual_count": mediafiles.exclude(updated_by__isnull=True).count(),
            "return_url": return_url,
            "source_label": source_label,
            "sample_mediafiles": sample_mediafiles,
            "example_path": example_path,
            "example_filename": example_filename,
            "example_directory_parts": _directory_parts_from_path(example_path),
            "regex_chatgpt_prompt": regex_chatgpt_prompt,
            "regex_chatgpt_url": f"https://chatgpt.com/?q={urllib.parse.quote(regex_chatgpt_prompt)}",
            "path_regex_chatgpt_prompt_prefix_lines": PATH_REGEX_CHATGPT_PROMPT_PREFIX_LINES,
            "path_regex_chatgpt_prompt_suffix": PATH_REGEX_CHATGPT_PROMPT_SUFFIX,
        },
    )


@login_required
def sequences(
    request,
    records_per_page: Optional[int] = None,
    album_hash=None,
    individual_identity_id=None,
    uploadedarchive_id=None,
    identity_is_representative=None,
    locality_hash=None,
    show_overview_button=False,
    taxon_verified: Optional[bool] = None,
    **filter_kwargs,
) -> HttpResponse:
    """List sequences with inline mediafile expansion."""
    logger.debug("Starting Sequence view")
    if not show_overview_button:
        show_overview_button = bool(_parse_bool_query_param(request.GET.get("show_overview_button")))
    if taxon_verified is None:
        taxon_verified = _parse_bool_query_param(request.GET.get("taxon_verified"))
    view_mode = request.GET.get("view", "cards")
    if view_mode not in {"cards", "list"}:
        view_mode = "cards"
    if show_overview_button:
        view_mode = "cards"
    records_per_page = _get_sequence_records_per_page(request, records_per_page)
    sort_key = _get_sequence_sort(request)

    albums_available = (
        Album.objects.filter(Q(albumsharerole__user=request.user.caiduser) | Q(owner=request.user.caiduser))
        .distinct()
        .order_by("created_at")
    )

    full_mediafiles, mediafile_filter, page_title, mediafiles_name_suggestion = _get_filtered_mediafiles_queryset(
        request,
        uploadedarchive_id=uploadedarchive_id,
        album_hash=album_hash,
        individual_identity_id=individual_identity_id,
        identity_is_representative=identity_is_representative,
        locality_hash=locality_hash,
        show_overview_button=show_overview_button,
        taxon_verified=taxon_verified,
        extra_filter_kwargs=filter_kwargs,
    )
    active_uploadedarchive = _get_active_uploadedarchive_from_request(request, uploadedarchive_id)
    active_taxon = _get_active_taxon_from_request(request)
    active_album = _get_active_album_from_request(request, album_hash)
    active_identity = _get_active_identity_from_request(request, individual_identity_id)
    active_identities = _get_active_identities_from_request(request)
    active_locality = _get_active_locality_from_request(request, locality_hash)

    sequence_queryset = _get_sequences_queryset_from_mediafiles(full_mediafiles)
    if show_overview_button:
        first_observed_taxon_name = (
            AnimalObservation.objects.filter(
                mediafile__sequence=OuterRef("pk"),
                mediafile__in=full_mediafiles,
                taxon__isnull=False,
            )
            .order_by("taxon__name", "taxon_id", "id")
            .values("taxon__name")[:1]
        )
        sequence_queryset = sequence_queryset.annotate(
            verification_taxon_sort=Coalesce(
                Subquery(first_observed_taxon_name),
                Value("zzzzzz_no_taxon", output_field=CharField()),
            )
        ).order_by("verification_taxon_sort", "first_captured_at", "first_mediafile_id", "pk")
    else:
        sequence_queryset = sequence_queryset.order_by(*SEQUENCE_SORT_OPTIONS[sort_key]["order_by"])
    paginator = Paginator(sequence_queryset, per_page=records_per_page)
    page_with_sequences, _, page_context = _prepare_page(
        paginator,
        request=request,
    )

    page_sequence_ids = [obj.id for obj in page_with_sequences.object_list]
    sequence_mediafiles = (
        MediaFile.objects.filter(sequence_id__in=page_sequence_ids)
        .select_related("parent", "taxon", "predicted_taxon", "locality", "identity", "updated_by", "sequence")
        .prefetch_related("observations__taxon")
        .order_by("captured_at", "id")
    )
    page_sequences = (
        models.Sequence.objects.filter(id__in=page_sequence_ids)
        .annotate(
            first_captured_at=Subquery(
                full_mediafiles.exclude(sequence__isnull=True)
                .filter(sequence=OuterRef("pk"))
                .order_by("captured_at", "id")
                .values("captured_at")[:1]
            ),
            first_mediafile_id=Subquery(
                full_mediafiles.exclude(sequence__isnull=True)
                .filter(sequence=OuterRef("pk"))
                .order_by("captured_at", "id")
                .values("id")[:1]
            ),
            mediafile_count=Count("mediafile"),
        )
        .prefetch_related(Prefetch("mediafile_set", queryset=sequence_mediafiles))
        .order_by("first_captured_at", "first_mediafile_id", "pk")
    )
    sequence_by_id = {sequence.id: sequence for sequence in page_sequences}
    ordered_sequences = [sequence_by_id[sequence_id] for sequence_id in page_sequence_ids if sequence_id in sequence_by_id]
    matching_mediafile_ids_on_page = set(
        full_mediafiles.filter(sequence_id__in=page_sequence_ids).values_list("id", flat=True)
    )

    sequence_lookup = {sequence.id: sequence for sequence in ordered_sequences}
    for sequence in ordered_sequences:
        mediafiles_in_sequence = list(sequence.mediafile_set.all())
        sequence.cover_mediafile = mediafiles_in_sequence[0] if mediafiles_in_sequence else None
        sequence.has_multiple_taxa = len({mf.taxon_id for mf in mediafiles_in_sequence if mf.taxon_id}) > 1
        sequence.has_multiple_identities = len({mf.identity_id for mf in mediafiles_in_sequence if mf.identity_id}) > 1
        locality_counts = {}
        for mediafile in mediafiles_in_sequence:
            mediafile.matches_current_filter = mediafile.id in matching_mediafile_ids_on_page
            if mediafile.locality is None:
                continue
            locality_counts[mediafile.locality] = locality_counts.get(mediafile.locality, 0) + 1
        sequence.localities = [
            locality
            for locality, _count in sorted(locality_counts.items(), key=lambda item: (-item[1], item[0].name, item[0].id))
        ]
        sequence.primary_locality = sequence.localities[0] if sequence.localities else None
        sequence.additional_localities = sequence.localities[1:]
        sequence.has_multiple_localities = len(sequence.localities) > 1
    if show_overview_button:
        _annotate_verification_sequence_taxon_groups(ordered_sequences)
    page_mediafile_ids = [
        mediafile.id
        for sequence in ordered_sequences
        for mediafile in sequence.mediafile_set.all()
    ]
    request.session["mediafile_ids_page"] = page_mediafile_ids

    form_bulk_processing = MediaFileBulkForm(request.POST or None)

    if request.method == "POST" and any(
        (isinstance(key, str)) and key.startswith("btnBulkProcessing") for key in request.POST
    ):
        if form_bulk_processing.is_valid():
            selected_mediafile_ids = _resolve_selected_mediafile_ids_from_post(request)
            request.session["mediafile_ids"] = selected_mediafile_ids
            request.session["mediafiles_name_suggestion"] = mediafiles_name_suggestion
            selected_mediafiles = MediaFile.objects.filter(id__in=selected_mediafile_ids)
            selected_album_hash = request.POST.get("selectAlbum", "")
            for mediafile in selected_mediafiles:
                _single_mediafile_update(
                    request,
                    mediafile,
                    form_bulk_processing,
                    form_bulk_processing,
                    selected_album_hash,
                )
            return redirect(request.get_full_path())

    if request.method == "POST" and "btnDissolveSequences" in request.POST:
        selected_mediafile_ids = _resolve_selected_mediafile_ids_from_post(request)
        if not selected_mediafile_ids:
            messages.warning(request, "Select at least one sequence or media file to dissolve.")
            return redirect(request.get_full_path())

        dissolved_mediafile_count = _dissolve_mediafiles_into_singleton_sequences(
            request.user.caiduser,
            selected_mediafile_ids,
        )
        if dissolved_mediafile_count:
            messages.success(
                request,
                f"Dissolved {dissolved_mediafile_count} media files into single-media sequences.",
            )
        else:
            messages.info(request, "Selected media files are already in single-media sequences.")
        return redirect(request.get_full_path())

    if request.method == "POST" and "btnExtractFilenameMetadata" in request.POST:
        selected_mediafile_ids = _resolve_selected_mediafile_ids_from_post(request)
        if not selected_mediafile_ids:
            selected_mediafile_ids = list(full_mediafiles.values_list("id", flat=True))
        return _start_filename_metadata_session(
            request,
            selected_mediafile_ids,
            request.get_full_path(),
            "Sequences",
        )

    if request.method == "POST" and "btnDownloadSequences" in request.POST:
        selected_mediafile_ids = _resolve_selected_mediafile_ids_from_post(request)
        if not selected_mediafile_ids:
            selected_mediafile_ids = list(full_mediafiles.values_list("id", flat=True))
        request.session[SEQUENCE_DOWNLOAD_SESSION_KEY] = selected_mediafile_ids
        request.session[SEQUENCE_DOWNLOAD_RETURN_URL_SESSION_KEY] = request.get_full_path()
        return redirect("caidapp:download_sequences")

    context = {
        **page_context,
        "page_title": page_title.replace("Media files", "Sequences"),
        "user_is_staff": request.user.is_staff,
        "form_bulk_processing": form_bulk_processing,
        "albums_available": albums_available,
        "number_of_sequences": sequence_queryset.count(),
        "number_of_mediafiles": full_mediafiles.count(),
        "show_overview_button": show_overview_button,
        "verification_mediafiles_url": reverse("caidapp:media_files")
        + "?"
        + _build_mediafiles_scope_query_string(
            request,
            uploadedarchive_id=uploadedarchive_id,
            album_hash=album_hash,
            individual_identity_id=individual_identity_id,
            identity_is_representative=identity_is_representative,
            locality_hash=locality_hash,
            show_overview_button=show_overview_button,
            taxon_verified=taxon_verified,
        ),
        "filter": mediafile_filter,
        "view_mode": view_mode,
        "records_per_page": records_per_page,
        "sort_key": sort_key,
        "sequence_sort_options": SEQUENCE_SORT_OPTIONS,
        "sequence_per_page_options": SEQUENCE_PER_PAGE_OPTIONS,
        "sequence_objects": ordered_sequences,
        "sequence_lookup": sequence_lookup,
        "active_uploadedarchive": active_uploadedarchive,
        "active_taxon": active_taxon,
        "active_album": active_album,
        "active_identity": active_identity,
        "active_identities": active_identities,
        "active_locality": active_locality,
        "matching_mediafile_ids_on_page": sorted(matching_mediafile_ids_on_page),
        "has_active_sequence_search": bool(request.GET.get("search")),
        "sequences_stats_query_string": _build_sequence_scope_query_string(
            request,
            uploadedarchive_id=uploadedarchive_id,
            album_hash=album_hash,
            individual_identity_id=individual_identity_id,
            identity_is_representative=identity_is_representative,
            locality_hash=locality_hash,
            show_overview_button=show_overview_button,
            taxon_verified=taxon_verified,
        ),
    }
    context = add_querystring_to_context(request, context)
    return render(request, "caidapp/sequences.html", context)


@login_required
def media_files_update(
    request,
    records_per_page: Optional[int] = None,
    album_hash=None,
    individual_identity_id=None,
    uploadedarchive_id=None,
    identity_is_representative=None,
    locality_hash=None,
    show_overview_button=False,
    order_by=None,
    taxon_verified: Optional[bool] = None,
    **filter_kwargs,
) -> HttpResponse:
    """List of mediafiles based on query with bulk update of category."""
    # create list of mediafiles
    logger.debug("Starting Media files view")
    logger.debug(f"{request.GET=}")
    if records_per_page is None:
        records_per_page = request.session.get("mediafiles_records_per_page", 20)

    albums_available = (
        Album.objects.filter(Q(albumsharerole__user=request.user.caiduser) | Q(owner=request.user.caiduser))
        .distinct()
        .order_by("created_at")
    )

    # Order the queryset according to the view default, session, or fallback preference.
    if order_by is None:
        order_by = request.session.get("mediafiles_order_by", "-parent__uploaded_at")
    logger.debug("Selecting related")
    # Nová filtrace
    # Build the base queryset (including annotations)

    # mediafiles = MediaFile.objects.annotate(**_mediafiles_annotate())
    # Apply always-on filters (for example, access control)
    full_mediafiles, mediafile_filter, page_title, mediafiles_name_suggestion = _get_filtered_mediafiles_queryset(
        request,
        uploadedarchive_id=uploadedarchive_id,
        album_hash=album_hash,
        individual_identity_id=individual_identity_id,
        identity_is_representative=identity_is_representative,
        locality_hash=locality_hash,
        show_overview_button=show_overview_button,
        taxon_verified=taxon_verified,
        extra_filter_kwargs=filter_kwargs,
    )

    if show_overview_button and not full_mediafiles.exists():
        return message_view(
            request,
            "No mediafiles for verification.",
            headline="Verification",
            link=reverse_lazy("caidapp:uploads"),
        )

    # konec nové filtrace
    if show_overview_button:
        first_observed_taxon_name = (
            AnimalObservation.objects.filter(mediafile=OuterRef("pk"), taxon__isnull=False)
            .order_by("taxon__name", "taxon_id", "id")
            .values("taxon__name")[:1]
        )
        full_mediafiles = full_mediafiles.annotate(
            verification_taxon_sort=Coalesce(
                Subquery(first_observed_taxon_name),
                Value("zzzzzz_no_taxon", output_field=CharField()),
            )
        ).order_by("verification_taxon_sort", order_by, "id")
    else:
        full_mediafiles = full_mediafiles.order_by(order_by)

    full_mediafiles = full_mediafiles.select_related(
        "parent", "taxon", "predicted_taxon", "locality", "identity", "updated_by", "sequence"
    ).prefetch_related("observations__taxon")

    number_of_mediafiles = full_mediafiles.count()
    logger.debug(f"{number_of_mediafiles=}")

    mediafiles_ids = list(full_mediafiles.values_list("id", flat=True))
    # logger.debug(f"{mediafiles_ids=}")
    request.session["mediafile_ids"] = mediafiles_ids
    request.session["mediafiles_name_suggestion"] = mediafiles_name_suggestion
    paginator = Paginator(full_mediafiles, per_page=records_per_page)
    page_with_mediafiles, _, page_context = _prepare_page(
        paginator,
        request=request,
        # page_number=page_number
    )

    page_ids = [obj.id for obj in page_with_mediafiles.object_list]
    request.session["mediafile_ids_page"] = page_ids

    MediaFileFormSet = modelformset_factory(MediaFile, form=MediaFileSelectionForm, extra=0)
    logger.debug("Processing POST or GET request")
    if (request.method == "POST") and (
        any([(isinstance(key, str)) and (key.startswith("btnBulkProcessing")) for key in request.POST])
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

            select_all_in_the_pages = True if form.data.get("select_all", "") == "on" else False
            logger.debug(f"{select_all_in_the_pages=}")
            if "btnBulkProcessingAlbum" in form.data:
                if selected_album_hash == "new":
                    logger.debug("Creating new album")
                    logger.debug("Select Album :" + form.data["selectAlbum"])
                    album = create_new_album(request)
                    selected_album_hash = album.hash

            if select_all_in_the_pages:
                # selected all m media file processing
                for mediafile in full_mediafiles:

                    _single_mediafile_update(request, mediafile, form, form_bulk_processing, selected_album_hash)
                    # album.cover = mediafile
                    # album.save()
            else:
                for mediafileform in form:
                    # go over selected mediafiles
                    if mediafileform.is_valid():
                        if mediafileform.cleaned_data["selected"]:
                            logger.debug("mediafileform is valid")
                            # reset selected field for refreshed view
                            mediafileform.cleaned_data["selected"] = False
                            mediafileform.selected = False
                            instance: MediaFile = mediafileform.save(commit=False)
                            _single_mediafile_update(request, instance, form, form_bulk_processing, selected_album_hash)
                            # album.cover = instance
                            # album.save()

            if "btnBulkProcessingAlbum" in form.data:
                if selected_album_hash == "new":
                    album.cover = album.medifile_set.first()
                    album.save()

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

    if show_overview_button:
        _annotate_verification_taxon_groups(form)

    logger.debug("Setting the context for rendering the page")
    context = {
        # "page_obj": page_with_mediafiles,
        # "elided_page_range": elided_page_range,
        **page_context,
        "form_objects": form,
        "page_title": page_title,
        "user_is_staff": request.user.is_staff,
        "form_bulk_processing": form_bulk_processing,
        # "form_query": queryform,
        "albums_available": albums_available,
        "number_of_mediafiles": number_of_mediafiles,
        "show_overview_button": show_overview_button,
        "filter": mediafile_filter,
        "mediafiles_stats_query_string": _build_mediafiles_scope_query_string(
            request,
            uploadedarchive_id=uploadedarchive_id,
            album_hash=album_hash,
            individual_identity_id=individual_identity_id,
            identity_is_representative=identity_is_representative,
            locality_hash=locality_hash,
            show_overview_button=show_overview_button,
            taxon_verified=taxon_verified,
        ),
        "mediafiles_download_query_string": _build_mediafiles_scope_query_string(
            request,
            uploadedarchive_id=uploadedarchive_id,
            album_hash=album_hash,
            individual_identity_id=individual_identity_id,
            identity_is_representative=identity_is_representative,
            locality_hash=locality_hash,
            show_overview_button=show_overview_button,
            taxon_verified=taxon_verified,
        ),
        # "map_html": map_html,
        # "taxon_stats_html": taxon_stats_html,
    }
    context = add_querystring_to_context(request, context)
    logger.debug("ready to render page")

    return render(request, "caidapp/media_files_update.html", context)


def _single_mediafile_update(request, instance, form, form_bulk_processing, selected_album_hash):
    # logger.debug(f"{instance=}")
    # logger.debug(f"{instance.id=}")
    # logger.debug(f"{form.data=}")
    # logger.debug(f"{len(form.data)=}")
    if len(form.data) > 0:
        items = list(form.data.items())
        if items and len(items) > 0:
            logger.debug(f"{items[0]=} ... {items[-1]=}")
        else:
            logger.debug("No data found in form.")

    if "btnBulkProcessingAlbum" in form.data:
        if selected_album_hash == "new":
            logger.debug("Creating new album")
            logger.debug("Select Album :" + form.data["selectAlbum"])
            album = create_new_album(request)
            album.cover = instance
            album.save()
            instance.album_set.add(album)
            instance.save()
            selected_album_hash = album.hash
        else:
            # logger.debug("selectAlbum")
            # logger.debug(f"{selected_album_hash=}")
            album = get_object_or_404(Album, hash=selected_album_hash)

            # check if file is not already in album
            if instance.album_set.filter(pk=album.pk).count() == 0:
                # add file to album
                instance.album_set.add(album)
                instance.save()
    elif "btnBulkProcessing_id_taxon" in form.data:
        observation = instance.first_observation_get_or_create
        observation.taxon = form_bulk_processing.cleaned_data["taxon"]
        instance.taxon = observation.taxon
        instance.updated_by = request.user.caiduser
        instance.updated_at = django.utils.timezone.now()
        instance.save()
        observation.save()
    elif "btnBulkProcessing_id_identity" in form.data:
        observation = instance.first_observation_get_or_create
        observation.identity = form_bulk_processing.cleaned_data["identity"]
        instance.identity = observation.identity
        # instance.identity_is_representative = False
        instance.updated_by = request.user.caiduser
        instance.updated_at = django.utils.timezone.now()
        instance.save()
    elif "btnBulkProcessing_id_identity_is_representative" in form.data:
        observation = instance.first_observation_get_or_create
        observation.identity_is_representative = form_bulk_processing.cleaned_data["identity_is_representative"]
        instance.identity_is_representative = observation.identity_is_representative
        instance.updated_by = request.user.caiduser
        instance.updated_at = django.utils.timezone.now()
        instance.save()
    elif "btnBulkProcessingDelete" in form.data:
        instance.delete()
    elif "btnBulkProcessing_id_taxon_verified" in form.data:
        observation = instance.first_observation_get_or_create
        observation.taxon_verified = form_bulk_processing.cleaned_data["taxon_verified"]
        instance.taxon_verified = observation.taxon_verified
        instance.updated_by = request.user.caiduser
        instance.updated_at = django.utils.timezone.now()
        instance.save()

    elif "btnBulkProcessing_set_taxon_verified" in form.data:
        observations = list(instance.observations.all())
        if not observations:
            observations = [instance.first_observation_get_or_create]
        for observation in observations:
            observation.taxon_verified = True
            observation.save()
        # observation = instance.first_observation_get_or_create
        # observation.taxon_verified = True
        instance.taxon_verified = True
        instance.updated_by = request.user.caiduser
        instance.updated_at = django.utils.timezone.now()
        instance.save()


from dateutil.relativedelta import relativedelta  # Import relativedelta


@login_required
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
                prev_url = request.META.get("HTTP_REFERER", "/")
                # next_url = reverse_lazy("caidapp:")
                next_url = prev_url
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


@login_required
def mediafiles_stats_view(request):
    """Show mediafiles stats."""
    mediafiles, _, _, _ = _get_filtered_mediafiles_queryset(request)

    map_html = views_locality.create_map_from_mediafiles(mediafiles)
    # logger.debug(f"{map_html=}")
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
    workgroup = request.user.caiduser.workgroup
    next_url = request.GET.get("next") or request.POST.get("next") or reverse("caidapp:uploads_ready_for_identification")
    if request.method == "POST":
        form = UploadedArchiveSelectTaxonForIdentificationForm(request.POST)
        if form.is_valid():
            taxon = form.cleaned_data["taxon_for_identification"]
            uploaded_archive.taxon_for_identification = taxon
            uploaded_archive.identification_status = "IR"  # Ready for identification
            uploaded_archive.is_for_identification = True
            uploaded_archive.save()
            return redirect(next_url)
    else:
        initial_taxon = uploaded_archive.taxon_for_identification
        if initial_taxon is None and workgroup:
            initial_taxon = workgroup.default_taxon_for_identification
        form = UploadedArchiveSelectTaxonForIdentificationForm(initial={"taxon_for_identification": initial_taxon})
    return render(
        request,
        "caidapp/update_form.html",
        {
            "form": form,
            "headline": "Send to identification",
            "button": "Send to identification",
            "text_note": (
                "This screen sends media files with the selected taxon into the identification workflow. "
                "Only images whose observation taxon matches this selection will be used for identification."
            ),
            "cancel_button_url": next_url,
            "mediafile": uploaded_archive.mediafile_set.all().first(),
        },
    )


@login_required
def create_new_album(request, name="New Album"):
    """Create new album."""
    album = Album()
    album.name = name
    album.owner = request.user.caiduser
    album.save()
    return album


class WorkgroupAdminRequiredMixin(UserPassesTestMixin):
    def test_func(self):
        """Check if user is workgroup admin."""
        return self.request.user.caiduser.workgroup_admin


class WorkgroupUpdateView(WorkgroupAdminRequiredMixin, UpdateView):
    model = WorkGroup
    form_class = forms.WorkgroupForm
    template_name = "caidapp/update_form.html"
    # go_back = request.META.get("HTTP_REFERER", "/")
    success_url = reverse_lazy("caidapp:home")

    def get_object(self, queryset=None):
        """Get the workgroup object to be updated."""
        # workgroup_hash = self.kwargs.get("workgroup_hash")
        # return get_object_or_404(WorkGroup, hash=workgroup_hash)
        return self.request.user.caiduser.workgroup

    def form_valid(self, form):
        """If the form is valid, save the associated model."""
        response = super().form_valid(form)
        # Additional processing can be done here if needed
        return response

    def get_context_data(self, **kwargs):
        """Get context data for the template."""
        context = super().get_context_data(**kwargs)
        context["headline"] = "Update workgroup"
        context["button"] = "Save"
        context["nav_dict"] = {
            "Invitations": reverse_lazy("caidapp:workgroup_invitations"),
            "Invite User": reverse_lazy("caidapp:workgroup_invitation"),
        }
        return context


# remove, depreceated
# @login_required
# def workgroup_update(request, workgroup_hash: str):
#     """Update workgroup."""
#     workgroup = get_object_or_404(WorkGroup, hash=workgroup_hash)
#     if request.method == "POST":
#         form = WorkgroupUsersForm(request.POST)
#         logger.debug(request.POST)
#         logger.debug(form)
#         if form.is_valid():
#             logger.debug(form.cleaned_data)
#             workgroup_users_all = workgroup.caiduser_set.all()
#             logger.debug(f"Former all users {workgroup_users_all}")
#             workgroup.caiduser_set.set(form.cleaned_data["workgroup_users"])
#
#             pass
#             # logger
#             # form.save()
#             # return redirect("workgroup_list")
#     else:
#
#         workgroup_users = workgroup.caiduser_set.all()
#         data = {
#             # 'id': dog_request_id,
#             # 'color': dog_color,
#             "workgroup_users": workgroup_users,
#         }
#         form = WorkgroupUsersForm(data)
#         # form = WorkgroupUsersForm(instance=workgroup.)
#     return render(
#         request,
#         "caidapp/update_form.html",
#         {
#             "form": form,
#             "headline": "Update workgroup",
#             "button": "Save",
#             # "user_is_staff": request.user.is_staff,
#         },
#     )
#     return render(request, "caidapp/update_form.html", {"form": workgroup_hash})


def _update_csv_by_uploadedarchive(request, uploadedarchive_id: int):
    uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)

    if uploaded_archive.owner.workgroup == request.user.caiduser.workgroup:
        updated_at = uploaded_archive.output_updated_at
        logger.debug(f"{updated_at=}")
        if updated_at is None:
            # set updated_at to old date
            updated_at = datetime.datetime(2000, 1, 1, 0, 0, 0, 0).replace(tzinfo=ZoneInfo(settings.TIME_ZONE))
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


@login_required
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


@login_required
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


def _get_mediafiles_for_export(request, uploadedarchive_id: Optional[int]) -> Tuple[QuerySet, Optional[str]]:
    """Get mediafiles for export based on explicit URL scope and filters."""
    mediafiles, _, _, name_suggestion = _get_filtered_mediafiles_queryset(
        request,
        uploadedarchive_id=uploadedarchive_id,
    )
    mediafiles = mediafiles.select_related("parent", "locality").prefetch_related("observations__taxon", "observations__identity")
    return mediafiles, name_suggestion


def _sanitize_export_component(value: Optional[object], default: str = "unknown") -> str:
    """Sanitize a single path component used in exported filenames."""
    text = str(value).strip() if value is not None else ""
    if not text:
        text = default
    text = model_tools.remove_diacritics(text)
    text = re.sub(r'[<>:"/\\\\|?*\x00-\x1f]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip(" ._")
    if text in {"", ".", ".."}:
        return default
    return text


def _build_export_value(unique_values: List[str], unknown_label: str, mixed_label: str) -> Tuple[str, str]:
    """Return aggregate export value and joined list representation."""
    sanitized_values = sorted({_sanitize_export_component(value, default=unknown_label) for value in unique_values if value})
    if not sanitized_values:
        return unknown_label, unknown_label
    if len(sanitized_values) == 1:
        return sanitized_values[0], sanitized_values[0]
    return mixed_label, "+".join(sanitized_values)


def _get_mediafile_export_context(mediafile: models.MediaFile) -> Dict[str, str]:
    """Return filename template context based on mediafile and its observations."""
    observation_taxa = []
    observation_identities = []
    for observation in mediafile.observations.all():
        if observation.taxon_id and observation.taxon:
            observation_taxa.append(observation.taxon.name)
        if observation.identity_id and observation.identity:
            observation_identities.append(observation.identity.name)

    if not observation_taxa and mediafile.taxon_id and mediafile.taxon:
        observation_taxa.append(mediafile.taxon.name)
    if not observation_identities and mediafile.identity_id and mediafile.identity:
        observation_identities.append(mediafile.identity.name)

    species, species_list = _build_export_value(observation_taxa, "unknown_species", "mixed_species")
    identity, identity_list = _build_export_value(observation_identities, "unknown_identity", "mixed_identity")

    mediafile_name = Path(mediafile.mediafile.name) if mediafile.mediafile else Path(str(mediafile.pk))
    original_stem = Path(mediafile.original_filename).stem if mediafile.original_filename else mediafile_name.stem
    extension = mediafile_name.suffix or Path(mediafile.original_filename).suffix
    locality = _sanitize_export_component(mediafile.locality.name if mediafile.locality else None, default="unknown_locality")
    captured_date = mediafile.captured_at.strftime("%Y-%m-%d") if mediafile.captured_at else "unknown_date"

    return {
        "hash": _sanitize_export_component(mediafile_name.stem or mediafile.pk, default=str(mediafile.pk)),
        "species": species,
        "identity": identity,
        "species_list": species_list,
        "identity_list": identity_list,
        "ext": extension.lstrip("."),
        "dotext": extension,
        "original_name": _sanitize_export_component(original_stem, default="original"),
        "locality": locality,
        "date": captured_date,
        "mediafile_id": str(mediafile.pk),
    }


def _render_mediafile_export_path(template: str, mediafile: models.MediaFile) -> str:
    """Render and sanitize export path for a single mediafile."""
    formatter = Formatter()
    context = _get_mediafile_export_context(mediafile)
    try:
        rendered = formatter.vformat(template, args=(), kwargs=context)
    except KeyError as exc:
        raise ValueError(f"Unknown export placeholder: {exc.args[0]}") from exc

    normalized_parts = []
    for part in rendered.replace("\\", "/").split("/"):
        if not part or part == ".":
            continue
        normalized_parts.append(_sanitize_export_component(part, default="item"))

    if not normalized_parts:
        raise ValueError("Export template produced an empty path.")
    return "/".join(normalized_parts)


def _get_mediafile_export_template(request: HttpRequest) -> str:
    """Resolve export template from predefined schema or explicit custom template."""
    export_scheme = request.GET.get("export_scheme", "species_identity")
    export_path_template = request.GET.get("export_path_template", "").strip().replace("{.ext}", "{dotext}")

    if export_path_template:
        return export_path_template
    if export_scheme == "custom":
        raise ValueError("Custom export template is empty.")
    if export_scheme in MEDIAFILE_EXPORT_SCHEMAS:
        return MEDIAFILE_EXPORT_SCHEMAS[export_scheme]
    raise ValueError(f"Unknown export scheme: {export_scheme}")


def _build_export_mediafiles_data(request: HttpRequest, mediafiles: QuerySet) -> List[Dict[str, str]]:
    """Prepare mediafiles list for ZIP export with unique output paths."""
    template = _get_mediafile_export_template(request)
    seen_paths: Dict[str, int] = {}
    mediafiles_data = []
    for mediafile in mediafiles:
        output_name = _render_mediafile_export_path(template, mediafile)
        count = seen_paths.get(output_name, 0)
        seen_paths[output_name] = count + 1
        if count:
            output_path = Path(output_name)
            output_name = str(output_path.with_name(f"{output_path.stem}__{count + 1}{output_path.suffix}")).replace(
                "\\", "/"
            )
        mediafiles_data.append({"path": mediafile.mediafile.name, "output_name": output_name})
    return mediafiles_data


def _get_sequence_download_mediafiles(request: HttpRequest) -> QuerySet:
    """Return mediafiles selected for the sequence download workflow."""
    mediafile_ids = request.session.get(SEQUENCE_DOWNLOAD_SESSION_KEY, [])
    return (
        MediaFile.objects.for_user(request.user.caiduser)
        .filter(id__in=mediafile_ids)
        .select_related("parent", "locality", "sequence", "taxon", "predicted_taxon", "identity")
        .prefetch_related("observations__taxon", "observations__predicted_taxon", "observations__identity")
        .order_by("sequence_id", "captured_at", "id")
    )


def _identity_export_values(identity: Optional[models.IndividualIdentity]) -> Dict[str, str]:
    if identity is None:
        return {"unique_name": "", "code": "", "juv_code": ""}
    return {
        "unique_name": identity.name or "",
        "code": identity.code or "",
        "juv_code": identity.juv_code or "",
    }


def _location_export_values(mediafile: models.MediaFile) -> Dict[str, str]:
    effective_location = mediafile.effective_location
    if effective_location and "," in str(effective_location):
        latitude, longitude = [part.strip() for part in str(effective_location).split(",", 1)]
    else:
        latitude, longitude = "", ""
    return {
        "locality coordinates": str(effective_location) if effective_location else "",
        "latitude": latitude,
        "longitude": longitude,
    }


def _build_sequence_observation_export_records(
    mediafiles: QuerySet,
    request: Optional[HttpRequest] = None,
    columns: Optional[List[str]] = None,
    include_export_path: bool = True,
) -> List[Dict[str, object]]:
    """Build observation-level export records from mediafiles."""
    selected_columns = columns or SEQUENCE_EXPORT_DEFAULT_COLUMNS
    mediafile_export_paths = {}
    if include_export_path and request is not None:
        for mediafile in mediafiles:
            mediafile_export_paths[mediafile.id] = _render_mediafile_export_path(_get_mediafile_export_template(request), mediafile)

    records = []
    for mediafile in mediafiles:
        observations = list(mediafile.observations.all()) or [None]
        mediafile_location_values = _location_export_values(mediafile)
        for observation in observations:
            taxon = observation.taxon if observation and observation.taxon_id else mediafile.taxon
            predicted_taxon = (
                observation.predicted_taxon
                if observation and observation.predicted_taxon_id
                else mediafile.predicted_taxon
            )
            identity = observation.identity if observation and observation.identity_id else mediafile.identity
            row = {
                "mediafile_id": mediafile.id,
                "observation_id": observation.id if observation else "",
                "original_path": mediafile.original_filename or mediafile.mediafile.name,
                "export_path": mediafile_export_paths.get(mediafile.id, ""),
                "uploaded_archive": mediafile.parent.name if mediafile.parent else "",
                "sequence_id": mediafile.sequence_id or "",
                "datetime": mediafile.captured_at.isoformat() if mediafile.captured_at else "",
                "media_type": mediafile.media_type,
                "locality name": mediafile.locality.name if mediafile.locality else "",
                "predicted_category": taxon.name if taxon else "",
                "taxon_verified": observation.taxon_verified if observation else mediafile.taxon_verified,
                "predicted_taxon": predicted_taxon.name if predicted_taxon else "",
                "predicted_taxon_confidence": (
                    observation.predicted_taxon_confidence
                    if observation and observation.predicted_taxon_confidence is not None
                    else mediafile.predicted_taxon_confidence
                ),
                "identity_is_representative": (
                    observation.identity_is_representative if observation else mediafile.identity_is_representative
                ),
                "orientation": observation.orientation if observation else mediafile.orientation,
                "bbox_x_center": observation.bbox_x_center if observation else "",
                "bbox_y_center": observation.bbox_y_center if observation else "",
                "bbox_width": observation.bbox_width if observation else "",
                "bbox_height": observation.bbox_height if observation else "",
                "note": mediafile.note,
                "locality_check_at": (
                    mediafile.parent.locality_check_at.isoformat()
                    if mediafile.parent and mediafile.parent.locality_check_at
                    else ""
                ),
            }
            row.update(mediafile_location_values)
            row.update(_identity_export_values(identity))
            records.append({column: row.get(column, "") for column in selected_columns})
    return records


def _sequence_export_dataframe(mediafiles: QuerySet, request: HttpRequest, columns: Optional[List[str]] = None) -> pd.DataFrame:
    records = _build_sequence_observation_export_records(mediafiles, request=request, columns=columns)
    return pd.DataFrame.from_records(records, columns=columns or SEQUENCE_EXPORT_DEFAULT_COLUMNS)


def _get_sequence_export_columns(request: HttpRequest) -> List[str]:
    requested_columns = request.GET.getlist("columns")
    valid_columns = [column for column, _label in SEQUENCE_EXPORT_COLUMNS]
    selected_columns = [column for column in valid_columns if column in requested_columns]
    return selected_columns or SEQUENCE_EXPORT_DEFAULT_COLUMNS


@login_required
def download_sequences_view(request) -> HttpResponse:
    """Configure downloads created from the current Sequences selection."""
    mediafiles = _get_sequence_download_mediafiles(request)
    mediafile_count = mediafiles.count()
    return_url = request.session.get(SEQUENCE_DOWNLOAD_RETURN_URL_SESSION_KEY) or reverse_lazy("caidapp:sequences")
    if mediafile_count == 0:
        return message_view(
            request,
            "No media files were selected for download.",
            headline="Download sequences",
            link=return_url,
            button_label="Back to sequences",
        )

    return render(
        request,
        "caidapp/sequences_download.html",
        {
            "mediafile_count": mediafile_count,
            "return_url": return_url,
            "export_columns": SEQUENCE_EXPORT_COLUMNS,
            "default_export_columns": SEQUENCE_EXPORT_DEFAULT_COLUMNS,
            "export_schemas": MEDIAFILE_EXPORT_SCHEMAS,
        },
    )


@login_required
def download_csv_for_sequences_view(request) -> HttpResponse:
    """Download observation-level CSV from the current sequence download selection."""
    mediafiles = _get_sequence_download_mediafiles(request)
    columns = _get_sequence_export_columns(request)
    df = _sequence_export_dataframe(mediafiles, request, columns)
    if df.empty:
        return HttpResponse("No data available to export.", content_type="text/plain")

    response = HttpResponse(df.to_csv(index=False), content_type="text/csv")
    response["Content-Disposition"] = "attachment; filename=sequence_metadata.csv"
    return response


@login_required
def download_xlsx_for_sequences_view(request) -> HttpResponse:
    """Download observation-level XLSX from the current sequence download selection."""
    mediafiles = _get_sequence_download_mediafiles(request)
    columns = _get_sequence_export_columns(request)
    df = _sequence_export_dataframe(mediafiles, request, columns)
    if df.empty:
        return HttpResponse("No data available to export.", content_type="text/plain")

    df = model_tools.convert_datetime_to_naive(df)
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Media files")
    output.seek(0)

    response = HttpResponse(output, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response["Content-Disposition"] = "attachment; filename=sequence_metadata.xlsx"
    return response


@login_required
def download_zip_for_sequences_view(request) -> JsonResponse:
    """Prepare ZIP with selected sequence media files and observation-level metadata."""
    mediafiles = _get_sequence_download_mediafiles(request)
    if not mediafiles.exists():
        return JsonResponse({"message": "No media files were selected for download."}, status=400)

    try:
        mediafiles_data = _build_export_mediafiles_data(request, mediafiles)
    except ValueError as exc:
        return JsonResponse({"message": str(exc)}, status=400)

    mediafiles = _get_sequence_download_mediafiles(request)
    export_path_by_mediafile_id = {}
    for mediafile_data, mediafile in zip(mediafiles_data, mediafiles):
        export_path_by_mediafile_id[mediafile.id] = mediafile_data["output_name"]
    metadata_records = _build_sequence_observation_export_records(
        mediafiles,
        request=request,
        columns=SEQUENCE_EXPORT_DEFAULT_COLUMNS,
    )
    for record in metadata_records:
        record["export_path"] = export_path_by_mediafile_id.get(record["mediafile_id"], record.get("export_path", ""))

    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    user_hash = request.user.caiduser.hash
    abs_zip_path = Path(settings.MEDIA_ROOT) / "users" / user_hash / f"sequence_mediafiles.{datetime_str}.zip"
    task = tasks.create_mediafiles_zip_with_metadata.delay(user_hash, mediafiles_data, str(abs_zip_path), metadata_records)
    _ = tasks.clean_old_mediafile_zips.delay(str(abs_zip_path.parent), glob_pattern="sequence_mediafiles.*.zip", max_age_days=7)
    return JsonResponse({"task_id": task.id})


@login_required
def download_csv_for_mediafiles_view(request, uploadedarchive_id: Optional[int] = None):
    """Download csv for media files."""
    mediafiles, name_suggestion = _get_mediafiles_for_export(request, uploadedarchive_id)
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


@login_required
def download_xlsx_for_mediafiles_view(request, uploadedarchive_id: Optional[int] = None):
    """Download xlsx for media files."""
    mediafiles, name_suggestion = _get_mediafiles_for_export(request, uploadedarchive_id)
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
        df.to_excel(writer, index=False, sheet_name="Localities")

    # Rewind the buffer
    output.seek(0)

    response = HttpResponse(output, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response["Content-Disposition"] = f"attachment; filename={fn}.xlsx"
    return response


@login_required
def download_xlsx_for_mediafiles_view_NDOP(request, uploadedarchive_id: Optional[int] = None):
    """Download xlsx for media files."""
    logger.debug("download_xlsx_for_mediafiles_view_NDOP")
    mediafiles, name_suggestion = _get_mediafiles_for_export(request, uploadedarchive_id)
    fn = ("metadata_CaID_NDOP_" + name_suggestion) if name_suggestion is not None else "metadata_CaID_NDOP"

    try:
        df = tasks.create_dataframe_from_mediafiles_NDOP(mediafiles)
        if df.empty:
            return HttpResponse("No data available to export.", content_type="text/plain")
    except Exception:
        logger.error(traceback.format_exc())
        return HttpResponse("Error during export.", content_type="text/plain")

    # convert timezone-aware datetime to naive datetime
    df = model_tools.convert_datetime_to_naive(df)

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Localities")

    # Rewind the buffer
    output.seek(0)

    response = HttpResponse(output, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response["Content-Disposition"] = f"attachment; filename={fn}.xlsx"
    return response


@login_required
def download_zip_for_mediafiles_view(request, uploadedarchive_id: Optional[int] = None) -> JsonResponse:
    """Download zip for media files."""
    mediafiles, name_suggestion = _get_mediafiles_for_export(request, uploadedarchive_id)
    # remove diacritics from name_suggestion
    if name_suggestion is not None:
        name_suggestion = model_tools.remove_diacritics(name_suggestion)
    fn = ("mediafiles_" + name_suggestion) if name_suggestion is not None else "mediafiles"
    # number_of_mediafiles = len(mediafiles)
    logger.debug(f"{len(mediafiles)=}")
    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    user_hash = request.user.caiduser.hash
    abs_zip_path = (
        # Path(settings.MEDIA_ROOT) / "users" / request.user.caiduser.hash / f"mediafiles.zip"
        Path(settings.MEDIA_ROOT)
        / "users"
        / request.user.caiduser.hash
        / f"{fn}.{datetime_str}.zip"
    )

    try:
        mediafiles_data = _build_export_mediafiles_data(request, mediafiles)
    except ValueError as exc:
        return JsonResponse({"message": str(exc)}, status=400)

    if not mediafiles_data:
        return JsonResponse({"message": "No media files matched the selected filters."}, status=400)

    # Start the Celery task

    task = tasks.create_mediafiles_zip.delay(user_hash, mediafiles_data, str(abs_zip_path))

    _ = tasks.clean_old_mediafile_zips.delay(str(abs_zip_path.parent), glob_pattern="mediafiles_*.zip", max_age_days=7)

    # Return the task ID so the frontend can poll for completion
    return JsonResponse({"task_id": task.id})


@login_required
def check_zip_status_view(request, task_id):
    """Check the status of the zip creation task."""
    # find task based on task_id
    task = AsyncResult(task_id)
    logger.debug(f"Check status of the task: {task_id=}: {task.state=}")

    # Define response mappings
    status_mapping = {
        "PENDING": "pending",
        "STARTED": "pending",  # Treat STARTED as pending
        "SUCCESS": "ready",
        "FAILURE": "error",
    }

    response = {
        "status": status_mapping.get(task.state, "unknown"),
    }

    if task.state == "SUCCESS":
        logger.debug(f"{task.result=}")
        fn_name = Path(task.result).name
        # Task is complete, return the download link
        # download_url = f"/media/users/{request.user.caiduser.hash}/mediafiles.zip"
        # download_url = f"{settings.MEDIA_URL}users/{request.user.caiduser.hash}/mediafiles.zip"
        download_url = f"{settings.MEDIA_URL}users/{request.user.caiduser.hash}/{fn_name}"
        logger.debug(f"{download_url=}")
        # download_url = request.build_absolute_uri(download_url)
        # logger.debug(f"{download_url=}")

        response["download_url"] = download_url
        # return JsonResponse()

    # elif task.state == "PENDING":
    # response = {"status": "pending"}
    # return JsonResponse({"status": "pending"})
    elif task.state == "FAILURE":
        response["message"] = str(task.result)
        # response = {"status": "error", "message": str(task.result)}
        # return JsonResponse({"status": "error", "message": str(task.result)})

    logger.debug(f"{response=}")
    return JsonResponse(response)


def _generate_new_hash_for_localities():
    for locality in Locality.objects.all():
        locality.hash = models.get_hash8()
        locality.save()


@login_required
def refresh_data(request):
    """Update new calculations for formerly uploaded archives."""
    uploaded_archives = UploadedArchive.objects.all()
    for uploaded_archive in uploaded_archives:
        uploaded_archive.update_earliest_and_latest_captured_at()
        uploaded_archive.make_sequences()

        if uploaded_archive.contains_single_taxon and uploaded_archive.taxon_for_identification is None:
            # this fixes the compatibility with the old version before 2024-05
            uploaded_archive.taxon_for_identification = models.get_taxon("Animalia")
            uploaded_archive.save()

        # uploaded_archive.refresh_status_after_migration(request)

    # this was used to fix same hashes generated by wrong function
    # _generate_new_hash_for_localities()

    # _refresh_media_file_original_name(request)
    # tasks.refresh_thumbnails()

    # get taxon (and create it if it does not exist
    models.get_taxon("Unclassifiable")

    return redirect("caidapp:home")


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


# def set_sort_identities_by(request, sort_by: str):
#     """Sort uploaded archives by."""
#     request.session["sort_identities_by"] = sort_by
#
#     # go back to previous page
#     return redirect(request.META.get("HTTP_REFERER", "/"))


# def set_sort_localities_by(request, sort_by: str):
#     """Sort uploaded archives by."""
#     request.session["sort_localities_by"] = sort_by
#
#     # go back to previous page
#     return redirect(request.META.get("HTTP_REFERER", "/"))


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


@login_required
def switch_private_mode(request):
    """Switch private mode."""
    actual_mode = request.session.get("private_mode", False)
    request.session["private_mode"] = not actual_mode

    return redirect(request.META.get("HTTP_REFERER", "/"))


class ImageUploadGraphView(View):
    def get(self, request):
        """Render the image upload graph."""
        # Fetch data from MediaFile model
        mediafiles = MediaFile.objects.all().values("parent__uploaded_at", "parent__owner__user__username")

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


def _prepare_merged_individual_identity_object(
    # request,
    individual_from: models.IndividualIdentity,
    individual_to: models.IndividualIdentity,
    # individual_identity_from_id:int, individual_identity_to_id:int
) -> Tuple[models.IndividualIdentity, Dict[str, str]]:

    # individual_from, individual_to = get_individuals(request, individual_identity_from_id,
    #                                                  individual_identity_to_id)
    today = datetime.date.today()
    today_str = today.strftime("%Y-%m-%d")

    differences = generate_differences(individual_to, individual_from)
    differences_str = f"merged: {individual_to.name} + {individual_from.name}, {today_str}\n" + "\n  ".join(
        f"{key}: {value}" for key, value in differences.items()
    )

    # Suggestion based on merging logic
    suggestion = IndividualIdentity(
        name=f"{individual_to.name}",
        sex=individual_to.sex if individual_to.sex != "U" else individual_from.sex,
        coat_type=(individual_to.coat_type if individual_to.coat_type != "U" else individual_from.coat_type),
        birth_date=individual_to.birth_date or individual_from.birth_date,
        death_date=individual_to.death_date or individual_from.death_date,
        note=f"{individual_to.note}\n{individual_from.note}\n" + differences_str,
        code=f"{individual_to.code}",
        juv_code=f"{individual_to.juv_code}",
    )

    return suggestion, differences


def generate_differences(individual1, individual2):
    """Generate differences between two identities."""
    differences = {}
    fields_to_compare = ["sex", "coat_type", "birth_date", "death_date", "code", "juv_code"]
    for field in fields_to_compare:
        value1 = getattr(individual1, field)
        value2 = getattr(individual2, field)
        if value1 != value2:
            differences[field] = f"{value1} , {value2}"
    return differences


def get_individuals(request, id1, id2) -> Tuple[models.IndividualIdentity, models.IndividualIdentity]:
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


class MergeIdentitiesWithPreview(View):
    def get(self, request, individual_identity_from_id, individual_identity_to_id):
        """Render the merge form."""
        individual_from, individual_to = get_individuals(
            request, individual_identity_from_id, individual_identity_to_id
        )
        suggestion, differences = _prepare_merged_individual_identity_object(
            individual_from,
            individual_to,
            # individual_identity_from_id, individual_identity_to_id
        )
        differences_html = (
            "<h3>Differences</h3><ul>"
            + "".join(f"<li>{key}: {value}</li>" for key, value in differences.items())
            + "</ul>"
        )

        # Differences for the right column

        form = IndividualIdentityForm(instance=suggestion)
        media_file = MediaFile.objects.filter(identity=individual_to, identity_is_representative=True).first()

        return render(
            request,
            "caidapp/update_form.html",
            {
                "form": form,
                "headline": "Merge Individual Identity",
                "button": "Save",
                "link": request.META.get("HTTP_REFERER", "/"),
                "cancel_button_url": request.META.get("HTTP_REFERER", "/"),
                "individual_identity": individual_to,
                "mediafile": media_file,
                "delete_button_url": reverse_lazy(
                    "caidapp:delete_individual_identity",
                    kwargs={"individual_identity_id": individual_identity_to_id},
                ),
                "right_col_raw_html": differences_html,
            },
        )

    def post(self, request, individual_identity_from_id, individual_identity_to_id):
        """Handle form submission."""
        individual_to, individual_from = get_individuals(
            request, individual_identity_to_id, individual_identity_from_id
        )

        form = IndividualIdentityForm(request.POST, instance=individual_to)
        if form.is_valid():
            individual_identity = form.save(commit=False)
            individual_identity.updated_by = request.user.caiduser
            individual_identity.save()

            # mediafiles of identity2 are reassigned to identity1
            individual_from.mediafile_set.update(identity=individual_to)

            models.MediafileIdentificationSuggestion.objects.filter(identity=individual_from).update(
                identity=individual_to
            )

            # remove old identity

            individual_from.delete()

            return redirect("caidapp:individual_identities")

        # On failure, re-render the form with errors
        return self.get(request, individual_identity_to_id, individual_identity_from_id)


class MergeIdentitiesNoPreview(View):
    def get(self, request, individual_identity_from_id, individual_identity_to_id):
        """Merge two individual identities without preview."""
        individual_from, individual_to = get_individuals(
            request, individual_identity_from_id, individual_identity_to_id
        )

        merge_identities_helper(request, individual_from, individual_to)

        # go back to prev page
        return redirect(request.META.get("HTTP_REFERER", "/"))


def merge_identities_helper(request, individual_from, individual_to):
    """Merge two individual identities."""
    if individual_to is None or individual_from is None:
        messages.warning(request, "Individual identity not found.")
        return

    # # TODO check if it has been finished already and show here time of last update

    suggestion, _ = _prepare_merged_individual_identity_object(
        individual_from,
        individual_to,
        # individual_identity_from_id, individual_identity_to_id
    )
    # set individual_to to suggestion
    # Convert the suggestion to a dict, excluding the primary key (and any other fields you want to skip)
    suggestion_data = model_to_dict(
        suggestion, exclude=["id", "updated_by", "id_worker", "owner", "owner_workgroup", "hash"]
    )
    for field, value in suggestion_data.items():
        logger.debug(f"{field=}, {value=}")
        setattr(individual_to, field, value)
    # Reassign media files and identification suggestions from individual_from to individual_to.
    individual_from.mediafile_set.update(identity=individual_to)
    models.MediafileIdentificationSuggestion.objects.filter(identity=individual_from).update(identity=individual_to)
    # Remove the redundant identity.
    individual_from.delete()
    individual_to.save()


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
            # remove file if it already exists
            if file_path.exists():
                file_path.unlink()
            with open(file_path, "wb+") as destination:
                for chunk in spreadsheet_file.chunks():
                    destination.write(chunk)

            logger.debug(f"{file_path.exists()=}")

            if file_path.suffix == ".csv":
                # read csv file with utf-8 encoding
                df = pd.read_csv(file_path, encoding="utf-8-sig")
            elif file_path.suffix == ".xlsx":
                df = pd.read_excel(file_path)
            else:
                df = None
                return messages.error(request, "Only CSV and XLSX files are supported.")

            # load metadata

            logger.debug(f"{uploaded_archive.csv_file.name=}")
            # metadata = pd.read_csv(Path(settings.MEDIA_ROOT) / uploaded_archive.csv_file.name, index_col=0)

            # metadata = merge_update_spreadsheet_with_metadata_spreadsheet(df, metadata)
            logger.debug("deleting uploaded file")
            Path(file_path).unlink()
            logger.debug(f"{df.columns=}")

            # metadata.to_csv(uploaded_archive.csv_file.name, encoding="utf-8-sig")
            df.rename(
                columns={
                    "original path": "original_path",
                    "taxon": "taxon",
                    "category": "taxon",
                    "unique name": "unique_name",
                    "identity code": "code",
                    "juvenile code": "juv_code",
                    "juv code": "juv_code",
                    "location_name": "locality_name",
                    "locality name": "locality_name",
                    "lat": "latitude",
                    "lon": "longitude",
                    "datetime": "datetime",
                },
                inplace=True,
            )
            # check if the column names are unique

            counter0 = 0
            counter_fields_updated = 0
            counter_file_in_spreadsheet_does_not_exist = 0
            counter_locality = 0
            counter_individuality = 0
            self.prev_url = request.META.get("HTTP_REFERER", "/")
            if "original_path" not in df.columns:
                logger.debug(f"{df.columns=}")
                logger.warning("The 'original_path' column is required in the uploaded spreadsheet.")

                return message_view(
                    request,
                    "The 'original_path' column is required in the uploaded spreadsheet.",
                    headline="Update metadata",
                    link=self.prev_url,
                    button_label="Ok",
                )

            for i, row in tqdm(df.iterrows(), total=len(df), desc="Updating metadata"):
                # turn \ into / in path
                original_path = row["original_path"].replace("\\", "/").strip()

                # get or None
                mf = MediaFile.objects.filter(parent=uploaded_archive, original_filename=original_path).first()
                if mf:
                    try:
                        ao = mf.observations.first()
                        if ao is None:
                            ao = models.Observation()
                            ao.mediafile = mf
                            ao.owner = uploaded_archive.owner
                            ao.owner_workgroup = uploaded_archive.owner.workgroup
                            ao.save()
                            mf.observations.add(ao)
                            mf.save()
                        # logger.debug(f"{mf=}")
                        counter0 += 1
                        # mf.category = row['category']
                        if "predicted_category" in row:
                            ao.taxon = models.get_taxon(row["predicted_category"])  # remove this
                            counter_fields_updated += 1

                        code = row["code"] if "code" in row else ""
                        unique_name = row["unique_name"] if "unique_name" in row else ""
                        juv_code = row["juv_code"] if "juv_code" in row else ""
                        identity = None
                        if code:
                            identity = models.get_unique_code(code, workgroup=uploaded_archive.owner.workgroup)
                        elif unique_name:
                            identity = models.get_unique_name(
                                row["unique_name"], workgroup=uploaded_archive.owner.workgroup
                            )
                        if identity is not None:
                            identity_updated = False
                            if unique_name and identity.name != unique_name.strip():
                                identity.name = unique_name.strip()
                                identity_updated = True
                            if code and identity.code != str(code).strip():
                                identity.code = str(code).strip()
                                identity_updated = True
                            if juv_code and identity.juv_code != str(juv_code).strip():
                                identity.juv_code = str(juv_code).strip()
                                identity_updated = True
                            if identity_updated:
                                identity.save()
                                counter_fields_updated += 1
                            ao.identity = identity
                            mf.identity = identity
                            counter_fields_updated += 1
                            counter_individuality += 1

                        if "locality_name" in row:
                            locality_obj = models.get_locality(
                                caiduser=request.user.caiduser, name=row["locality_name"]
                            )
                            if locality_obj:
                                mf.locality = locality_obj
                                counter_fields_updated += 1
                                counter_locality += 1
                        if ("latitude" in row) and ("longitude" in row):
                            latitude = row["latitude"]
                            longitude = row["longitude"]
                            if not pd.isna(latitude) and not pd.isna(longitude):
                                mf.location = f"{round(float(latitude), 3)},{round(float(longitude), 3)}"
                                counter_fields_updated += 1
                        if "datetime" in row:
                            # check if it is in django compatible datetime format
                            row_datetime = row["datetime"]
                            if isinstance(row_datetime, str):
                                # datetime_str = row["datetime"]
                                # mf.captured_at = datetime_str
                                mf.captured_at = row_datetime
                                counter_fields_updated += 1
                            elif isinstance(row_datetime, float) and np.isnan(row_datetime):
                                pass  # do nothing
                            # else if it is pandas datetime
                            elif isinstance(row_datetime, pd.Timestamp):
                                mf.captured_at = row_datetime.to_pydatetime()
                                counter_fields_updated += 1
                            else:
                                logger.debug(f"{row['datetime']=}")
                                logger.debug(f"{type(row['datetime'])=}")
                                logger.warning(f"Could not update datetime for {mf=}")

                        if "identity__coat_type" in row:
                            coat_type = row["coat_type"]
                            if coat_type:
                                counter_fields_updated += 1
                                ao.identity.coat_type = coat_type

                        if "orientation" in row:
                            orientation = row["orientation"]
                            if orientation:
                                # ORIENTATION_CHOICES = (
                                #     ("L", "Left"),
                                #     ("R", "Right"),
                                #     ("F", "Front"),
                                #     ("B", "Back"),
                                #     ("N", "None"),
                                #     ("U", "Unknown"),
                                # )
                                orientation = orientation.upper().strip()
                                orientation = orientation[0]
                                counter_fields_updated += 1
                                ao.orientation = orientation
                        ao.save()
                        mf.save()

                    except Exception as e:
                        logger.debug(f"{mf=}")
                        logger.debug(traceback.format_exc())
                        logger.debug(f"{row=}")
                        if "datetime" in row:
                            logger.debug(f"{row['datetime']=}")
                            logger.debug(f"{type(row['datetime'])=}")
                        logger.error(e)
                else:
                    counter_file_in_spreadsheet_does_not_exist += 1
            msg = (
                "Updated metadata for "
                + str(counter0)
                + " mediafiles. "
                + str(counter_fields_updated)
                + " fields updated "
                + f"(individualities={counter_individuality}, localities={counter_locality}). "
                + str(counter_file_in_spreadsheet_does_not_exist)
                + " files in spreadsheet do not exist. "
                + f"The spreadsheet has {len(df)} rows. "
            )
            if counter0 == 0:
                # show a few examples of original_path from table
                sample_size = min(3, len(df))
                if sample_size > 0:
                    msg += (
                        "Sample of `original_path` in spreadsheet: "
                        + ", ".join(df.sample(sample_size)["original_path"].dropna().astype(str).values)
                        + " ; "
                    )
                # add example of up to 3 original filenames from uploaded archives
                mfs = list(
                    MediaFile.objects.filter(parent=uploaded_archive).values_list("original_filename", flat=True)
                )
                if mfs:
                    msg += "Sample of `original_filename` in uploaded archive: " + ", ".join(
                        random.sample(mfs, min(3, len(mfs)))
                    )

            logger.info(msg)

            return message_view(
                request,
                msg,
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
                    "text_note": "The 'original_path' is required in the uploaded spreadsheet. "
                    + "The 'predicted_category', 'unique_name', 'code', 'juv_code', 'locality name', "
                    + "'latitude', 'longitude', 'datetime' are optional.",
                },
            )

    def get(self, request, uploaded_archive_id):
        """Render the form for updating the uploaded archive."""
        uploaded_archive = get_object_or_404(UploadedArchive, pk=uploaded_archive_id)
        output_dir = Path(settings.MEDIA_ROOT) / uploaded_archive.outputdir
        assert output_dir.exists()
        form = forms.UploadedArchiveUpdateBySpreadsheetForm()

        prev_url = request.META.get("HTTP_REFERER", "/")
        return render(
            request,
            "caidapp/update_form.html",
            {
                "form": form,
                "headline": "Upload XLSX or CSV",
                "button": "Save",
                "next": prev_url,
                "text_note": "The 'original_path' is required in the uploaded spreadsheet. "
                + "The 'predicted_category', 'unique_name', 'code', 'juv_code', 'locality name', "
                + "'latitude', 'longitude', 'datetime' are optional.",
            },
        )


class SplitPart(Func):
    """Custom database function to split a string by a delimiter and return the N-th part."""

    function = "SPLIT_PART"
    arity = 3  # Number of arguments the function takes


class MyPygWalkerView(PygWalkerView):
    template_name = "caidapp/custom_pygwalker.html"

    # mediafile_ids = request.session.get("mediafile_ids", [])
    # mediafiles = MediaFile.objects.filter(id__in=mediafile_ids)

    title = "Media File Analysis"
    theme = "light"  # 'light', 'dark', 'media'

    field_list = [
        "id",
        "captured_at",
        "locality",
        "identity",
        "taxon",
        "taxon__name",
        "identity__name",
        "locality__name",
        "latitude",
        "longitude",
    ]

    def get(self, request):
        """Process GET request."""
        # Access mediafile_ids from the session
        mediafile_ids = request.session.get("mediafile_ids", [])
        # Filter MediaFile objects based on the retrieved IDs
        self.queryset = MediaFile.objects.filter(id__in=mediafile_ids).annotate(
            latitude=Cast(
                SplitPart(F("locality__location"), Value(","), 1),
                output_field=django.db.models.FloatField(),
            ),
            longitude=Cast(
                SplitPart(F("locality__location"), Value(","), 2),
                output_field=django.db.models.FloatField(),
            ),
        )
        # Call the parent class's get method to maintain existing functionality
        return super().get(request)


class PygWalkerLocalitiesView(PygWalkerView):
    template_name = "caidapp/custom_pygwalker.html"

    # mediafile_ids = request.session.get("mediafile_ids", [])
    # mediafiles = MediaFile.objects.filter(id__in=mediafile_ids)

    title = "Localities"
    theme = "light"  # 'light', 'dark', 'media'

    # field_list = ["name", "some_field", "some_other__related_field", "id", "created_at", "updated_at"]
    field_list = ["name", "latitude", "longitude", "mediafile_count"]

    def get(self, request):
        """Process GET request."""
        # Access mediafile_ids from the session
        # mediafile_ids = request.session.get("mediafile_ids", [])
        # Filter MediaFile objects based on the retrieved IDs
        params = user_has_access_filter_params(request.user.caiduser, "owner")
        # logger.debug(f"{params=}")
        # localities = (
        #     Locality.objects.filter(**params)
        self.queryset = Locality.objects.filter(
            **params
            # owner__workgroup=request.user.caiduser.workgroup
        ).annotate(
            latitude=Cast(SplitPart(F("location"), Value(","), 1), output_field=django.db.models.FloatField()),
            longitude=Cast(SplitPart(F("location"), Value(","), 2), output_field=django.db.models.FloatField()),
            # there is locality
            mediafile_count=Count("mediafiles"),
        )
        # Call the parent class's get method to maintain existing functionality
        return super().get(request)


@login_required
def select_second_id_for_identification_merge(request, individual_identity1_id: int):
    """Select taxon for identification."""
    individual_identity1 = get_object_or_404(IndividualIdentity, pk=individual_identity1_id)
    identities = IndividualIdentity.objects.filter(owner_workgroup=request.user.caiduser.workgroup).exclude(
        pk=individual_identity1_id
    )
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


def refresh_identities_suggestions_view(request):
    """Refresh identity suggestions view."""
    # call background task
    refresh_identities_suggestions(request)
    return redirect(request.META.get("HTTP_REFERER", "/"))


def refresh_identities_suggestions(request, limit: int = 100, redirect: bool = True):
    """Refresh identity suggestions."""
    inspect = current_app.control.inspect(timeout=1.0)
    worker_stats = inspect.stats() if inspect else None
    if not worker_stats:
        result_id = compute_identity_suggestions(request.user.caiduser.workgroup.id, limit)
        request.session.pop("refresh_job_id", None)
        request.session["refresh_job_started_at"] = timezone.now().isoformat()
        request.session["refresh_result_id"] = result_id
        logger.debug(
            "No Celery worker available for merge suggestions. Computed synchronously for workgroup %s.",
            request.user.caiduser.workgroup_id,
        )
        return result_id

    job = tasks.refresh_identities_suggestions_task.delay(request.user.caiduser.workgroup.id)
    logger.debug(
        f"{job.id=}, {request.user.id=}, {request.user=}, {request.user.caiduser=}, {request.user.caiduser.workgroup=}"
    )
    request.session["refresh_job_id"] = job.id
    request.session["refresh_job_started_at"] = timezone.now().isoformat()
    request.session.pop("refresh_result_id", None)
    return None


def get_identity_suggestions(request):
    """Get identity suggestions status and data."""
    job_id = request.session.get("refresh_job_id")
    sync_result_id = request.session.get("refresh_result_id")

    sugg_obj = (
        models.MergeIdentitySuggestionResult.objects.filter(workgroup=request.user.caiduser.workgroup)
        .order_by("id")
        .last()
    )
    suggestions = sugg_obj.suggestions if sugg_obj else None
    created_at = sugg_obj.created_at if sugg_obj else None

    if sync_result_id and not job_id:
        status = "SUCCESS"
        job_started_at = request.session.get("refresh_job_started_at")
        try:
            sugg_obj2 = models.MergeIdentitySuggestionResult.objects.get(id=sync_result_id)
            if sugg_obj2.workgroup == request.user.caiduser.workgroup:
                suggestions = sugg_obj2.suggestions
                created_at = sugg_obj2.created_at
        except Exception as e:
            logger.warning("Could not fetch sync job result: " + str(e))
            messages.warning(request, "Could not fetch sync job result: " + str(e))
    elif not job_id:
        status = "no-job"
        job_started_at = None

        # return {"status": "no-job", "suggestions": suggestions, "created_at"}
    else:

        job_started_at = request.session.get("refresh_job_started_at")

        result = AsyncResult(job_id)
        status = result.status
        if result.successful():
            result_id = result.result  # ID uloženého výsledku
            try:
                sugg_obj2 = models.MergeIdentitySuggestionResult.objects.get(id=result_id)
                if sugg_obj2.workgroup == request.user.caiduser.workgroup:
                    suggestions = sugg_obj2.suggestions
                    created_at = sugg_obj2.created_at
                    status = result.status
                else:
                    logger.warning("Job result workgroup does not match user workgroup.")
                    messages.warning(request, "Job result workgroup does not match user workgroup.")
            except Exception as e:
                logger.warning("Could not fetch job result: " + str(e))
                messages.warning(request, "Could not fetch job result: " + str(e))

        # return {"status": "done", "suggestions": suggestions, 'started_at': job_started_at}
    # return {"status": result.status, 'started_at': job_started_at}
    return dict(
        status=status,
        suggestions=suggestions,
        started_at=job_started_at,
        created_at=created_at,
    )


@login_required
def suggest_merge_identities_view(request, limit: int = 100):
    """Suggest merge identities."""
    response = get_identity_suggestions(request)
    if response["status"] == "no-job" and response["suggestions"] is None:
        refresh_identities_suggestions(request)
        messages.info(request, "Generation of merge suggestions has started. Check back in a moment.")
        return message_view(
            request,
            "Generation of merge suggestions has started.",
            link=reverse_lazy("caidapp:suggest_merge_identities"),
            button_label="Check now",
            headline="Generating suggestions",
            link_secondary=reverse_lazy("caidapp:refresh_merge_identities_suggestions"),
            button_label_secondary="Start again",
        )

    if "started_at" in response and response["started_at"]:
        started_at = datetime.datetime.fromisoformat(response["started_at"])
        messages.info(request, f"Suggestion refreshed {timesince_now(started_at)} ago.")
    if "created_at" in response and response["created_at"]:
        created_at = response["created_at"]
        messages.info(request, f"This data created {timesince_now(created_at)} ago.")
    if response["status"] == "no-job":
        logger.debug("No job found for suggestions.")
    if response["suggestions"] is None:
        messages.info(
            request,
            "No suggestions available. Check if they are generated now or regenerate suggestions.",
        )
        return message_view(
            request,
            "No suggestions found. Check if the job is",
            link=reverse_lazy("caidapp:suggest_merge_identities"),
            button_label="Check now",
            headline="No suggestions found",
            link_secondary=reverse_lazy("caidapp:refresh_merge_identities_suggestions"),
            button_label_secondary="Regenerate suggestions",
        )
    suggestions_ids = response["suggestions"]
    try:
        # assert "merge_identity_suggestions_ids" in request.session

        # suggestions_ids = request.session["merge_identity_suggestions_ids"]
        if suggestions_ids:

            from django.core.exceptions import ObjectDoesNotExist

            logger.debug(f"{len(suggestions_ids)=}")

            suggestions = []
            for identity_a_id, identity_b_id, distance in suggestions_ids[:limit]:
                try:
                    identity_a = IndividualIdentity.objects.get(id=identity_a_id)
                    identity_b = IndividualIdentity.objects.get(id=identity_b_id)
                except ObjectDoesNotExist:
                    continue  # přeskočí, pokud některý objekt neexistuje
                suggestions.append((identity_a, identity_b, distance))
            # suggestions = [
            #     (
            #         IndividualIdentity.objects.get(id=identity_a_id),
            #         IndividualIdentity.objects.get(id=identity_b_id),
            #         distance
            #       )
            #     for identity_a_id, identity_b_id, distance in suggestions_ids
            # ]

            # if limit and limit > 0:
            #     suggestions = suggestions[:limit]
        else:
            suggestions = None

        return render(request, "caidapp/suggest_merge_identities.html", {"suggestions": suggestions})
    except Exception as e:

        logger.warning(e)
        logger.debug(traceback.format_exc())
        return message_view(
            request,
            "An error occurred while fetching suggestions. They might be being refreshed. Please try again later.",
        )
        # # TODO show time of last update and maybe init
        # refresh_identities_suggestions(request)
        #
        # suggestions_ids = request.session["merge_identity_suggestions_ids"]
        # suggestions = [
        #     (
        #         IndividualIdentity.objects.get(id=identity_a_id),
        #         IndividualIdentity.objects.get(id=identity_b_id),
        #         distance
        #     )
        #     for identity_a_id, identity_b_id, distance in suggestions_ids
        # ]
        #
        # if limit and limit > 0:
        #     suggestions = suggestions[:limit]
        #
        #
        # return render(request, "caidapp/suggest_merge_identities.html",
        #               {"suggestions": suggestions})


@login_required
def run_identification_outlier_detection_view(request):
    """Start background detection of suspicious identity assignments for the current workgroup."""
    if not (request.user.caiduser.workgroup_admin or request.user.is_staff):
        raise PermissionDenied

    workgroup = request.user.caiduser.workgroup
    if workgroup is None:
        return message_view(request, "No workgroup assigned.")

    result, metadata_file = tasks.run_identification_outlier_detection_for_workgroup(workgroup)
    messages.info(request, "Identification outlier detection has started.")
    return redirect("caidapp:identification_outlier_suggestions_result", result_id=result.id)


@login_required
def identification_outlier_suggestions_view(request, result_id: int = None):
    """Display latest suspicious identity assignments for the current workgroup."""
    if not (request.user.caiduser.workgroup_admin or request.user.is_staff):
        raise PermissionDenied

    workgroup = request.user.caiduser.workgroup
    if workgroup is None:
        return message_view(request, "No workgroup assigned.")

    queryset = models.IdentificationOutlierSuggestionResult.objects.filter(workgroup=workgroup).order_by("-id")
    if result_id is not None:
        result = get_object_or_404(queryset, id=result_id)
    else:
        result = queryset.first()

    if result is None:
        return message_view(
            request,
            "No identification outlier suggestions found yet.",
            headline="No results yet",
            link=reverse_lazy("caidapp:run_identification_outlier_detection"),
            button_label="Start detection",
        )

    def _mediafile_card_data(mediafile: MediaFile | None) -> dict | None:
        if mediafile is None:
            return None
        raw_name = str(mediafile.original_filename or mediafile.mediafile or mediafile.id)
        short_name = Path(raw_name).name if raw_name else str(mediafile.id)
        tooltip_parts = []
        if mediafile.taxon:
            tooltip_parts.append(f"Taxon: {mediafile.taxon}")
        if mediafile.note:
            tooltip_parts.append(f"Comments: {mediafile.note}")
        return {
            "object": mediafile,
            "display_name": short_name,
            "full_name": raw_name,
            "tooltip_text": "\n".join(tooltip_parts),
        }

    suggestion_cards = []
    for raw_item in result.suggestions or []:
        suspicious_mediafile = None
        current_identity = None
        candidate_cards = []

        suspicious_mediafile_id = raw_item.get("suspicious_mediafile_id")
        current_identity_id = raw_item.get("current_identity_id")

        if suspicious_mediafile_id:
            suspicious_mediafile = MediaFile.objects.filter(id=suspicious_mediafile_id).first()
        if current_identity_id:
            current_identity = IndividualIdentity.objects.filter(id=current_identity_id).first()
        elif suspicious_mediafile and suspicious_mediafile.identity_id:
            current_identity = suspicious_mediafile.identity

        for raw_candidate in raw_item.get("suggestions", []):
            candidate_mediafile = MediaFile.objects.filter(id=raw_candidate.get("mediafile_id")).first()
            candidate_cards.append(
                {
                    "identity": IndividualIdentity.objects.filter(id=raw_candidate.get("identity_id")).first(),
                    "mediafile": _mediafile_card_data(candidate_mediafile),
                    "score": raw_candidate.get("score"),
                    "reason": raw_candidate.get("reason", ""),
                }
            )

        suggestion_cards.append(
            {
                "suspicious_mediafile": _mediafile_card_data(suspicious_mediafile),
                "current_identity": current_identity,
                "reason": raw_item.get("reason", ""),
                "candidates": candidate_cards,
            }
        )

    csv_url = workgroup.file_url("identification_outliers.csv")
    extended_csv_path = workgroup.file_path("identification_outliers_extended.csv")
    extended_csv_url = workgroup.file_url("identification_outliers_extended.csv")
    logger.debug(f"{csv_url=}")

    similarity_plot_html = None
    tsne_plot_html = None
    umap_plot_html = None

    if extended_csv_path.exists():
        metadata = pd.read_csv(extended_csv_path)
        metadata["class_label"] = metadata.get("label", metadata.get("class_id", "")).fillna("").astype(str)
        hover_fields = [
            col
            for col in [
                "mediafile_id",
                "class_id",
                "class_label",
                "best_other_label",
                "own_similarity",
                "best_other_similarity",
                "delta",
            ]
            if col in metadata.columns
        ]

        if {"own_similarity", "best_other_similarity", "is_suspect"}.issubset(metadata.columns):
            suspect_labels = metadata["is_suspect"].fillna(False).astype(bool).map(
                lambda value: "Suspect" if bool(value) else "Not suspect"
            )
            similarity_fig = px.scatter(
                metadata,
                x="own_similarity",
                y="best_other_similarity",
                color=suspect_labels,
                color_discrete_map={"Suspect": "#d62728", "Not suspect": "#1f77b4"},
                hover_name="class_label" if "class_label" in metadata.columns else None,
                hover_data=hover_fields,
                opacity=0.75,
                title="Likely Mislabeled Candidates",
                labels={
                    "own_similarity": "Similarity to Own Center",
                    "best_other_similarity": "Similarity to Best Other Center",
                    "color": "",
                },
                render_mode="webgl",
            )
            similarity_fig.add_shape(
                type="line",
                x0=-1,
                y0=-1,
                x1=1,
                y1=1,
                line={"dash": "dash", "color": "gray", "width": 1},
            )
            similarity_fig.update_layout(template="plotly_white", legend_title_text="")
            similarity_plot_html = similarity_fig.to_html(full_html=False, include_plotlyjs=False)

        if {"tsne_x", "tsne_y"}.issubset(metadata.columns):
            tsne_fig = px.scatter(
                metadata,
                x="tsne_x",
                y="tsne_y",
                color="class_label",
                hover_name="class_label",
                hover_data=hover_fields,
                opacity=0.7,
                title="t-SNE Embedding",
                render_mode="webgl",
            )
            tsne_fig.update_traces(marker={"size": 7})
            tsne_fig.update_layout(template="plotly_white", showlegend=False)
            tsne_plot_html = tsne_fig.to_html(full_html=False, include_plotlyjs=False)

        if {"umap_x", "umap_y"}.issubset(metadata.columns):
            umap_fig = px.scatter(
                metadata,
                x="umap_x",
                y="umap_y",
                color="class_label",
                hover_name="class_label",
                hover_data=hover_fields,
                opacity=0.7,
                title="UMAP Embedding",
                render_mode="webgl",
            )
            umap_fig.update_traces(marker={"size": 7})
            umap_fig.update_layout(template="plotly_white", showlegend=False)
            umap_plot_html = umap_fig.to_html(full_html=False, include_plotlyjs=False)

    return render(
        request,
        "caidapp/identification_outlier_suggestions.html",
        {
            "result": result,
            "suggestion_cards": suggestion_cards,
            "csv_url": csv_url,
            "extended_csv_url": extended_csv_url,
            "similarity_plot_html": similarity_plot_html,
            "tsne_plot_html": tsne_plot_html,
            "umap_plot_html": umap_plot_html,
        },
    )


@login_required
def accept_identification_outlier_suggestion_view(request):
    """Accept one suggested identity for a suspicious media file."""
    if request.method != "POST":
        raise PermissionDenied

    if not (request.user.caiduser.workgroup_admin or request.user.is_staff):
        raise PermissionDenied

    suspicious_mediafile_id = request.POST.get("suspicious_mediafile_id")
    suggested_identity_id = request.POST.get("suggested_identity_id")
    result_id = request.POST.get("result_id")
    next_url = request.POST.get("next") or reverse_lazy("caidapp:identification_outlier_suggestions")

    suspicious_mediafile = get_object_or_404(MediaFile, id=suspicious_mediafile_id)
    suggested_identity = get_object_or_404(IndividualIdentity, id=suggested_identity_id)

    if suspicious_mediafile.parent.owner.workgroup != request.user.caiduser.workgroup:
        return HttpResponseNotAllowed("Not allowed to work with this media file.")
    if suggested_identity.owner_workgroup != request.user.caiduser.workgroup:
        return HttpResponseNotAllowed("Not allowed to use this identity.")

    suspicious_mediafile.identity = suggested_identity
    suspicious_mediafile.updated_by = request.user.caiduser
    suspicious_mediafile.save(update_fields=["identity", "updated_by"])

    if result_id:
        result = models.IdentificationOutlierSuggestionResult.objects.filter(
            id=result_id,
            workgroup=request.user.caiduser.workgroup,
        ).first()
        if result is not None:
            result.suggestions = [
                item
                for item in (result.suggestions or [])
                if int(item.get("suspicious_mediafile_id") or -1) != suspicious_mediafile.id
            ]
            result.save(update_fields=["suggestions"])

    messages.success(
        request,
        f"Identity for media file '{suspicious_mediafile}' was updated to '{suggested_identity}'.",
    )
    return redirect(next_url)


@login_required
def merge_selected_identities_view(request):
    """Merge selected identities based on suggestions."""
    if request.method == "POST":
        selected_suggestions = request.POST.getlist("suggestions")
        if not selected_suggestions:
            messages.info(request, "No suggestions were selected for merging.")
            return redirect("caidapp:suggest_merge_identities")

        for suggestion in selected_suggestions:
            try:
                id1, id2 = suggestion.split("|")
                # Retrieve the identities ensuring they belong to the user's workgroup
                identity1 = get_object_or_404(
                    IndividualIdentity, pk=id1, owner_workgroup=request.user.caiduser.workgroup
                )
                identity2 = get_object_or_404(
                    IndividualIdentity, pk=id2, owner_workgroup=request.user.caiduser.workgroup
                )
                # Order the identities by media file count (if that’s how your merge logic expects it)
                # identity_a, identity_b = order_identity_by_mediafile_count(identity1, identity2)

                # Perform the merge.
                # Replace the following call with your actual merge logic.
                merge_identities_helper(request, identity1, identity2)
                # For example, if you have a function that handles merging:
                # merge_identities_no_preview(request, identity_a.id, identity_b.id)

            except Exception:
                logger.debug(f"{suggestion=}")
                logger.debug(traceback.format_exc())
                logger.warning("Skipping this suggestion. Probably the identities were already merged.")

                messages.debug(
                    request,
                    "Skipping this suggestion. Probably the identities were already merged.",
                )
                # Skip this suggestion if it doesn't have the correct format
                continue

        messages.success(request, "Selected identities merged successfully.")
        return redirect("caidapp:suggest_merge_identities")
    else:
        messages.error(request, "Invalid request method.")
        return redirect("caidapp:suggest_merge_identities")


@login_required
def show_identity_code_suggestions(request):
    """Show identity code suggestions."""
    all_identities = IndividualIdentity.objects.filter(
        owner_workgroup=request.user.caiduser.workgroup,
        # **user_has_access_filter_params(request.user.caiduser, "owner")
    )
    suggestions = []
    for identity in all_identities:
        suggested_code = identity.suggested_code_from_name()
        if suggested_code:
            identity.suggested_code = suggested_code
            identity.suggested_name = identity.suggested_name_without_code()
            suggestions.append(identity)

    workgroup = request.user.caiduser.workgroup
    active_regex = workgroup.get_identity_code_regex() if workgroup else models.DEFAULT_IDENTITY_CODE_REGEX

    return render(
        request,
        "caidapp/suggest_identity_codes.html",
        {
            "identities": suggestions,
            "active_regex": active_regex,
        },
    )


@login_required
def apply_identity_code_suggestion(request, identity_id: int, rename: bool = True):
    """Use the suggested individuality code."""
    identity = get_object_or_404(IndividualIdentity, pk=identity_id, owner_workgroup=request.user.caiduser.workgroup)

    code = identity.suggested_code_from_name()
    if code:
        identity.note = identity.note + f"\nformer code: {str(identity.code)} \nformer name: {str(identity.name)}"
        identity.code = code
        if rename:
            identity.name = identity.name.replace(code, "").strip()
        identity.save()

    return redirect(request.META.get("HTTP_REFERER", "/"))


@login_required
def uploads_status_api(request, group: str):
    """Get JSON with ingormation about statuses.

    Vrátí JSON s informacemi o statusech (např. pro všechny archivy daného uživatele).
    """
    species = True if group == "species" else False
    # Můžete vrátit jen pro aktuálně přihlášeného uživatele:
    user = request.user
    if not user.is_authenticated:
        return JsonResponse({"error": "Unauthorized"}, status=401)

    # Získat archivy usera (dle vaší logiky, v příkladu jen pro demonstraci)
    uploaded_archives = UploadedArchive.objects.filter(**user_has_access_filter_params(user.caiduser, "owner"))

    data = []
    for ua in uploaded_archives:
        if species:
            st = ua.get_status()
        else:
            st = ua.get_identification_status()
        # status = st["status"]
        # status_message = st["status_message"]
        data.append({"id": ua.id, **st})

    return JsonResponse({"archives": data})


@login_required
def export_identities_csv(request):
    """Export identities to CSV."""

    all_identities = IndividualIdentity.objects.filter(
        owner_workgroup=request.user.caiduser.workgroup,
        # **user_has_access_filter_params(request.user.caiduser, "owner")
    )
    df = pd.DataFrame.from_records(all_identities.values())[
        ["id", "name", "code", "juv_code", "sex", "coat_type", "birth_date", "death_date", "note"]
    ]

    return views_general.csv_response(df, "identities")


@login_required
def export_identities_xlsx(request):
    """Export identities to Excel."""
    all_identities = IndividualIdentity.objects.filter(
        owner_workgroup=request.user.caiduser.workgroup,
        # **user_has_access_filter_params(request.user.caiduser, "owner")
    )
    df = pd.DataFrame.from_records(all_identities.values())[
        ["id", "name", "code", "juv_code", "sex", "coat_type", "birth_date", "death_date", "note"]
    ]

    return views_general.excel_response(df, "identities")


def _spreadsheet_cell_has_value(value) -> bool:
    """Return True when spreadsheet cell contains a meaningful value."""
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def _spreadsheet_row_id(value) -> Optional[int]:
    """Parse integer primary key from spreadsheet cell."""
    if not _spreadsheet_cell_has_value(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def import_identities_view(request):
    """Import identities."""
    logger.debug(f"Importing identities, method {request.method}")
    if request.method == "POST":
        form = forms.SpreadsheetFileImportForm(request.POST, request.FILES)
        if form.is_valid():
            logger.debug("form is valid")
            file = form.cleaned_data["spreadsheet_file"]

            file_ext = Path(file.name).suffix.lower()
            file_content = file.read()
            rename_columns = {
                "Location": "location",
                "Latitude": "latitude",
                "Longitude": "longitude",
                "lat": "latitude",
                "lon": "longitude",
                "Lat": "latitude",
                "Lon": "longitude",
            }

            if file_ext == ".xlsx":
                df = pd.read_excel(BytesIO(file_content))
                df.rename(columns=rename_columns, inplace=True)
            elif file_ext == ".csv":
                df = pd.read_csv(BytesIO(file_content))
                df.rename(columns=rename_columns, inplace=True)
            else:
                return HttpResponse("Only .xlsx and .csv files are supported.")

            for index, row in df.iterrows():
                # row as dict
                row = row.to_dict()
                identity = None
                try:
                    identity_id = _spreadsheet_row_id(row.get("id"))
                    if identity_id is not None:
                        identity = IndividualIdentity.objects.filter(
                            id=identity_id,
                            owner_workgroup=request.user.caiduser.workgroup,
                        ).first()

                    if identity is None and _spreadsheet_cell_has_value(row.get("code")):
                        identity, created_new = IndividualIdentity.objects.get_or_create(
                            code=row["code"], owner_workgroup=request.user.caiduser.workgroup
                        )
                    elif identity is None and _spreadsheet_cell_has_value(row.get("name")):
                        identity, created_new = IndividualIdentity.objects.get_or_create(
                            name=row["name"], owner_workgroup=request.user.caiduser.workgroup
                        )
                    elif identity is not None:
                        created_new = False
                    else:
                        logger.warning(f"No identification (name or code) found for {row} ")
                        continue
                except models.IndividualIdentity.MultipleObjectsReturned:
                    logger.debug(f"{row=}")
                    logger.warning(traceback.format_exc())
                    messages.warning(request, f"Skipping row. Multiple identities found for {row}")
                    continue

                logger.debug(f"{identity=}")

                if _spreadsheet_cell_has_value(row.get("name")):
                    print(f"{row}")
                    print(f"{row['name']}")
                    identity.name = row["name"]
                if _spreadsheet_cell_has_value(row.get("code")):
                    identity.code = row["code"]
                if _spreadsheet_cell_has_value(row.get("sex")):
                    sex = row["sex"][0].upper()
                    if sex in ["M", "F", "U"]:
                        identity.sex = sex
                    else:
                        logger.warning(f"Invalid sex: {row['sex']}")
                if _spreadsheet_cell_has_value(row.get("coat_type")):
                    rename_coat = {
                        "Spotted": "S",
                        "Marbled": "M",
                        "Unspotted": "N",
                        "Unkown": "U",
                    }
                    if row["coat_type"] in rename_coat:
                        coat_type = rename_coat[row["coat_type"]]
                    else:
                        coat_type = row["coat_type"][0]

                    if coat_type in ["S", "M", "N", "U"]:
                        identity.coat_type = coat_type
                    else:
                        logger.warning(f"Invalid coat_type: {row['coat_type']}")

                if "note" in row:
                    note = row["note"]
                    if isinstance(note, str):
                        identity.note = note

                if _spreadsheet_cell_has_value(row.get("juv_code")):
                    identity.juv_code = row["juv_code"]

                if "birth_date" in row and not pd.isna(row["birth_date"]):
                    identity.birth_date = row["birth_date"]
                if "death_date" in row and not pd.isna(row["death_date"]):
                    identity.death_date = row["death_date"]

                if identity.owner_workgroup is None:
                    identity.owner_workgroup = request.user.caiduser.workgroup

                identity.save()
            return redirect("caidapp:individual_identities")
    else:
        form = forms.SpreadsheetFileImportForm()
    return render(
        request,
        # "caidapp/model_form_upload.html",
        "caidapp/update_form.html",
        {
            "form": form,
            "headline": "Import identities",
            "button": "Import",
            "text_note": "Upload CSV or XLSX file. "
            + "There should be columns 'id', 'name' or 'code' in the file. "
            + "Optional columns are 'sex', 'coat_type', 'birth_date', 'death_date', 'note'.",
            "next": "caidapp:individual_identities",
        },
    )


# create view which will be shown just before the identify to make sure that the user wants to identify
@login_required
def pre_identify_view(request):
    """Show pre-identification confirmation page."""
    return render(
        request,
        "caidapp/pre_identify.html",
        {
            "prev_page_url": request.META.get("HTTP_REFERER", "/"),
        },
    )


@login_required
@require_POST
def toggle_identity_representative(request, mediafile_id: int):
    """Toggle identity representative flag for a media file."""
    mf = get_object_or_404(models.MediaFile, id=mediafile_id)

    # Povolení jen v rámci stejné workgroup + musí mít identitu
    if mf.identity is None:
        return JsonResponse({"ok": False, "error": "Mediafile nemá přiřazenou identitu."}, status=400)
    if request.user.caiduser.workgroup != mf.parent.owner.workgroup:
        return HttpResponseNotAllowed("Not allowed")

    mf.identity_is_representative = not mf.identity_is_representative
    mf.updated_by = request.user.caiduser  # pokud máš tohle pole
    mf.save(update_fields=["identity_is_representative", "updated_by"])

    logger.debug("almost done")
    return JsonResponse({"ok": True, "representative": mf.identity_is_representative})


class NotificationCreateView(CreateView):
    model = models.Notification
    fields = ["message", "level"]
    # could use a form class instead
    title = "Create Notification"
    # form_class = forms.NotificationForm
    template_name = "caidapp/generic_form.html"
    success_url = reverse_lazy("caidapp:notifications")

    def form_valid(self, form):
        """Assign the current user to the notification before saving."""
        form.instance.user = self.request.user.caiduser
        return super().form_valid(form)


class NotificationListView(ListView):
    model = models.Notification
    template_name = "caidapp/generic_list_table.html"
    context_object_name = "notifications"
    # title = "Notifications"

    def get_queryset(self):
        """Limit queryset to notifications of the current user."""
        user = self.request.user.caiduser

        recipient_qs = models.NotificationRecipient.objects.filter(notification=OuterRef("pk"), user=user)

        return (
            models.Notification.objects.filter(recipients__user=user)
            .annotate(
                recipient=Subquery(recipient_qs.values("user__user__username")[:1]),
                read=Subquery(recipient_qs.values("read")[:1]),
            )
            .order_by("-created_at")
        )

        # return models.Notification.objects.filter(user=self.request.user.caiduser).order_by("-created_at")
        # return (
        #     models.Notification.objects
        #     .filter(notificationrecipient__user=self.request.user.caiduser)
        #     .distinct()
        #     .order_by("-notification__created_at")
        # )

    def get_context_data(self, **kwargs):
        """Set up context data for the list view."""
        context = super().get_context_data(**kwargs)
        context["title"] = _("Notifications")
        context["list_display"] = [
            "created_at",
            # "level",
            "message",
            # "recipient",
            # "read",
        ]
        # context["object_detail_url"] = "caidapp:notification-detail"
        # context["object_update_url"] = "caidapp:notification-update"
        # context["object_delete_url"] = "caidapp:notification-delete"
        # context["object_create_url"] = "caidapp:notification-create"
        return context


class NotificationDetailView(DetailView):
    model = models.Notification
    template_name = "caidapp/generic_detail.html"
    context_object_name = "notification"
    title = "Notification Detail"
    # paginate_by = 20
    fields = ["message",  "level", "created_at"]
    cancel_url = reverse_lazy("caidapp:notifications")

    # def get_queryset(self):
    #     """Limit queryset to notifications of the current user."""
    #     qs = super().get_queryset()
    #     # např. jen zprávy pro aktuálního uživatele
    #     return qs.filter(user=self.request.user.caiduser)

    def get_queryset(self):
        user = self.request.user.caiduser
        return (
            super()
            .get_queryset()
            .filter(recipients__user=user)
            .prefetch_related("recipients")
            .distinct()
        )

    def get(self, request, *args, **kwargs):
        """Handle GET request and mark notification as read."""
        response = super().get(request, *args, **kwargs)
        # Mark as read when viewed
        me_as_recipient = self.object.recipients.filter(user=request.user.caiduser).first()
        if me_as_recipient and not me_as_recipient.read:
            me_as_recipient.read = True
            me_as_recipient.save(update_fields=["read"])
        # if not self.object.read:
        #     self.object.read = True
        #     self.object.save(update_fields=["read"])
        return response

    def get_context_data(self, **kwargs):
        """Set up context data for the detail view."""
        context = super().get_context_data(**kwargs)
        field_data = []

        for field_name in self.fields:
            field = self.model._meta.get_field(field_name)
            value = getattr(self.object, field_name)
            field_data.append(
                {
                    "name": field_name,
                    "verbose_name": field.verbose_name,
                    "value": value,
                }
            )
        context["fields"] = field_data
        return context


class NotificationUpdateView(UpdateView):
    model = models.Notification
    fields = ["message", "level"]
    template_name = "caidapp/generic_form.html"
    success_url = reverse_lazy("caidapp:notifications")
    title = "Update Notification"


class NotificationDeleteView(DeleteView):
    model = models.Notification
    template_name = "caidapp/generic_form.html"
    success_url = reverse_lazy("caidapp:notifications")
    title = "Delete Notification"


class WorkGroupInvitationCreateView(LoginRequiredMixin, UserPassesTestMixin, CreateView):
    model = models.WorkGroupInvitation
    template_name = "caidapp/generic_form.html"
    fields = ["invited_user"]
    success_url = reverse_lazy("caidapp:workgroup_invitations")

    def test_func(self):
        return self.request.user.caiduser.workgroup_admin

    # def dispatch(self, request, *args, **kwargs):
    #     """Check if the user is a workgroup admin and set the target workgroup for the invitation."""
    #     response
    #     if not request.user.caiduser.workgroup_admin:
    #         raise PermissionDenied
    #
    #     self.target_workgroup = request.user.caiduser.workgroup
    #     return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        """Set the inviter and target workgroup before saving the form."""
        form.instance.invited_by = self.request.user.caiduser
        form.instance.target_workgroup = self.target_workgroup
        return super().form_valid(form)


class WorkGroupInvitationListView(LoginRequiredMixin, ListView):
    model = models.WorkGroupInvitation
    template_name = "caidapp/generic_list_table.html"
    context_object_name = "WorkGroupInvitation"
    title = "Workgroup Invitations"

    def dispatch(self, request, *args, **kwargs):
        user = request.user
        """Check if the user is a workgroup admin and set the target workgroup for filtering invitations."""
        if not user.caiduser or not user.caiduser.workgroup_admin or not user.caiduser.workgroup_admin:
            raise PermissionDenied

        self.target_workgroup = request.user.caiduser.workgroup
        return super().dispatch(request, *args, **kwargs)

    def get_queryset(self):
        """Limit queryset to invitations for the current user's workgroup."""
        logger.debug(f"{self.target_workgroup=}")
        invs = models.WorkGroupInvitation.objects.filter(target_workgroup=self.target_workgroup)
        logger.debug(f"{invs.count()=}")
        return invs.order_by("-created_at")

    def get_context_data(self, **kwargs):
        """Set up context data for the list view."""
        context = super().get_context_data(**kwargs)
        context["title"] = _("Workgroup Invitations")
        context["list_display"] = [
            "invited_user",
            "invited_by",
            "created_at",
            "status",
        ]
        context["object_detail_url"] = "caidapp:workgroup_invitation_detail"
        return context


class WorkGroupInvitationForUserListView(LoginRequiredMixin, ListView):
    model = models.WorkGroupInvitation
    template_name = "caidapp/generic_list_table.html"
    context_object_name = "workgroup_invitations"
    title = "Your Workgroup Invitations"

    def get_queryset(self):
        """Limit queryset to invitations for the current user."""
        return models.WorkGroupInvitation.objects.filter(invited_user=self.request.user.caiduser).order_by(
            "-created_at"
        )

    def get_context_data(self, **kwargs):
        """Set up context data for the list view."""
        context = super().get_context_data(**kwargs)
        context["title"] = _("Your Workgroup Invitations")
        context["list_display"] = [
            "invited_by",
            "target_workgroup",
            "created_at",
            "status",
        ]
        context["object_detail_url"] = "caidapp:workgroup_invitation_detail"

        return context


class WorkGroupInvitationDetailView(LoginRequiredMixin, DetailView):
    model = models.WorkGroupInvitation
    template_name = "caidapp/generic_detail.html"
    context_object_name = "workgroup_invitation"
    title = "Workgroup Invitation Detail"
    fields = [
        "invited_user",
        "invited_by",
        "target_workgroup",
        "created_at",
        "status",
    ]
    cancel_url = reverse_lazy("caidapp:workgroup_invitations")

    def dispatch(self, request, *args, **kwargs):
        """Check permissions and set the invitation object for later use in the view."""
        invitation_id = kwargs.get("pk")
        invitation = get_object_or_404(models.WorkGroupInvitation, pk=invitation_id)
        self.invitation = invitation

        if request.user.caiduser.workgroup_admin and invitation.target_workgroup == request.user.caiduser.workgroup:
            pass
        elif invitation.invited_user == request.user.caiduser:
            pass
        else:
            raise PermissionDenied
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        """Set up context data for the detail view."""
        context = super().get_context_data(**kwargs)
        field_data = []

        for field_name in self.fields:
            field = self.model._meta.get_field(field_name)
            value = getattr(self.object, field_name)
            field_data.append(
                {
                    "name": field_name,
                    "verbose_name": field.verbose_name,
                    "value": value,
                }
            )
        context["fields"] = field_data
        logger.debug(f"{self.request.user=}, {self.invitation.invited_user=}, {self.invitation.status=}")
        if self.request.user.caiduser == self.invitation.invited_user and self.invitation.status == "pending":
            context["bottom_button_list"] = [
                {
                    "label": "Accept Invitation",
                    "style": "primary",
                    "url": reverse_lazy(
                        "caidapp:workgroup_invitation_accept",
                        args=[self.object.pk],
                    ),
                    "method": "post",
                },
                {
                    "label": "Decline Invitation",
                    "style": "danger",
                    "url": reverse_lazy(
                        "caidapp:workgroup_invitation_decline",
                        args=[self.object.pk],
                    ),
                    "method": "post",
                },
            ]
        return context


class WorkGroupInvitationDeclineView(LoginRequiredMixin, UpdateView):
    model = models.WorkGroupInvitation
    fields = []
    template_name = "caidapp/generic_form.html"

    title = "Decline Workgroup Invitation"
    description = "This invitation will be declined."

    cancel_url = reverse_lazy("caidapp:workgroup_invitations")

    def get_queryset(self):
        """Limit queryset to pending invitations for the current user."""
        return models.WorkGroupInvitation.objects.filter(
            invited_user=self.request.user.caiduser,
            status="pending",
        )

    def form_valid(self, form):
        """Decline the invitation by updating its status to 'rejected'."""
        invitation = self.object

        if invitation.invited_user != self.request.user.caiduser:
            raise PermissionDenied

        invitation.status = "rejected"
        invitation.responded_at = timezone.now()
        invitation.save(update_fields=["status", "responded_at"])

        return redirect(self.get_success_url())


class WorkGroupInvitationAcceptView(LoginRequiredMixin, UpdateView):
    model = models.WorkGroupInvitation
    fields = []  # žádná pole ve formuláři
    template_name = "caidapp/generic_form.html"

    title = "Accept Workgroup Invitation"
    description = "By accepting this invitation, you will be moved to the new workgroup " "together with all your data."
    cancel_url = reverse_lazy("caidapp:workgroup_invitations")

    def get_queryset(self):
        """Limit queryset to pending invitations for the current user."""
        return models.WorkGroupInvitation.objects.filter(
            invited_user=self.request.user.caiduser,
            status="pending",
        )

    def form_valid(self, form):
        """Accept the invitation and migrate the user to the new workgroup."""
        invitation = self.object

        # 🔐 bezpečnost – ještě jednou pro jistotu
        if invitation.invited_user != self.request.user.caiduser:
            raise PermissionDenied

        # 🔥 migrace uživatele
        migrate_user_to_workgroup(
            user=invitation.invited_user,
            target_workgroup=invitation.target_workgroup,
            approved_by=invitation.invited_by,
        )

        invitation.status = "accepted"
        invitation.responded_at = timezone.now()
        invitation.save(update_fields=["status", "responded_at"])

        return redirect(self.get_success_url())

