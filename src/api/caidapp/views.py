import datetime
import io
import math
import logging
import os
import random
import re
import time
import traceback
import urllib.parse
import uuid
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
from django.core.exceptions import PermissionDenied, ValidationError
from django.core.files.base import ContentFile
from django.core.paginator import Page, Paginator
from django.db import transaction
from django.db.models import CharField, Count, F, Func, IntegerField, Max, Min, OuterRef, Prefetch, Q, QuerySet, Subquery, Value, Window
from django.db.models.functions import Cast, Coalesce
from django.forms import modelformset_factory
from django.forms.models import model_to_dict
from django.http import FileResponse, HttpRequest, HttpResponseNotAllowed, JsonResponse
from django.shortcuts import Http404, HttpResponse, get_object_or_404, redirect, render
from django.template.loader import render_to_string
from django.urls import reverse, reverse_lazy
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.utils.http import url_has_allowed_host_and_scheme
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
from .model_extra import (
    best_identity_merge_candidate,
    compute_identity_suggestions,
    user_has_rw_acces_to_uploadedarchive,
    user_has_rw_access_to_mediafile,
)
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


def _annotate_identity_mediafile_stats(queryset: QuerySet) -> QuerySet:
    """Annotate identity stats using observation-linked mediafiles."""
    return queryset.annotate(
        mediafile_count=Count("animalobservation__mediafile", distinct=True),
        representative_mediafile_count=Count(
            "animalobservation__mediafile",
            filter=Q(animalobservation__identity_is_representative=True),
            distinct=True,
        ),
        locality_count=Count("animalobservation__mediafile__locality", distinct=True),
        last_seen=Max("animalobservation__mediafile__captured_at"),
    )


IDENTITY_PER_PAGE_OPTIONS = (6, 12, 24, 48, 96)
IDENTITY_SORT_FIELDS = {
    "name",
    "sex",
    "birth_date",
    "death_date",
    "coat_type",
    "mediafile_count",
    "representative_mediafile_count",
    "locality_count",
    "last_seen",
}


def _get_identity_records_per_page(request, class_prefix: str, default: int = 24) -> int:
    """Return validated per-page value for identity views."""
    raw_value = request.GET.get("per_page")
    if raw_value is not None:
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = default
        if value in IDENTITY_PER_PAGE_OPTIONS:
            request.session[f"item_number_{class_prefix}"] = value
            return value
    return views_general.get_item_number_anything(request, class_prefix, default=default)


def _get_identity_sort_and_direction(request, class_prefix: str) -> Tuple[str, str]:
    """Return validated sort and direction for identity views."""
    default_sort = "name"
    default_direction = "asc"
    sort = request.GET.get("sort") or request.session.get(f"sort_{class_prefix}") or default_sort
    direction = request.GET.get("dir") or request.session.get(f"dir_{class_prefix}") or default_direction
    if sort not in IDENTITY_SORT_FIELDS:
        sort = default_sort
    if direction not in {"asc", "desc"}:
        direction = default_direction
    request.session[f"sort_{class_prefix}"] = sort
    request.session[f"dir_{class_prefix}"] = direction
    return sort, direction


MEDIAFILE_EXPORT_SCHEMAS = {
    "species_identity": "{species}/{identity}/{hash}_{species}_{identity}{dotext}",
    "identity_dirs": "{identity}/{hash}_{species}_{identity}{dotext}",
    "flat": "{hash}_{species}_{identity}{dotext}",
}

SEQUENCE_DOWNLOAD_SESSION_KEY = "sequence_download_mediafile_ids"
SEQUENCE_DOWNLOAD_RETURN_URL_SESSION_KEY = "sequence_download_return_url"
OBSERVATION_DOWNLOAD_SESSION_KEY = "observation_download_mediafile_ids"
OBSERVATION_DOWNLOAD_RETURN_URL_SESSION_KEY = "observation_download_return_url"
OBSERVATION_PER_PAGE_OPTIONS = (24, 48, 96, 192)

SEQUENCE_EXPORT_COLUMNS = [
    ("unique_name", "Identity"),
    ("code", "Identity code"),
    ("juv_code", "Juvenile code"),
    ("locality name", "Locality name"),
    ("locality_id", "Locality ID"),
    ("mediafile_location", "Media file location (explicit)"),
    ("locality_location", "Locality location"),
    ("location_source", "Effective location source"),
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
    ("taxon_id", "Taxon ID"),
    ("predicted_category", "Taxon"),
    ("media_type", "Media type"),
    ("taxon_verified", "Taxon verified"),
    ("predicted_taxon", "Predicted taxon"),
    ("predicted_taxon_id", "Predicted taxon ID"),
    ("predicted_taxon_confidence", "Predicted taxon confidence"),
    ("identity_is_representative", "Identity is representative"),
    ("identity_id", "Identity ID"),
    ("orientation", "Orientation"),
    ("bbox_cx", "BBox center X (relative)"),
    ("bbox_cy", "BBox center Y (relative)"),
    ("bbox_w", "BBox width (relative)"),
    ("bbox_h", "BBox height (relative)"),
    ("mediafile_note", "Media file note"),
    ("note", "Media file note (legacy column name)"),
]
SEQUENCE_EXPORT_DEFAULT_COLUMNS = [
    "unique_name",
    "code",
    "juv_code",
    "locality name",
    "locality_id",
    "mediafile_location",
    "locality_location",
    "location_source",
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
    "taxon_id",
    "predicted_category",
    "media_type",
    "taxon_verified",
    "predicted_taxon_id",
    "predicted_taxon",
    "predicted_taxon_confidence",
    "identity_id",
    "identity_is_representative",
    "orientation",
    "bbox_cx",
    "bbox_cy",
    "bbox_w",
    "bbox_h",
    "mediafile_note",
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


def _append_created_selection_to_next_url(request, next_url: str, created_param: str, created_id: int) -> str:
    """Append created-object selection params to a validated next URL."""
    parsed_url = urllib.parse.urlsplit(next_url)
    query_params = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parsed_url.query, keep_blank_values=True)
        if key
        not in {
            "created_taxon_id",
            "created_identity_id",
            "select_observation_prefix",
        }
    ]
    query_params.append((created_param, str(created_id)))
    observation_prefix = request.GET.get("select_observation_prefix") or request.POST.get("select_observation_prefix")
    if observation_prefix:
        query_params.append(("select_observation_prefix", observation_prefix))
    return urllib.parse.urlunsplit(parsed_url._replace(query=urllib.parse.urlencode(query_params)))


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
            next_url = request.POST.get("next")
            if next_url and url_has_allowed_host_and_scheme(
                next_url,
                allowed_hosts={request.get_host()},
                require_https=request.is_secure(),
            ):
                if taxon_id is None:
                    next_url = _append_created_selection_to_next_url(request, next_url, "created_taxon_id", taxon.id)
                return redirect(next_url)
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
        form = forms.CaIDUserSettingsForm(
            instance=request.user.caiduser,
            can_edit_workflows=request.user.caiduser.workgroup_admin,
        )
        context = {
            "form": form,
            "headline": "User settings",
            "button": "Save",
        }
        context["nav_dict"] = {
            "Invitations": reverse("caidapp:workgroup_invitations_for_user"),
        }
        if request.user.caiduser.workgroup_admin and request.user.caiduser.workgroup_id:
            context["nav_dict"]["Workgroup Settings"] = reverse(
                "caidapp:workgroup-update", args=[request.user.caiduser.workgroup_id]
            )
        return render(request, self.template_name, context)

    def post(self, request):
        """Handle the form submission for user settings."""
        form = forms.CaIDUserSettingsForm(
            request.POST,
            instance=request.user.caiduser,
            can_edit_workflows=request.user.caiduser.workgroup_admin,
        )
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
        if request.user.caiduser.workgroup_admin and request.user.caiduser.workgroup_id:
            context["nav_dict"]["Workgroup Settings"] = reverse(
                "caidapp:workgroup-update", args=[request.user.caiduser.workgroup_id]
            )
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
        contains_identities=False,
    ).exclude(is_for_identification=True, taxon_for_identification__isnull=True)
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
    _reconcile_identification_task_status(workgroup, "init")
    _reconcile_identification_task_status(workgroup, "run")
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
            representative_mediafile_count=Count(
                "animalobservation__mediafile",
                filter=Q(animalobservation__identity_is_representative=True),
                distinct=True,
            ),
            non_representative_mediafile_count=Count(
                "animalobservation__mediafile",
                filter=Q(animalobservation__identity_is_representative=False),
                distinct=True,
            ),
        )
        .filter(non_representative_mediafile_count__gt=0)
        .order_by("representative_mediafile_count", "-non_representative_mediafile_count")
    )
    manual_identification_count = models.get_mediafiles_with_missing_identity(request.user.caiduser).count()
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
            manual_identification_count=manual_identification_count,
            suggestion_candidate_mediafile_count=suggestion_candidate_mediafile_count,
            suggestion_candidate_archive_count=suggestion_candidate_archive_count,
            suggestion_run_info=suggestion_run_info,
        ),
    )


def _identification_status_style(status: str) -> str:
    """Return a Bootstrap status color for identification state labels."""
    normalized = str(status or "").strip().lower()
    if normalized in {"done", "finished", "success", "succeeded"}:
        return "success"
    if normalized in {"error", "failed", "failure"}:
        return "danger"
    if normalized in {"processing", "scheduled", "started", "progress"}:
        return "primary"
    return "secondary"


@login_required
def identification_information(request) -> HttpResponse:
    """Show workgroup-admin diagnostics for the re-identification workflow."""
    if not user_can_manage_identification(request.user):
        return HttpResponseNotAllowed("Identification information is for workgroup admins only.")

    workgroup = request.user.caiduser.workgroup
    run_statistics = workgroup.identification_run_statistics.all()
    recent_runs = list(run_statistics[:50])
    for run in recent_runs:
        run.status_style = _identification_status_style(run.status)

    latest_init_run = run_statistics.filter(operation="init").first()
    latest_identify_run = run_statistics.filter(operation="identify").first()
    current_model = workgroup.identification_model
    initialized_model = workgroup.identification_initialized_model

    mediafiles = MediaFile.objects.filter(parent__owner__workgroup=workgroup)
    counts = {
        "identities": IndividualIdentity.objects.filter(owner_workgroup=workgroup).count(),
        "mediafiles": mediafiles.count(),
        "identified_mediafiles": mediafiles.filter(observations__identity__isnull=False).distinct().count(),
        "representative_mediafiles": mediafiles.filter(
            observations__identity__isnull=False,
            observations__identity_is_representative=True,
        ).distinct().count(),
        "initialized_reference_mediafiles": mediafiles.filter(used_for_init_identification=True).count(),
        "identification_uploads": UploadedArchive.objects.filter(
            owner__workgroup=workgroup,
            is_for_identification=True,
        ).count(),
        "queued_mediafiles": MediafilesForIdentification.objects.filter(
            mediafile__parent__owner__workgroup=workgroup
        ).values("mediafile_id").distinct().count(),
    }

    return render(
        request,
        "caidapp/identification_information.html",
        {
            "workgroup": workgroup,
            "current_model": current_model,
            "initialized_model": initialized_model,
            "model_ready": workgroup.identification_model_is_initialized(),
            "init_status_style": _identification_status_style(workgroup.identification_init_status),
            "identify_status_style": _identification_status_style(workgroup.identification_reid_status),
            "latest_init_run": latest_init_run,
            "latest_identify_run": latest_identify_run,
            "recent_runs": recent_runs,
            "counts": counts,
        },
    )


@login_required
def download_init_identification_csv(request):
    """Download the latest workgroup init identification CSV."""
    caiduser = request.user.caiduser
    if not caiduser.workgroup_admin:
        return HttpResponseNotAllowed("Only workgroup admins can download init identification CSV.")

    csv_path = Path(settings.MEDIA_ROOT) / caiduser.workgroup.name / "init_identification.csv"
    return _download_identification_csv(
        csv_path,
        f"init_identification_workgroup_{caiduser.workgroup.id}.csv",
        "Init identification CSV does not exist yet.",
    )


@login_required
def download_run_identification_csv(request):
    """Download the latest workgroup run identification CSV."""
    caiduser = request.user.caiduser
    if not caiduser.workgroup_admin:
        return HttpResponseNotAllowed("Only workgroup admins can download run identification CSV.")

    reid_runs_dir = Path(settings.MEDIA_ROOT) / caiduser.workgroup.name / "reid_runs"
    candidates = list(reid_runs_dir.glob("*/identification_metadata.csv")) if reid_runs_dir.exists() else []
    csv_path = max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
    return _download_identification_csv(
        csv_path,
        f"run_identification_workgroup_{caiduser.workgroup.id}.csv",
        "Run identification CSV does not exist yet.",
    )


def _download_identification_csv(csv_path: Path | None, filename: str, missing_message: str) -> HttpResponse:
    if csv_path is None or not csv_path.exists():
        return HttpResponse(missing_message, content_type="text/plain", status=404)
    return FileResponse(open(csv_path, "rb"), as_attachment=True, filename=filename, content_type="text/csv")


def _celery_progress_payload(task_id: str, default_message: str) -> dict | None:
    if not task_id:
        return None
    try:
        task = AsyncResult(task_id)
        state = task.state
        info = task.info
        if state == "PROGRESS" and isinstance(info, dict):
            raw_percent = info.get("percent")
            percent = max(0, min(int(raw_percent), 99)) if raw_percent is not None else None
            return {
                "state": state,
                "percent": percent,
                "stage": str(info.get("stage", "")),
                "message": str(info.get("message", default_message)),
            }
        if state == "SUCCESS" and isinstance(info, dict) and info.get("status") == "ERROR":
            return {
                "state": state,
                "percent": None,
                "stage": "failed",
                "message": "Identification failed",
            }
        if state == "SUCCESS":
            return {
                "state": state,
                "percent": 99,
                "stage": "finalize",
                "message": "Finalizing identification results",
            }
        return {
            "state": state,
            "percent": None,
            "stage": "queued" if state == "PENDING" else "starting",
            "message": "Queued for identification" if state == "PENDING" else default_message,
        }
    except Exception:
        logger.warning("Could not read identification progress for task %s", task_id, exc_info=True)
        return None


def _identification_task_fields(operation: str) -> dict:
    if operation == "init":
        return {
            "status": "identification_init_status",
            "at": "identification_init_at",
            "message": "identification_init_message",
            "task_id": "identification_scheduled_init_task_id",
            "eta": "identification_scheduled_init_eta",
            "label": "Init identification",
        }
    return {
        "status": "identification_reid_status",
        "at": "identification_reid_at",
        "message": "identification_reid_message",
        "task_id": "identification_scheduled_run_task_id",
        "eta": "identification_scheduled_run_eta",
        "label": "Compare with identity database",
    }


def _identification_task_state(task_id: str | None) -> str | None:
    if not task_id:
        return None
    try:
        return AsyncResult(task_id).state
    except Exception:
        logger.warning("Could not read identification task state for task %s", task_id, exc_info=True)
        return None


def _identification_task_info(task_id: str | None):
    if not task_id:
        return None
    try:
        return AsyncResult(task_id).info
    except Exception:
        logger.warning("Could not read identification task info for task %s", task_id, exc_info=True)
        return None


def _reconcile_identification_task_status(workgroup: WorkGroup, operation: str) -> None:
    fields = _identification_task_fields(operation)
    status = getattr(workgroup, fields["status"])
    task_id = getattr(workgroup, fields["task_id"])
    if status not in {"Processing", "Scheduled"} or not task_id:
        return

    task_state = _identification_task_state(task_id)
    if task_state not in {"SUCCESS", "FAILURE", "REVOKED"}:
        return

    now_value = django.utils.timezone.now()
    task_info = _identification_task_info(task_id) if task_state == "SUCCESS" else None
    task_returned_error = isinstance(task_info, dict) and task_info.get("status") == "ERROR"

    if task_state == "SUCCESS" and not task_returned_error:
        next_status = "Finished"
        message = (
            f"{fields['label']} worker task ended with Celery state SUCCESS at "
            f"{now_value:%Y-%m-%d %H:%M}. Dashboard status was reconciled automatically."
        )
    elif task_state == "REVOKED":
        next_status = "Not initiated"
        message = (
            f"{fields['label']} worker task was revoked at {now_value:%Y-%m-%d %H:%M}. "
            "Dashboard status was reconciled automatically."
        )
    else:
        next_status = "Failed"
        error_message = ""
        if task_returned_error and isinstance(task_info, dict):
            error_message = f" Worker returned: {task_info.get('error', 'ERROR')}."
        message = (
            f"{fields['label']} worker task ended with Celery state {task_state} at "
            f"{now_value:%Y-%m-%d %H:%M}.{error_message} Check API callback and worker logs."
        )

    setattr(workgroup, fields["status"], next_status)
    setattr(workgroup, fields["at"], now_value)
    setattr(workgroup, fields["message"], message)
    setattr(workgroup, fields["task_id"], None)
    setattr(workgroup, fields["eta"], None)
    workgroup.save(
        update_fields=[
            fields["status"],
            fields["at"],
            fields["message"],
            fields["task_id"],
            fields["eta"],
        ]
    )


def _workgroup_identification_progress(workgroup: WorkGroup, operation: str) -> dict:
    _reconcile_identification_task_status(workgroup, operation)
    if operation == "init":
        status = workgroup.identification_init_status
        message = workgroup.identification_init_message
        task_id = workgroup.identification_scheduled_init_task_id
        eta = workgroup.identification_scheduled_init_eta
        started_at = workgroup.identification_init_at if status == "Processing" else None
        default_message = "Initializing identification database"
    else:
        status = workgroup.identification_reid_status
        message = workgroup.identification_reid_message
        task_id = workgroup.identification_scheduled_run_task_id
        eta = workgroup.identification_scheduled_run_eta
        started_at = workgroup.identification_reid_at if status == "Processing" else None
        default_message = "Generating identification suggestions"

    progress = None
    task_state = _identification_task_state(task_id) if task_id else None
    if status in {"Processing", "Scheduled"}:
        progress = _celery_progress_payload(task_id, message or default_message)
        if progress is None:
            progress = {
                "state": "PENDING",
                "percent": None,
                "stage": "scheduled" if status == "Scheduled" else "starting",
                "message": message or default_message,
            }

    return {
        "status": status,
        "message": message,
        "task_id": task_id or "",
        "eta": eta.isoformat() if eta else None,
        "started_at": started_at.isoformat() if started_at else None,
        "task_state": task_state,
        "progress": progress,
    }


@login_required
def identification_progress_api(request):
    workgroup = request.user.caiduser.workgroup
    if workgroup is None:
        return JsonResponse({"error": "No workgroup assigned."}, status=400)
    return JsonResponse(
        {
            "init": _workgroup_identification_progress(workgroup, "init"),
            "run": _workgroup_identification_progress(workgroup, "run"),
        }
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
        mediafile_count=Count("mediafile", distinct=True),  # Count of all related MediaFiles
        mediafile_count_with_taxon=Count(
            "mediafile",
            filter=Q(mediafile__observations__taxon=F("taxon_for_identification")),
            distinct=True,
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
        identification_mediafiles = MediaFile.objects.filter(
            parent__owner__workgroup=workgroup,
            parent__is_for_identification=True,
            parent__import_finished=True,
        )
        identification_mediafiles = models.filter_mediafiles_by_identification_taxon(
            identification_mediafiles, workgroup
        )
        has_assigned_identity = identification_mediafiles.filter(
            observations__identity__isnull=False
        ).exists()
        if identification_mediafiles.exists() and not has_assigned_identity:
            return {
                "label": "Manual identification",
                "url": reverse("caidapp:manual_identification"),
                "section": "identification",
            }

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
            observations__identity_is_representative=True,
            parent__taxon_for_identification__isnull=False,
        ).distinct().count()

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
        return _annotate_identity_mediafile_stats(
            IndividualIdentity.objects.filter(
                pk__in=selected_ids,
                owner_workgroup=self.request.user.caiduser.workgroup,
            )
        ).order_by("name", "id")

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
            return_url = request.POST.get("return_url")
            if not return_url or not url_has_allowed_host_and_scheme(
                return_url,
                allowed_hosts={request.get_host()},
                require_https=request.is_secure(),
            ):
                return redirect(reverse("caidapp:individual_identities"))

            parsed_url = urllib.parse.urlsplit(return_url)
            query_params = [
                (key, value)
                for key, value in urllib.parse.parse_qsl(parsed_url.query, keep_blank_values=True)
                if key != "page"
            ]
            query_params.append(("page", "1"))
            return_url = urllib.parse.urlunsplit(parsed_url._replace(query=urllib.parse.urlencode(query_params)))
            return redirect(return_url)

        messages.warning(request, "Choose a bulk action.")
        return redirect(request.get_full_path())

    def get_queryset(self):
        """Get queryset for the view."""
        class_prefix = "identities_" + self.request.GET.get("view", "cards")

        self.paginate_by = _get_identity_records_per_page(self.request, class_prefix, default=24)
        qs = IndividualIdentity.objects.filter(Q(owner_workgroup=self.request.user.caiduser.workgroup) & ~Q(name="nan"))
        qs = _annotate_identity_mediafile_stats(qs)

        self.filterset = filters.IndividualIdentityFilter(self.request.GET, queryset=qs)
        qs = self.filterset.qs

        sort, direction = _get_identity_sort_and_direction(self.request, class_prefix)
        order_by = sort if direction == "asc" else f"-{sort}"
        logger.debug(f"Sorting identities by {order_by}")
        qs = qs.order_by(order_by, "id")
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
        context["filter"] = self.filterset
        context["list_display"] = []
        class_prefix = "identities_" + self.request.GET.get("view", "cards")
        context["sort_key"] = self.request.session.get(f"sort_{class_prefix}", "name")
        context["sort_direction"] = self.request.session.get(f"dir_{class_prefix}", "asc")
        context["per_page_options"] = IDENTITY_PER_PAGE_OPTIONS
        context["records_per_page"] = self.paginate_by
        context = add_querystring_to_context(self.request, context)
        return context


@login_required
def individual_identity_create(request):
    """Create new individual_identity."""
    if request.method == "POST":
        form = IndividualIdentityForm(request.POST)
        if form.is_valid():
            individual_identity = form.save(commit=False)
            individual_identity.owner_workgroup = request.user.caiduser.workgroup
            individual_identity.updated_by = request.user.caiduser
            individual_identity.save()
            url = request.META.get("HTTP_REFERER", reverse("caidapp:individual_identities"))
            next_url = request.GET.get("next") or request.POST.get("next") or url
            if next_url and url_has_allowed_host_and_scheme(
                next_url,
                allowed_hosts={request.get_host()},
                require_https=request.is_secure(),
            ):
                next_url = _append_created_selection_to_next_url(
                    request, next_url, "created_identity_id", individual_identity.id
                )
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
        mediafiles_count = individual_identity.count_of_mediafiles()
        sequences_count = (
            individual_identity.observation_mediafiles().exclude(sequence_id__isnull=True).values("sequence_id").distinct().count()
        )
        media_files = individual_identity.observation_mediafiles().filter(
            observations__identity=individual_identity,
            observations__identity_is_representative=True,
        )
        # media_file = media_files.first()

        nav_dict = {}
        if individual_identity:
            nav_dict[f"Media Files ({mediafiles_count})"] = reverse_lazy(
                "caidapp:individual_identity_mediafiles",
                kwargs={"individual_identity_id": individual_identity.id},
            )
            nav_dict[f"Sequences ({sequences_count})"] = (
                f"{reverse('caidapp:sequences')}?individual_identity_id={individual_identity.id}"
            )
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
                "mediafiles_count": mediafiles_count,
                "sequences_count": sequences_count,
                "mediafiles_url": reverse_lazy(
                    "caidapp:individual_identity_mediafiles",
                    kwargs={"individual_identity_id": individual_identity.id},
                ),
                "delete_button_url": reverse_lazy(
                    "caidapp:delete_individual_identity",
                    kwargs={"individual_identity_id": individual_identity.id},
                ),
                "merge_button_url": reverse_lazy(
                    "caidapp:merge_identities",
                    kwargs={"individual_identity1_id": individual_identity.id},
                ),
                "nav_dict": nav_dict,
                "right_nav": right_nav,
            }
        )
        return context

    def get_success_url(self):
        """Return to the originating page when possible."""
        next_url = self.request.GET.get("next") or self.request.POST.get("next")
        if next_url and url_has_allowed_host_and_scheme(
            next_url,
            allowed_hosts={self.request.get_host()},
            require_https=self.request.is_secure(),
        ):
            return next_url
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
    top_mediafile = (
        identity.observation_mediafiles()
        .filter(observations__identity=identity, observations__identity_is_representative=True)
        .first()
    )
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
            "top_identity": identity,
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
            "top_identity": reid_suggestion.identity,
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


def get_best_representative_mediafiles(identity, orientation=None, max_count=5) -> list[MediaFile]:
    """Get best representative mediafiles for identity."""
    # Pokud máme předem načtené reprezentativní mediafiles
    candidates = getattr(identity, "representative_mediafiles_candidates", None)
    if candidates is not None:
        if candidates:
            return candidates[:max_count]
        # fallback na reprezentativní bez orientace
        fallback = list(
            identity.observation_mediafiles()
            .filter(observations__identity=identity, observations__identity_is_representative=True)
            .order_by("-captured_at")[:max_count]
        )
        if fallback:
            return fallback
        return list(identity.observation_mediafiles().order_by("-captured_at")[:max_count])
    else:
        # fallback: přímé dotazy jako dřív
        qs = identity.observation_mediafiles()
        mf = qs.filter(
            observations__identity=identity,
            observations__identity_is_representative=True,
            observations__orientation=orientation,
        )
        if not mf.exists():
            mf = qs.filter(observations__identity=identity, observations__identity_is_representative=True)
        if not mf.exists():
            mf = qs.all()
        return list(mf.distinct().order_by("-captured_at")[:max_count])


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
        identity_ids = foridentification.top_mediafiles.values_list("identity_id", flat=True)
        logger.debug(f"{identity_ids=}")

        identity_ids = [i for i in identity_ids if i is not None]
        logger.debug(f"{identity_ids=}")

        reid_observation_id = (foridentification.mediafile.metadata_json or {}).get("reid_observation_id")
        unknown_observation = foridentification.mediafile.observations.filter(id=reid_observation_id).first()
        if unknown_observation is None and foridentification.mediafile.observations.count() == 1:
            unknown_observation = foridentification.mediafile.observations.first()
        orientation_of_unknown = unknown_observation.orientation if unknown_observation is not None else None

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
        remaining_identities = (
            IndividualIdentity.objects.filter(
                Q(owner_workgroup=request.user.caiduser.workgroup),
                ~Q(name="nan"),
                ~Q(id__in=identity_ids),
            )
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
            identity.representative_mediafiles = list(
                identity.observation_mediafiles().order_by("captured_at", "id")[:3]
            )

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

    identity.representative_mediafiles = identity.observation_mediafiles().filter(
        observations__identity=identity,
        observations__identity_is_representative=True,
    )

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
    """Set identity for the observation being identified."""
    mediafiles_for_identification = get_object_or_404(MediafilesForIdentification, id=mediafiles_for_identification_id)
    representative = request.GET.get("representative") == "1"
    individual_identity = get_object_or_404(IndividualIdentity, id=individual_identity_id)

    # if request.user.caiduser.workgroup != mediafile.parent.owner.workgroup:
    #     return HttpResponseNotAllowed("Not allowed to work with this media file.")
    if request.user.caiduser.workgroup != individual_identity.owner_workgroup:
        return HttpResponseNotAllowed("Not allowed to work with this media file.")
    if request.user.caiduser.workgroup != mediafiles_for_identification.mediafile.parent.owner.workgroup:
        return HttpResponseNotAllowed("Not allowed to work with this media file.")

    mediafile = mediafiles_for_identification.mediafile
    reid_observation_id = (mediafile.metadata_json or {}).get("reid_observation_id")
    observation = mediafile.observations.filter(id=reid_observation_id).first() if reid_observation_id else None
    if observation is None:
        observations = list(mediafile.observations.order_by("id")[:2])
        if len(observations) == 1:
            observation = observations[0]
    if observation is None:
        messages.warning(
            request,
            "This media file has multiple or no observations. Choose the target animal in the media file editor.",
        )
        return redirect("caidapp:media_file_update", pk=mediafile.id)

    observation.identity = individual_identity
    observation.identity_is_representative = representative
    observation.updated_by = request.user.caiduser
    observation.updated_at = django.utils.timezone.now()
    observation.save(update_fields=["identity", "identity_is_representative", "updated_by", "updated_at"])
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
    source_model = request.user.caiduser.workgroup.identification_model
    now = django.utils.timezone.now()
    new_name = models.build_trained_identification_model_name(
        source_model.name,
        request.user.caiduser.workgroup.name,
        now,
    )
    clean_new_name = re.sub(r"[^a-zA-Z0-9 _-]", "", new_name)

    group_dir = Path(settings.MEDIA_ROOT) / request.user.caiduser.workgroup.name
    output_dir = group_dir / "models" / clean_new_name
    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_path = output_dir / f"{clean_new_name}.pth"
    identity_metadata_file = group_dir / "train_identification.csv"

    csv_data = _prepare_dataframe_for_identification(
        mediafiles_qs,
        representative_only=True,
        require_identity=True,
    )

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
        name=new_name,
        model_path=str(output_model_path),
        base_model_path=source_model.get_runtime_model_source(),
        workgroup=request.user.caiduser.workgroup,
        source_identification_model=source_model,
    )
    new_identification_model.save()
    sig = signature(
        "train_identification",
        kwargs={
            # csv file should contain image_path, class_id, label
            "input_metadata_file": str(identity_metadata_file),
            "organization_id": request.user.caiduser.workgroup.id,
            "identification_model": {
                "name": new_identification_model.name,
                "base_model_source": source_model.get_runtime_model_source(),
                "initial_weights_path": source_model.get_runtime_checkpoint_path(),
                "output_weights_path": str(new_identification_model.model_path),
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
    if not user_can_manage_identification(request.user):
        return HttpResponseNotAllowed("Stopping identification is for workgroup admins only.")
    workgroup = request.user.caiduser.workgroup
    redirect_url = request.META.get("HTTP_REFERER", reverse("caidapp:dash_identities"))
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
    return redirect(redirect_url)


# TODO rename to identification button style
def _single_species_button_style(request) -> dict:
    workgroup = request.user.caiduser.workgroup

    is_initiated = request.user.caiduser.workgroup.identification_init_at is not None

    n_representative = workgroup.mediafiles_for_train_or_init_identification().count()
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

    init_disabled = (
        workgroup.identification_init_status in {"Processing", "Scheduled"}
        or workgroup.identification_reid_status in {"Processing", "Scheduled"}
    )
    logger.debug(
        f"{init_disabled=}, {workgroup.identification_init_status=}, "
        f"{workgroup.identification_reid_status=}, {exists_representative=}"
    )
    btn_styles["init_identification"]["class"] += " disabled" if init_disabled else ""
    btn_styles["init_identification"][
        "tooltip"
    ] = f"Identification initialization with {n_representative} representative media files."
    btn_styles["init_identification"][
        "confirm"
    ] = f"Identification initialization with {n_representative} media files will take some time. Continue?"

    btn_styles["run_identification"]["class"] += (
        " disabled"
        if (
            (not is_initiated)
            or (workgroup.identification_init_status in {"Processing", "Scheduled"})
            or (not workgroup.identification_model_is_initialized())
        )
        else ""
    )
    if workgroup.identification_model_is_initialized():
        btn_styles["run_identification"]["tooltip"] = (
            f"Identification suggestion for {n_unidentified} archives."
        )
    else:
        btn_styles["run_identification"]["tooltip"] = (
            "Identification is waiting for initialization of the selected model."
        )
    btn_styles["run_identification"][
        "confirm"
    ] = f"Identification of {n_unidentified} archives will take some time. Continue?"
    btn_styles["n_representative"] = n_representative
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
    if not user_can_manage_identification(request.user):
        return HttpResponseNotAllowed("Identification suggestions are for workgroup admins only.")
    workgroup = request.user.caiduser.workgroup
    if not workgroup.identification_model_is_initialized():
        messages.error(request, "Identification is waiting for initialization of the selected model.")
        return redirect(request.META.get("HTTP_REFERER", "/"))
    task_state = None
    if workgroup.identification_scheduled_run_task_id:
        task_state = AsyncResult(workgroup.identification_scheduled_run_task_id).state
    if workgroup.identification_reid_status == "Processing" and task_state in {"STARTED", "PROGRESS"}:
        messages.info(request, "Identification suggestion generation is already running.")
        return redirect(request.META.get("HTTP_REFERER", "/"))

    if workgroup.identification_scheduled_run_task_id:
        current_app.control.revoke(workgroup.identification_scheduled_run_task_id, terminate=True)

    tasks.run_identification_on_unidentified_for_workgroup(workgroup.id, request=request)
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
    if not workgroup.identification_model_is_initialized():
        messages.error(request, "Identification is waiting for initialization of the selected model.")
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
    if not workgroup.identification_model_is_initialized():
        logger.warning(
            "Cannot run identification for workgroup %s: selected model %s is not initialized (initialized model %s).",
            workgroup.id,
            workgroup.identification_model_id,
            workgroup.identification_initialized_model_id,
        )
        return False
    initialized_model = workgroup.identification_initialized_model
    selection = selection or {}
    if uploaded_archives is None:
        uploaded_archives = tasks.get_uploaded_archives_pending_identification(workgroup)

    uploaded_archives = list(uploaded_archives)
    if not uploaded_archives:
        return False

    uploaded_archive_ids = [uploaded_archive.id for uploaded_archive in uploaded_archives]
    bulk_selection = {**selection, "uploaded_archive_ids": uploaded_archive_ids}
    mediafiles, observation_taxon, _require_observations = tasks.resolve_identification_selection(
        workgroup,
        selection=bulk_selection,
    )
    logger.debug(f"Generating CSV for bulk identification with {len(mediafiles)} records...")

    csv_data = _prepare_dataframe_for_identification(mediafiles, observation_taxon=observation_taxon)
    df = pd.DataFrame(csv_data)
    if df.shape[0] == 0:
        logger.warning("No records found for bulk identification in workgroup %s.", workgroup.id)
        return False
    image_number, video_number = tasks.count_identification_media_types(mediafiles)
    statistic = tasks.create_identification_run_statistic(
        workgroup=workgroup,
        operation="identify",
        image_number=image_number,
        video_number=video_number,
    )

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
                "name": initialized_model.name,
                "model_source": initialized_model.get_runtime_model_source(),
                "weights_path": initialized_model.get_runtime_checkpoint_path(),
            },
        ),
    )
    identify_task = identify_signature.apply_async(
        link=tasks.identify_bulk_on_success.s(
            workgroup_id=workgroup.id,
            uploaded_archive_ids=uploaded_archive_ids,
            statistic_id=statistic.id,
        ),
        link_error=tasks.identify_bulk_on_error.s(
            workgroup_id=workgroup.id,
            uploaded_archive_ids=uploaded_archive_ids,
            statistic_id=statistic.id,
        ),
    )
    statistic.task_id = identify_task.id
    statistic.save(update_fields=["task_id"])
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
    if not workgroup.identification_model_is_initialized():
        logger.warning(
            "Cannot run identification for workgroup %s: selected model %s is not initialized (initialized model %s).",
            workgroup.id,
            workgroup.identification_model_id,
            workgroup.identification_initialized_model_id,
        )
        return False
    initialized_model = workgroup.identification_initialized_model
    logger.debug("Generating CSV for run_identification...")
    mediafiles, observation_taxon, _require_observations = tasks.resolve_identification_selection(
        workgroup,
        uploaded_archive=uploaded_archive,
        selection=selection,
    )
    logger.debug(f"Generating CSV for init_identification with {len(mediafiles)} records...")
    uploaded_archive.identification_status = "IAIP"

    csv_data = _prepare_dataframe_for_identification(mediafiles, observation_taxon=observation_taxon)
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
            level=Notification.WARNING,
            workgroups=[workgroup],
        )

        return False
        # return redirect(request.META.get("HTTP_REFERER", "/"))
    image_number, video_number = tasks.count_identification_media_types(mediafiles)
    statistic = tasks.create_identification_run_statistic(
        workgroup=workgroup,
        operation="identify",
        image_number=image_number,
        video_number=video_number,
    )

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
                "name": initialized_model.name,
                "model_source": initialized_model.get_runtime_model_source(),
                "weights_path": initialized_model.get_runtime_checkpoint_path(),
            },
        ),
    )
    identify_task = identify_signature.apply_async(
        link=identify_on_success.s(
            uploaded_archive_id=uploaded_archive.id,
            statistic_id=statistic.id,
        ),
        link_error=on_error_in_upload_processing.s(statistic_id=statistic.id),
    )
    statistic.task_id = identify_task.id
    statistic.save(update_fields=["task_id"])
    workgroup.identification_reid_status = "Processing"
    workgroup.identification_reid_at = django.utils.timezone.now()
    workgroup.identification_reid_message = (
        f"Running identification for {uploaded_archive.name} and {df.shape[0]} media files."
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


def user_can_manage_identification(user) -> bool:
    """Return whether the user may start or stop workgroup identification jobs."""
    if not user.is_authenticated:
        return False
    caiduser = getattr(user, "caiduser", None)
    return bool(caiduser and caiduser.workgroup and caiduser.workgroup_admin)


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
        filter_kwargs.update({"observations__taxon_verified": taxon_verified})
    if album_hash is not None:
        album = get_object_or_404(Album, hash=album_hash)
        filter_kwargs.update(dict(album=album))
    if individual_identity_id is not None:
        individual_identity = get_object_or_404(IndividualIdentity, pk=individual_identity_id)
        filter_kwargs.update({"observations__identity": individual_identity})
    if taxon_id is not None:
        taxon = get_object_or_404(Taxon, pk=taxon_id)
        filter_kwargs.update({"observations__taxon": taxon})
    if uploadedarchive_id is not None:
        uploadedarchive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
        filter_kwargs.update(dict(parent=uploadedarchive))
    if identity_is_representative is not None:
        filter_kwargs.update({"observations__identity_is_representative": identity_is_representative})
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

        vector = SearchVector("observations__taxon__name", "locality__name")
        query = SearchQuery(query)
        logger.debug(str(query))
        mediafiles = mediafiles.annotate(rank=SearchRank(vector, query)).filter(rank__gt=0).order_by("-rank")
        # return mediafiles
    mediafiles = mediafiles.select_related(
        "parent", "locality", "updated_by", "sequence"
    ).prefetch_related("observations__taxon", "observations__identity")

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
    init_identification_candidates = bool(
        _parse_bool_query_param(request.GET.get("init_identification_candidates"))
    )

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
    elif init_identification_candidates:
        page_title = "Media files - init identification candidates"
    else:
        page_title = "Media files"

    sequence_id = request.GET.get("sequence")
    sequence = get_object_or_404(models.Sequence, pk=sequence_id) if sequence_id else None

    mediafiles_name_suggestion = None
    if taxon_verified is not None:
        filter_kwargs["observations__taxon_verified"] = taxon_verified

    if show_overview_button:
        filter_kwargs["observations__taxon_verified"] = False
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
        filter_kwargs["observations__identity"] = individual_identity
        mediafiles_name_suggestion = f"individual_identity_{individual_identity.name}"
    elif individual_identity_ids:
        active_identities = _get_active_identities_from_request(request)
        identity_names = ", ".join(str(identity) for identity in active_identities[:3])
        if len(active_identities) > 3:
            identity_names += f", +{len(active_identities) - 3}"
        page_title = f"Media files - {identity_names}"
        filter_kwargs["observations__identity_id__in"] = [identity.id for identity in active_identities]
        mediafiles_name_suggestion = "individual_identities"
    elif locality_hash is not None:
        locality = _get_locality_for_user_or_404(request, locality_hash)
        page_title = f"Media files - {locality.name}"
        filter_kwargs["locality"] = locality
        mediafiles_name_suggestion = f"locality_{locality.name}"
    elif identity_is_representative is not None:
        page_title = "Media files - representative"
        filter_kwargs["observations__identity_is_representative"] = identity_is_representative
        mediafiles_name_suggestion = f"representative_identity_{str(identity_is_representative)}"

    if individual_identity_id is not None and identity_is_representative is not None:
        filter_kwargs["observations__identity_is_representative"] = identity_is_representative

    if init_identification_candidates:
        mediafiles = request.user.caiduser.workgroup.mediafiles_for_train_or_init_identification()
        mediafiles_name_suggestion = "init_identification_candidates"
    else:
        mediafiles = MediaFile.objects.filter(
            Q(album__albumsharerole__user=request.user.caiduser)
            | Q(**models.user_has_access_filter_params(request.user.caiduser, "parent__owner")),
            **filter_kwargs,
        )
    mediafile_filter_data = request.GET.copy()
    if identity_is_representative is not None:
        mediafile_filter_data.pop("identity_is_representative", None)
    if init_identification_candidates:
        mediafile_filter_data.pop("init_identification_candidates", None)
    mediafile_filter = filters.MediaFileFilter(mediafile_filter_data, queryset=mediafiles, request=request)
    full_mediafiles = mediafile_filter.qs.filter(sequence=sequence) if sequence else mediafile_filter.qs
    if identity_is_representative and not init_identification_candidates:
        full_mediafiles = models.filter_mediafiles_by_identification_taxon(
            full_mediafiles,
            request.user.caiduser.workgroup,
        )
    full_mediafiles = full_mediafiles.distinct()

    return full_mediafiles, mediafile_filter, page_title, mediafiles_name_suggestion


def _get_observation_records_per_page(request: HttpRequest) -> int:
    """Return a validated per-page setting for the observation review view."""
    raw_value = request.GET.get("per_page")
    if raw_value is not None:
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = OBSERVATION_PER_PAGE_OPTIONS[0]
        if value in OBSERVATION_PER_PAGE_OPTIONS:
            request.session["observations_records_per_page"] = value
            return value
    return request.session.get("observations_records_per_page", OBSERVATION_PER_PAGE_OPTIONS[0])


def _get_observations_queryset(request: HttpRequest) -> Tuple[QuerySet, filters.AnimalObservationFilter]:
    """Return observations accessible to the current user, filtered per observation.

    This must not use the mediafile list helper: a mediafile-level identity
    filter would include every sibling observation from a matching image.
    """
    caiduser = request.user.caiduser
    observations = AnimalObservation.objects.filter(
        Q(mediafile__album__albumsharerole__user=caiduser)
        | Q(**models.user_has_access_filter_params(caiduser, "mediafile__parent__owner"))
    ).select_related(
        "mediafile",
        "mediafile__parent",
        "mediafile__locality",
        "mediafile__sequence",
        "taxon",
        "predicted_taxon",
        "identity",
        "updated_by",
    )
    observation_filter = filters.AnimalObservationFilter(request.GET, queryset=observations, request=request)
    return observation_filter.qs.distinct(), observation_filter


def _get_mediafiles_for_observation_export(request: HttpRequest) -> QuerySet:
    """Return complete mediafiles represented by the current observation filter.

    The filter selects observations, but the spreadsheet is intentionally
    mediafile-complete: every matching mediafile exports all its observations
    (or one blank row when it has none).  This preserves the established
    import/export round-trip format.
    """
    observation_queryset, observation_filter = _get_observations_queryset(request)
    sequence_id = _parse_int_query_param_or_404(request.GET.get("sequence"), "sequence id")
    if sequence_id is not None:
        observation_queryset = observation_queryset.filter(mediafile__sequence_id=sequence_id)

    observation_only_filter_names = (
        "taxon",
        "identity",
        "orientation",
        "taxon_verified",
        "identity_is_representative",
        "has_bbox",
        "has_identity",
        "multiple_in_mediafile",
        "objects_per_image",
        "objects_per_image_min",
        "objects_per_image_max",
    )
    has_observation_only_filter = any(
        observation_filter.form.cleaned_data.get(name) not in (None, "")
        for name in observation_only_filter_names
    )
    if has_observation_only_filter:
        mediafiles = MediaFile.objects.filter(id__in=observation_queryset.order_by().values("mediafile_id"))
    else:
        # File-context filters do not require an observation to exist.  This
        # branch is what preserves a blank spreadsheet row for empty mediafiles.
        mediafiles = MediaFile.objects.filter(
            Q(album__albumsharerole__user=request.user.caiduser)
            | Q(**models.user_has_access_filter_params(request.user.caiduser, "parent__owner"))
        )
        uploaded_archive = observation_filter.form.cleaned_data.get("uploadedarchive")
        locality = observation_filter.form.cleaned_data.get("locality")
        captured_at = observation_filter.form.cleaned_data.get("captured_at")
        if uploaded_archive is not None:
            mediafiles = mediafiles.filter(parent=uploaded_archive)
        if locality is not None:
            mediafiles = mediafiles.filter(locality=locality)
        if captured_at is not None:
            if captured_at.start is not None:
                mediafiles = mediafiles.filter(captured_at__date__gte=captured_at.start)
            if captured_at.stop is not None:
                mediafiles = mediafiles.filter(captured_at__date__lte=captured_at.stop)
        if sequence_id is not None:
            mediafiles = mediafiles.filter(sequence_id=sequence_id)
        search_value = observation_filter.form.cleaned_data.get("search")
        if search_value:
            mediafiles = mediafiles.filter(
                Q(original_filename__icontains=search_value)
                | Q(locality__name__icontains=search_value)
                | Q(observations__taxon__name__icontains=search_value)
                | Q(observations__identity__name__icontains=search_value)
            )

    return (
        mediafiles.distinct()
        .select_related("parent", "locality", "sequence")
        .prefetch_related(
            Prefetch(
                "observations",
                queryset=AnimalObservation.objects.select_related("taxon", "predicted_taxon", "identity").order_by("id"),
            )
        )
        .order_by("sequence_id", "captured_at", "id")
    )


@login_required
def download_csv_for_observations_view(request: HttpRequest) -> HttpResponse:
    """Download import-compatible metadata for mediafiles matched by observations."""
    mediafiles = _get_mediafiles_for_observation_export(request)
    df = _sequence_export_dataframe(mediafiles, request, SEQUENCE_EXPORT_DEFAULT_COLUMNS)
    if df.empty:
        return HttpResponse("No data available to export.", content_type="text/plain")
    response = HttpResponse(df.to_csv(index=False), content_type="text/csv")
    response["Content-Disposition"] = "attachment; filename=observation_metadata.csv"
    return response


@login_required
def download_xlsx_for_observations_view(request: HttpRequest) -> HttpResponse:
    """Download import-compatible XLSX for mediafiles matched by observations."""
    mediafiles = _get_mediafiles_for_observation_export(request)
    df = _sequence_export_dataframe(mediafiles, request, SEQUENCE_EXPORT_DEFAULT_COLUMNS)
    if df.empty:
        return HttpResponse("No data available to export.", content_type="text/plain")
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        model_tools.convert_datetime_to_naive(df).to_excel(writer, index=False, sheet_name="Observations")
    output.seek(0)
    response = HttpResponse(output, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response["Content-Disposition"] = "attachment; filename=observation_metadata.xlsx"
    return response


@login_required
def prepare_observation_download(request: HttpRequest) -> HttpResponse:
    """Start the configurable image/metadata export for the current observation scope."""
    mediafile_ids = list(_get_mediafiles_for_observation_export(request).values_list("id", flat=True))
    request.session[OBSERVATION_DOWNLOAD_SESSION_KEY] = mediafile_ids
    return_url = reverse("caidapp:observations")
    if request.GET:
        return_url = f"{return_url}?{request.GET.urlencode()}"
    request.session[OBSERVATION_DOWNLOAD_RETURN_URL_SESSION_KEY] = return_url
    return redirect("caidapp:download_observations")


@login_required
def observations(request: HttpRequest) -> HttpResponse:
    """Review individual animal observations, optionally grouped by their context."""
    group_by = request.GET.get("group", "sequence")
    if group_by not in {"sequence", "mediafile", "none"}:
        group_by = "sequence"
    view_mode = request.GET.get("view", "cards")
    if view_mode not in {"cards", "large_cards", "list"}:
        view_mode = "cards"
    sort_by = request.GET.get("sort", "captured_asc")
    if sort_by not in {
        "captured_desc",
        "captured_asc",
        "locality",
        "filename",
        "taxon",
        "identity",
        "observation_id",
    }:
        sort_by = "captured_asc"
    grouping_disabled_for_sort = group_by != "none" and sort_by in {"taxon", "identity"}
    if grouping_disabled_for_sort:
        group_by = "none"
    records_per_page = _get_observation_records_per_page(request)
    observation_queryset, observation_filter = _get_observations_queryset(request)

    sequence_id = _parse_int_query_param_or_404(request.GET.get("sequence"), "sequence id")
    if sequence_id is not None:
        observation_queryset = observation_queryset.filter(mediafile__sequence_id=sequence_id)

    matching_mediafiles = _get_mediafiles_for_observation_export(request)
    if sequence_id is not None:
        matching_mediafiles = matching_mediafiles.filter(sequence_id=sequence_id)
    editable_mediafiles = MediaFile.objects.for_user(request.user.caiduser).filter(
        id__in=matching_mediafiles.order_by().values("id")
    )
    editable_observations = observation_queryset.filter(
        mediafile_id__in=editable_mediafiles.order_by().values("id")
    )
    editable_sequence_ids = list(
        editable_observations.exclude(mediafile__sequence_id__isnull=True)
        .order_by()
        .values_list("mediafile__sequence_id", flat=True)
        .distinct()
    )
    editable_sequence_direct_mediafile_count = (
        editable_observations.exclude(mediafile__sequence_id__isnull=True)
        .order_by()
        .values("mediafile_id")
        .distinct()
        .count()
    )
    editable_sequence_mediafile_count = MediaFile.objects.for_user(request.user.caiduser).filter(
        sequence_id__in=editable_sequence_ids
    ).count()
    form_bulk_processing = MediaFileBulkForm(
        request.POST or None,
        workgroup=request.user.caiduser.workgroup,
    )
    if request.method == "POST":
        if request.POST.get("select_all_filtered") == "on":
            selected_observations = editable_observations
        else:
            requested_observation_ids = {
                int(value)
                for value in request.POST.getlist("selected_observation_ids")
                if str(value).isdigit() and int(value) > 0
            }
            selected_observations = editable_observations.filter(id__in=requested_observation_ids)
        selected_observation_ids = list(selected_observations.values_list("id", flat=True))
        selected_mediafile_ids = sorted(
            set(selected_observations.values_list("mediafile_id", flat=True))
        )
        selected_sequence_ids = sorted(
            set(
                selected_observations.exclude(mediafile__sequence_id__isnull=True).values_list(
                    "mediafile__sequence_id", flat=True
                )
            )
        )
        observation_field_actions = {
            "btnBulkProcessing_id_taxon": "taxon",
            "btnBulkProcessing_id_identity": "identity",
            "btnBulkProcessing_id_identity_is_representative": "identity_is_representative",
            "btnBulkProcessing_id_taxon_verified": "taxon_verified",
        }
        selection_actions = {
            "btnBulkProcessing_set_full_image_bbox",
            "btnBulkProcessing_remove_bbox",
            "btnCreateSequence",
            "btnDissolveSequences",
            "btnExtractFilenameMetadata",
            "btnDownloadObservations",
            *observation_field_actions,
        }
        requested_selection_action = next((key for key in selection_actions if key in request.POST), None)
        if requested_selection_action and not selected_observation_ids:
            messages.warning(request, "Select at least one editable observation.")
            return redirect(request.get_full_path())

        if requested_selection_action in {
            "btnBulkProcessing_set_full_image_bbox",
            "btnBulkProcessing_remove_bbox",
        }:
            updated_at = django.utils.timezone.now()
            bbox_values = (0.5, 0.5, 1.0, 1.0) if requested_selection_action.endswith("full_image_bbox") else (None, None, None, None)
            for observation in selected_observations:
                (
                    observation.bbox_x_center,
                    observation.bbox_y_center,
                    observation.bbox_width,
                    observation.bbox_height,
                ) = bbox_values
                observation.updated_by = request.user.caiduser
                observation.updated_at = updated_at
                observation.save(update_fields=[*OBSERVATION_BBOX_MODEL_FIELDS, "updated_by", "updated_at"])
            MediaFile.objects.filter(id__in=selected_mediafile_ids).update(
                updated_by=request.user.caiduser,
                updated_at=updated_at,
            )
            changed_observation_count = len(selected_observation_ids)
            action_label = "Set full-image bbox on" if requested_selection_action.endswith("full_image_bbox") else "Removed bbox from"
            messages.success(request, f"{action_label} {changed_observation_count} observations.")
            return redirect(request.get_full_path())

        if requested_selection_action in observation_field_actions:
            if not form_bulk_processing.is_valid():
                messages.error(request, "The bulk value is not valid.")
                return redirect(request.get_full_path())
            field_name = observation_field_actions[requested_selection_action]
            field_value = form_bulk_processing.cleaned_data[field_name]
            updated_at = django.utils.timezone.now()
            changed_count = 0
            skipped_count = 0
            for observation in selected_observations.select_related("identity"):
                if field_name == "identity_is_representative" and field_value and not observation.identity_id:
                    skipped_count += 1
                    continue
                if (
                    field_name == "taxon_verified"
                    and field_value
                    and observation.is_no_detection_placeholder
                    and not observation.taxon_id
                ):
                    skipped_count += 1
                    continue
                previous_value = getattr(observation, f"{field_name}_id", None) if field_name in {"taxon", "identity"} else getattr(observation, field_name)
                new_value = field_value.id if field_name in {"taxon", "identity"} and field_value is not None else field_value
                setattr(observation, field_name, field_value)
                update_fields = [field_name, "updated_by", "updated_at"]
                if field_name == "taxon" and previous_value != new_value:
                    observation.taxon_verified = False
                    observation.taxon_verified_at = None
                    update_fields.extend(["taxon_verified", "taxon_verified_at"])
                if field_name == "identity" and previous_value != new_value:
                    observation.identity_is_representative = False
                    update_fields.append("identity_is_representative")
                if field_name == "taxon_verified":
                    observation.taxon_verified_at = updated_at if field_value else None
                    update_fields.append("taxon_verified_at")
                observation.updated_by = request.user.caiduser
                observation.updated_at = updated_at
                observation.save(update_fields=update_fields)
                changed_count += 1
            MediaFile.objects.filter(
                id__in=selected_observations.values("mediafile_id")
            ).update(updated_by=request.user.caiduser, updated_at=updated_at)
            messages.success(request, f"Updated {field_name.replace('_', ' ')} on {changed_count} observations.")
            if skipped_count:
                if field_name == "taxon_verified":
                    messages.warning(
                        request,
                        f"Skipped {skipped_count} no-detection placeholders without a taxon. "
                        "Set their taxon to Nothing to confirm an empty image.",
                    )
                else:
                    messages.warning(request, f"Skipped {skipped_count} observations without an identity.")
            return redirect(request.get_full_path())

        if "btnCreateSequence" in request.POST:
            result = _create_sequence_from_mediafiles(request.user.caiduser, selected_mediafile_ids)
            if result["status"] == "empty":
                messages.warning(request, "No editable media files matched the selection.")
            elif result["status"] == "multiple_archives":
                messages.error(request, "Selected media files must belong to the same upload to create a sequence.")
            elif result["status"] == "already_one_sequence":
                messages.info(request, "Selected media files already form one complete sequence.")
            else:
                messages.success(request, f"Created a new sequence from {result['mediafile_count']} media files.")
            return redirect(request.get_full_path())

        if "btnDissolveSequences" in request.POST:
            if not selected_sequence_ids:
                messages.warning(request, "The selected observations do not belong to a sequence.")
                return redirect(request.get_full_path())
            complete_sequence_mediafile_ids = list(
                MediaFile.objects.for_user(request.user.caiduser)
                .filter(sequence_id__in=selected_sequence_ids)
                .values_list("id", flat=True)
            )
            dissolved_count = _dissolve_mediafiles_into_singleton_sequences(
                request.user.caiduser, complete_sequence_mediafile_ids
            )
            if dissolved_count:
                messages.success(
                    request,
                    f"Split {len(selected_sequence_ids)} complete containing sequences; "
                    f"moved {dissolved_count} media files into single-media sequences.",
                )
            else:
                messages.info(request, "The containing sequences are already single-media sequences.")
            return redirect(request.get_full_path())

        if "btnExtractFilenameMetadata" in request.POST:
            return _start_filename_metadata_session(
                request,
                selected_mediafile_ids,
                request.get_full_path(),
                "Observations",
            )

        if "btnDownloadObservations" in request.POST:
            request.session[OBSERVATION_DOWNLOAD_SESSION_KEY] = selected_mediafile_ids
            request.session[OBSERVATION_DOWNLOAD_RETURN_URL_SESSION_KEY] = request.get_full_path()
            return redirect("caidapp:download_observations")

    group_ordering = []
    if group_by != "none":
        partition_field = "mediafile__sequence_id" if group_by == "sequence" else "mediafile_id"
        if sort_by == "captured_desc":
            group_sort_expression = Max("mediafile__captured_at")
        elif sort_by == "captured_asc":
            group_sort_expression = Min("mediafile__captured_at")
        elif sort_by == "locality":
            group_sort_expression = Min("mediafile__locality__name")
        elif sort_by == "filename":
            group_sort_expression = Min("mediafile__original_filename")
        else:
            group_sort_expression = Min("id")
        observation_queryset = observation_queryset.annotate(
            observation_group_sort=Window(
                expression=group_sort_expression,
                partition_by=[F(partition_field)],
            )
        )
        group_sort_field = F("observation_group_sort")
        group_ordering = [
            group_sort_field.desc(nulls_last=True) if sort_by == "captured_desc" else group_sort_field.asc(nulls_last=True),
            F(partition_field).asc(nulls_last=True),
        ]

    if sort_by == "captured_desc":
        observation_ordering = [F("mediafile__captured_at").desc(nulls_last=True), "mediafile_id", "id"]
    elif sort_by == "captured_asc":
        observation_ordering = [F("mediafile__captured_at").asc(nulls_last=True), "mediafile_id", "id"]
    elif sort_by == "locality":
        observation_ordering = [F("mediafile__locality__name").asc(nulls_last=True), "mediafile__captured_at", "mediafile_id", "id"]
    elif sort_by == "filename":
        observation_ordering = [F("mediafile__original_filename").asc(nulls_last=True), "mediafile_id", "id"]
    elif sort_by == "taxon":
        observation_ordering = [F("taxon__name").asc(nulls_last=True), "mediafile__captured_at", "mediafile_id", "id"]
    elif sort_by == "identity":
        observation_ordering = [F("identity__name").asc(nulls_last=True), "mediafile__captured_at", "mediafile_id", "id"]
    else:
        observation_ordering = ["id"]
    observation_queryset = observation_queryset.order_by(*group_ordering, *observation_ordering)
    paginator = Paginator(observation_queryset, per_page=records_per_page)
    page_obj, _, page_context = _prepare_page(paginator, request=request)
    observations_on_page = list(page_obj.object_list)
    page_observation_ids = [observation.id for observation in observations_on_page]
    editable_observation_ids = set(
        editable_observations.filter(id__in=page_observation_ids).values_list("id", flat=True)
    )
    page_sequence_ids = {
        observation.mediafile.sequence_id
        for observation in observations_on_page
        if observation.mediafile.sequence_id is not None
    }
    page_sequence_mediafile_counts = dict(
        MediaFile.objects.for_user(request.user.caiduser)
        .filter(sequence_id__in=page_sequence_ids)
        .values("sequence_id")
        .annotate(mediafile_count=Count("id"))
        .values_list("sequence_id", "mediafile_count")
    )
    for observation in observations_on_page:
        observation.bulk_editable = observation.id in editable_observation_ids
        observation.containing_sequence_mediafile_count = page_sequence_mediafile_counts.get(
            observation.mediafile.sequence_id, 0
        )

    groups = []
    group_lookup = {}
    for observation in observations_on_page:
        if group_by == "sequence":
            key = ("sequence", observation.mediafile.sequence_id)
            label = (
                f"Sequence {observation.mediafile.sequence.local_id or observation.mediafile.sequence_id}"
                if observation.mediafile.sequence_id
                else "No sequence"
            )
        elif group_by == "mediafile":
            key = ("mediafile", observation.mediafile_id)
            label = observation.mediafile.original_filename or f"Media file {observation.mediafile_id}"
        else:
            key = ("observation", observation.id)
            label = None
        group = group_lookup.get(key)
        if group is None:
            group_id = "-".join(str(part) if part is not None else "none" for part in key)
            group = {
                "id": group_id,
                "label": label,
                "observations": [],
                "is_collapsible": group_by != "none",
            }
            group_lookup[key] = group
            groups.append(group)
        group["observations"].append(observation)

    for group in groups:
        group_observations = group["observations"]
        group["mediafile_count"] = len({observation.mediafile_id for observation in group_observations})
        group["bbox_count"] = sum(observation.bbox_x_center is not None for observation in group_observations)
        group["taxon_count"] = len({observation.taxon_id for observation in group_observations if observation.taxon_id})
        group["identity_count"] = len(
            {observation.identity_id for observation in group_observations if observation.identity_id}
        )
        captured_values = [
            observation.mediafile.captured_at
            for observation in group_observations
            if observation.mediafile.captured_at is not None
        ]
        group["captured_at_start"] = min(captured_values) if captured_values else None
        group["captured_at_end"] = max(captured_values) if captured_values else None

    context = {
        **page_context,
        "page_title": "Observations",
        "filter": observation_filter,
        "observation_groups": groups,
        "group_by": group_by,
        "sort_by": sort_by,
        "grouping_disabled_for_sort": grouping_disabled_for_sort,
        "view_mode": view_mode,
        "records_per_page": records_per_page,
        "observation_per_page_options": OBSERVATION_PER_PAGE_OPTIONS,
        "number_of_observations": observation_queryset.count(),
        "number_of_mediafiles": matching_mediafiles.count(),
        "show_observation_result_summary": True,
        "number_of_editable_observations": editable_observations.count(),
        "number_of_editable_observation_mediafiles": editable_observations.values("mediafile_id").distinct().count(),
        "number_of_editable_observation_sequences": len(editable_sequence_ids),
        "number_of_editable_observation_sequence_mediafiles": editable_sequence_direct_mediafile_count,
        "number_of_mediafiles_in_editable_observation_sequences": editable_sequence_mediafile_count,
        "form_bulk_processing": form_bulk_processing,
    }
    return render(request, "caidapp/observations.html", add_querystring_to_context(request, context))


@login_required
def representative_mediafiles_redirect(request):
    """Backward-compatible redirect to the unified media files view."""
    query_params = request.GET.copy()
    query_params["identity_is_representative"] = "true"
    return redirect(f"{reverse('caidapp:media_files')}?{query_params.urlencode()}")


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


def _resolve_selected_mediafile_ids_from_post(request, filtered_mediafiles: Optional[QuerySet] = None) -> List[int]:
    """Resolve bulk-selected mediafiles from sequence and mediafile checkboxes."""
    if request.POST.get("select_all_filtered") == "on":
        if filtered_mediafiles is None:
            raise ValueError("Filtered mediafiles are required for an all-filtered selection")
        return list(filtered_mediafiles.values_list("id", flat=True))
    selected_sequence_ids = [int(v) for v in request.POST.getlist("selected_sequence_ids") if str(v).isdigit()]
    selected_mediafile_ids = {int(v) for v in request.POST.getlist("selected_mediafile_ids") if str(v).isdigit()}
    deselected_mediafile_ids = {int(v) for v in request.POST.getlist("deselected_mediafile_ids") if str(v).isdigit()}

    if selected_sequence_ids:
        sequence_mediafile_ids = MediaFile.objects.for_user(request.user.caiduser).filter(
            sequence_id__in=selected_sequence_ids
        ).values_list("id", flat=True)
        selected_mediafile_ids.update(sequence_mediafile_ids)

    selected_mediafile_ids.difference_update(deselected_mediafile_ids)
    accessible_ids = MediaFile.objects.for_user(request.user.caiduser).filter(
        id__in=selected_mediafile_ids
    ).values_list("id", flat=True)
    return sorted(accessible_ids)


def _resolve_selected_mediafile_ids_from_formset(form, full_mediafiles: QuerySet) -> List[int]:
    """Resolve selected mediafile ids from the media files formset selection."""
    if form.data.get("select_all") == "on":
        return list(full_mediafiles.values_list("id", flat=True))

    selected_mediafile_ids = []
    for mediafile_form in form:
        if mediafile_form.is_valid() and mediafile_form.cleaned_data.get("selected"):
            selected_mediafile_ids.append(mediafile_form.instance.id)
    return selected_mediafile_ids


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


def _create_sequence_from_mediafiles(caiduser, mediafile_ids: List[int]) -> dict:
    """Move selected mediafiles into one new sequence within a single upload."""
    selected_mediafiles = list(
        MediaFile.objects.for_user(caiduser)
        .filter(id__in=mediafile_ids)
        .select_related("parent", "sequence")
        .order_by("captured_at", "id")
    )
    if not selected_mediafiles:
        return {"status": "empty"}

    archive_ids = {mediafile.parent_id for mediafile in selected_mediafiles if mediafile.parent_id}
    if len(archive_ids) != 1:
        return {"status": "multiple_archives"}

    selected_ids = {mediafile.id for mediafile in selected_mediafiles}
    affected_sequence_ids = {mediafile.sequence_id for mediafile in selected_mediafiles if mediafile.sequence_id}
    existing_sequence_ids = {mediafile.sequence_id for mediafile in selected_mediafiles}
    if len(existing_sequence_ids) == 1:
        existing_sequence_id = next(iter(existing_sequence_ids))
        if existing_sequence_id is not None:
            existing_sequence_size = MediaFile.objects.filter(sequence_id=existing_sequence_id).count()
            if existing_sequence_size == len(selected_mediafiles):
                return {"status": "already_one_sequence"}

    archive_id = next(iter(archive_ids))
    current_max_local_id = models.Sequence.objects.filter(uploaded_archive_id=archive_id).aggregate(
        max_local_id=Max("local_id")
    )["max_local_id"]
    next_local_id = 0 if current_max_local_id is None else current_max_local_id + 1

    with transaction.atomic():
        new_sequence = models.Sequence.objects.create(uploaded_archive_id=archive_id, local_id=next_local_id)
        MediaFile.objects.filter(id__in=selected_ids).update(sequence=new_sequence)
        deleted_sequence_count = (
            models.Sequence.objects.filter(id__in=affected_sequence_ids)
            .annotate(mediafile_count=Count("mediafile"))
            .filter(mediafile_count=0)
            .delete()[0]
        )

    return {
        "status": "created",
        "sequence_id": new_sequence.id,
        "mediafile_count": len(selected_mediafiles),
        "deleted_sequence_count": deleted_sequence_count,
    }


def _set_mediafile_bbox_to_full_image(mediafile: MediaFile, caiduser) -> int:
    """Set full-image bbox for all observations on one mediafile."""
    observations = list(mediafile.observations.all())
    if not observations:
        observations = [mediafile.first_observation_get_or_create]

    for observation in observations:
        observation.bbox_x_center = 0.5
        observation.bbox_y_center = 0.5
        observation.bbox_width = 1.0
        observation.bbox_height = 1.0
        observation.updated_by = caiduser
        observation.updated_at = django.utils.timezone.now()
        observation.save(
            update_fields=[
                "bbox_x_center",
                "bbox_y_center",
                "bbox_width",
                "bbox_height",
                "updated_by",
                "updated_at",
            ]
        )

    mediafile.updated_by = caiduser
    mediafile.updated_at = django.utils.timezone.now()
    mediafile.save(update_fields=["updated_by", "updated_at"])
    return len(observations)


def _remove_mediafile_bbox(mediafile: MediaFile, caiduser) -> int:
    """Remove bboxes from a media file without altering its observations' identities."""
    observations = list(mediafile.observations.all())
    updated_at = django.utils.timezone.now()

    for observation in observations:
        observation.bbox_x_center = None
        observation.bbox_y_center = None
        observation.bbox_width = None
        observation.bbox_height = None
        observation.updated_by = caiduser
        observation.updated_at = updated_at
        observation.save(
            update_fields=[
                "bbox_x_center",
                "bbox_y_center",
                "bbox_width",
                "bbox_height",
                "updated_by",
                "updated_at",
            ]
        )

    if observations:
        mediafile.updated_by = caiduser
        mediafile.updated_at = updated_at
        mediafile.save(update_fields=["updated_by", "updated_at"])
    return len(observations)


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
            if observation_changed:
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
    elif observation is not None and observation.identity_id and not force_rewrite_filled_data:
        identity = observation.identity
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
        .select_related("parent", "locality", "updated_by", "sequence")
        .prefetch_related("observations__taxon", "observations__identity")
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
        sequence_taxon_ids = {
            observation.taxon_id
            for mediafile in mediafiles_in_sequence
            for observation in mediafile.observations.all()
            if observation.taxon_id
        }
        sequence_identity_ids = {
            observation.identity_id
            for mediafile in mediafiles_in_sequence
            for observation in mediafile.observations.all()
            if observation.identity_id
        }
        sequence.has_multiple_taxa = len(sequence_taxon_ids) > 1
        sequence.has_multiple_identities = len(sequence_identity_ids) > 1
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

    form_bulk_processing = MediaFileBulkForm(request.POST or None, workgroup=request.user.caiduser.workgroup)

    if request.method == "POST" and any(
        (isinstance(key, str)) and key.startswith("btnBulkProcessing") for key in request.POST
    ):
        if form_bulk_processing.is_valid():
            selected_mediafile_ids = _resolve_selected_mediafile_ids_from_post(request, full_mediafiles)
            request.session["mediafile_ids"] = selected_mediafile_ids
            request.session["mediafiles_name_suggestion"] = mediafiles_name_suggestion
            selected_mediafiles = MediaFile.objects.filter(id__in=selected_mediafile_ids)
            selected_album_hash = request.POST.get("selectAlbum", "")
            bulk_result_counts = {}
            for mediafile in selected_mediafiles:
                result = _single_mediafile_update(
                    request,
                    mediafile,
                    form_bulk_processing,
                    form_bulk_processing,
                    selected_album_hash,
                )
                bulk_result_counts[result] = bulk_result_counts.get(result, 0) + 1
            _add_bulk_processing_result_messages(request, bulk_result_counts)
            return redirect(request.get_full_path())

    if request.method == "POST" and "btnDissolveSequences" in request.POST:
        selected_mediafile_ids = _resolve_selected_mediafile_ids_from_post(request, full_mediafiles)
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

    if request.method == "POST" and "btnCreateSequence" in request.POST:
        selected_mediafile_ids = _resolve_selected_mediafile_ids_from_post(request, full_mediafiles)
        if not selected_mediafile_ids:
            messages.warning(request, "Select at least one sequence or media file to combine.")
            return redirect(request.get_full_path())

        result = _create_sequence_from_mediafiles(request.user.caiduser, selected_mediafile_ids)
        if result["status"] == "empty":
            messages.warning(request, "No accessible media files were found in the selection.")
        elif result["status"] == "multiple_archives":
            messages.error(request, "Selected media files must belong to the same upload to create a sequence.")
        elif result["status"] == "already_one_sequence":
            messages.info(request, "Selected media files already form one complete sequence.")
        else:
            deleted_text = ""
            if result["deleted_sequence_count"]:
                deleted_text = f" Removed {result['deleted_sequence_count']} empty original sequences."
            messages.success(
                request,
                f"Created a new sequence from {result['mediafile_count']} media files.{deleted_text}",
            )
        return redirect(request.get_full_path())

    if request.method == "POST" and "btnExtractFilenameMetadata" in request.POST:
        selected_mediafile_ids = _resolve_selected_mediafile_ids_from_post(request, full_mediafiles)
        if not selected_mediafile_ids:
            selected_mediafile_ids = list(full_mediafiles.values_list("id", flat=True))
        return _start_filename_metadata_session(
            request,
            selected_mediafile_ids,
            request.get_full_path(),
            "Sequences",
        )

    if request.method == "POST" and "btnDownloadSequences" in request.POST:
        selected_mediafile_ids = _resolve_selected_mediafile_ids_from_post(request, full_mediafiles)
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
    active_uploadedarchive = _get_active_uploadedarchive_from_request(request, uploadedarchive_id)
    active_taxon = _get_active_taxon_from_request(request)
    active_album = _get_active_album_from_request(request, album_hash)
    active_identity = _get_active_identity_from_request(request, individual_identity_id)
    active_locality = _get_active_locality_from_request(request, locality_hash)

    if show_overview_button and not full_mediafiles.exists():
        return message_view(
            request,
            "No mediafiles for verification.",
            headline="Verification",
            link=reverse_lazy("caidapp:home"),
            button_label="Go home",
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

    first_observation = AnimalObservation.objects.filter(mediafile=OuterRef("pk")).order_by("id")
    observation_count = (
        AnimalObservation.objects.filter(mediafile=OuterRef("pk"))
        .order_by()
        .values("mediafile")
        .annotate(count=Count("pk"))
        .values("count")[:1]
    )
    full_mediafiles = full_mediafiles.annotate(
        observation_count=Coalesce(Subquery(observation_count, output_field=IntegerField()), Value(0)),
        first_observation_identity_id=Subquery(first_observation.values("identity_id")[:1]),
        first_observation_identity_name=Subquery(first_observation.values("identity__name")[:1]),
        first_observation_identity_is_representative=Subquery(
            first_observation.values("identity_is_representative")[:1]
        ),
    )

    full_mediafiles = full_mediafiles.select_related(
        "parent", "locality", "updated_by", "sequence"
    ).prefetch_related("observations__taxon", "observations__predicted_taxon", "observations__identity")

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
    if request.method == "POST" and "btnCreateSequence" in request.POST:
        form_bulk_processing = MediaFileBulkForm(workgroup=request.user.caiduser.workgroup)
        page_query = full_mediafiles.filter(id__in=[object.id for object in page_with_mediafiles])
        form = MediaFileFormSet(request.POST, queryset=page_query)
        selected_mediafile_ids = _resolve_selected_mediafile_ids_from_formset(form, full_mediafiles)

        if not selected_mediafile_ids:
            messages.warning(request, "Select at least one media file to combine.")
            return redirect(request.get_full_path())

        result = _create_sequence_from_mediafiles(request.user.caiduser, selected_mediafile_ids)
        if result["status"] == "empty":
            messages.warning(request, "No accessible media files were found in the selection.")
        elif result["status"] == "multiple_archives":
            messages.error(request, "Selected media files must belong to the same upload to create a sequence.")
        elif result["status"] == "already_one_sequence":
            messages.info(request, "Selected media files already form one complete sequence.")
        else:
            deleted_text = ""
            if result["deleted_sequence_count"]:
                deleted_text = f" Removed {result['deleted_sequence_count']} empty original sequences."
            messages.success(
                request,
                f"Created a new sequence from {result['mediafile_count']} media files.{deleted_text}",
            )
        return redirect(request.get_full_path())

    if (request.method == "POST") and (
        any([(isinstance(key, str)) and (key.startswith("btnBulkProcessing")) for key in request.POST])
        # ("btnBulkProcessing" in request.POST) or ("btnBulkProcessingAlbum" in request.POST)
    ):
        logger.debug("btnBulkProcessing")
        form_bulk_processing = MediaFileBulkForm(request.POST, workgroup=request.user.caiduser.workgroup)

        form = MediaFileFormSet(request.POST)
        logger.debug("form")
        logger.debug(request.POST)
        if form.is_valid() and form_bulk_processing.is_valid():
            logger.debug("form is valid")
            # if 'newsletter_sub' in .data:
            #     # do subscribe
            #     elif 'newsletter_unsub' in self.data:
            selected_album_hash = form.data.get("selectAlbum", "")

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
                bulk_result_counts = {}
                for mediafile in full_mediafiles:

                    result = _single_mediafile_update(request, mediafile, form, form_bulk_processing, selected_album_hash)
                    bulk_result_counts[result] = bulk_result_counts.get(result, 0) + 1
                    # album.cover = mediafile
                    # album.save()
            else:
                bulk_result_counts = {}
                for mediafileform in form:
                    # go over selected mediafiles
                    if mediafileform.is_valid():
                        if mediafileform.cleaned_data["selected"]:
                            logger.debug("mediafileform is valid")
                            # reset selected field for refreshed view
                            mediafileform.cleaned_data["selected"] = False
                            mediafileform.selected = False
                            instance: MediaFile = mediafileform.save(commit=False)
                            result = _single_mediafile_update(request, instance, form, form_bulk_processing, selected_album_hash)
                            bulk_result_counts[result] = bulk_result_counts.get(result, 0) + 1
                            # album.cover = instance
                            # album.save()
            _add_bulk_processing_result_messages(request, bulk_result_counts)

            if "btnBulkProcessingAlbum" in form.data:
                if selected_album_hash == "new":
                    album.cover = album.medifile_set.first()
                    album.save()

                    # mediafileform.save()
            # form.save()
        else:
            logger.debug("bulk form is not valid")
            logger.debug(form.errors)
            logger.debug(form_bulk_processing.errors)
        # queryform = MediaFileSetQueryForm(request.POST)
        form_bulk_processing = MediaFileBulkForm(workgroup=request.user.caiduser.workgroup)
        page_query = full_mediafiles.filter(id__in=[object.id for object in page_with_mediafiles])
        form = MediaFileFormSet(queryset=page_query)
    else:

        logger.debug("initial form processing")
        form_bulk_processing = MediaFileBulkForm(workgroup=request.user.caiduser.workgroup)
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
        "active_uploadedarchive": active_uploadedarchive,
        "active_taxon": active_taxon,
        "active_album": active_album,
        "active_identity": active_identity,
        "active_locality": active_locality,
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
        instance.updated_by = request.user.caiduser
        instance.updated_at = django.utils.timezone.now()
        instance.save(update_fields=["updated_by", "updated_at"])
        observation.save()
    elif "btnBulkProcessing_id_identity" in form.data:
        observations = list(instance.observations.all()[:2])
        if len(observations) > 1:
            return "skipped_identity_multiple_observations"
        observation = observations[0] if observations else instance.first_observation_get_or_create
        observation.identity = form_bulk_processing.cleaned_data["identity"]
        instance.updated_by = request.user.caiduser
        instance.updated_at = django.utils.timezone.now()
        instance.save(update_fields=["updated_by", "updated_at"])
        observation.save()
    elif "btnBulkProcessing_id_identity_is_representative" in form.data:
        observations = list(instance.observations.all())
        if len(observations) > 1:
            return "skipped_representative_multiple_observations"
        observation = observations[0] if observations else instance.first_observation_get_or_create
        observation.identity_is_representative = form_bulk_processing.cleaned_data["identity_is_representative"]
        observation.updated_by = request.user.caiduser
        observation.updated_at = django.utils.timezone.now()
        observation.save(update_fields=["identity_is_representative", "updated_by", "updated_at"])
        instance.updated_by = request.user.caiduser
        instance.updated_at = django.utils.timezone.now()
        instance.save(update_fields=["updated_by", "updated_at"])
    elif "btnBulkProcessingDelete" in form.data:
        instance.delete()
    elif "btnBulkProcessing_id_taxon_verified" in form.data:
        observation = instance.first_observation_get_or_create
        requested_verified = form_bulk_processing.cleaned_data["taxon_verified"]
        if requested_verified and observation.is_no_detection_placeholder and not observation.taxon_id:
            return "skipped_verify_no_detection_placeholder"
        observation.taxon_verified = requested_verified
        instance.updated_by = request.user.caiduser
        instance.updated_at = django.utils.timezone.now()
        instance.save(update_fields=["updated_by", "updated_at"])
        observation.save()

    elif "btnBulkProcessing_set_taxon_verified" in form.data:
        observations = list(instance.observations.all())
        if not observations:
            observations = [instance.first_observation_get_or_create]
        for observation in observations:
            if observation.is_no_detection_placeholder and not observation.taxon_id:
                continue
            observation.taxon_verified = True
            observation.save()
        # observation = instance.first_observation_get_or_create
        # observation.taxon_verified = True
        instance.updated_by = request.user.caiduser
        instance.updated_at = django.utils.timezone.now()
        instance.save(update_fields=["updated_by", "updated_at"])
    elif "btnBulkProcessing_set_full_image_bbox" in form.data:
        _set_mediafile_bbox_to_full_image(instance, request.user.caiduser)
    elif "btnBulkProcessing_remove_bbox" in form.data:
        _remove_mediafile_bbox(instance, request.user.caiduser)

    return "updated"


def _add_bulk_processing_result_messages(request, result_counts: dict) -> None:
    skipped_representative = result_counts.get("skipped_representative_multiple_observations", 0)
    skipped_identity = result_counts.get("skipped_identity_multiple_observations", 0)
    skipped_placeholder_verify = result_counts.get("skipped_verify_no_detection_placeholder", 0)
    if skipped_representative:
        messages.warning(
            request,
            "Representative identity was not changed for "
            f"{skipped_representative} media files with multiple observations. "
            "Edit representative flags in the media file detail for those files.",
        )
    if skipped_identity:
        messages.warning(
            request,
            "Identity was not changed for "
            f"{skipped_identity} media files with multiple observations. "
            "Choose the animal in the media file detail instead.",
        )
    if skipped_placeholder_verify:
        messages.warning(
            request,
            f"Skipped {skipped_placeholder_verify} no-detection placeholders without a taxon. "
            "Set their taxon to Nothing to confirm an empty image.",
        )


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
            return redirect("caidapp:home")
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


def _build_identity_merge_regex_chatgpt_prompt(current_regex: str) -> str:
    return f"""Help me configure a Python regular expression for wildlife identity merge suggestions.

The application applies the regex separately to two identity names BEFORE calculating Levenshtein distance. The regex extracts an identity-distinguishing token:
- If both names match and their captured values differ, the pair must never be suggested for merging.
- If both names match and captured values are identical, normal Levenshtein comparison continues.
- If only one or neither name matches, normal Levenshtein comparison continues.
- When the regex has capture groups, the application compares the groups. Without capture groups, it compares the full match.

Example: Sara_juv.22-1 must not merge with Sara_juv.22-2 or Sara_juv.23-1. A suitable regex captures year 22 and juvenile index 1 separately.

Current regex:
{current_regex}

First consult me: ask for representative identity names that must not be paired, names that should still be allowed to pair, and naming variants or separators that occur. Then explain the proposed capture groups and test the regex against my examples. Do not finalize until ambiguities have been discussed.

Conduct the consultation and all explanations in the language used by the user. If the user's language is not yet clear, ask which language they prefer.

When we agree, put the final regex by itself on the LAST line of your response. The last line must contain only the regex: no Markdown code fence, no quotation marks, and no Python r prefix."""


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
        identification_model_changed = "identification_model" in form.changed_data
        response = super().form_valid(form)
        if identification_model_changed:
            transaction.on_commit(
                lambda: tasks.schedule_init_identification_for_workgroup(
                    models.WorkGroup.objects.get(pk=self.object.pk)
                )
            )
        return response

    def get_context_data(self, **kwargs):
        """Get context data for the template."""
        context = super().get_context_data(**kwargs)
        context["headline"] = "Update workgroup"
        context["button"] = "Save"
        regex_prompt = _build_identity_merge_regex_chatgpt_prompt(
            self.object.get_identity_merge_distinguishing_regex()
        )
        context["identity_merge_regex_chatgpt_prompt"] = regex_prompt
        context["identity_merge_regex_chatgpt_url"] = (
            f"https://chatgpt.com/?q={urllib.parse.quote(regex_prompt)}"
        )
        context["nav_dict"] = {
            "Personal Settings": reverse_lazy("caidapp:update_caiduser"),
            "Users": reverse_lazy("caidapp:workgroup_members"),
            "Invitations": reverse_lazy("caidapp:workgroup_invitations"),
            "Invite User": reverse_lazy("caidapp:workgroup_invitation"),
        }
        if self.request.user.is_staff:
            context["nav_dict"]["Add model from HuggingFace"] = (
                reverse_lazy("admin:caidapp_identificationmodel_add")
                + f"?workgroup={self.request.user.caiduser.workgroup_id}"
            )
        return context


class WorkgroupMemberListView(WorkgroupAdminRequiredMixin, ListView):
    model = models.CaIDUser
    template_name = "caidapp/workgroup_members.html"
    context_object_name = "workgroup_members"

    def get_queryset(self):
        return (
            models.CaIDUser.objects.filter(workgroup=self.request.user.caiduser.workgroup)
            .select_related("user")
            .order_by("user__username")
        )

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["headline"] = "Workgroup users"
        context["nav_dict"] = {
            "Workgroup Settings": reverse_lazy(
                "caidapp:workgroup-update", args=[self.request.user.caiduser.workgroup_id]
            ),
            "Invitations": reverse_lazy("caidapp:workgroup_invitations"),
            "Invite User": reverse_lazy("caidapp:workgroup_invitation"),
        }
        return context


class WorkgroupMemberUpdateView(WorkgroupAdminRequiredMixin, UpdateView):
    model = models.CaIDUser
    form_class = forms.WorkgroupMemberWorkflowForm
    template_name = "caidapp/update_form.html"
    success_url = reverse_lazy("caidapp:workgroup_members")

    def get_queryset(self):
        return models.CaIDUser.objects.filter(workgroup=self.request.user.caiduser.workgroup).select_related("user")

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["headline"] = f"Workflow access: {self.object}"
        context["button"] = "Save"
        context["cancel_button_url"] = reverse("caidapp:workgroup_members")
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
    mediafiles = mediafiles.select_related(
        "parent", "locality", "sequence"
    ).prefetch_related(
        Prefetch(
            "observations",
            queryset=AnimalObservation.objects.select_related("taxon", "predicted_taxon", "identity").order_by("id"),
        )
    )
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
        .select_related("parent", "locality", "sequence")
        .prefetch_related(
            Prefetch(
                "observations",
                queryset=AnimalObservation.objects.select_related("taxon", "predicted_taxon", "identity").order_by("id"),
            )
        )
        .order_by("sequence_id", "captured_at", "id")
    )


def _get_observation_download_mediafiles(request: HttpRequest) -> QuerySet:
    """Return mediafiles captured by the current Observations export scope."""
    mediafile_ids = request.session.get(OBSERVATION_DOWNLOAD_SESSION_KEY, [])
    caiduser = request.user.caiduser
    return (
        MediaFile.objects.filter(
            Q(album__albumsharerole__user=caiduser)
            | Q(**models.user_has_access_filter_params(caiduser, "parent__owner")),
            id__in=mediafile_ids,
        )
        .distinct()
        .select_related("parent", "locality", "sequence")
        .prefetch_related(
            Prefetch(
                "observations",
                queryset=AnimalObservation.objects.select_related("taxon", "predicted_taxon", "identity").order_by("id"),
            )
        )
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


def _annotate_identities_with_mediafile_count(queryset):
    """Annotate identities with the number of linked distinct media files."""
    return queryset.annotate(mediafile_count=Count("animalobservation__mediafile", distinct=True))


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
        "locality_id": mediafile.locality_id or "",
        "mediafile_location": str(mediafile.location) if mediafile.location else "",
        "locality_location": str(mediafile.locality.location) if mediafile.locality and mediafile.locality.location else "",
        "location_source": mediafile.effective_location_source or "",
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
        observations = [
            observation
            for observation in mediafile.observations.all()
            if not observation.is_no_detection_placeholder
        ] or [None]
        mediafile_location_values = _location_export_values(mediafile)
        for observation in observations:
            taxon = observation.taxon if observation else None
            predicted_taxon = observation.predicted_taxon if observation else None
            identity = observation.identity if observation else None
            row = {
                "mediafile_id": mediafile.id,
                "observation_id": observation.id if observation else "",
                "taxon_id": taxon.id if taxon else "",
                "original_path": mediafile.original_filename or mediafile.mediafile.name,
                "export_path": mediafile_export_paths.get(mediafile.id, ""),
                "uploaded_archive": mediafile.parent.name if mediafile.parent else "",
                "sequence_id": mediafile.sequence_id or "",
                "datetime": mediafile.captured_at.isoformat() if mediafile.captured_at else "",
                "media_type": mediafile.media_type,
                "locality name": mediafile.locality.name if mediafile.locality else "",
                "predicted_category": taxon.name if taxon else "",
                "taxon_verified": observation.taxon_verified if observation else "",
                "predicted_taxon": predicted_taxon.name if predicted_taxon else "",
                "predicted_taxon_id": predicted_taxon.id if predicted_taxon else "",
                "predicted_taxon_confidence": (
                    observation.predicted_taxon_confidence if observation else ""
                ),
                "identity_is_representative": (
                    observation.identity_is_representative if observation else ""
                ),
                "identity_id": identity.id if identity else "",
                "orientation": observation.orientation if observation else "",
                "bbox_cx": observation.bbox_x_center if observation else "",
                "bbox_cy": observation.bbox_y_center if observation else "",
                "bbox_w": observation.bbox_width if observation else "",
                "bbox_h": observation.bbox_height if observation else "",
                "mediafile_note": mediafile.note,
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
            "download_title": "Download sequences",
            "breadcrumb_url": "caidapp:sequences",
            "breadcrumb_label": "Sequences",
            "download_csv_url_name": "caidapp:download_csv_for_sequences",
            "download_xlsx_url_name": "caidapp:download_xlsx_for_sequences",
            "download_zip_url_name": "caidapp:download_zip_for_sequences",
            "show_import_link": True,
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
def download_observations_view(request: HttpRequest) -> HttpResponse:
    """Configure the image and metadata export prepared from Observations."""
    mediafiles = _get_observation_download_mediafiles(request)
    mediafile_count = mediafiles.count()
    return_url = request.session.get(OBSERVATION_DOWNLOAD_RETURN_URL_SESSION_KEY) or reverse_lazy("caidapp:observations")
    if mediafile_count == 0:
        return message_view(
            request,
            "No media files matched the current observation filter.",
            headline="Download observations",
            link=return_url,
            button_label="Back to observations",
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
            "download_title": "Download observations",
            "breadcrumb_url": "caidapp:observations",
            "breadcrumb_label": "Observations",
            "download_csv_url_name": "caidapp:download_csv_for_observations_selection",
            "download_xlsx_url_name": "caidapp:download_xlsx_for_observations_selection",
            "download_zip_url_name": "caidapp:download_zip_for_observations",
            "show_import_link": False,
        },
    )


@login_required
def download_csv_for_observations_selection_view(request) -> HttpResponse:
    mediafiles = _get_observation_download_mediafiles(request)
    df = _sequence_export_dataframe(mediafiles, request, _get_sequence_export_columns(request))
    if df.empty:
        return HttpResponse("No data available to export.", content_type="text/plain")
    response = HttpResponse(df.to_csv(index=False), content_type="text/csv")
    response["Content-Disposition"] = "attachment; filename=observation_metadata.csv"
    return response


@login_required
def download_xlsx_for_observations_selection_view(request) -> HttpResponse:
    mediafiles = _get_observation_download_mediafiles(request)
    df = _sequence_export_dataframe(mediafiles, request, _get_sequence_export_columns(request))
    if df.empty:
        return HttpResponse("No data available to export.", content_type="text/plain")
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        model_tools.convert_datetime_to_naive(df).to_excel(writer, index=False, sheet_name="Observations")
    output.seek(0)
    response = HttpResponse(output, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response["Content-Disposition"] = "attachment; filename=observation_metadata.xlsx"
    return response


@login_required
def download_zip_for_observations_view(request) -> JsonResponse:
    """Prepare a ZIP with images and observation-level metadata."""
    mediafiles = _get_observation_download_mediafiles(request)
    if not mediafiles.exists():
        return JsonResponse({"message": "No media files were selected for download."}, status=400)
    try:
        mediafiles_data = _build_export_mediafiles_data(request, mediafiles)
    except ValueError as exc:
        return JsonResponse({"message": str(exc)}, status=400)

    mediafiles = _get_observation_download_mediafiles(request)
    export_path_by_mediafile_id = {
        mediafile.id: mediafile_data["output_name"] for mediafile_data, mediafile in zip(mediafiles_data, mediafiles)
    }
    metadata_records = _build_sequence_observation_export_records(
        mediafiles, request=request, columns=SEQUENCE_EXPORT_DEFAULT_COLUMNS
    )
    for record in metadata_records:
        record["export_path"] = export_path_by_mediafile_id.get(record["mediafile_id"], record.get("export_path", ""))

    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    user_hash = request.user.caiduser.hash
    abs_zip_path = Path(settings.MEDIA_ROOT) / "users" / user_hash / f"observation_mediafiles.{datetime_str}.zip"
    task = tasks.create_mediafiles_zip_with_metadata.delay(user_hash, mediafiles_data, str(abs_zip_path), metadata_records)
    _ = tasks.clean_old_mediafile_zips.delay(str(abs_zip_path.parent), glob_pattern="observation_mediafiles.*.zip", max_age_days=7)
    return JsonResponse({"task_id": task.id})


@login_required
def download_csv_for_mediafiles_view(request, uploadedarchive_id: Optional[int] = None):
    """Download observation-level CSV for the filtered media files."""
    mediafiles, name_suggestion = _get_mediafiles_for_export(request, uploadedarchive_id)
    fn = ("metadata_" + name_suggestion) if name_suggestion is not None else "metadata"

    try:
        df = _sequence_export_dataframe(mediafiles, request, SEQUENCE_EXPORT_DEFAULT_COLUMNS)
        if df.empty:
            return HttpResponse("No data available to export.", content_type="text/plain")
    except Exception:
        logger.error(traceback.format_exc())
        return HttpResponse("Error during export.", content_type="text/plain")
    # df = tasks.create_dataframe_from_mediafiles(mediafiles)
    response = HttpResponse(df.to_csv(index=False), content_type="text/csv")
    response["Content-Disposition"] = f"attachment; filename={fn}.csv"
    return response


@login_required
def download_xlsx_for_mediafiles_view(request, uploadedarchive_id: Optional[int] = None):
    """Download observation-level XLSX for the filtered media files."""
    mediafiles, name_suggestion = _get_mediafiles_for_export(request, uploadedarchive_id)
    fn = ("metadata_" + name_suggestion) if name_suggestion is not None else "metadata"

    try:
        df = _sequence_export_dataframe(mediafiles, request, SEQUENCE_EXPORT_DEFAULT_COLUMNS)
        if df.empty:
            return HttpResponse("No data available to export.", content_type="text/plain")
    except Exception:
        logger.error(traceback.format_exc())
        return HttpResponse("Error during export.", content_type="text/plain")

    # convert timezone-aware datetime to naive datetime
    df = model_tools.convert_datetime_to_naive(df)

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Observations")

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
    mediafiles = MediaFile.objects.filter(
        observations__identity=identity,
        observations__identity_is_representative=True,
    ).distinct()

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


def _build_image_upload_graph_figure(mediafiles_df: pd.DataFrame):
    """Build the upload statistics figure with a date axis."""
    grouped = (
        mediafiles_df.groupby(["date", "parent__owner__user__username"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["date", "parent__owner__user__username"])
    )
    fig = px.bar(
        grouped,
        x="date",
        y="count",
        color="parent__owner__user__username",
        title="Media Files Uploaded Over Time by User",
        labels={
            "date": "Upload Date",
            "count": "Number of Uploaded Files",
            "parent__owner__user__username": "User",
        },
    )

    fig.update_xaxes(type="date", title_text="Upload Date", rangeslider=dict(visible=True))
    fig.update_yaxes(title_text="Number of Uploads")
    fig.update_layout(hovermode="x unified")
    return fig


class ImageUploadGraphView(View):
    def get(self, request):
        """Render the image upload graph."""
        mediafiles = MediaFile.objects.all().values("parent__uploaded_at", "parent__owner__user__username")
        df = pd.DataFrame(mediafiles)
        df["parent__uploaded_at"] = pd.to_datetime(df["parent__uploaded_at"])
        df["date"] = df["parent__uploaded_at"].dt.date

        fig = _build_image_upload_graph_figure(df)
        graph = fig.to_html(full_html=False, config={"scrollZoom": True, "displaylogo": False, "responsive": True})

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
        media_file = (
            individual_to.observation_mediafiles()
            .filter(
                observations__identity=individual_to,
                observations__identity_is_representative=True,
            )
            .first()
        )

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

            AnimalObservation.objects.filter(identity=individual_from).update(identity=individual_to)

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
    # Reassign observations and identification suggestions from individual_from to individual_to.
    AnimalObservation.objects.filter(identity=individual_from).update(identity=individual_to)
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
                            if _spreadsheet_cell_requests_clear(row.get("predicted_category")):
                                ao.taxon = None
                            elif _spreadsheet_cell_has_value(row.get("predicted_category")):
                                ao.taxon = models.get_taxon(row["predicted_category"])  # remove this
                            counter_fields_updated += 1

                        code_supplied, code_value = _spreadsheet_optional_text_update(row.get("code"))
                        unique_name_supplied, unique_name_value = _spreadsheet_optional_text_update(row.get("unique_name"))
                        juv_code_supplied, juv_code_value = _spreadsheet_optional_text_update(row.get("juv_code"))
                        identity = None
                        if code_value:
                            identity = models.get_unique_code(code_value, workgroup=uploaded_archive.owner.workgroup)
                        elif unique_name_value:
                            identity = models.get_unique_name(
                                unique_name_value, workgroup=uploaded_archive.owner.workgroup
                            )
                        elif (code_supplied or juv_code_supplied) and ao.identity is not None:
                            identity = ao.identity
                        if identity is not None:
                            identity_updated = False
                            if unique_name_value and identity.name != unique_name_value:
                                identity.name = unique_name_value
                                identity_updated = True
                            if code_supplied and identity.code != code_value:
                                identity.code = code_value
                                identity_updated = True
                            if juv_code_supplied and identity.juv_code != juv_code_value:
                                identity.juv_code = juv_code_value
                                identity_updated = True
                            if identity_updated:
                                identity.save()
                                counter_fields_updated += 1
                            ao.identity = identity
                            counter_fields_updated += 1
                            counter_individuality += 1

                        if "locality_name" in row:
                            if _spreadsheet_cell_requests_clear(row.get("locality_name")):
                                mf.locality = None
                                counter_fields_updated += 1
                                counter_locality += 1
                            elif _spreadsheet_cell_has_value(row.get("locality_name")):
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
                            if _spreadsheet_cell_requests_clear(latitude) and _spreadsheet_cell_requests_clear(longitude):
                                mf.location = None
                                counter_fields_updated += 1
                            elif not pd.isna(latitude) and not pd.isna(longitude):
                                mf.location = f"{round(float(latitude), 3)},{round(float(longitude), 3)}"
                                counter_fields_updated += 1
                        if "datetime" in row:
                            # check if it is in django compatible datetime format
                            row_datetime = row["datetime"]
                            if _spreadsheet_cell_requests_clear(row_datetime):
                                mf.captured_at = None
                                counter_fields_updated += 1
                            elif isinstance(row_datetime, str):
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
                    + "'latitude', 'longitude', 'datetime' are optional. "
                    + "Blank cells keep existing values. "
                    + f"Use {forms.SPREADSHEET_CLEAR_TOKEN} to clear supported nullable fields such as predicted_category, code, juv_code, locality name, latitude+longitude and datetime.",
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
                + "'latitude', 'longitude', 'datetime' are optional. "
                + "Blank cells keep existing values. "
                + f"Use {forms.SPREADSHEET_CLEAR_TOKEN} to clear supported nullable fields such as predicted_category, code, juv_code, locality name, latitude+longitude and datetime.",
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
        "observations__identity",
        "observations__taxon",
        "observations__taxon__name",
        "observations__identity__name",
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
    individual_identity1 = get_object_or_404(
        IndividualIdentity,
        pk=individual_identity1_id,
        owner_workgroup=request.user.caiduser.workgroup,
    )

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
        candidate = best_identity_merge_candidate(individual_identity1, identities)
        form = forms.IndividualIdentitySelectSecondForMergeForm(
            identities=identities,
            initial={"identity": candidate},
        )
    return render(
        request,
        "caidapp/update_form.html",
        {
            "form": form,
            "headline": "Select identity for merge",
            "button": "Select",
            "text_note": "The selected identity will be merged into the first one and then deleted.",
            # "next": "caidapp:uploads_identities",
            "mediafile": individual_identity1.cover_mediafile(),
            "select2_enabled": True,
        },
    )


@login_required
def refresh_identities_suggestions_view(request):
    """Refresh identity suggestions view."""
    state = get_merge_identity_suggestions_state(request)
    if state["status"] in {"pending", "progress"}:
        messages.info(request, "Merge suggestion generation is already running.")
    else:
        refresh_identities_suggestions(request)
        messages.info(request, "Merge suggestion generation has started.")
    return redirect("caidapp:suggest_merge_identities")


def refresh_identities_suggestions(request, limit: int = 100, redirect: bool = True):
    """Refresh identity suggestions."""
    if not _celery_worker_available():
        result_id = compute_identity_suggestions(request.user.caiduser.workgroup.id, limit)
        request.session.pop("refresh_job_id", None)
        request.session["refresh_job_started_at"] = timezone.now().isoformat()
        request.session["refresh_result_id"] = result_id
        logger.debug(
            "No Celery worker available for merge suggestions. Computed synchronously for workgroup %s.",
            request.user.caiduser.workgroup_id,
        )
        return result_id

    job = tasks.refresh_identities_suggestions_task.delay(request.user.caiduser.workgroup.id, limit)
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


def _clear_merge_identity_suggestions_job(request):
    request.session.pop("refresh_job_id", None)
    request.session.pop("refresh_job_started_at", None)
    request.session.modified = True


def get_merge_identity_suggestions_state(request):
    """Return current merge suggestion generation state and latest data."""
    job_id = request.session.get("refresh_job_id")
    result_id = request.session.get("refresh_result_id")
    latest = models.MergeIdentitySuggestionResult.objects.filter(
        workgroup=request.user.caiduser.workgroup,
        **({"id": result_id} if result_id else {}),
    ).order_by("id").last()
    suggestions = latest.suggestions if latest else None
    created_at = latest.created_at if latest else None
    started_at = request.session.get("refresh_job_started_at")
    progress = {"current": 0, "total": 0, "suggestions_count": 0, "message": ""}

    if not job_id:
        return {
            "status": "success" if suggestions is not None else "idle",
            "suggestions": suggestions,
            "created_at": created_at,
            "started_at": started_at,
            "progress": progress,
        }

    result = AsyncResult(job_id)
    job_age = None
    if started_at:
        try:
            job_age = timezone.now() - datetime.datetime.fromisoformat(started_at)
        except ValueError:
            logger.warning("Invalid merge suggestion job timestamp: %s", started_at)
    missing_worker_is_stale = not _celery_worker_available() and (
        job_age is None or job_age > datetime.timedelta(seconds=30)
    )
    pending_job_is_stale = result.state == "PENDING" and job_age is not None and job_age > datetime.timedelta(minutes=15)
    if result.state in {"PENDING", "STARTED", "PROGRESS"} and (missing_worker_is_stale or pending_job_is_stale):
        _clear_merge_identity_suggestions_job(request)
        messages.warning(request, "The previous merge suggestion job is no longer running. You can start it again.")
        return {
            "status": "success" if suggestions is not None else "idle",
            "suggestions": suggestions,
            "created_at": created_at,
            "started_at": None,
            "progress": progress,
        }

    if result.state == "SUCCESS":
        payload = result.result or {}
        result_id = payload.get("result_id") if isinstance(payload, dict) else payload
        completed = models.MergeIdentitySuggestionResult.objects.filter(
            id=result_id,
            workgroup=request.user.caiduser.workgroup,
        ).first()
        _clear_merge_identity_suggestions_job(request)
        if completed is None:
            messages.error(request, "The merge suggestion result could not be found.")
            return {
                "status": "error",
                "suggestions": suggestions,
                "created_at": created_at,
                "started_at": started_at,
                "progress": progress,
            }
        request.session["refresh_result_id"] = completed.id
        return {
            "status": "success",
            "suggestions": completed.suggestions,
            "created_at": completed.created_at,
            "started_at": started_at,
            "progress": progress,
        }

    if result.state in {"FAILURE", "REVOKED"}:
        status = "error" if result.state == "FAILURE" else "cancelled"
        _clear_merge_identity_suggestions_job(request)
        return {
            "status": status,
            "suggestions": suggestions,
            "created_at": created_at,
            "started_at": started_at,
            "progress": progress,
        }

    meta = result.info if isinstance(result.info, dict) else {}
    progress.update(
        current=meta.get("current", 0),
        total=meta.get("total", 0),
        suggestions_count=meta.get("suggestions_count", 0),
        message=meta.get("message", "Preparing merge suggestions..."),
    )
    return {
        "status": "progress" if result.state == "PROGRESS" else "pending",
        "suggestions": suggestions,
        "created_at": created_at,
        "started_at": started_at,
        "progress": progress,
    }


@login_required
def merge_identity_suggestions_status(request):
    state = get_merge_identity_suggestions_state(request)
    return JsonResponse(
        {
            "status": state["status"],
            "progress": state["progress"],
            "redirect_url": reverse("caidapp:suggest_merge_identities") if state["status"] == "success" else "",
        }
    )


@login_required
def start_merge_identity_suggestions(request):
    if request.method != "POST":
        messages.error(request, "Invalid request method.")
        return redirect("caidapp:suggest_merge_identities")

    state = get_merge_identity_suggestions_state(request)
    if state["status"] in {"pending", "progress"}:
        messages.info(request, "Merge suggestion generation is already running.")
    else:
        refresh_identities_suggestions(request)
        messages.info(request, "Merge suggestion generation has started.")
    return redirect("caidapp:suggest_merge_identities")


@login_required
def cancel_merge_identity_suggestions(request):
    if request.method != "POST":
        messages.error(request, "Invalid request method.")
        return redirect("caidapp:suggest_merge_identities")

    job_id = request.session.get("refresh_job_id")
    if job_id:
        current_app.control.revoke(job_id, terminate=True)
        _clear_merge_identity_suggestions_job(request)
        messages.info(request, "Merge suggestion generation cancel requested.")
    else:
        messages.info(request, "No merge suggestion generation is running.")
    return redirect("caidapp:suggest_merge_identities")


@login_required
@require_POST
def exclude_merge_identity_suggestion(request):
    """Persistently suppress one identity pair from merge suggestions."""
    value = request.POST.get("excluded_suggestion", "")
    try:
        identity_id1, identity_id2 = (int(item) for item in value.split("|", 1))
    except (TypeError, ValueError):
        messages.error(request, "Invalid merge suggestion.")
        return redirect("caidapp:suggest_merge_identities")

    workgroup = request.user.caiduser.workgroup
    identities = {
        identity.id: identity
        for identity in IndividualIdentity.objects.filter(
            id__in=[identity_id1, identity_id2],
            owner_workgroup=workgroup,
        )
    }
    if len(identities) != 2 or identity_id1 == identity_id2:
        messages.error(request, "The identities could not be excluded.")
        return redirect("caidapp:suggest_merge_identities")

    identity_a_id, identity_b_id = sorted((identity_id1, identity_id2))
    models.MergeIdentitySuggestionExclusion.objects.get_or_create(
        workgroup=workgroup,
        identity_a_id=identity_a_id,
        identity_b_id=identity_b_id,
    )
    messages.success(
        request,
        f"'{identities[identity_id1].name}' and '{identities[identity_id2].name}' will no longer be suggested together.",
    )
    return redirect("caidapp:suggest_merge_identities")


@login_required
def suggest_merge_identities_view(request, limit: int = 100):
    """Suggest merge identities."""
    response = get_merge_identity_suggestions_state(request)
    if response["status"] == "idle" and response["suggestions"] is None:
        refresh_identities_suggestions(request)
        response = get_merge_identity_suggestions_state(request)

    if response["status"] in {"pending", "progress"}:
        return render(
            request,
            "caidapp/suggest_merge_identities.html",
            {
                "suggestions": [],
                "job_running": True,
                "job_progress": response["progress"],
                "status_url": reverse("caidapp:merge_identity_suggestions_status"),
                "cancel_url": reverse("caidapp:cancel_merge_identity_suggestions"),
            },
        )

    if "started_at" in response and response["started_at"]:
        started_at = datetime.datetime.fromisoformat(response["started_at"])
        messages.info(request, f"Suggestion refreshed {timesince_now(started_at)} ago.")
    if "created_at" in response and response["created_at"]:
        created_at = response["created_at"]
        messages.info(request, f"This data created {timesince_now(created_at)} ago.")
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

            logger.debug(f"{len(suggestions_ids)=}")

            all_identity_ids = {
                identity_id
                for identity_a_id, identity_b_id, _distance in suggestions_ids
                for identity_id in (identity_a_id, identity_b_id)
            }
            valid_identity_ids = set(
                IndividualIdentity.objects.filter(
                    id__in=all_identity_ids,
                    owner_workgroup=request.user.caiduser.workgroup,
                ).values_list("id", flat=True)
            )
            excluded_pairs = {
                tuple(sorted((identity_a_id, identity_b_id)))
                for identity_a_id, identity_b_id in models.MergeIdentitySuggestionExclusion.objects.filter(
                    workgroup=request.user.caiduser.workgroup
                ).values_list("identity_a_id", "identity_b_id")
            }
            valid_suggestions_ids = [
                suggestion
                for suggestion in suggestions_ids
                if suggestion[0] in valid_identity_ids and suggestion[1] in valid_identity_ids
                and tuple(sorted((suggestion[0], suggestion[1]))) not in excluded_pairs
            ]
            paginator = Paginator(valid_suggestions_ids, limit)
            page_obj = paginator.get_page(request.GET.get("page"))
            identity_ids = {
                identity_id
                for identity_a_id, identity_b_id, _distance in page_obj.object_list
                for identity_id in (identity_a_id, identity_b_id)
            }
            identities_by_id = {
                identity.id: identity
                for identity in _annotate_identities_with_mediafile_count(
                    IndividualIdentity.objects.filter(
                        id__in=identity_ids,
                        owner_workgroup=request.user.caiduser.workgroup,
                    )
                )
            }
            suggestions = []
            for identity_a_id, identity_b_id, distance in page_obj.object_list:
                try:
                    identity_a = identities_by_id[identity_a_id]
                    identity_b = identities_by_id[identity_b_id]
                except KeyError:
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
            page_obj = None

        return render(
            request,
            "caidapp/suggest_merge_identities.html",
            {
                "suggestions": suggestions,
                "page_obj": page_obj,
                "job_running": False,
                "start_url": reverse("caidapp:start_merge_identity_suggestions"),
            },
        )
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
        if mediafile.taxons_display:
            tooltip_parts.append(f"Taxon: {mediafile.taxons_display}")
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
        elif suspicious_mediafile:
            current_identity = suspicious_mediafile.identity_from_observations

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

    observations = list(suspicious_mediafile.observations.order_by("id")[:2])
    if len(observations) != 1:
        messages.warning(
            request,
            "Identity was not changed because this media file does not have exactly one observation.",
        )
        return redirect(next_url)
    observation = observations[0]
    observation.identity = suggested_identity
    observation.updated_by = request.user.caiduser
    observation.updated_at = django.utils.timezone.now()
    observation.save(update_fields=["identity", "updated_by", "updated_at"])

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
    workgroup = request.user.caiduser.workgroup
    active_regex = workgroup.get_identity_code_regex() if workgroup else models.DEFAULT_IDENTITY_CODE_REGEX
    job_state = _get_identity_code_suggestions_state(request)

    if job_state["status"] == "idle" and job_state["suggestion_ids"] is None:
        start_response = _start_identity_code_suggestions(request, clear_existing=False)
        if start_response["mode"] == "async":
            job_state = _get_identity_code_suggestions_state(request)
        else:
            return render(
                request,
                "caidapp/suggest_identity_codes.html",
                {
                    "identities": start_response["suggestions"],
                    "active_regex": active_regex,
                    "job_status": "success",
                    "job_running": False,
                    "job_progress": {"current": 0, "total": 0, "matches": 0, "message": ""},
                    "start_url": reverse("caidapp:start_identity_code_suggestions"),
                },
            )

    if job_state["status"] in {"pending", "progress"}:
        return render(
            request,
            "caidapp/suggest_identity_codes.html",
            {
                "identities": [],
                "active_regex": active_regex,
                "job_status": job_state["status"],
                "job_running": True,
                "job_progress": job_state["progress"],
                "job_started_at": job_state["started_at"],
                "status_url": reverse("caidapp:identity_code_suggestions_status"),
                "cancel_url": reverse("caidapp:cancel_identity_code_suggestions"),
            },
        )

    suggestions = _load_identity_code_suggestions_from_ids(
        request.user.caiduser.workgroup,
        job_state["suggestion_ids"],
    )

    return render(
        request,
        "caidapp/suggest_identity_codes.html",
        {
            "identities": suggestions,
            "active_regex": active_regex,
            "job_status": job_state["status"],
            "job_running": False,
            "job_progress": job_state["progress"],
            "start_url": reverse("caidapp:start_identity_code_suggestions"),
        },
    )


def _celery_worker_available() -> bool:
    """Return True when a Celery worker is available."""
    inspect = current_app.control.inspect(timeout=1.0)
    worker_stats = inspect.stats() if inspect else None
    return bool(worker_stats)


def _compute_identity_code_suggestions_sync(workgroup):
    """Compute identity code suggestions synchronously."""
    all_identities = _annotate_identities_with_mediafile_count(
        IndividualIdentity.objects.filter(
            owner_workgroup=workgroup,
        )
    )
    suggestions = []
    for identity in all_identities:
        suggested_code = identity.suggested_code_from_name()
        if suggested_code:
            identity.suggested_code = suggested_code
            identity.suggested_name = identity.suggested_name_without_code()
            suggestions.append(identity)
    return suggestions


def _clear_identity_code_suggestions_session_state(request, clear_suggestions: bool = False):
    """Clear job state for identity code suggestions from the session."""
    request.session.pop("identity_code_suggestions_job_id", None)
    request.session.pop("identity_code_suggestions_started_at", None)
    if clear_suggestions:
        request.session.pop("identity_code_suggestions_ids", None)
        request.session.pop("identity_code_suggestions_generated_at", None)
    request.session.modified = True


def _start_identity_code_suggestions(request, clear_existing: bool):
    """Start identity code suggestion generation asynchronously or synchronously."""
    if clear_existing:
        _clear_identity_code_suggestions_session_state(request, clear_suggestions=True)

    if _celery_worker_available():
        job = tasks.compute_identity_code_suggestions_task.delay(request.user.caiduser.workgroup.id)
        request.session["identity_code_suggestions_job_id"] = job.id
        request.session["identity_code_suggestions_started_at"] = timezone.now().isoformat()
        request.session.pop("identity_code_suggestions_ids", None)
        request.session.pop("identity_code_suggestions_generated_at", None)
        request.session.modified = True
        return {"mode": "async", "job_id": job.id}

    suggestions = _compute_identity_code_suggestions_sync(request.user.caiduser.workgroup)
    return {"mode": "sync", "suggestions": suggestions}


def _load_identity_code_suggestions_from_ids(workgroup, suggestion_ids):
    """Load cached identity code suggestions from IDs."""
    if not suggestion_ids:
        return []

    identities_by_id = {
        identity.id: identity
        for identity in _annotate_identities_with_mediafile_count(
            IndividualIdentity.objects.filter(
                owner_workgroup=workgroup,
                id__in=suggestion_ids,
            )
        )
    }
    suggestions = []
    for identity_id in suggestion_ids:
        identity = identities_by_id.get(identity_id)
        if identity is None:
            continue
        suggested_code = identity.suggested_code_from_name()
        if not suggested_code:
            continue
        identity.suggested_code = suggested_code
        identity.suggested_name = identity.suggested_name_without_code()
        suggestions.append(identity)
    return suggestions


def _get_identity_code_suggestions_state(request):
    """Return current state of identity code suggestion generation."""
    job_id = request.session.get("identity_code_suggestions_job_id")
    suggestion_ids = request.session.get("identity_code_suggestions_ids")
    started_at = request.session.get("identity_code_suggestions_started_at")
    progress = {"current": 0, "total": 0, "matches": 0, "message": ""}

    if not job_id:
        return {
            "status": "success" if suggestion_ids is not None else "idle",
            "suggestion_ids": suggestion_ids,
            "started_at": started_at,
            "progress": progress,
        }

    result = AsyncResult(job_id)
    if result.state in {"PENDING", "STARTED", "PROGRESS"} and not _celery_worker_available():
        _clear_identity_code_suggestions_session_state(request, clear_suggestions=False)
        messages.warning(
            request,
            "The previous suggestion generation job is no longer running. You can start it again.",
        )
        return {
            "status": "success" if suggestion_ids is not None else "idle",
            "suggestion_ids": suggestion_ids,
            "started_at": None,
            "progress": progress,
        }

    if result.state == "SUCCESS":
        payload = result.result or {}
        suggestion_ids = payload.get("suggestion_ids", [])
        request.session["identity_code_suggestions_ids"] = suggestion_ids
        request.session["identity_code_suggestions_generated_at"] = timezone.now().isoformat()
        _clear_identity_code_suggestions_session_state(request, clear_suggestions=False)
        return {
            "status": "success",
            "suggestion_ids": suggestion_ids,
            "started_at": started_at,
            "progress": {
                "current": payload.get("total", 0),
                "total": payload.get("total", 0),
                "matches": payload.get("matches", 0),
                "message": "Suggestion generation finished.",
            },
        }

    if result.state == "FAILURE":
        _clear_identity_code_suggestions_session_state(request, clear_suggestions=False)
        messages.error(request, f"Identity code suggestion generation failed: {result.result}")
        return {
            "status": "error",
            "suggestion_ids": suggestion_ids,
            "started_at": started_at,
            "progress": progress,
        }

    if result.state == "REVOKED":
        _clear_identity_code_suggestions_session_state(request, clear_suggestions=False)
        messages.info(request, "Identity code suggestion generation was cancelled.")
        return {
            "status": "cancelled",
            "suggestion_ids": suggestion_ids,
            "started_at": started_at,
            "progress": progress,
        }

    meta = result.info if isinstance(result.info, dict) else {}
    progress.update(
        {
            "current": meta.get("current", 0),
            "total": meta.get("total", 0),
            "matches": meta.get("matches", 0),
            "message": meta.get("message", "Preparing identity code suggestions..."),
        }
    )
    return {
        "status": "progress" if result.state == "PROGRESS" else "pending",
        "suggestion_ids": suggestion_ids,
        "started_at": started_at,
        "progress": progress,
    }


@login_required
def identity_code_suggestions_status(request):
    """Return JSON status of identity code suggestion generation."""
    state = _get_identity_code_suggestions_state(request)
    return JsonResponse(
        {
            "status": state["status"],
            "progress": state["progress"],
            "redirect_url": reverse("caidapp:show_identity_code_suggestions") if state["status"] == "success" else "",
        }
    )


@login_required
def start_identity_code_suggestions(request):
    """Explicitly start or regenerate identity code suggestions."""
    if request.method != "POST":
        messages.error(request, "Invalid request method.")
        return redirect("caidapp:show_identity_code_suggestions")

    start_response = _start_identity_code_suggestions(request, clear_existing=True)
    if start_response["mode"] == "async":
        messages.info(request, "Identity code suggestion generation has started.")
    else:
        messages.info(request, "Identity code suggestions were generated synchronously because no worker is available.")
    return redirect("caidapp:show_identity_code_suggestions")


@login_required
def cancel_identity_code_suggestions(request):
    """Cancel the running identity code suggestion generation."""
    if request.method != "POST":
        messages.error(request, "Invalid request method.")
        return redirect("caidapp:show_identity_code_suggestions")

    job_id = request.session.get("identity_code_suggestions_job_id")
    if not job_id:
        messages.info(request, "No identity code suggestion generation is running.")
        return redirect("caidapp:show_identity_code_suggestions")

    current_app.control.revoke(job_id, terminate=True)
    _clear_identity_code_suggestions_session_state(request, clear_suggestions=False)
    messages.info(request, "Identity code suggestion generation cancel requested.")
    return redirect("caidapp:show_identity_code_suggestions")


def _apply_identity_code_to_identity(identity: IndividualIdentity, rename: bool = True) -> bool:
    """Apply suggested code to one identity."""
    code = identity.suggested_code_from_name()
    if not code:
        return False

    identity.note = identity.note + f"\nformer code: {str(identity.code)} \nformer name: {str(identity.name)}"
    identity.code = code
    if rename:
        identity.name = identity.name.replace(code, "").strip()
    identity.save()
    return True


@login_required
def apply_identity_code_suggestion(request, identity_id: int, rename: bool = True):
    """Use the suggested individuality code."""
    identity = get_object_or_404(IndividualIdentity, pk=identity_id, owner_workgroup=request.user.caiduser.workgroup)
    _apply_identity_code_to_identity(identity, rename=rename)
    return redirect(request.META.get("HTTP_REFERER", "/"))


@login_required
def apply_selected_identity_code_suggestions(request, rename: bool = True):
    """Apply suggested codes to selected identities."""
    if request.method != "POST":
        messages.error(request, "Invalid request method.")
        return redirect("caidapp:show_identity_code_suggestions")

    selected_identity_ids = request.POST.getlist("identity_ids")
    if not selected_identity_ids:
        messages.info(request, "No identities were selected.")
        return redirect("caidapp:show_identity_code_suggestions")

    identities = IndividualIdentity.objects.filter(
        owner_workgroup=request.user.caiduser.workgroup,
        id__in=selected_identity_ids,
    )
    applied_count = 0
    for identity in identities:
        applied_count += int(_apply_identity_code_to_identity(identity, rename=rename))

    if applied_count == 0:
        messages.info(request, "No selected identities had an applicable code suggestion.")
    else:
        messages.success(request, f"Applied code suggestions to {applied_count} identities.")

    return redirect("caidapp:show_identity_code_suggestions")


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
    requested_ids = request.GET.get("ids", "")
    if requested_ids:
        try:
            uploaded_archives = uploaded_archives.filter(
                id__in=[int(value) for value in requested_ids.split(",") if value]
            )
        except ValueError:
            return JsonResponse({"error": "Invalid archive ids"}, status=400)

    data = []
    for ua in uploaded_archives:
        if species:
            st = ua.get_status()
        else:
            st = ua.get_identification_status()
        progress = None
        if ua.taxon_status == "TAIP" and ua.taxon_task_id:
            try:
                task = AsyncResult(ua.taxon_task_id)
                state = task.state
                info = task.info
                if state == "PROGRESS" and isinstance(info, dict):
                    raw_percent = info.get("percent")
                    percent = max(0, min(int(raw_percent), 99)) if raw_percent is not None else None
                    progress = {
                        "state": state,
                        "percent": percent,
                        "stage": str(info.get("stage", "")),
                        "message": str(info.get("message", "Processing upload")),
                    }
                elif state == "SUCCESS" and isinstance(info, dict) and info.get("status") == "ERROR":
                    progress = {
                        "state": state,
                        "percent": None,
                        "stage": "failed",
                        "message": "Processing failed",
                    }
                elif state == "SUCCESS":
                    progress = {
                        "state": state,
                        "percent": 99,
                        "stage": "import_results",
                        "message": "Importing processed results",
                    }
                else:
                    progress = {
                        "state": state,
                        "percent": None,
                        "stage": "queued" if state == "PENDING" else "starting",
                        "message": "Queued for processing" if state == "PENDING" else "Starting processing",
                    }
            except Exception:
                logger.warning("Could not read taxon progress for archive %s", ua.id, exc_info=True)
        # status = st["status"]
        # status_message = st["status_message"]
        data.append({"id": ua.id, **st, "progress": progress})

    return JsonResponse({"archives": data})


@login_required
def export_identities_csv(request):
    """Export identities to CSV."""

    all_identities = IndividualIdentity.objects.filter(
        owner_workgroup=request.user.caiduser.workgroup,
        # **user_has_access_filter_params(request.user.caiduser, "owner")
    ).annotate(mediafile_count=Count("animalobservation__mediafile", distinct=True)).order_by("id")
    df = pd.DataFrame.from_records(all_identities.values())[
        ["id", "name", "code", "juv_code", "sex", "coat_type", "birth_date", "death_date", "note", "mediafile_count"]
    ]

    return views_general.csv_response(df, "identities")


@login_required
def export_identities_xlsx(request):
    """Export identities to Excel."""
    all_identities = IndividualIdentity.objects.filter(
        owner_workgroup=request.user.caiduser.workgroup,
        # **user_has_access_filter_params(request.user.caiduser, "owner")
    ).annotate(mediafile_count=Count("animalobservation__mediafile", distinct=True)).order_by("id")
    df = pd.DataFrame.from_records(all_identities.values())[
        ["id", "name", "code", "juv_code", "sex", "coat_type", "birth_date", "death_date", "note", "mediafile_count"]
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


def _spreadsheet_cell_requests_clear(value) -> bool:
    """Return True when spreadsheet cell explicitly asks to clear a nullable value."""
    if pd.isna(value):
        return False
    return str(value).strip().upper() == forms.SPREADSHEET_CLEAR_TOKEN.upper()


def _spreadsheet_optional_text_update(value) -> tuple[bool, Optional[str]]:
    """Return whether a text cell requests an update and the normalized value."""
    if _spreadsheet_cell_requests_clear(value):
        return True, None
    if not _spreadsheet_cell_has_value(value):
        return False, None
    return True, str(value).strip()


def _spreadsheet_optional_value_update(value) -> tuple[bool, object]:
    """Return whether a cell requests an update and the raw replacement value."""
    if _spreadsheet_cell_requests_clear(value):
        return True, None
    if not _spreadsheet_cell_has_value(value):
        return False, None
    return True, value


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

                    if identity is None and _spreadsheet_cell_has_value(row.get("code")) and not _spreadsheet_cell_requests_clear(row.get("code")):
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
                if "code" in row:
                    code_supplied, code_value = _spreadsheet_optional_text_update(row.get("code"))
                    if code_supplied:
                        identity.code = code_value
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
                    note_supplied, note_value = _spreadsheet_optional_text_update(row.get("note"))
                    if note_supplied:
                        identity.note = note_value or ""

                if "juv_code" in row:
                    juv_code_supplied, juv_code_value = _spreadsheet_optional_text_update(row.get("juv_code"))
                    if juv_code_supplied:
                        identity.juv_code = juv_code_value

                if "birth_date" in row:
                    birth_date_supplied, birth_date_value = _spreadsheet_optional_value_update(row.get("birth_date"))
                    if birth_date_supplied:
                        identity.birth_date = birth_date_value
                if "death_date" in row:
                    death_date_supplied, death_date_value = _spreadsheet_optional_value_update(row.get("death_date"))
                    if death_date_supplied:
                        identity.death_date = death_date_value

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
            + "Optional columns are 'sex', 'coat_type', 'birth_date', 'death_date', 'note'. "
            + "Blank cells keep existing values. "
            + f"Use {forms.SPREADSHEET_CLEAR_TOKEN} to clear supported nullable fields such as code, juv_code, note, birth_date and death_date.",
            "next": "caidapp:individual_identities",
        },
    )


OBSERVATION_IMPORT_CLEAR = forms.SPREADSHEET_CLEAR_TOKEN
OBSERVATION_BBOX_COLUMNS = ("bbox_cx", "bbox_cy", "bbox_w", "bbox_h")
OBSERVATION_BBOX_MODEL_FIELDS = ("bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height")


def _spreadsheet_optional_float(value, column: str) -> Optional[float]:
    if not _spreadsheet_cell_has_value(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must be a number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{column} must be finite")
    return parsed


def _spreadsheet_optional_bool(value, column: str) -> Optional[bool]:
    if not _spreadsheet_cell_has_value(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "y", "1", "ano"}:
        return True
    if normalized in {"false", "no", "n", "0", "ne"}:
        return False
    raise ValueError(f"{column} must be true or false")


def _spreadsheet_strict_id(value, column: str, required: bool = False) -> Optional[int]:
    """Parse a positive integer ID without truncating or treating invalid input as blank."""
    if not _spreadsheet_cell_has_value(value):
        if required:
            raise ValueError(f"{column} is required")
        return None
    if isinstance(value, bool):
        raise ValueError(f"{column} must be a positive integer")
    text = str(value).strip()
    if re.fullmatch(r"[1-9]\d*", text):
        return int(text)
    if re.fullmatch(r"[1-9]\d*\.0+", text):
        return int(float(text))
    raise ValueError(f"{column} must be a positive integer")


def _resolve_imported_related_object(row, id_column, name_columns, queryset, label):
    """Resolve an optional related object, using a stable ID whenever supplied."""
    raw_id = row.get(id_column)
    if _spreadsheet_cell_requests_clear(raw_id):
        return None, True
    object_id = _spreadsheet_strict_id(raw_id, id_column)
    if object_id is not None:
        obj = queryset.filter(id=object_id).first()
        if obj is None:
            raise ValueError(f"unknown or inaccessible {label} ID {object_id}")
        for column in name_columns:
            value = row.get(column)
            if not _spreadsheet_cell_has_value(value):
                continue
            model_field = "name" if column in {"predicted_category", "predicted_taxon", "unique_name", "locality name"} else column
            if str(getattr(obj, model_field) or "").strip() != str(value).strip():
                raise ValueError(f"{label} ID {object_id} does not match {column} '{value}'")
        return obj, True
    for column in name_columns:
        value = row.get(column)
        if not _spreadsheet_cell_has_value(value):
            continue
        model_field = "name" if column in {"predicted_category", "predicted_taxon", "unique_name", "locality name"} else column
        matches = queryset.filter(**{model_field: str(value).strip()})
        match_count = matches.count()
        if match_count != 1:
            message = (
                f"{label} {column} '{value}' matched {match_count} records; "
                "it must identify exactly one record"
            )
            if label == "identity" and column == "unique_name":
                if _looks_like_original_file_path(value):
                    message += (
                        ". This value looks like an original file path, not an identity name; "
                        "check that the spreadsheet columns have not shifted"
                    )
                message += ". Prefer identity_id for a known individual, or leave identity_id and unique_name blank"
            raise ValueError(message)
        return matches.first(), True
    return None, False


def _looks_like_original_file_path(value) -> bool:
    value = str(value)
    return "/" in value or "\\" in value or Path(value).suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _resolve_imported_identity(row, queryset, caiduser, create_missing=False):
    """Resolve an identity, optionally creating one from an explicit unique_name."""
    raw_id = row.get("identity_id")
    if _spreadsheet_cell_requests_clear(raw_id):
        return None, True
    identity_id = _spreadsheet_strict_id(raw_id, "identity_id")
    if identity_id is not None:
        return _resolve_imported_related_object(row, "identity_id", ("unique_name", "code"), queryset, "identity")

    unique_name = row.get("unique_name")
    if not _spreadsheet_cell_has_value(unique_name):
        return _resolve_imported_related_object(row, "identity_id", ("unique_name", "code"), queryset, "identity")

    unique_name = str(unique_name).strip()
    matches = queryset.filter(name=unique_name)
    match_count = matches.count()
    if match_count == 1:
        return matches.first(), True
    if match_count > 1:
        raise ValueError(f"identity unique_name '{unique_name}' matched {match_count} records; it must identify exactly one record")
    if not create_missing:
        message = f"identity unique_name '{unique_name}' matched 0 records; it must identify exactly one record"
        if _looks_like_original_file_path(unique_name):
            message += ". This value looks like an original file path, not an identity name; check that the spreadsheet columns have not shifted"
        raise ValueError(message)
    if _looks_like_original_file_path(unique_name):
        raise ValueError(
            "identity unique_name "
            f"'{unique_name}' looks like an original file path, so it cannot be used to create an identity"
        )
    if len(unique_name) > IndividualIdentity._meta.get_field("name").max_length:
        raise ValueError("identity unique_name is too long")

    code = row.get("code")
    code = str(code).strip() if _spreadsheet_cell_has_value(code) else None
    if code:
        if len(code) > IndividualIdentity._meta.get_field("code").max_length:
            raise ValueError("identity code is too long")
        code_match_count = queryset.filter(code=code).count()
        if code_match_count:
            raise ValueError(f"identity code '{code}' is already used by {code_match_count} records")
    return IndividualIdentity.objects.create(
        name=unique_name,
        code=code,
        owner_workgroup=caiduser.workgroup,
        updated_by=caiduser,
    ), True


def _resolve_imported_locality(row, queryset, caiduser, create_missing=False):
    """Resolve a locality, optionally creating a missing one from an explicit name."""
    raw_id = row.get("locality_id")
    if _spreadsheet_cell_requests_clear(raw_id):
        return None, True
    locality_id = _spreadsheet_strict_id(raw_id, "locality_id")
    locality_name = row.get("locality name")
    if locality_id is not None:
        locality = queryset.filter(id=locality_id).first()
        if locality is None:
            raise ValueError(f"unknown or inaccessible locality ID {locality_id}")
        if _spreadsheet_cell_has_value(locality_name) and locality.name.strip() != str(locality_name).strip():
            raise ValueError(f"locality ID {locality_id} does not match locality name '{locality_name}'")
        return locality, True
    if not _spreadsheet_cell_has_value(locality_name):
        return None, False

    locality_name = str(locality_name).strip()
    matches = queryset.filter(name=locality_name)
    match_count = matches.count()
    if match_count == 1:
        return matches.first(), True
    if match_count == 0 and create_missing:
        if len(locality_name) > Locality._meta.get_field("name").max_length:
            raise ValueError("locality name is too long")
        return Locality.objects.create(name=locality_name, owner=caiduser), True
    if match_count == 0:
        raise ValueError(f"locality name '{locality_name}' matched 0 records; it must identify exactly one record")
    raise ValueError(f"locality name '{locality_name}' matched {match_count} records; it must identify exactly one record")


def _observation_bbox_from_row(row) -> Tuple[Optional[List[Optional[float]]], bool]:
    raw_values = [row.get(column) for column in OBSERVATION_BBOX_COLUMNS]
    present = [_spreadsheet_cell_has_value(value) for value in raw_values]
    if not any(present):
        return None, False
    if all(_spreadsheet_cell_requests_clear(value) for value in raw_values):
        return [None, None, None, None], True
    if not all(present):
        raise ValueError("bbox requires bbox_cx, bbox_cy, bbox_w and bbox_h together")
    values = [_spreadsheet_optional_float(value, column) for value, column in zip(raw_values, OBSERVATION_BBOX_COLUMNS)]
    cx, cy, width, height = values
    if any(value is None or value < 0 or value > 1 for value in values):
        raise ValueError("bbox values must be between 0 and 1")
    if cx - width / 2 < 0 or cx + width / 2 > 1 or cy - height / 2 < 0 or cy + height / 2 > 1:
        raise ValueError("bbox must fit within the normalized image bounds")
    return values, True


def _read_uploaded_spreadsheet(file) -> pd.DataFrame:
    content = file.read()
    suffix = Path(file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(BytesIO(content))
    if suffix == ".xlsx":
        return pd.read_excel(BytesIO(content))
    raise ValueError("Only .csv and .xlsx files are supported.")


def _observation_mediafile_note_from_row(row) -> Tuple[bool, str]:
    """Return a mediafile note update, accepting the legacy note column."""
    updates = []
    for column in ("mediafile_note", "note"):
        supplied, value = _spreadsheet_optional_text_update(row.get(column))
        if supplied:
            updates.append((column, value or ""))
    if not updates:
        return False, ""
    if len({value for _column, value in updates}) > 1:
        raise ValueError("mediafile_note and legacy note contain conflicting values")
    return True, updates[0][1]


OBSERVATION_IMPORT_OBSERVATION_VALUE_COLUMNS = (
    "taxon_id",
    "name",
    "predicted_category",
    "predicted_taxon_id",
    "predicted_taxon",
    "predicted_taxon_confidence",
    "identity_id",
    "unique_name",
    "code",
    "juv_code",
    "taxon_verified",
    "identity_is_representative",
    "orientation",
    *OBSERVATION_BBOX_COLUMNS,
)


def _row_has_observation_values(row) -> bool:
    """Return whether a blank-observation export row carries an animal update."""
    return any(_spreadsheet_cell_has_value(row.get(column)) for column in OBSERVATION_IMPORT_OBSERVATION_VALUE_COLUMNS)


def _row_includes_observation_columns(row) -> bool:
    """Distinguish an exported blank observation row from a minimal legacy import."""
    return any(column in row for column in OBSERVATION_IMPORT_OBSERVATION_VALUE_COLUMNS)


def _import_observation_dataframe(
    df: pd.DataFrame,
    caiduser,
    create_missing_localities=False,
    create_missing_identities=False,
    progress_callback=None,
) -> Tuple[int, int]:
    """Atomically create or update observations from an exported spreadsheet."""
    if "mediafile_id" not in df.columns:
        raise ValueError("Missing required column mediafile_id")
    accessible_mediafiles = MediaFile.objects.filter(
        Q(album__albumsharerole__user=caiduser)
        | Q(**models.user_has_access_filter_params(caiduser, "parent__owner"))
    ).distinct()
    accessible_observations = AnimalObservation.objects.filter(mediafile__in=accessible_mediafiles)
    identity_queryset = IndividualIdentity.objects.filter(owner_workgroup=caiduser.workgroup)
    locality_queryset = Locality.objects.filter(**models.user_has_access_filter_params(caiduser, "owner"))
    seen_observation_ids = set()
    mediafile_note_updates = {}
    created = updated = 0

    with transaction.atomic():
        total_rows = len(df.index)
        for row_number, series in enumerate(df.to_dict(orient="records"), start=2):
            try:
                mediafile_id = _spreadsheet_strict_id(series.get("mediafile_id"), "mediafile_id", required=True)
                mediafile = accessible_mediafiles.select_related("locality").filter(id=mediafile_id).first()
                if mediafile is None:
                    raise ValueError(f"unknown or inaccessible mediafile ID {mediafile_id}")

                mediafile_note_supplied, mediafile_note = _observation_mediafile_note_from_row(series)
                if mediafile_note_supplied:
                    previous_note = mediafile_note_updates.get(mediafile_id)
                    if previous_note is not None and previous_note != mediafile_note:
                        raise ValueError(f"conflicting mediafile_note values for mediafile_id {mediafile_id}")
                    mediafile_note_updates[mediafile_id] = mediafile_note

                observation_id = _spreadsheet_strict_id(series.get("observation_id"), "observation_id")
                skip_blank_observation = (
                    observation_id is None
                    and _row_includes_observation_columns(series)
                    and not _row_has_observation_values(series)
                )
                if observation_id is not None:
                    if observation_id in seen_observation_ids:
                        raise ValueError(f"duplicate observation_id {observation_id}")
                    seen_observation_ids.add(observation_id)
                    observation = accessible_observations.filter(id=observation_id).first()
                    if observation is None:
                        raise ValueError(f"unknown or inaccessible observation ID {observation_id}")
                    if observation.mediafile_id != mediafile.id:
                        raise ValueError("observation_id does not belong to mediafile_id")
                    updated += 1
                else:
                    observation = AnimalObservation(mediafile=mediafile)
                    if not skip_blank_observation:
                        created += 1

                taxon, supplied = _resolve_imported_related_object(
                    series, "taxon_id", ("name", "predicted_category"), Taxon.objects.all(), "taxon"
                )
                if supplied:
                    observation.taxon = taxon
                predicted_taxon, supplied = _resolve_imported_related_object(
                    series, "predicted_taxon_id", ("predicted_taxon",), Taxon.objects.all(), "predicted taxon"
                )
                if supplied:
                    observation.predicted_taxon = predicted_taxon
                identity, supplied = _resolve_imported_identity(
                    series,
                    identity_queryset,
                    caiduser,
                    create_missing=create_missing_identities,
                )
                if supplied:
                    observation.identity = identity

                for column in ("taxon_verified", "identity_is_representative"):
                    parsed = _spreadsheet_optional_bool(series.get(column), column)
                    if parsed is not None:
                        setattr(observation, column, parsed)
                if _spreadsheet_cell_has_value(series.get("orientation")):
                    orientation = str(series["orientation"]).strip().upper()
                    if orientation not in dict(models.ORIENTATION_CHOICES):
                        raise ValueError(f"invalid orientation '{orientation}'")
                    observation.orientation = orientation
                confidence = _spreadsheet_optional_float(
                    series.get("predicted_taxon_confidence"), "predicted_taxon_confidence"
                )
                if confidence is not None:
                    if not 0 <= confidence <= 1:
                        raise ValueError("predicted_taxon_confidence must be between 0 and 1")
                    observation.predicted_taxon_confidence = confidence
                bbox, supplied = _observation_bbox_from_row(series)
                if supplied:
                    for model_field, value in zip(OBSERVATION_BBOX_MODEL_FIELDS, bbox):
                        setattr(observation, model_field, value)
                if not skip_blank_observation:
                    observation.updated_by = caiduser
                    observation.updated_at = timezone.now()
                    observation.save()

                locality, supplied = _resolve_imported_locality(
                    series,
                    locality_queryset,
                    caiduser,
                    create_missing=create_missing_localities,
                )
                mediafile_fields = []
                if supplied:
                    mediafile.locality = locality
                    mediafile_fields.append("locality")
                raw_location = series.get("mediafile_location")
                if _spreadsheet_cell_requests_clear(raw_location):
                    mediafile.location = None
                    mediafile_fields.append("location")
                elif _spreadsheet_cell_has_value(raw_location):
                    location = str(raw_location).strip()
                    parts = location.split(",")
                    if len(parts) != 2:
                        raise ValueError("mediafile_location must be 'latitude,longitude'")
                    latitude, longitude = [_spreadsheet_optional_float(part, "mediafile_location") for part in parts]
                    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                        raise ValueError("mediafile_location is outside valid latitude/longitude bounds")
                    mediafile.location = f"{latitude},{longitude}"
                    mediafile_fields.append("location")
                if mediafile_note_supplied:
                    mediafile.note = mediafile_note
                    mediafile_fields.append("note")
                if mediafile_fields:
                    mediafile.updated_by = caiduser
                    mediafile.updated_at = timezone.now()
                    mediafile.save(update_fields=[*mediafile_fields, "updated_by", "updated_at"])
            except ValueError as exc:
                raise ValueError(f"Row {row_number}: {exc}") from exc
            if progress_callback and (row_number == total_rows + 1 or row_number % 25 == 0):
                progress_callback(row_number - 1, total_rows)
    return created, updated


def _collect_observation_import_errors(
    df: pd.DataFrame,
    caiduser,
    create_missing_localities=False,
    create_missing_identities=False,
) -> list[str]:
    """Return row errors for a failed import without persisting diagnostic writes."""
    errors = []
    with transaction.atomic():
        for row_number, record in enumerate(df.to_dict(orient="records"), start=2):
            try:
                _import_observation_dataframe(
                    pd.DataFrame([record], columns=df.columns),
                    caiduser,
                    create_missing_localities=create_missing_localities,
                    create_missing_identities=create_missing_identities,
                )
            except Exception as exc:
                message = str(exc)
                if message.startswith("Row 2: "):
                    message = f"Row {row_number}: {message[7:]}"
                errors.append(message)
        transaction.set_rollback(True)
    return errors


@login_required
def import_observations_view(request):
    """Import observation and related mediafile metadata from CSV or XLSX."""
    if request.method == "POST":
        form = forms.SpreadsheetFileImportForm(request.POST, request.FILES)
        if form.is_valid():
            observation_import = None
            stored_file = None
            try:
                uploaded_file = form.cleaned_data["spreadsheet_file"]
                suffix = Path(uploaded_file.name).suffix.lower()
                if suffix not in {".csv", ".xlsx"}:
                    raise ValueError("Only .csv and .xlsx files are supported.")
                import_dir = Path(settings.PRIVATE_DATA_PATH) / "observation_imports"
                import_dir.mkdir(parents=True, exist_ok=True)
                stored_file = import_dir / f"{uuid.uuid4().hex}{suffix}"
                with stored_file.open("wb") as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)
                observation_import = models.ObservationImport.objects.create(
                    caiduser=request.user.caiduser,
                    source_filename=Path(uploaded_file.name).name[:255],
                    stored_file=str(stored_file),
                    create_missing_localities=form.cleaned_data["create_missing_localities"],
                    create_missing_identities=form.cleaned_data["create_missing_identities"],
                )
            except Exception as exc:
                if observation_import is not None:
                    observation_import.status = models.ObservationImport.STATUS_FAILED
                    observation_import.error_message = f"Could not queue import: {exc}"
                    observation_import.finished_at = timezone.now()
                    observation_import.save(update_fields=["status", "error_message", "finished_at"])
                if stored_file is not None:
                    stored_file.unlink(missing_ok=True)
                form.add_error("spreadsheet_file", f"Import cancelled — no rows were changed. {exc}")
            else:
                task = tasks.import_observations_task.delay(observation_import.id)
                observation_import.task_id = task.id
                observation_import.save(update_fields=["task_id"])
                return redirect("caidapp:observation_import_status", pk=observation_import.id)
    else:
        form = forms.SpreadsheetFileImportForm()
    return render(
        request,
        "caidapp/update_form.html",
        {
            "form": form,
            "headline": "Import observations",
            "button": "Import",
            "text_note": (
                "Upload an exported CSV or XLSX. observation_id updates an existing observation; "
                "a blank observation_id creates one for mediafile_id, except for a completely blank observation row "
                "from an export (which preserves an image without observations). Blank cells leave values unchanged. "
                f"Use {OBSERVATION_IMPORT_CLEAR} to clear supported nullable values. "
                "mediafile_note updates the note on the media file; the legacy note column is also accepted. "
                "Rows for the same mediafile_id must not contain conflicting note values. "
                "Optional checkboxes can create missing localities from locality name and identities from unique_name. "
                "Identity creation rejects values that look like original file paths. "
                "The import is atomic: if any row is invalid, no rows from the file are saved. "
                "BBox uses YOLO-style normalized bbox_cx, bbox_cy, bbox_w and bbox_h values in the range 0-1."
            ),
            "next": "caidapp:media_files",
        },
    )


@login_required
def observation_import_status_view(request, pk):
    """Show a persisted observation-import status to its submitting user only."""
    observation_import = get_object_or_404(models.ObservationImport, pk=pk, caiduser=request.user.caiduser)
    return render(request, "caidapp/observation_import_status.html", {"observation_import": observation_import})


# create view which will be shown just before the identify to make sure that the user wants to identify
@login_required
def pre_identify_view(request):
    """Show pre-identification confirmation page."""
    if not user_can_manage_identification(request.user):
        return HttpResponseNotAllowed("Identification is for workgroup admins only.")
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

    if request.user.caiduser.workgroup != mf.parent.owner.workgroup:
        return HttpResponseNotAllowed("Not allowed")

    observations = list(mf.observations.order_by("id")[:2])
    if len(observations) != 1:
        return JsonResponse(
            {
                "ok": False,
                "error": "Representative status can be changed here only for a media file with exactly one observation.",
            },
            status=400,
        )
    observation = observations[0]
    if observation.identity_id is None:
        return JsonResponse({"ok": False, "error": "The observation has no identity."}, status=400)

    representative = not observation.identity_is_representative
    observation.identity_is_representative = representative
    observation.updated_by = request.user.caiduser
    observation.updated_at = django.utils.timezone.now()
    observation.save(update_fields=["identity_is_representative", "updated_by", "updated_at"])

    logger.debug("almost done")
    return JsonResponse({"ok": True, "representative": representative})


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


class NotificationListView(LoginRequiredMixin, ListView):
    model = models.Notification
    template_name = "caidapp/notification_list.html"
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
        context["object_detail_url"] = "caidapp:notification-detail"
        # context["object_detail_url"] = "caidapp:notification-detail"
        # context["object_update_url"] = "caidapp:notification-update"
        # context["object_delete_url"] = "caidapp:notification-delete"
        # context["object_create_url"] = "caidapp:notification-create"
        return context


@login_required
@require_POST
def mark_all_notifications_as_read(request):
    """Mark every unread notification belonging to the current user as read."""
    models.NotificationRecipient.objects.filter(
        user=request.user.caiduser,
        read=False,
    ).update(read=True, read_at=timezone.now())
    return redirect("caidapp:notifications")


class NotificationDetailView(LoginRequiredMixin, DetailView):
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
            me_as_recipient.read_at = timezone.now()
            me_as_recipient.save(update_fields=["read", "read_at"])
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
        link_url = self.object.get_link_url()
        if link_url:
            context["bottom_button_list"] = [
                {
                    "label": self.object.link_label or _("Open"),
                    "style": "primary",
                    "url": link_url,
                }
            ]
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
    form_class = forms.WorkGroupInvitationForm
    success_url = reverse_lazy("caidapp:workgroup_invitations")

    def test_func(self):
        user = self.request.user.caiduser
        return user.workgroup_admin and user.workgroup_id is not None

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs["target_workgroup"] = self.request.user.caiduser.workgroup
        return kwargs

    # def dispatch(self, request, *args, **kwargs):
    #     """Check if the user is a workgroup admin and set the target workgroup for the invitation."""
    #     response
    #     if not request.user.caiduser.workgroup_admin:
    #         raise PermissionDenied
    #
    #     self.target_workgroup = request.user.caiduser.workgroup
    #     return super().dispatch(request, *args, **kwargs)

    @django.db.transaction.atomic
    def form_valid(self, form):
        """Set the inviter and target workgroup before saving the form."""
        form.instance.invited_by = self.request.user.caiduser
        form.instance.target_workgroup = self.request.user.caiduser.workgroup
        response = super().form_valid(form)
        models.Notification.create_for(
            message=_("You have been invited to join %(workgroup)s.")
            % {"workgroup": form.instance.target_workgroup.name},
            users=[form.instance.invited_user],
            link_url_name="caidapp:workgroup_invitation_detail",
            link_url_kwargs={"pk": form.instance.pk},
            link_label=_("View invitation"),
        )
        return response


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
    success_url = reverse_lazy("caidapp:workgroup_invitations_for_user")

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
    success_url = reverse_lazy("caidapp:workgroup_invitations_for_user")

    def get_queryset(self):
        """Limit queryset to pending invitations for the current user."""
        return models.WorkGroupInvitation.objects.filter(
            invited_user=self.request.user.caiduser,
            status="pending",
        )

    @django.db.transaction.atomic
    def form_valid(self, form):
        """Accept the invitation and migrate the user to the new workgroup."""
        invitation = self.object

        # 🔐 bezpečnost – ještě jednou pro jistotu
        if invitation.invited_user != self.request.user.caiduser:
            raise PermissionDenied

        # 🔥 migrace uživatele
        try:
            migrate_user_to_workgroup(
                user=invitation.invited_user,
                target_workgroup=invitation.target_workgroup,
                approved_by=invitation.invited_by,
            )
        except ValidationError as exc:
            form.add_error(None, exc)
            return self.form_invalid(form)

        invitation.status = "accepted"
        invitation.responded_at = timezone.now()
        invitation.save(update_fields=["status", "responded_at"])

        return redirect(self.get_success_url())

