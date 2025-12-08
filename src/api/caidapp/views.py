import datetime
import logging
import os
import random
import re
import time
import traceback
from functools import wraps
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import zipfile
from django.core.files.base import ContentFile
import io

import django
import django.db
import django.utils.timezone
import numpy as np
import pandas as pd
import plotly.express as px
from celery import signature
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
from django.core.paginator import Page, Paginator
from django.db.models import Count, F, Func, Max, Min, OuterRef, Q, QuerySet, Subquery, Value
from django.db.models.functions import Cast
from django.forms import modelformset_factory
from django.forms.models import model_to_dict
from django.http import HttpResponseNotAllowed, JsonResponse, HttpRequest
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

from . import (
    filters,
    forms,
    model_tools,
    models,
    tasks,
    views_general,
    views_locality,
    views_uploads,
)
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
from .model_extra import user_has_rw_acces_to_uploadedarchive, user_has_rw_access_to_mediafile
from .model_tools import timesince_now
from .models import (
    Album,
    AnimalObservation,
    ArchiveCollection,
    IndividualIdentity,
    Locality,
    MediaFile,
    MediafilesForIdentification,
    Taxon,
    UploadedArchive,
    WorkGroup,
    get_all_relevant_localities,
    user_has_access_filter_params,
    Notification
)
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
    return render(
        request,
        "caidapp/home.html",
    )


def login(request):
    """Login page."""
    if request.user.is_authenticated:
        return redirect("caidapp:home")
    else:
        return render(
            request,
            "caidapp/login.html",
        )


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
        form = forms.CaIDUserSettingsForm(request.POST, instance=request.user.caiduser)
        if form.is_valid():
            form.save()
            messages.success(request, "Settings updated successfully.")
            url = request.META.get("HTTP_REFERER", "/")
            return redirect(url)
        else:
            messages.error(request, "Please correct the errors below.")
        return render(request, self.template_name, {"form": form})


def get_filtered_mediafiles(
    user,
    contains_single_taxon: Optional[bool] = None,
    taxon_for_identification__isnull: Optional[bool] = None,
    contains_identities: Optional[bool] = None,
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
        taxon_for_identification__isnull=False,
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
    queryset = get_filtered_mediafiles(
        request.user,
        # contains_single_taxon=True,
        contains_identities=False,
        taxon_for_identification__isnull=False,
    )
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

    return render(
        request,
        "caidapp/dash_identities.html",
        dict(
            # **page_context,
            btn_styles=_single_species_button_style(request),
            identities_by_representative_mediafiles=identities,
        ),
    )


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


@login_required
def select_reid_model(request):
    """Select reid model."""
    form = forms.UserIdentificationModelForm()
    if request.method == "POST":
        form = forms.UserIdentificationModelForm(request.POST)
        if form.is_valid():
            request.user.caiduser.identification_model = form.cleaned_data["identification_model"]
            request.user.caiduser.save()

            messages.info(request, "Identification model set.")
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
    models.user_has_access_filter_params(request.user.caiduser, "owner")
    n_non_classified_taxons = len(models.get_mediafiles_with_missing_taxon(request.user.caiduser))
    n_missing_verifications = len(models.get_mediafiles_with_missing_verification(request.user.caiduser))

    some_missing_taxons = n_non_classified_taxons > 0
    some_missing_verifications = n_missing_verifications > 0

    btn_tooltips = {
        "annotate_missing_taxa": f"Annotate {n_non_classified_taxons} media files " + "with missing taxon.",
        "verify_taxa": f"Go to verification of {n_missing_verifications} media files.",
    }
    btn_styles = {
        "upload_species": "secondary",
        "annotate_missing_taxa": "secondary",
        "verify_taxa": "secondary",
    }
    if not some_missing_taxons and not some_missing_verifications:
        btn_styles["upload_species"] = "primary"
    elif some_missing_taxons:
        btn_styles["annotate_missing_taxa"] = "primary"
    elif some_missing_verifications:
        btn_styles["verify_taxa"] = "primary"

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
    if not request.user.caiduser.identification_model:
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
    messages.info(request, f"Scheduling identification initialization for workgroup {caiduser.workgroup.name} with {mf_count} media files.")

    if not request.user.caiduser.identification_model:

        caiduser.workgroup.identification_model = models.IdentificationModel.objects.filter(public=True).first()
        messages.warning(request, f"Setting default identification model: {caiduser.workgroup.identification_model.name}")

    from .tasks import schedule_init_identification_for_workgroup



    schedule_init_identification_for_workgroup(caiduser.workgroup, delay_minutes=0)
    # return redirect("caidapp:individual_identities")
    return redirect("caidapp:dash_identities")


def stop_init_identification(request):
    """Stop identification initialization."""
    workgroup = request.user.caiduser.workgroup
    if workgroup.identification_init_status == "Processing":
        workgroup.identification_init_status = "Not initiated"
        workgroup.save()
    elif workgroup.identification_reid_status == "Processing":
        workgroup.identification_reid_status = "Not initiated"
        workgroup.save()
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
    """Run identification in all uploaded archives."""
    workgroup = request.user.caiduser.workgroup

    tasks.run_identification_on_unidentified_for_workgroup(workgroup.id)
    return redirect(request.META.get("HTTP_REFERER", "/"))


@login_required
def run_identification_view(request, uploadedarchive_id):
    """Run identification of uploaded archive."""
    uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    # check if user is owner member of the workgroup
    if uploaded_archive.owner.workgroup != request.user.caiduser.workgroup:
        return HttpResponseNotAllowed("Identification is for workgroup members only.")
    status_ok = run_identification(uploaded_archive, workgroup=request.user.caiduser.workgroup)
    if status_ok:
        messages.info(request, f"Identification started for {uploaded_archive.name}.")
    else:
        messages.error(request, "No records for identification with the expected taxon.")
    return redirect(request.META.get("HTTP_REFERER", "/"))


def run_identification(uploaded_archive: UploadedArchive, workgroup: models.WorkGroup) -> bool:
    """Run identification of uploaded archive."""
    logger.debug("Generating CSV for run_identification...")
    # if uploaded_archive.taxon_for_identification:
    #     taxon_str = uploaded_archive.taxon_for_identification.name
    # else:
    #     taxon_str = "Lynx lynx"

    # find media files with observations of the expected taxon
    kwargs = {}
    if workgroup.default_taxon_for_identification and workgroup.check_taxon_before_identification:
        kwargs.update(dict(taxon=workgroup.default_taxon_for_identification))

    mf_ids = (
        AnimalObservation.objects.filter(
            mediafile__parent=uploaded_archive,
            **kwargs,
        )
        .values_list("mediafile_id", flat=True)
        .distinct()
    )

    mediafiles = uploaded_archive.mediafile_set.filter(
        id__in=mf_ids,
        # taxon__name=taxon_str
    ).all()
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
            workgroups=[workgroup]
        )


        return False
        # return redirect(request.META.get("HTTP_REFERER", "/"))

    from celery import current_app

    tasks = current_app.tasks.keys()
    logger.debug(f"tasks={tasks}")

    logger.debug("Calling run_detection and run_identification ...")

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


def _one_zip_from_request_FILES(request:HttpRequest) -> HttpRequest:
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
        zipped_file = ContentFile(
            buffer.read(), name=f"uploaded_multiple_files.{now_str}.zip"
        )

        # substitute the original request.FILES with the zip file
        request.FILES.setlist("archivefile", [zipped_file])

    return request


@login_required
def upload_archive(
    request,
    contains_single_taxon=False,
    contains_identities=False,
):
    """Process the uploaded zip file."""
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
                default_taxon = models.get_taxon("Lynx lynx")
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
    taxon_stats = ""
    if len(mediafiles) > 0:
        taxon_stats = mediafiles.values("taxon__name").annotate(count=Count("taxon__name")).order_by("-count")
        logger.debug(f"{taxon_stats=}")
        df = pd.DataFrame.from_records(taxon_stats)
        df.rename(columns={"taxon__name": "Taxon", "count": "Count"}, inplace=True)
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
    if ffk.get("filter_orientation", None):
        orientation = ffk.get("filter_orientation")
        if orientation != "All":
            filter_kwargs["orientation"] = orientation

    if ffk.get("filter_hide_empty", None):
        # MediaFile.category.name is not "Empty"
        exclude_filter_kwargs.update(dict(taxon__name="Nothing"))

    return filter_kwargs, exclude_filter_kwargs


@login_required
def media_files_update(
    request,
    records_per_page: Optional[int] = None,
    album_hash=None,
    individual_identity_id=None,
    taxon_id=None,
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
    album = None

    # page_number = 1
    # exclude_filter_kwargs = {}
    # form_filter_kwargs = {}
    # query = None
    if records_per_page is None:
        records_per_page = request.session.get("mediafiles_records_per_page", 20)

    albums_available = (
        Album.objects.filter(Q(albumsharerole__user=request.user.caiduser) | Q(owner=request.user.caiduser))
        .distinct()
        .order_by("created_at")
    )

    filter_kwargs = {}

    mediafiles_name_suggestion = None
    if show_overview_button:
        # mediafiles = mediafiles.filter(taxon_verified=False)
        filter_kwargs["taxon_verified"] = False
        mediafiles_name_suggestion = "taxon_not_verified"

        # page_title = "Media files - verification"

    if uploadedarchive_id is not None:
        uploaded_archive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
        # datetime format YYYY-MM-DD HH:MM:SS
        if uploaded_archive.locality_check_at is not None:
            locality_check_at = " - " + uploaded_archive.locality_check_at.strftime("%Y-%m-%d %H:%M:%S")
        else:
            locality_check_at = ""
        page_title = f"Media files - {uploaded_archive.locality_at_upload}{locality_check_at}"
        # mediafiles = mediafiles.filter(parent=uploaded_archive)
        filter_kwargs["parent"] = uploaded_archive
        mediafiles_name_suggestion = f"uploaded_archive_{uploaded_archive.locality_at_upload}{locality_check_at}"

    elif album_hash is not None:
        album = get_object_or_404(Album, hash=album_hash)
        page_title = f"Media files - {album.name}"
        # mediafiles = mediafiles.filter(album=album)
        filter_kwargs = {"album": album}
        mediafiles_name_suggestion = f"album_{album.name}"
    elif individual_identity_id is not None:
        individual_identity = get_object_or_404(IndividualIdentity, pk=individual_identity_id)
        page_title = f"Media files - {individual_identity.name}"
        # mediafiles = mediafiles.filter(identity=individual_identity)
        filter_kwargs = {"identity": individual_identity}
        mediafiles_name_suggestion = f"individual_identity_{individual_identity.name}"
    elif taxon_id is not None:
        taxon = get_object_or_404(Taxon, pk=taxon_id)
        page_title = f"Media files - {taxon.name}"
        # mediafiles = mediafiles.filter(taxon=taxon)
        filter_kwargs = {"taxon": taxon}
        mediafiles_name_suggestion = f"taxon_{taxon.name}"
    elif locality_hash is not None:
        locality = get_object_or_404(Locality, hash=locality_hash)
        page_title = f"Media files - {locality.name}"
        # mediafiles = mediafiles.filter(locality=locality)
        filter_kwargs = {"locality": locality}
        mediafiles_name_suggestion = f"locality_{locality.name}"
    elif identity_is_representative is not None:
        page_title = "Media files - representative"
        # mediafiles = mediafiles.filter(identity_is_representative=identity_is_representative)
        filter_kwargs = {"identity_is_representative": identity_is_representative}
        mediafiles_name_suggestion = f"representative_identity_{str(identity_is_representative)}"
    else:
        page_title = "Media files"

    # logger.debug(f"{len(mediafiles)=}")
    # if request.user.caiduser.workgroup:
    #     mediafiles = mediafiles.filter(Q(parent__owner__workgroup=request.user.caiduser.workgroup))

    # Order the queryset according to your session or default preference
    order_by = request.session.get("mediafiles_order_by", "-parent__uploaded_at")
    logger.debug("Selecting related")
    # Nová filtrace
    # Build the base queryset (including annotations)

    # mediafiles = MediaFile.objects.annotate(**_mediafiles_annotate())
    # Apply always-on filters (for example, access control)
    mediafiles = MediaFile.objects.filter(
        Q(album__albumsharerole__user=request.user.caiduser)
        | Q(**models.user_has_access_filter_params(request.user.caiduser, "parent__owner")),
        **filter_kwargs,
    )
    logger.debug(f"{request.GET=}")

    # Instantiate the filter with GET parameters and your base queryset
    mediafile_filter = filters.MediaFileFilter(request.GET, queryset=mediafiles, request=request)

    # The filtered queryset is available as .qs
    full_mediafiles = mediafile_filter.qs

    if show_overview_button and not full_mediafiles.exists():
        return message_view(
            request,
            "No mediafiles for verification.",
            headline="Verification",
            link=reverse_lazy("caidapp:uploads"),
        )

    # konec nové filtrace
    full_mediafiles = full_mediafiles.order_by(order_by).select_related(
        "parent", "taxon", "predicted_taxon", "locality", "identity", "updated_by", "sequence"
    )

    number_of_mediafiles = mediafile_filter.qs.count()
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
        observation = instance.first_observation_get_or_create
        observation.taxon_verified = True
        instance.taxon_verified = observation.taxon_verified
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
    mediafile_ids = request.session.get("mediafile_ids", [])
    mediafiles = MediaFile.objects.filter(id__in=mediafile_ids)

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
        name_suggestion = request.session.get("mediafiles_name_suggestion", None)

    return mediafiles, name_suggestion


@login_required
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


@login_required
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
    mediafiles, name_suggestion = _get_mediafiles(request, uploadedarchive_id)
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
    mediafiles, name_suggestion = _get_mediafiles(request, uploadedarchive_id)
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

    # Prepare the mediafiles list for serialization (e.g., paths and output names)
    mediafiles_data = [{"path": mf.mediafile.name, "output_name": _make_output_name(mf)} for mf in mediafiles]

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


def _make_output_name(mediafile: models.MediaFile):
    """Create output name.

    pattern: {locality}_{date}_{original_name}_{taxon}_{identity}
    """
    locality = mediafile.locality.name if mediafile.locality else "no_locality"
    date = mediafile.captured_at.strftime("%Y-%m-%d") if mediafile.captured_at else "no_date"
    original_name = mediafile.original_filename if mediafile.original_filename else "no_original_name"
    # remove extension
    original_name = Path(original_name).stem
    taxon = mediafile.taxon.name if mediafile.taxon else "no_taxon"
    identity = mediafile.identity.name if mediafile.identity else "no_identity"
    suffix = Path(mediafile.mediafile.name).suffix
    output_name = f"{locality}_{date}_{original_name}_{taxon}_{identity}.{suffix}"
    return output_name


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
            uploaded_archive.taxon_for_identification = models.get_taxon("Lynx lynx")
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
        fig.update_xaxes(type="taxon", title_text="Upload Date")
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
                        # logger.debug(f"{mf=}")
                        counter0 += 1
                        # mf.category = row['category']
                        if "predicted_category" in row:
                            mf.taxon = models.get_taxon(row["predicted_category"])  # remove this
                            counter_fields_updated += 1

                        code = row["code"] if "code" in row else ""
                        unique_name = row["unique_name"] if "unique_name" in row else ""
                        if code:
                            identity = models.get_unique_code(code, workgroup=uploaded_archive.owner.workgroup)
                            if identity is None:
                                logger.warning("Could not find identity with code: " + code)
                            counter_fields_updated += 1
                            counter_individuality += 1
                            if unique_name:
                                mf.identity.name = unique_name.strip()
                                counter_fields_updated += 1
                                mf.identity.save()
                        elif unique_name:
                            mf.identity = models.get_unique_name(
                                row["unique_name"], workgroup=uploaded_archive.owner.workgroup
                            )
                            counter_fields_updated += 1
                            counter_individuality += 1

                        if "locality_name" in row:
                            locality_obj = models.get_locality(
                                caiduser=request.user.caiduser, name=row["locality_name"]
                            )
                            if locality_obj:
                                mf.locality = locality_obj
                                if ("latitude" in row) and ("longitude" in row):
                                    mf.locality.set_location(float(row["latitude"]), float(row["longitude"]))
                                    counter_fields_updated += 1
                                counter_fields_updated += 1
                                counter_locality += 1
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
                                mf.identity.coat_type = coat_type

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
                                mf.orientation = orientation
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
                    + "The 'predicted_category', 'unique_name', 'locality name', 'latitude', "
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
        return render(
            request,
            "caidapp/update_form.html",
            {
                "form": form,
                "headline": "Upload XLSX or CSV",
                "button": "Save",
                "next": prev_url,
                "text_note": "The 'original_path' is required in the uploaded spreadsheet. "
                + "The 'predicted_category', 'unique_name', 'locality name', 'latitude', "
                + "'longitude', 'datetime' are optional.",
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
    job = tasks.refresh_identities_suggestions_task.delay(request.user.caiduser.workgroup.id)
    logger.debug(
        f"{job.id=}, {request.user.id=}, {request.user=}, {request.user.caiduser=}, {request.user.caiduser.workgroup=}"
    )
    request.session["refresh_job_id"] = job.id
    request.session["refresh_job_started_at"] = timezone.now().isoformat()


def get_identity_suggestions(request):
    """Get identity suggestions status and data."""
    job_id = request.session.get("refresh_job_id")

    sugg_obj = (
        models.MergeIdentitySuggestionResult.objects.filter(workgroup=request.user.caiduser.workgroup)
        .order_by("id")
        .last()
    )
    suggestions = sugg_obj.suggestions if sugg_obj else None
    created_at = sugg_obj.created_at if sugg_obj else None

    if not job_id:
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

    return render(request, "caidapp/suggest_identity_codes.html", {"identities": list(all_identities)})


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
        ["name", "code", "juv_code", "sex", "coat_type", "birth_date", "death_date", "note"]
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
        ["name", "code", "juv_code", "sex", "coat_type", "birth_date", "death_date", "note"]
    ]

    return views_general.excel_response(df, "identities")


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
                    if "code" in row and len(row["code"]) > 0:
                        identity, created_new = IndividualIdentity.objects.get_or_create(
                            code=row["code"], owner_workgroup=request.user.caiduser.workgroup
                        )
                    elif "name" in row and len(row["name"]) > 0:
                        identity, created_new = IndividualIdentity.objects.get_or_create(
                            name=row["name"], owner_workgroup=request.user.caiduser.workgroup
                        )
                    else:
                        logger.warning(f"No identification (name or code) found for {row} ")
                        continue
                except models.IndividualIdentity.MultipleObjectsReturned:
                    logger.debug(f"{row=}")
                    logger.warning(traceback.format_exc())
                    messages.warning(request, f"Skipping row. Multiple identities found for {row}")
                    continue

                logger.debug(f"{identity=}")

                if "name" in row and len(row["name"]) > 0:
                    print(f"{row}")
                    print(f"{row['name']}")
                    identity.name = row["name"]
                if "code" in row and len(row["code"]) > 0:
                    identity.code = row["code"]
                if "sex" in row and len(row["sex"]) > 0:
                    sex = row["sex"][0].upper()
                    if sex in ["M", "F", "U"]:
                        identity.sex = sex
                    else:
                        logger.warning(f"Invalid sex: {row['sex']}")
                if "coat_type" in row and len(row["coat_type"]) > 0:
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

                if "juv_code" in row and len(row["juv_code"]) > 0:
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
            + "There should be columns 'name' or 'code' in the file. "
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
    fields = ["message", "read", "level"]
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
    title = "Notifications"

    def get_queryset(self):
        """Limit queryset to notifications of the current user."""
        # return models.Notification.objects.filter(user=self.request.user.caiduser).order_by("-created_at")
        return (
            models.NotificationRecipient.objects
            .filter(user=self.request.user.caiduser)
            .select_related("notification")
            .order_by("-notification__created_at")
        )

    def get_context_data(self, **kwargs):
        """Set up context data for the list view."""
        context = super().get_context_data(**kwargs)
        context["title"] = _("Issue")
        context["list_display"] = ["message", "user", "read", "created_at"]
        context["object_detail_url"] = "caidapp:notification-detail"
        context["object_update_url"] = "caidapp:notification-update"
        context["object_delete_url"] = "caidapp:notification-delete"
        context["object_create_url"] = "caidapp:notification-create"
        return context


class NotificationDetailView(DetailView):
    model = models.Notification
    template_name = "caidapp/generic_detail.html"
    context_object_name = "notification"
    title = "Notification Detail"
    paginate_by = 20
    fields = ["message", "read", "level", "created_at"]
    cancel_url = reverse_lazy("caidapp:notifications")

    def get_queryset(self):
        """Limit queryset to notifications of the current user."""
        qs = super().get_queryset()
        # např. jen zprávy pro aktuálního uživatele
        return qs.filter(user=self.request.user.caiduser)

    def get(self, request, *args, **kwargs):
        """Handle GET request and mark notification as read."""
        response = super().get(request, *args, **kwargs)
        # Mark as read when viewed
        if not self.object.read:
            self.object.read = True
            self.object.save(update_fields=["read"])
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
    fields = ["message", "read", "level"]
    template_name = "caidapp/generic_form.html"
    success_url = reverse_lazy("caidapp:notifications")
    title = "Update Notification"


class NotificationDeleteView(DeleteView):
    model = models.Notification
    template_name = "caidapp/generic_form.html"
    success_url = reverse_lazy("caidapp:notifications")
    title = "Delete Notification"
