import logging
import os
from typing import Optional

import django
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db import transaction
from django.db.models.query import QuerySet
from django.http import Http404, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, resolve_url
from django.urls import reverse, reverse_lazy
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.generic import DeleteView
from extra_views import InlineFormSetFactory, UpdateWithInlinesView

from . import forms, model_extra, models
from .forms import MediaFileForm
from .models import AnimalObservation, MediaFile
from .views import media_files_update, sequences

logger = logging.getLogger(__name__)


@login_required
def stream_video(request, mediafile_id):
    """Stream video file."""
    mediafile = get_object_or_404(MediaFile, id=mediafile_id)
    if mediafile.media_type != "video":
        raise Http404("Not a video file")

    if (mediafile.preview is not None) and os.path.exists(mediafile.preview.path):
        video_path = mediafile.preview.path
    else:
        video_path = mediafile.mediafile.path
        logger.warning(f"Preview does not exist for mediafile {mediafile.id=}")
    if not os.path.exists(video_path):
        raise Http404()

    def file_iterator(file_name, chunk_size=8192):
        with open(file_name, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    # response = StreamingHttpResponse(file_iterator(video_path), content_type='/video/mp4')
    response = StreamingHttpResponse(file_iterator(video_path), content_type="video/x-m4v")
    response["Content-Length"] = os.path.getsize(video_path)
    response["Accept-Ranges"] = "bytes"

    return response


def _set_taxon_for_sequence(mediafile: MediaFile, taxon, caiduser, commit=True):
    """Set given taxon for all AnimalObservation in the same sequence as mediafile.

    Uses bulk update for performance. Returns number of observations updated.
    """
    logger.debug(f"{taxon=}")
    now = timezone.now()
    # If mediafile has sequence, get all mediafiles in it; otherwise only this mediafile
    if mediafile.sequence is not None:
        obs_qs = AnimalObservation.objects.filter(mediafile__sequence=mediafile.sequence)
    else:
        obs_qs = AnimalObservation.objects.filter(mediafile=mediafile)

    logger.debug(f"{len(obs_qs)=}")

    # Restrict to observations user can access? We assume view permission checked earlier.
    if taxon is not None:
        taxon_id = taxon.id
    else:
        taxon_id = None

    with transaction.atomic():
        # bulk update
        updated_count = obs_qs.update(taxon_id=taxon_id, updated_by=caiduser, updated_at=now)

    return updated_count


def _replace_observations_with_nothing(mediafile: MediaFile, caiduser):
    """Replace all observations for a mediafile with a single Nothing observation."""
    now = timezone.now()
    nothing_taxon = models.get_taxon("Nothing")

    with transaction.atomic():
        deleted_count, _ = AnimalObservation.objects.filter(mediafile=mediafile).delete()
        AnimalObservation.objects.create(
            mediafile=mediafile,
            taxon=nothing_taxon,
            taxon_verified=True,
            taxon_verified_at=now,
            updated_by=caiduser,
            updated_at=now,
        )

    return deleted_count


class ObservationInline(InlineFormSetFactory):
    model = AnimalObservation
    form_class = forms.AnimalObservationForm
    # fields = forms.AnimalObservationForm.Meta.fields
    fields = [
        "taxon",
        "identity",
        "identity_is_representative",
        "orientation",
        "taxon_verified",
        "bbox_x_center",
        "bbox_y_center",
        "bbox_width",
        "bbox_height",
        # "orientation"
    ]
    can_delete = True
    extra = 0
    fk_name = "mediafile"
    # widgets = {
    #     "bbox_x_center": HiddenInput(),
    #     "bbox_y_center": HiddenInput(),
    #     "bbox_width": HiddenInput(),
    #     "bbox_height": HiddenInput(),
    # }

    def get_factory_kwargs(self):
        """Pass extra kwargs to factory."""
        kwargs = super().get_factory_kwargs()
        kwargs["extra"] = self.extra
        kwargs["can_delete"] = self.can_delete

        return kwargs


class MediaFileUpdateView(LoginRequiredMixin, UpdateWithInlinesView):
    model = MediaFile
    form_class = MediaFileForm
    inlines = [ObservationInline]
    template_name = "caidapp/media_file_update.html"
    context_object_name = "mediafile"

    def get_queryset(self):
        """Restrict to mediafiles that user can access."""
        # user or his workgroup can access to mediafiles
        return MediaFile.objects.for_user(self.request.user.caiduser)

    def _get_next_url(self):
        """Get next URL from GET or POST parameters, or fallback to referer or media files list."""
        next_url = self.request.POST.get("next") or self.request.GET.get("next")

        if next_url and url_has_allowed_host_and_scheme(
                next_url,
                allowed_hosts={self.request.get_host()},
        ):
            return next_url

        # return self.request.GET.get("next") or self.request.META.get("HTTP_REFERER", "/")
        return resolve_url("caidapp:media_files")

    def get_success_url(self):
        """After successful update, return to previous page."""
        if self.request.POST.get("mark_empty_image"):
            return reverse("caidapp:media_file_update", args=[self.object.id])
        return self._get_next_url()

    def _get_taxon_from_inline_observations(self):
        """Get taxon from inline observations, prefer first non-null."""
        total_forms = int(self.request.POST.get("observations-TOTAL_FORMS", 0) or 0)
        for index in range(total_forms):
            taxon_id_str = self.request.POST.get(f"observations-{index}-taxon")
            logger.debug("observations-%s-taxon=%s", index, taxon_id_str)
            if not taxon_id_str:
                continue
            try:
                taxon_id = int(taxon_id_str)
                return models.Taxon.objects.get(id=taxon_id)
            except (ValueError, models.Taxon.DoesNotExist):
                continue
        return None

    def form_valid(self, form):
        """Set updated_by and updated_at on save."""
        form.instance.updated_by = self.request.user.caiduser
        form.instance.updated_at = django.utils.timezone.now()
        response = super().form_valid(form)
        # Save all valid inline formsets
        # inlines = self.get_inlines()

        # If the user clicked "save and set taxon for sequence", set the taxon on all observations
        if self.request.POST.get("save_set_taxon_sequence"):
            logger.debug("User clicked save and set taxon for sequence")
            try:

                # Prefer taxon set on an observation form (first non-null), fallback to mediafile.taxon
                # obs_qs = form.instance.observations.all()
                # first observation that has taxon set
                # obs = obs_qs.filter(taxon__isnull=False).first()
                # vezmi první observation s vyplněným taxonem

                taxon = self._get_taxon_from_inline_observations()
                # observations jsou v inlines
                # získej taxon z první observace

                logger.debug(f"{taxon=}")
                updated = _set_taxon_for_sequence(form.instance, taxon, self.request.user.caiduser)
                if taxon is not None:
                    messages.success(
                        self.request, f"Updated taxon on {updated} observations in sequence (using observation taxon)."
                    )
                else:
                    messages.success(self.request, f"Cleared taxon on {updated} observations in sequence.")
            except Exception as e:
                logger.exception(f"Failed to set taxon for sequence, {e}")
                messages.error(self.request, "Failed to set taxon for sequence.")

        if self.request.POST.get("mark_empty_image"):
            try:
                deleted_count = _replace_observations_with_nothing(self.object, self.request.user.caiduser)
                if deleted_count:
                    messages.warning(
                        self.request,
                        f"Replaced {deleted_count} existing observations with a single empty-image observation.",
                    )
                else:
                    messages.success(self.request, "Marked this media file as an empty image.")
            except Exception:
                logger.exception("Failed to mark mediafile as empty image")
                messages.error(self.request, "Failed to mark this media file as an empty image.")

        return response

    def get_context_data(self, **kwargs):
        """Add next URL to context."""
        context = super().get_context_data(**kwargs)
        next_url = self.request.POST.get("next") or self.request.GET.get("next")
        next_url = self.request.GET.get("next", "")
        context["next"] = next_url
        context["effective_location"] = self.object.effective_location
        context["effective_location_source"] = self.object.effective_location_source
        return context


class ObservationDeleteView(LoginRequiredMixin, DeleteView):
    model = AnimalObservation
    template_name = "caidapp/generic_form.html"
    context_object_name = "observation"

    def get_success_url(self):
        """After deletion, return to media file edit."""
        # po smazání se vrátíš na editaci mediafile
        mediafile = self.object.mediafile
        return reverse("caidapp:media_file_update", args=[mediafile.id])


def _manual_identification_next_url(request, current_mediafile=None):
    """Return the next unidentified identification media file URL."""
    mediafiles = models.get_mediafiles_with_missing_identity(request.user.caiduser)
    if current_mediafile is not None:
        mediafiles = mediafiles.exclude(pk=current_mediafile.pk)
    next_mediafile = mediafiles.first()
    if next_mediafile is None:
        return None
    return reverse("caidapp:manual_identification_mediafile", args=[next_mediafile.pk])


class MediaFileManualIdentificationView(MediaFileUpdateView):
    """Assign identities while walking through identification media files."""

    def get_queryset(self):
        return models.get_mediafiles_with_missing_identity(self.request.user.caiduser)

    def get_success_url(self):
        next_url = _manual_identification_next_url(self.request, self.object)
        if next_url is not None:
            return next_url
        messages.success(self.request, "Manual identification is complete.")
        return reverse("caidapp:dash_identities")

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update(
            {
                "headline": "Manual identification",
                "manual_identification_mode": True,
                "button": "Save and continue",
                "skip_url": _manual_identification_next_url(self.request, self.object),
                "cancel_url": reverse("caidapp:dash_identities"),
                "cancel_label": "Identification dashboard",
            }
        )
        return context


@login_required
def start_manual_identification(request):
    """Open the first accessible identification media file without identity."""
    next_url = _manual_identification_next_url(request)
    if next_url is None:
        messages.info(request, "All identification media files already have an identity.")
        return redirect("caidapp:dash_identities")
    return redirect(next_url)


# @login_required
# def missing_taxon_annotation(
#     request,
#     uploaded_archive_id: Optional[int] = None,
#     # prev_mediafile_id: Optional[int] = None
# ):
#     """List of uploads."""
#     # get uploadeda archive or None
#     if uploaded_archive_id is not None:
#         uploadedarchive = get_object_or_404(
#             models.UploadedArchive,
#             id=uploaded_archive_id,
#             # **get_content_owner_filter_params(request.user.caiduser, "owner"),
#         )
#     else:
#         uploadedarchive = None
#
#     # pick random non-classified media file
#     mediafiles = models.get_mediafiles_with_missing_taxon(request.user.caiduser, uploadedarchive=uploadedarchive)
#     mediafile = mediafiles.order_by("-parent__uploaded_at", "-captured_at").first()
#     # missing_count = mediafiles.count()
#     # last_ten_mediafiles = mediafiles.order_by("-parent__uploaded_at", "-captured_at")
#     # #
#     # # Select a random media file from the last 10
#     # if len(list(last_ten_mediafiles)) > 0:
#     #     mediafile = last_ten_mediafiles.first()
#     # else:
#     #     mediafile = None  # Handle the case when there are no media files
#     #
#     if uploadedarchive is not None:
#         # kwargs = {"uploaded_archive_id": uploadedarchive.id}
#         # # if mediafile:
#         # #     kwargs["prev_mediafile_id"] = mediafile.id
#         # next_url = reverse_lazy( "caidapp:missing_taxon_annotation", kwargs=kwargs )
#         # if missing_count > 1:
#         #     skip_url = reverse_lazy( "caidapp:missing_taxon_annotation", kwargs=kwargs )
#         # else:
#         #     skip_url = None
#         cancel_url = reverse_lazy(
#             "caidapp:uploadedarchive_mediafiles", kwargs={"uploadedarchive_id": uploadedarchive.id}
#         )
#     else:
#         # kwargs = {"prev_mediafile_id": prev_mediafile_id}
#         # next_url = reverse_lazy( "caidapp:missing_taxon_annotation", kwargs={})
#         # skip_url = next_url
#         cancel_url = reverse_lazy("caidapp:taxon_processing")
#     #
#     #     # Everything done
#     if mediafile is None:
#         if uploadedarchive is not None:
#             message = f"All taxa known for {uploadedarchive.name}"
#         else:
#             message = "All taxa known"
#         return message_view(request, message, link=cancel_url, headline="No missing taxa")
#
#     logger.debug(f'Redirecting to mediafile {mediafile.id} for taxon annotation')
#     logger.debug(f'{uploaded_archive_id=}')
#     if uploaded_archive_id:
#         return redirect(
#             "caidapp:missing_taxon_annotation_for_mediafile",
#             mediafile_id=mediafile.id,
#             uploaded_archive_id=uploaded_archive_id,
#         )
#     else:
#         return redirect("caidapp:missing_taxon_annotation_for_mediafile", mediafile_id=mediafile.id)


def get_next_in_queryset(queryset, instance):
    """Get next instance in queryset after given instance."""
    ids = list(queryset.values_list("id", flat=True))
    logger.debug(f"Getting next in queryset for instance {instance.id}")
    logger.debug(f"IDs in queryset: {ids}")
    try:
        idx = ids.index(instance.id)
    except ValueError:
        return None
    if idx + 1 < len(ids):
        return queryset.model.objects.get(id=ids[idx + 1])
    return None


def _mta_get_next_url(
    request,
    current_mediafile: Optional[models.MediaFile],
    uploadedarchive: Optional[models.UploadedArchive],
) -> Optional[str]:
    """Get next URL for missing taxon annotation."""
    caiduser = request.user.caiduser
    next_mediafile = get_next_missing_taxon_mediafile(
        caiduser,
        uploadedarchive=uploadedarchive,
        current=current_mediafile,
    )

    if current_mediafile and (next_mediafile is None):
        return None
    if next_mediafile:
        url_name = "caidapp:missing_taxon_annotation_for_mediafile"
        kwargs = {"pk": next_mediafile.id}
    else:
        url_name = "caidapp:missing_taxon_annotation"
        kwargs = {}

    if uploadedarchive:
        url = reverse_lazy(url_name, kwargs=kwargs) + f"?uploadedarchive_id={uploadedarchive.id}"
    else:
        url = reverse_lazy(url_name, kwargs=kwargs)

    return url


# class FinishMissingTaxonAnnotationView(LoginRequiredMixin, django.views.TemplateView):
#     """View to finish missing taxon annotation process."""
#     template_name = "caidapp/message.html"
#
#     def get_context_data(self, **kwargs):
#         context = super().get_context_data(**kwargs)
#
#         uploadedarchive_id = self.request.GET.get("uploadedarchive_id")
#         uploadedarchive = None
#
#         if uploadedarchive_id:
#             uploadedarchive = get_object_or_404(
#                 models.UploadedArchive, id=uploadedarchive_id
#             )
#
#         if uploadedarchive:
#             context.update({
#                 "headline": "All taxa known",
#                 "message": f"No missing taxa for {uploadedarchive.name}",
#                 "link": reverse("caidapp:uploads"),
#                 "button_label": "Ok",
#             })
#         else:
#             context.update({
#                 "headline": "All taxa known",
#                 "message": "No missing taxa",
#                 "link": reverse("caidapp:taxon_processing"),
#                 "button_label": "Ok",
#             })
#
#         return context
#
#
# class FinishMissingTaxonAnnotationView(LoginRequiredMixin, django.views.View):
#     template_name = "caidapp/finish_taxon_annotation.html"
#
#
#     def get(self, request, *args, **kwargs):
#         # uploadedarchive_id from GET
#         uploadedarchive_id = request.GET.get("uploadedarchive_id")
#
#         if uploadedarchive_id:
#             uploadedarchive = get_object_or_404(models.UploadedArchive, id=uploadedarchive_id)
#         else:
#             uploadedarchive = None
#
#         if uploadedarchive:
#             return message_view(
#                 request,
#                 f"No missing taxa for {uploadedarchive.name}",
#                 link=reverse_lazy("caidapp:uploads"),
#                 headline="All taxa known",
#             )
#         else:
#             return message_view(
#                 request,
#                 "No missing taxa",
#                 link=reverse_lazy("caidapp:taxon_processing"),
#                 headline="All taxa known",
#             )
#


class MediaFileGetMissingTaxonView(LoginRequiredMixin, UpdateWithInlinesView):
    model = MediaFile
    form_class = MediaFileForm
    inlines = [ObservationInline]
    template_name = "caidapp/media_file_update.html"
    context_object_name = "mediafile"

    def get_uploadedarchive(self) -> Optional[models.UploadedArchive]:
        """Get uploaded archive from GET parameters, or None."""
        ua_id = self.request.GET.get("uploadedarchive_id")
        logger.debug(f"{ua_id=}")
        if not ua_id:
            return None
        return get_object_or_404(models.UploadedArchive, id=ua_id)

    def dispatch(self, request, *args, **kwargs):
        """Check that mediafile has missing taxon and user has access before dispatching."""
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        """Add next and cancel URLs to context."""
        context = super().get_context_data(**kwargs)
        uploadedarchive = self.get_uploadedarchive()
        context["uploadedarchive"] = uploadedarchive
        context["missing_taxon_annotation_mode"] = True

        messages.debug(self.request, f"In get_context_data of MediaFileGetMissingTaxonView {uploadedarchive=}")

        skip_url = _mta_get_next_url(
            self.request,
            current_mediafile=self.object,
            uploadedarchive=uploadedarchive,
        )

        if uploadedarchive:
            cancel_url = reverse_lazy("caidapp:uploads")
        else:
            cancel_url = reverse_lazy("caidapp:taxon_processing")
        logger.debug(f"{cancel_url=}")
        logger.debug(f"{skip_url=}")

        context["button"] = "Save and continue"
        context["skip_url"] = skip_url
        context["cancel_url"] = (cancel_url,)
        # uploadedarchive["next_url"] = next_url,
        return context

    def get_success_url(self):
        """After successful update, return to previous page."""
        uploadedarchive = self.get_uploadedarchive()
        if self.request.POST.get("mark_empty_image"):
            url = reverse("caidapp:missing_taxon_annotation_for_mediafile", args=[self.object.id])
            if uploadedarchive:
                url += f"?uploadedarchive_id={uploadedarchive.id}"
            return url
        next_url = _mta_get_next_url(
            self.request,
            current_mediafile=self.object,
            uploadedarchive=uploadedarchive,
        )

        if next_url is None:
            if uploadedarchive:
                messages.info(self.request, f"All taxa known for {uploadedarchive.name}")
                next_url = reverse_lazy("caidapp:uploads")

                # return message_view(
                #     request,
                #     f"No missing taxa for {uploadedarchive.name}",
                #     link=reverse_lazy("caidapp:uploads"),
                #     headline="All taxa known",
                # )
            else:
                messages.info(self.request, "All taxa known")
                next_url = reverse_lazy("caidapp:taxon_processing")
                # return message_view(
                #     request,
                #     "No missing taxa",
                #     link=reverse_lazy("caidapp:taxon_processing"),
                #     headline="All taxa known",
                # )

        return next_url
        # return self.request.GET.get("next") or self.request.META.get("HTTP_REFERER", "/")

    def form_valid(self, form):
        """Set updated_by and updated_at on save."""
        logger.debug("In form_valid of MediaFileUpdateView")
        form.instance.updated_by = self.request.user.caiduser
        form.instance.updated_at = django.utils.timezone.now()
        response = super().form_valid(form)
        # Save all valid inline formsets
        inlines = self.get_inlines()
        logger.debug(f"{len(inlines)=}")

        # Handle "save and set taxon for sequence" here as well when using missing-taxons flow
        if self.request.POST.get("save_set_taxon_sequence"):
            try:
                # Prefer taxon set on an observation form (first non-null), fallback to mediafile.taxon
                taxon = None
                try:
                    obs_qs = form.instance.observations.all()
                    obs_with_taxon = obs_qs.filter(taxon__isnull=False).first()
                    if obs_with_taxon:
                        taxon = obs_with_taxon.taxon
                    else:
                        taxon = form.instance.taxon
                except Exception:
                    taxon = form.instance.taxon

                updated = _set_taxon_for_sequence(form.instance, taxon, self.request.user.caiduser)
                if taxon is not None:
                    messages.success(
                        self.request, f"Updated taxon on {updated} observations in sequence (using observation taxon)."
                    )
                else:
                    messages.success(self.request, f"Cleared taxon on {updated} observations in sequence.")
            except Exception as e:
                logger.exception(f"Failed to set taxon for sequence {e}")
                messages.error(self.request, "Failed to set taxon for sequence.")

        if self.request.POST.get("mark_empty_image"):
            try:
                deleted_count = _replace_observations_with_nothing(self.object, self.request.user.caiduser)
                if deleted_count:
                    messages.warning(
                        self.request,
                        f"Replaced {deleted_count} existing observations with a single empty-image observation.",
                    )
                else:
                    messages.success(self.request, "Marked this media file as an empty image.")
            except Exception:
                logger.exception("Failed to mark mediafile as empty image")
                messages.error(self.request, "Failed to mark this media file as an empty image.")

        return response


def resolve_missing_taxon_context(request):
    """Resolve missing taxon annotation context from request parameters."""
    uploaded_archive_id = request.POST.get("uploaded_archive_id") or request.GET.get("uploaded_archive_id")

    uploadedarchive = None
    if uploaded_archive_id:
        uploadedarchive = get_object_or_404(
            models.UploadedArchive,
            id=uploaded_archive_id,
        )

    return uploadedarchive


def get_next_missing_taxon_mediafile(
    caiduser: models.CaIDUser,
    uploadedarchive: Optional[models.UploadedArchive] = None,
    current: Optional[models.MediaFile] = None,
) -> Optional[models.MediaFile]:
    """Get next media file with missing taxon for user after current mediafile."""
    qs: "QuerySet[models.MediaFile]" = models.get_mediafiles_with_missing_taxon(
        caiduser,
        uploadedarchive=uploadedarchive,
    ).order_by("-parent__uploaded_at", "-captured_at")

    if current:
        return get_next_in_queryset(qs, current)

    return qs.first()


@login_required
def start_missing_taxon_annotation(
    request,
    uploaded_archive_id: Optional[int] = None,
    # prev_mediafile_id: Optional[int] = None
):
    """Entry point for missing taxon annotation - redirect to first mediafile."""
    # get uploadeda archive or None
    if uploaded_archive_id is not None:
        uploadedarchive = get_object_or_404(
            models.UploadedArchive,
            id=uploaded_archive_id,
            # **get_content_owner_filter_params(request.user.caiduser, "owner"),
        )
    else:
        uploadedarchive = None

    #
    # mediafile = get_next_missing_taxon_mediafile(
    #     request.user.caiduser,
    #     uploadedarchive=uploadedarchive,
    # )
    #
    # if mediafile is None:
    #     return message_view(
    #         request,
    #         "All taxa known",
    #         link=reverse("caidapp:taxon_processing"),
    #     )

    # if no mediafile found, show message and go back
    url = _mta_get_next_url(
        request,
        None,
        # mediafile,
        uploadedarchive=uploadedarchive,
    )

    return redirect(url)


@login_required
def verify_taxa_view(request, uploaded_archive_id: Optional[int] = None):
    """See media files for verification."""
    return sequences(
        request,
        show_overview_button=True,
        taxon_verified=False,
        uploadedarchive_id=uploaded_archive_id,
        parent__contains_single_taxon=False,
    )
    # views.


@login_required
def taxons_on_page_are_verified(request):
    """Mark taxons on page as verified."""
    # get 'mediafiles_ids_page' from session
    mediafile_ids = request.session.get("mediafile_ids_page", [])
    mediafiles = MediaFile.objects.filter(id__in=mediafile_ids)
    for mediafile in mediafiles:
        __verify_taxon_in_observations(mediafile, request.user.caiduser, commit=True)

    # get previous url
    next_url = request.META.get("HTTP_REFERER", "/")

    return redirect(next_url)


@login_required
def set_mediafiles_order_by(request, order_by: str):
    """Set order by for media files."""
    request.session["mediafiles_order_by"] = order_by
    # go back to the same page
    return redirect(request.META.get("HTTP_REFERER", "/"))


@login_required
def set_mediafiles_records_per_page(request, records_per_page: int):
    """Set records per page for media files."""
    request.session["mediafiles_records_per_page"] = records_per_page

    return redirect(request.META.get("HTTP_REFERER", "/"))


def __verify_taxon_in_observations(mediafile: models.MediaFile, caiduser: models.CaIDUser, commit=True):
    # Update the MediaFile instance
    now = timezone.now()
    mediafile.updated_at = now
    # TODO make this in clever way
    for ao in mediafile.observations.all():
        ao.taxon_verified = True
        ao.taxon_verified_at = now
        ao.save()
    mediafile.updated_by = caiduser
    mediafile.taxon_verified = True
    mediafile.taxon_verified_at = now

    # mediafile.taxon_verified = True
    if commit:
        mediafile.save()


@login_required
def confirm_prediction(request, mediafile_id: int) -> JsonResponse:
    """Confirm prediction for media file with low confidence."""
    try:
        mediafile = get_object_or_404(MediaFile, id=mediafile_id)
        # zkontrolovat přístup před změnami
        if not model_extra.user_has_rw_access_to_mediafile(request.user.caiduser, mediafile, accept_none=True):
            return JsonResponse({"success": False, "message": "No read/write access to the file"})

        # nastavíme taxon mediafile i všech jeho observations na predicted_taxon
        # mediafile.taxon = mediafile.predicted_taxon
        for ao in mediafile.observations.all():
            ao.taxon = mediafile.predicted_taxon
            logger.debug(f"Confirming prediction for observation {ao.id} to taxon {ao.taxon}")
            ao.save()
        __verify_taxon_in_observations(mediafile, request.user.caiduser)
        return JsonResponse({"success": True, "message": "Prediction confirmed."})
    except Exception:
        return JsonResponse({"success": False, "message": "Invalid request."})
