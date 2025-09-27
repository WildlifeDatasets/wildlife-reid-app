import os
import random
from typing import Optional

import django
from django.http import Http404, HttpResponseNotAllowed, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse_lazy
from django.utils import timezone
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages

from . import model_extra, models, forms
from .forms import MediaFileForm, MediaFileMissingTaxonForm
from .models import MediaFile
from .views import logger, media_files_update, message_view


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


@login_required
def missing_taxon_annotation(request, uploaded_archive_id: Optional[int] = None,
                             # prev_mediafile_id: Optional[int] = None
                             ):
    """List of uploads."""
    # get uploadeda archive or None
    if uploaded_archive_id is not None:
        uploadedarchive = get_object_or_404(
            models.UploadedArchive,
            id=uploaded_archive_id,
            # **get_content_owner_filter_params(request.user.caiduser, "owner"),
        )
    else:
        uploadedarchive = None

    # pick random non-classified media file
    mediafiles = models.get_mediafiles_with_missing_taxon(
        request.user.caiduser, uploadedarchive=uploadedarchive
    )
    missing_count = mediafiles.count()
    last_ten_mediafiles = mediafiles.order_by("-parent__uploaded_at", "-captured_at")
#
        # Select a random media file from the last 10
    if len(list(last_ten_mediafiles)) > 0:
        mediafile = last_ten_mediafiles.first()
    else:
        mediafile = None  # Handle the case when there are no media files
#
    if uploadedarchive is not None:
        # kwargs = {"uploaded_archive_id": uploadedarchive.id}
        # # if mediafile:
        # #     kwargs["prev_mediafile_id"] = mediafile.id
        # next_url = reverse_lazy( "caidapp:missing_taxon_annotation", kwargs=kwargs )
        # if missing_count > 1:
        #     skip_url = reverse_lazy( "caidapp:missing_taxon_annotation", kwargs=kwargs )
        # else:
        #     skip_url = None
        cancel_url = reverse_lazy(
            "caidapp:uploadedarchive_mediafiles", kwargs={"uploadedarchive_id": uploadedarchive.id}
        )
    else:
        # kwargs = {"prev_mediafile_id": prev_mediafile_id}
        # next_url = reverse_lazy( "caidapp:missing_taxon_annotation", kwargs={})
        # skip_url = next_url
        cancel_url = reverse_lazy("caidapp:taxon_processing")
#
#     # Everything done
    if mediafile is None:
        if uploadedarchive is not None:
            message = f"All taxa known for {uploadedarchive.name}"
        else:
            message = "All taxa known"
        return message_view(request, message, link=cancel_url , headline="No missing taxa")

    if uploaded_archive_id:
        return redirect("caidapp:missing_taxon_annotation_for_mediafile", mediafile_id=mediafile.id, uploaded_archive_id=uploaded_archive_id)
    else:
        return redirect("caidapp:missing_taxon_annotation_for_mediafile", mediafile_id=mediafile.id)

def get_next_in_queryset(queryset, instance):
    ids = list(queryset.values_list("id", flat=True))
    try:
        idx = ids.index(instance.id)
    except ValueError:
        return None
    if idx + 1 < len(ids):
        return queryset.model.objects.get(id=ids[idx + 1])
    return None

@login_required
def missing_taxon_annotation_for_mediafile(request, mediafile_id: int, uploaded_archive_id: Optional[int] = None):
    """Do taxon annotation on selected media file."""
    # Načíst uploaded archive, pokud byl předán
    if uploaded_archive_id:
        uploadedarchive = get_object_or_404(models.UploadedArchive, id=uploaded_archive_id)
    else:
        uploadedarchive = None

    mediafile = get_object_or_404(MediaFile, id=mediafile_id)


    # pick random non-classified media file
    mediafiles = models.get_mediafiles_with_missing_taxon(
        request.user.caiduser, uploadedarchive=uploadedarchive
    )
    missing_count = mediafiles.count()
    mediafiles_to_be_annotated = mediafiles.order_by("-parent__uploaded_at", "-captured_at")
    # find position of current mediafile and select the next one
    next_mediafile = get_next_in_queryset(mediafiles_to_be_annotated, mediafile)

    kwargs = {}
    if next_mediafile:
        kwargs = {"mediafile_id": next_mediafile.id}

    if request.method == "POST":
        # Očekáváme, že formulář obsahuje hidden input "mediafile_id"
        form = forms.MediaFileMissingTaxonForm(request.POST, instance=mediafile)
        if form.is_valid():
            mediafile = form.save(commit=False)
            mediafile.updated_by = request.user.caiduser
            mediafile.updated_at = django.utils.timezone.now()
            mediafile.save()
            # Po uložení se přesměrujeme na další mediafile s chybějícím taxonem
            if uploadedarchive:
                kwargs["uploaded_archive_id"] = uploadedarchive.id
            return redirect(
                reverse_lazy("caidapp:missing_taxon_annotation", kwargs=kwargs)
            )
        else:
            messages.error(request, "Form is not valid")
    else:
        # GET: vybrat náhodný media file z dostupných
        pass

    # Nastavit URL pro další načtení / přeskočení
    if uploadedarchive:
        kwargs["uploaded_archive_id"] = uploadedarchive.id
        next_url = reverse_lazy("caidapp:missing_taxon_annotation", kwargs=kwargs)
        # Pokud je k dispozici více než jeden soubor, můžeme nabídnout možnost přeskočení
        skip_url = next_url if mediafiles.count() > 1 else None
        cancel_url = reverse_lazy("caidapp:uploadedarchive_mediafiles", kwargs={"uploadedarchive_id": uploadedarchive.id})
    else:
        next_url = reverse_lazy("caidapp:missing_taxon_annotation", kwargs={})
        skip_url = next_url
        cancel_url = reverse_lazy("caidapp:taxon_processing")

    # Připravíme formulář s instancí mediafile
    form = forms.MediaFileMissingTaxonForm(instance=mediafile)
    # Pozor: V šabloně je vhodné mít hidden input s hodnotou mediafile.id,
    # aby bylo možné při POST identifikovat, který soubor se upravuje.
    return render(
        request,
        "caidapp/media_file_set_taxon.html",
        {
            "form": form,
            "headline": "Media File",
            "button": "Save and continue",
            "mediafile": mediafile,
            "skip_url": skip_url,
            "cancel_url": cancel_url,
            "next_url": next_url,
        },
    )


@login_required
def verify_taxa_view(request, uploaded_archive_id: Optional[int] = None):
    """See media files for verification."""
    return media_files_update(
        request,
        show_overview_button=True,
        taxon_verified=False,
        uploadedarchive_id=uploaded_archive_id,
        order_by="taxon__name",
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
        mediafile.taxon_verified = True
        mediafile.save()

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


@login_required
def confirm_prediction(request, mediafile_id: int) -> JsonResponse:
    """Confirm prediction for media file with low confidence."""
    try:
        mediafile = get_object_or_404(MediaFile, id=mediafile_id)
        # user has rw access
        if model_extra.user_has_rw_access_to_mediafile(
            request.user.caiduser, mediafile, accept_none=True
        ):
            # Update the MediaFile instance
            mediafile.taxon = mediafile.predicted_taxon
            mediafile.updated_at = timezone.now()
            mediafile.updated_by = request.user.caiduser
            # mediafile.taxon_verified = True
            mediafile.save()

            return JsonResponse({"success": True, "message": "Prediction confirmed."})
        return JsonResponse({"success": False, "message": "No read/write access to the file"})
    except Exception:
        return JsonResponse({"success": False, "message": "Invalid request."})


# todo deprecated
@login_required
def media_file_update(
    request, media_file_id, next_text="Save", next_url=None, skip_url=None, cancel_url=None
):
    """Show and update media file."""
    # | Q(parent__owner=request.user.caiduser)
    # | Q(parent__owner__workgroup=request.user.caiduser.workgroup)
    mediafile = get_object_or_404(MediaFile, pk=media_file_id)
    if (mediafile.parent.owner.id != request.user.id) and (
        mediafile.parent.owner.workgroup != request.user.caiduser.workgroup
    ):
        return HttpResponseNotAllowed("Not allowed to see this media file.")

    if request.method == "POST":

        logger.debug(f"{next_url=}")
        if next_url:
            pass
            # return HttpResponseRedirect(next_url)
        else:
            next_url = request.GET.get("next")

            if next_url is None:
                # next url is where i am comming from
                next_url = request.META.get("HTTP_REFERER", "/")

        form = MediaFileForm(request.POST, instance=mediafile)
        logger.debug(f"Form in POST: {mediafile=}")
        if form.is_valid():
            logger.debug(f"Form is valid")

            mediafile.updated_by = request.user.caiduser
            mediafile.updated_at = django.utils.timezone.now()
            # get uploaded archive
            mediafile = form.save()
            logger.debug(f"{mediafile.taxon=}")
            logger.debug(f"{mediafile.mediafile.path=}")
            logger.debug(f"{mediafile.updated_at=}")
            logger.debug(f"{next_url=}")
            return redirect(next_url)
        else:
            logger.error("Form is not valid.")
            from django.contrib import messages
            messages.error(request, "Form is not valid.")

    else:
        logger.debug(f"{next_url=}")
        if next_url:
            pass
            # return HttpResponseRedirect(next_url)
        else:
            next_url = request.GET.get("next")

            if next_url is None:
                # next url is where i am comming from
                next_url = request.META.get("HTTP_REFERER", "/")
        cancel_url = next_url
        logger.debug(f"{next_url=}")

        form = MediaFileForm(instance=mediafile)
    return render(
        request,
        "caidapp/media_file_update.html",
        {
            "form": form,
            "headline": "Media File",
            "button": next_text,
            "mediafile": mediafile,
            "skip_url": skip_url,
            "cancel_url": cancel_url,
        },
    )

from django.contrib.auth.mixins import LoginRequiredMixin
from extra_views import UpdateWithInlinesView, InlineFormSetFactory
from .models import MediaFile, AnimalObservation
from .forms import MediaFileForm

class ObservationInline(InlineFormSetFactory):
    model = AnimalObservation
    fields = [
        "taxon",
        "identity", "identity_is_representative",
        "bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height",
        "orientation"
    ]
    can_delete = True

    def get_factory_kwargs(self):
        kwargs = super().get_factory_kwargs()
        kwargs['extra'] = 1   # nebo 1, jak potřebuješ
        return kwargs

class MediaFileUpdateView(LoginRequiredMixin, UpdateWithInlinesView):
    model = MediaFile
    form_class = MediaFileForm
    inlines = [ObservationInline]
    template_name = "caidapp/media_file_update.html"
    context_object_name = "mediafile"

    def get_success_url(self):
        return self.request.GET.get("next") or self.request.META.get("HTTP_REFERER", "/")

    def form_valid(self, form):
        form.instance.updated_by = self.request.user.caiduser
        form.instance.updated_at = django.utils.timezone.now()
        return super().form_valid(form)

