import logging
import os
from pathlib import Path

import django
from celery import signature
from django.conf import settings
from django.contrib.auth import logout
from django.shortcuts import get_object_or_404, redirect, render
from django.contrib.auth.views import LoginView
from django.urls import reverse_lazy
from django.contrib import messages

from .forms import UploadedArchiveForm
from .models import UploadedArchive
from .tasks import predict_on_error, predict_on_success

logger = logging.getLogger("app")

# Create your views here.


def wellcome(request):
    """TODO add docstring."""
    pass


def media_files(request, ploadedarchive_id):
    """TODO add docstring."""
    # uploadedarchive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    # uploadedarchive
    pass


def uploads(request):
    """TODO add docstring."""
    uploadedarchives = UploadedArchive.objects.filter(
        owner=request.user.ciduser,
    ).all()  # \
    # .exclude(
    #    tag__in=hide_tags
    # )
    # print(uploadedarchives)
    context = {"uploadedarchives": uploadedarchives}
    return render(request, "caidapp/uploads.html", context)


def logout_view(request):
    """TODO add docstring."""
    logout(request)
    # Redirect to a success page.
    return redirect("/caidapp/login")


def model_form_upload(request):
    """TODO add docstring."""
    if request.method == "POST":
        form = UploadedArchiveForm(
            request.POST,
            request.FILES,
            # owner=request.user
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

            # update record in the database
            output_dir = Path(settings.MEDIA_ROOT) / uploaded_archive.outputdir
            uploaded_archive.owner = request.user.ciduser
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

            return redirect("/caidapp/uploads/")
    else:
        form = UploadedArchiveForm()
    return render(
        request,
        "caidapp/model_form_upload.html",
        {"form": form, "headline": "Upload", "button": "Upload"},
    )


def delete_upload(request, uploadedarchive_id):
    """TODO add docstring."""
    uploadedarchive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    uploadedarchive.delete()
    return redirect("/caidapp/uploads")

class MyLoginView(LoginView):
    redirect_authenticated_user = True

    def get_success_url(self):
        return reverse_lazy('caidapp:uploads')

    def form_invalid(self, form):
        messages.error(self.request, 'Invalid username or password')
        return self.render_to_response(self.get_context_data(form=form))
