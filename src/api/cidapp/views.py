import logging

import django
from django.contrib.auth import logout
from django.shortcuts import get_object_or_404, redirect, render

from .celery import tasks
from .forms import UploadedArchiveForm
from .models import UploadedArchive

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
    print(uploadedarchives)
    context = {"uploadedarchives": uploadedarchives}
    return render(request, "cidapp/uploads.html", context)


def logout_view(request):
    """TODO add docstring."""
    logout(request)
    # Redirect to a success page.
    return redirect("/cidapp/login")


def model_form_upload(request):
    """TODO add docstring."""
    if request.method == "POST":
        form = UploadedArchiveForm(
            request.POST,
            request.FILES,
            # owner=request.user
        )
        if form.is_valid():
            # logger.debug(f"imagefile.name={dir(form)}")
            # name = form.cleaned_data['imagefile']
            # if name is None or name == '':
            #     return render(request, 'uploader/model_form_upload.html', {
            #         'form': form,
            #         "headline": "Upload",
            #         "button": "Upload",
            #         "error_text": "Image File is mandatory"
            #     })

            uploaded_archive = form.save()
            # async_task("uploader.tasks.email_media_recived", uploaded_archive)

            # email_media_recived(archivefile)
            # print(f"user id={request.user.id}")
            uploaded_archive.owner = request.user.ciduser
            uploaded_archive.started_at = django.utils.timezone.now()
            uploaded_archive.save()
            # PIGLEGCV_HOSTNAME = os.getenv("PIGLEGCV_HOSTNAME", default="127.0.0.1")
            # PIGLEGCV_PORT= os.getenv("PIGLEGCV_PORT", default="5000")
            # make_preview(archivefile)
            # update_owner(archivefile)

            # send celery message to the data worker
            logger.info("Created data worker ")
            task = tasks.send_task(
                "upload",
                kwargs={
                    "input_archive_file": uploaded_archive.archivefile.name,
                    "output_dir": uploaded_archive.outputdir,
                },
            )
            logger.info(f"Created worker task with id '{task.task_id}'.")

            # result = task.get()
            # logger.info(f"Retrieved result: {result}")

            # uploaded_archive.finished_at = django.utils.timezone.now()
            # uploaded_archive.zip_file = os.path.relpath(outputdir_zip, settings.MEDIA_ROOT)
            # uploaded_archive.csv_file = os.path.relpath(outputdir_csv, settings.MEDIA_ROOT)
            # uploaded_archive.save()

            # async_task(
            #     "cidapp.tasks.run_processing",
            #     uploaded_archive,
            #     timeout=settings.COMPUTER_VISION_TIMEOUT,
            #     # hook="uploader.tasks.email_report_from_task",
            # )
            return redirect("/cidapp/uploads/")
    else:
        form = UploadedArchiveForm()
    return render(
        request,
        "cidapp/model_form_upload.html",
        {"form": form, "headline": "Upload", "button": "Upload"},
    )


def delete_upload(request, uploadedarchive_id):
    """TODO add docstring."""
    uploadedarchive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    uploadedarchive.delete()
    return redirect("/cidapp/uploads")
