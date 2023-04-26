import django
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import logout
from .forms import UploadedArchiveForm
from .models import UploadedArchive, CIDUser
from django.conf import settings
from . import tasks
from .cv.data_processing_pipeline import data_processing
from loguru import logger

# Create your views here.

def wellcome(request):
    pass

def media_files(request, uploadedarchive_id):
    uploadedarchive = get_object_or_404(UploadedArchive, pk=uploadedarchive_id)
    uploadedarchive



def uploads(request):
    uploadedarchives = UploadedArchive.objects.filter(
        owner=request.user.ciduser,
    ).all() #\
    # .exclude(
    #    tag__in=hide_tags
    #)
    print(uploadedarchives)
    context = {
        "uploadedarchives": uploadedarchives
    }
    return render(request, 'cidapp/uploads.html', context)

def logout_view(request):
    logout(request)
    # Redirect to a success page.
    return redirect('/cidapp/login')

def model_form_upload(request):
    if request.method == "POST":
        form = UploadedArchiveForm(
            request.POST,
            request.FILES,
            # owner=request.user
        )
        if form.is_valid():
            from django_q.tasks import async_task

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
            print("async task ....")

            logger.debug("Run processing ...")
            # tasks.run_processing(
            #     uploaded_archive
            # )
            async_task(
                "cidapp.tasks.run_processing",
                uploaded_archive,
                timeout=settings.COMPUTER_VISION_TIMEOUT,
                # hook="uploader.tasks.email_report_from_task",
            )
            return redirect("/cidapp/uploads/")
    else:
        form = UploadedArchiveForm()
    return render(
        request,
        "cidapp/model_form_upload.html",
        {"form": form, "headline": "Upload", "button": "Upload"},
    )
