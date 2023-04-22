import django
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import logout
from .forms import UploadedArchiveForm
from .models import UploadedArchive, CIDUser

# Create your views here.

def wellcome(request):
    pass

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
            async_task("uploader.tasks.email_media_recived", uploaded_archive)

            # email_media_recived(archivefile)
            # print(f"user id={request.user.id}")
            uploaded_archive.owner = request.user.ciduser
            uploaded_archive.started_at = django.utils.timezone.now()
            uploaded_archive.save()
            # PIGLEGCV_HOSTNAME = os.getenv("PIGLEGCV_HOSTNAME", default="127.0.0.1")
            # PIGLEGCV_PORT= os.getenv("PIGLEGCV_PORT", default="5000")
            # make_preview(archivefile)
            # update_owner(archivefile)
            # async_task(
            #     "uploader.tasks.run_processing",
            #     archivefile,
            #     request.build_absolute_uri("/"),
            #     PIGLEGCV_HOSTNAME,
            #     int(PIGLEGCV_PORT),
            #     timeout=settings.PIGLEGCV_TIMEOUT,
            #     hook="uploader.tasks.email_report_from_task",
            # )
            return redirect("/cidapp/uploads/")
    else:
        form = UploadedArchiveForm()
    return render(
        request,
        "cidapp/model_form_upload.html",
        {"form": form, "headline": "Upload", "button": "Upload"},
    )
