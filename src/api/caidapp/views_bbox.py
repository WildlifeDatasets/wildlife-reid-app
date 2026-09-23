"""Explicit media-file scope for redetection; proposals never write annotations."""

from datetime import timedelta
import uuid

from django import forms
from django.contrib.auth.decorators import login_required
from django.core import signing
from django.http import Http404, HttpResponseBadRequest, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_POST, require_GET

from . import bbox_services as service, models


class RedetectionForm(forms.Form):
    selection = forms.CharField(widget=forms.HiddenInput)
    detector = forms.ChoiceField(
        choices=[("auto", "MegaDetector + SAM3 fallback"), ("megadetector", "MegaDetector"), ("sam3", "SAM3")],
        initial="auto",
    )
    confidence = forms.FloatField(min_value=0, max_value=1, initial=0.5, label="Minimum detection confidence")
    min_iou = forms.FloatField(
        min_value=0, max_value=1, initial=0.2, label="Minimum IoU for matching existing observations"
    )
    create_new = forms.BooleanField(
        required=False, initial=True, label="Create observations for unmatched new detections"
    )
    unmatched = forms.ChoiceField(
        choices=[
            ("clear", "Clear bbox and keep the observation"),
            ("keep", "Keep the original bbox"),
            ("delete", "Delete the unmatched observation and its annotations"),
        ],
        initial="clear",
        label="Existing bboxes without a matching detection",
    )


def prepare_redetection(request, mediafile_ids):
    mediafiles = models.MediaFile.objects.for_user(request.user.caiduser).filter(pk__in=mediafile_ids)
    ids = list(mediafiles.filter(media_type="image").order_by("pk").values_list("pk", flat=True))
    token = signing.dumps(
        {"ids": ids, "user": request.user.pk, "batch": str(uuid.uuid4())}, salt="bbox-selection", compress=True
    )
    return render(
        request,
        "caidapp/bbox_redetection.html",
        {
            "form": RedetectionForm(initial={"selection": token}),
            "mediafile_count": len(ids),
            "observation_count": models.AnimalObservation.objects.filter(mediafile_id__in=ids).count(),
            "video_count": mediafiles.exclude(media_type="image").count(),
        },
    )


@login_required
@require_POST
def start_redetection(request):
    form = RedetectionForm(request.POST)
    try:
        selection = signing.loads(request.POST.get("selection", ""), salt="bbox-selection", max_age=3600)
        if selection["user"] != request.user.pk:
            raise signing.BadSignature("Wrong user")
    except (signing.BadSignature, KeyError, TypeError):
        return HttpResponseBadRequest("Selection expired. Select media files again.")
    mediafiles = models.MediaFile.objects.for_user(request.user.caiduser).filter(
        pk__in=selection["ids"], media_type="image"
    )
    if form.is_valid():
        if not mediafiles.exists():
            return HttpResponseBadRequest("No editable image media files were selected.")
        # A repeated submit of this confirmation resumes its existing batch.
        if not models.BboxDetectionJob.objects.filter(
            batch=selection["batch"], requested_by=request.user.caiduser
        ).exists():
            options = {key: form.cleaned_data[key] for key in service.DEFAULT_OPTIONS}
            service.create_jobs(request.user.caiduser, mediafiles, options, batch=selection["batch"])
        return redirect("caidapp:bbox_detection_status", batch=selection["batch"])
    return render(
        request,
        "caidapp/bbox_redetection.html",
        {
            "form": form,
            "mediafile_count": mediafiles.count(),
            "observation_count": models.AnimalObservation.objects.filter(mediafile__in=mediafiles).count(),
        },
    )


def _expire_jobs(jobs):
    # A lost worker/callback cannot leave this page polling forever or apply late results.
    jobs.filter(status="queued", created_at__lt=timezone.now() - timedelta(hours=24)).update(
        status="failed",
        message="Detection timed out. No observations were changed.",
        finished_at=timezone.now(),
    )


@login_required
@require_GET
def detection_status(request, batch):
    jobs = (
        models.BboxDetectionJob.objects.filter(batch=batch, requested_by=request.user.caiduser)
        .filter(
            mediafile__in=models.MediaFile.objects.for_user(request.user.caiduser),
        )
        .select_related("mediafile")
    )
    if not jobs.exists():
        raise Http404
    _expire_jobs(jobs)
    return render(
        request,
        "caidapp/bbox_detection_status.html",
        {
            "jobs": jobs.order_by("pk"),
            "pending": jobs.filter(status="queued").exists(),
            "completed": jobs.exclude(status="queued").count(),
            "total": jobs.count(),
        },
    )


@login_required
@require_POST
def propose_bbox(request, mediafile_id):
    mediafile = get_object_or_404(
        models.MediaFile.objects.for_user(request.user.caiduser), pk=mediafile_id, media_type="image"
    )
    try:
        snapshot = service.observation_snapshot(mediafile)
    except (OSError, ValueError, AttributeError):
        return JsonResponse({"message": "The source image is unavailable."}, status=400)
    recent = models.BboxDetectionJob.objects.filter(
        mediafile=mediafile,
        requested_by=request.user.caiduser,
        purpose="suggest",
        status__in=["queued", "succeeded"],
        created_at__gte=timezone.now() - timedelta(minutes=15),
    ).order_by("-pk")
    job = next((j for j in recent if j.snapshot == snapshot), None)
    if job is None:
        _, jobs = service.create_jobs(
            request.user.caiduser,
            models.MediaFile.objects.filter(pk=mediafile.pk),
            dict(service.DEFAULT_OPTIONS),
            purpose="suggest",
        )
        job = jobs[0]
    return JsonResponse({"url": reverse("caidapp:bbox_proposal_status", args=[job.pk])})


@login_required
@require_GET
def proposal_status(request, job_id):
    jobs = models.BboxDetectionJob.objects.filter(
        pk=job_id,
        requested_by=request.user.caiduser,
        purpose="suggest",
        mediafile__in=models.MediaFile.objects.for_user(request.user.caiduser),
    )
    _expire_jobs(jobs)
    job = get_object_or_404(jobs)
    return JsonResponse({"status": job.status, "message": job.message, "detections": job.result.get("detections", [])})
