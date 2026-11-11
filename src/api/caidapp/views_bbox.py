"""Explicit media-file scope for redetection; proposals never write annotations."""

import uuid
from datetime import timedelta

from django import forms
from django.contrib.auth.decorators import login_required
from django.core import signing
from django.http import Http404, HttpResponseBadRequest, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from . import bbox_services as service
from . import models
from .observation_geometry import HAS_BBOX


class DetectionForm(forms.Form):
    selection = forms.CharField(widget=forms.HiddenInput)
    detector = forms.ChoiceField(
        choices=[("auto", "MegaDetector + SAM3 fallback"), ("megadetector", "MegaDetector"), ("sam3", "SAM3")],
        initial="auto",
    )
    confidence = forms.FloatField(min_value=0, max_value=1, initial=0.5, label="Minimum detection confidence")


class RedetectionForm(DetectionForm):
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
    """Render redetection confirmation for accessible images with a signed user-bound selection."""
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


def prepare_fill_missing(request, mediafile_ids, observation_ids=None):
    """Confirm exact missing-observation scope, or all missing boxes in selected media files."""
    mediafiles = models.MediaFile.objects.for_user(request.user.caiduser).filter(pk__in=mediafile_ids)
    targets = models.AnimalObservation.objects.filter(mediafile__in=mediafiles).exclude(HAS_BBOX)
    if observation_ids is not None:
        targets = targets.filter(pk__in=observation_ids)
        empty_ids = []
    else:
        empty_ids = list(mediafiles.filter(observations__isnull=True, media_type="image").values_list("pk", flat=True))
    ids = list(targets.filter(mediafile__media_type="image").order_by("pk").values_list("pk", flat=True))
    image_ids = sorted(set(targets.filter(pk__in=ids).values_list("mediafile_id", flat=True)) | set(empty_ids))
    token = signing.dumps(
        {
            "ids": image_ids,
            "observations": ids,
            "empty": empty_ids,
            "user": request.user.pk,
            "batch": str(uuid.uuid4()),
        },
        salt="bbox-fill-selection",
        compress=True,
    )
    return render(
        request,
        "caidapp/bbox_fill_missing.html",
        {
            "form": DetectionForm(initial={"selection": token}),
            "mediafile_count": len(image_ids),
            "observation_count": len(ids),
            "empty_count": len(empty_ids),
            "video_count": mediafiles.exclude(media_type="image").count(),
        },
    )


@login_required
@require_POST
def start_fill_missing(request):
    """Start or resume a user-bound batch that only fills the signed missing-box selection."""
    form = DetectionForm(request.POST)
    try:
        selection = signing.loads(request.POST.get("selection", ""), salt="bbox-fill-selection", max_age=3600)
        if selection["user"] != request.user.pk:
            raise signing.BadSignature("Wrong user")
        batch = uuid.UUID(selection["batch"])
        ids, target_ids, empty_ids = selection["ids"], selection["observations"], selection["empty"]
    except (signing.BadSignature, KeyError, TypeError, ValueError):
        return HttpResponseBadRequest("Selection expired. Select observations or media files again.")
    mediafiles = models.MediaFile.objects.for_user(request.user.caiduser).filter(pk__in=ids, media_type="image")
    if form.is_valid():
        existing = models.BboxDetectionJob.objects.filter(
            batch=batch, requested_by=request.user.caiduser, purpose="fill"
        )
        if not existing.exists():
            current_targets = models.AnimalObservation.objects.filter(
                pk__in=target_ids, mediafile__in=mediafiles
            ).exclude(HAS_BBOX)
            eligible_ids = set(current_targets.values_list("mediafile_id", flat=True)) | set(
                mediafiles.filter(pk__in=empty_ids, observations__isnull=True).values_list("pk", flat=True)
            )
            options = {
                **service.DEFAULT_OPTIONS,
                "detector": form.cleaned_data["detector"],
                "confidence": form.cleaned_data["confidence"],
                "unmatched": "keep",
            }
            _, jobs = service.create_jobs(
                request.user.caiduser,
                mediafiles.filter(pk__in=eligible_ids),
                options,
                purpose="fill",
                batch=batch,
                observation_ids=list(current_targets.values_list("pk", flat=True)),
            )
            if not jobs:
                return HttpResponseBadRequest("No selected editable image observations still need a bbox.")
        return redirect("caidapp:bbox_detection_status", batch=batch)
    return render(
        request,
        "caidapp/bbox_fill_missing.html",
        {
            "form": form,
            "mediafile_count": mediafiles.count(),
            "observation_count": models.AnimalObservation.objects.filter(
                pk__in=target_ids, mediafile__in=mediafiles
            ).count(),
            "empty_count": mediafiles.filter(pk__in=empty_ids, observations__isnull=True).count(),
        },
    )


@login_required
@require_POST
def start_redetection(request):
    """Validate a signed selection and create or resume its image detection batch."""
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
    """Expire stale jobs and render status for the user's accessible detection batch."""
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
    """Reuse a recent matching bbox proposal or queue a new suggestion job for an accessible image."""
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
        purpose__in=["suggest", "fill"],
        status__in=["queued", "succeeded", "review"],
        created_at__gte=timezone.now() - timedelta(minutes=15),
    ).order_by("-pk")
    job = next((j for j in recent if j.snapshot == snapshot and (j.purpose == "suggest" or j.status == "review")), None)
    review_id = request.GET.get("bbox_job", "")
    if review_id.isascii() and review_id.isdecimal() and len(review_id) <= 18:
        review = models.BboxDetectionJob.objects.filter(
            pk=int(review_id),
            mediafile=mediafile,
            requested_by=request.user.caiduser,
            purpose="fill",
            status="review",
        ).first()
        if review is not None and review.snapshot == snapshot:
            job = review
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
    """Expire stale proposals and return detection data for the user's accessible suggestion job."""
    jobs = models.BboxDetectionJob.objects.filter(
        pk=job_id,
        requested_by=request.user.caiduser,
        purpose__in=["suggest", "fill"],
        mediafile__in=models.MediaFile.objects.for_user(request.user.caiduser),
    )
    _expire_jobs(jobs)
    job = get_object_or_404(jobs)
    return JsonResponse(
        {
            "status": "succeeded" if job.purpose == "fill" and job.status == "review" else job.status,
            "message": job.message,
            "detections": job.result.get("detections", []),
        }
    )
