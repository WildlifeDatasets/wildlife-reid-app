"""Start and inspect durable embedding jobs within the current workgroup."""

import uuid

from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError
from django.core.paginator import Paginator
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views.decorators.http import require_GET, require_POST

from . import embedding_runs, models

STATUS_LABELS = {
    "preparing": "Preparing inputs",
    "queued": "Queued",
    "running": "Processing",
    "finalizing": "Saving results",
    "succeeded": "Completed",
    "partial": "Completed with warnings",
    "failed": "Failed",
    "attention": "Status could not be confirmed",
}
TERMINAL_STATUSES = {"succeeded", "partial", "failed"}
ITEM_LABELS = {
    "pending": "Pending",
    "computed": "Embedding saved",
    "cached": "Embedding already existed",
    "stale": "Input changed or was deleted",
    "skipped": "Skipped",
    "failed": "Failed",
}


def _runs_for(request):
    """Share runs with members of their owning workgroup only."""
    workgroup_id = request.user.caiduser.workgroup_id
    return models.TaskRun.objects.filter(workgroup_id=workgroup_id) if workgroup_id else models.TaskRun.objects.none()


def _payload(run, include_items=False):
    """Add presentation labels and local URLs to the service's public payload."""
    payload = embedding_runs.serialize_run(run, include_items=include_items)
    payload.update(
        status_label=STATUS_LABELS.get(run.status, run.status),
        is_terminal=run.status in TERMINAL_STATUSES,
        operation_label="Embeddings",
        detail_url=reverse("caidapp:task_detail", kwargs={"run_uuid": run.uuid}),
        status_url=reverse("caidapp:task_status", kwargs={"run_uuid": run.uuid}),
        result_url=reverse("caidapp:task_result", kwargs={"run_uuid": run.uuid}),
    )
    result = payload.get("result") or {}
    payload["result_summary"] = " · ".join(
        f"{result[key]} {label}"
        for key, label in (
            ("computed", "saved"),
            ("cached", "already existed"),
            ("stale", "changed"),
            ("skipped", "skipped"),
        )
        if result.get(key)
    )
    payload["scope_label"] = f"{run.items.count()} observations"
    for item in payload.get("items", []):
        item["status_label"] = ITEM_LABELS.get(item.get("status"), item.get("status", ""))
        item.setdefault("mediafile_label", f"Media file #{item.get('mediafile_id', '')}")
        item["mediafile_url"] = reverse("caidapp:media_file_update", kwargs={"pk": item["mediafile_id"]})
    retry_ids = [
        item["observation_id"]
        for item in payload.get("items", [])
        if item.get("status") not in {"computed", "cached"} and item.get("observation_id")
    ]
    payload["retry_ids"] = (
        list(
            models.AnimalObservation.objects.filter(
                pk__in=retry_ids,
                mediafile__parent__owner__workgroup_id=run.workgroup_id,
                mediafile__media_type="image",
                is_no_detection_placeholder=False,
            ).values_list("pk", flat=True)
        )
        if run.status in TERMINAL_STATUSES | {"attention"}
        else []
    )
    return payload


@login_required
@require_POST
def embedding_run_start(request):
    """Queue a saved selection or all observations matching the current filter."""
    caiduser = request.user.caiduser
    if not caiduser.workgroup_id:
        return JsonResponse({"message": "Join a workgroup before generating embeddings."}, status=400)
    scope = request.POST.get("scope")
    if scope not in {"observations", "mediafiles", "filtered_observations"}:
        return JsonResponse({"message": "Choose a valid embedding scope."}, status=400)
    try:
        ids = []
        if scope in {"observations", "mediafiles"}:
            ids = sorted({int(value) for value in request.POST.getlist(f"{scope[:-1]}_ids")})
            if not ids or any(value <= 0 for value in ids):
                raise ValueError
        elif request.POST.getlist("observation_ids") or request.POST.getlist("mediafile_ids"):
            # A malformed explicit selection must never expand to the entire workgroup.
            raise ValueError
        request_key = request.POST.get("request_key") or None
        if request_key:
            request_key = str(uuid.UUID(request_key))
    except (ValueError, TypeError):
        return JsonResponse({"message": "Select at least one valid target."}, status=400)
    mediafiles = models.MediaFile.objects.for_user(caiduser).filter(parent__owner__workgroup_id=caiduser.workgroup_id)
    if scope == "filtered_observations":
        from .views import _get_observations_queryset, _parse_int_query_param_or_404

        accessible, observation_filter = _get_observations_queryset(request)
        if not observation_filter.is_valid():
            return JsonResponse(
                {"message": "Correct the observation filters before generating embeddings."}, status=400
            )
        accessible = accessible.filter(mediafile__in=mediafiles)
        sequence_id = _parse_int_query_param_or_404(request.GET.get("sequence"), "sequence id")
        if sequence_id is not None:
            accessible = accessible.filter(mediafile__sequence_id=sequence_id)
        ids = list(accessible.values_list("pk", flat=True))
    elif scope == "observations":
        accessible = models.AnimalObservation.objects.filter(mediafile__in=mediafiles, pk__in=ids)
    else:
        accessible = mediafiles.filter(pk__in=ids)
    if set(accessible.values_list("pk", flat=True)) != set(ids):
        return JsonResponse({"message": "Some selected targets are unavailable or inaccessible."}, status=403)
    try:
        run = embedding_runs.create_embedding_run(
            caiduser.workgroup_id,
            observation_ids=ids if scope != "mediafiles" else None,
            mediafile_ids=ids if scope == "mediafiles" else None,
            requested_by=caiduser,
            trigger="manual",
            request_key=request_key,
        )
    except ValidationError as exc:
        return JsonResponse({"message": "; ".join(exc.messages)}, status=400)
    return JsonResponse(_payload(run), status=202)


@login_required
@require_GET
def task_list(request):
    """List recent workgroup jobs alongside the existing automatic debounce timer."""
    runs = _runs_for(request).order_by("-created_at")
    state = request.GET.get("state", "all")
    if state == "active":
        runs = runs.exclude(status__in=TERMINAL_STATUSES)
    elif state == "history":
        runs = runs.filter(status__in=TERMINAL_STATUSES)
    page = Paginator(runs, 25).get_page(request.GET.get("page"))
    payloads = [_payload(embedding_runs.refresh_run(run)) for run in page]
    schedule = models.ObservationEmbeddingSchedule.objects.filter(
        workgroup_id=request.user.caiduser.workgroup_id, task_id__isnull=False
    ).first()
    return render(
        request, "caidapp/tasks.html", {"runs": payloads, "page_obj": page, "schedule": schedule, "state": state}
    )


@login_required
@require_GET
def task_detail(request, run_uuid):
    """Render the durable result, independently of the browser that started it."""
    run = embedding_runs.refresh_run(get_object_or_404(_runs_for(request), uuid=run_uuid))
    payload = _payload(run, include_items=True)
    return render(
        request,
        "caidapp/task_detail.html",
        {"run": payload, "status_url": payload["status_url"], "result_url": payload["result_url"]},
    )


@login_required
@require_GET
def task_status(request, run_uuid):
    """Expose state and item outcomes only through authorized public run UUIDs."""
    run = embedding_runs.refresh_run(get_object_or_404(_runs_for(request), uuid=run_uuid))
    return JsonResponse(_payload(run, include_items=True))
