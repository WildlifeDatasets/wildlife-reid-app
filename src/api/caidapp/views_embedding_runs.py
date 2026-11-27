"""Start and inspect durable embedding jobs within the current workgroup."""

import uuid

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError
from django.core.paginator import Paginator
from django.db.models import Q, Value
from django.http import FileResponse, Http404, HttpResponse, HttpResponseForbidden, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from . import embedding_runs, export_runs, identification_refresh, model_weight_runs, models, observation_comparisons
from .observation_embeddings import reset_workgroup_embeddings
from .services.identity_review import input_version

STATUS_LABELS = {
    "preparing": "Preparing inputs",
    "queued": "Queued",
    "running": "Processing",
    "finalizing": "Saving results",
    "succeeded": "Completed",
    "partial": "Completed with warnings",
    "failed": "Failed",
    "cancelled": "Stopped",
    "attention": "Status could not be confirmed",
}
TERMINAL_STATUSES = embedding_runs.TERMINAL
ITEM_LABELS = {
    "pending": "Pending",
    "computed": "Embedding saved",
    "cached": "Embedding already existed",
    "stale": "Input changed or was deleted",
    "skipped": "Skipped",
    "failed": "Failed",
    "cancelled": "Stopped",
}


def _runs_for(request):
    """Scope task visibility to the workgroup, requester or installation administrators."""
    workgroup_id = request.user.caiduser.workgroup_id
    scope = Q(
        kind__in=("embedding", identification_refresh.KIND, *observation_comparisons.COMPARISON_LABELS),
        workgroup_id=workgroup_id,
    ) | Q(kind="export", requested_by=request.user.caiduser)
    if model_weight_runs.can_manage(request.user):
        scope |= Q(kind=model_weight_runs.KIND)
    return models.TaskRun.objects.filter(scope)


def _refresh(run):
    """Dispatch status checks without applying embedding lifecycle rules to exports."""
    if run.kind == model_weight_runs.KIND:
        return model_weight_runs.refresh(run)
    if run.kind in observation_comparisons.COMPARISON_LABELS:
        return observation_comparisons.refresh(run)
    if run.kind == identification_refresh.KIND:
        return run
    return export_runs.refresh_export(run) if run.kind == "export" else embedding_runs.refresh_run(run)


def _payload(run, include_items=False, *, staff_links=False):
    """Add presentation labels and local URLs to the service's public payload."""
    payload = embedding_runs.serialize_run(run, include_items=include_items)
    payload.update(
        status_label=STATUS_LABELS.get(run.status, run.status),
        is_terminal=run.status in TERMINAL_STATUSES,
        operation_label="Calculate missing image features" if run.trigger == "backfill" else "Embeddings",
        detail_url=reverse("caidapp:task_detail", kwargs={"run_uuid": run.uuid}),
        status_url=reverse("caidapp:task_status", kwargs={"run_uuid": run.uuid}),
        result_url=reverse("caidapp:task_result", kwargs={"run_uuid": run.uuid}),
    )
    result = payload.get("result") or {}
    if run.kind == model_weight_runs.KIND:
        payload.update(
            operation_label="Download model weights",
            scope_label="Installation",
            steps=[
                {
                    "key": key,
                    "label": label,
                    "status_label": STATUS_LABELS[result[key]["status"]],
                    "message": result[key]["message"],
                }
                for key, label in model_weight_runs.WORKERS.items()
            ],
            result_summary=run.message if not include_items else "",
            result={},
        )
        return payload
    if run.kind == identification_refresh.KIND:
        steps = []
        for key, label in identification_refresh.STEPS.items():
            entry = result["steps"].get(key)
            if entry is None:
                continue
            detail_url = ""
            if include_items and entry.get("record_id") and (staff_links or key not in {"outliers", "merges"}):
                detail_url = _refresh_step_url(key, entry["record_id"], run.workgroup_id)
            status = entry["status"]
            message = entry.get("message", "")
            if include_items and key == "features" and status == "failed" and entry.get("record_id"):
                child = models.TaskRun.objects.filter(
                    pk=entry["record_id"], workgroup_id=run.workgroup_id, kind="embedding",
                ).first()
                if child:
                    failure = embedding_runs.run_diagnostics(child)["failure"]
                    message = f"{failure['stage_label']} failed. {failure['reason']}" if failure else child.message
            steps.append({
                "key": key, "label": label, "status_label": STATUS_LABELS.get(status, status.capitalize()),
                "message": message, "detail_url": detail_url,
            })
        payload.update(
            operation_label="Refresh identification suggestions", scope_label="Workgroup",
            steps=steps, result_summary="", result={},
        )
        return payload
    if run.kind in observation_comparisons.COMPARISON_LABELS:
        payload.update(
            operation_label=observation_comparisons.COMPARISON_LABELS[run.kind],
            scope_label="Workgroup",
            result_summary=run.message,
            result={},
        )
        return payload
    if run.kind == "export":
        artifact = run.artifact
        try:
            available = (
                run.status == "succeeded"
                and artifact.expires_at is not None
                and artifact.expires_at > timezone.now()
                and export_runs.artifact_path(artifact).is_file()
            )
        except (ValidationError, OSError):
            available = False
        payload.update(
            operation_label=result.get("label", "Export"),
            scope_label=artifact.filename,
            filename=artifact.filename,
            size=artifact.size,
            expires_at=artifact.expires_at.isoformat() if artifact.expires_at else None,
            download_url=reverse("caidapp:task_download", kwargs={"run_uuid": run.uuid}) if available else None,
            result_summary=(
                ("Ready to download" if available else "File is no longer available")
                if run.status == "succeeded"
                else run.message
            ),
            retry_ids=[],
        )
        return payload
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
    if run.status == "failed" and not include_items:
        payload["result_summary"] = run.message
    if include_items and run.kind == "embedding":
        payload["diagnostics"] = embedding_runs.run_diagnostics(run)
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


def _refresh_step_url(step, record_id, workgroup_id):
    if step in {"features", "unknown", "known"}:
        uuid = models.TaskRun.objects.filter(pk=record_id, workgroup_id=workgroup_id).values_list("uuid", flat=True).first()
        return reverse("caidapp:task_detail", args=[uuid]) if uuid else ""
    if step == "identify":
        return reverse("caidapp:not_identified_mediafiles")
    # The standalone analysis pages still enforce staff access.
    route = {
        "outliers": "caidapp:identification_outlier_suggestions_result",
        "merges": "caidapp:similar_identity_pairs_result",
    }[step]
    return reverse(route, args=[record_id])


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
@require_POST
def manage_workgroup_embeddings(request):
    """Start a workgroup backfill or clear its feature data after confirmation."""
    caiduser = request.user.caiduser
    if not caiduser.workgroup_id or not caiduser.workgroup_admin:
        return HttpResponseForbidden()
    action = request.POST.get("action")
    if action == "stop_refresh":
        try:
            run_uuid = uuid.UUID(request.POST.get("run_uuid", ""))
        except ValueError:
            return HttpResponse(status=400)
        identification_refresh.stop_refresh(caiduser.workgroup_id, run_uuid)
        return redirect("caidapp:dash_identities")
    if action == "refresh":
        try:
            run = identification_refresh.request_refresh(caiduser.workgroup_id, caiduser)
        except ValidationError as exc:
            messages.error(request, exc.message)
        else:
            if request.POST.get("dashboard") != "yes":
                return redirect("caidapp:task_detail", run_uuid=run.uuid)
        if request.POST.get("dashboard") == "yes":
            return redirect("caidapp:dash_identities")
    elif action == "backfill":
        try:
            run = embedding_runs.create_embedding_run(
                caiduser.workgroup_id, requested_by=caiduser, trigger="backfill"
            )
        except ValidationError as exc:
            messages.error(request, "; ".join(exc.messages))
        else:
            return redirect("caidapp:task_detail", run_uuid=run.uuid)
    elif action == "reset" and request.POST.get("confirm_reset") == "yes":
        count = reset_workgroup_embeddings(caiduser.workgroup_id)
        messages.success(request, f"Reset embeddings and ALIKED features in this workgroup. Removed {count} feature sets.")
    else:
        messages.error(request, "Confirm the reset before removing feature data.")
    return redirect("caidapp:task_list")


@login_required
@require_GET
def task_list(request):
    """List recent authorized tasks alongside the automatic embedding schedule."""
    runs = _runs_for(request)
    state = request.GET.get("state", "all")
    if state == "active":
        runs = runs.exclude(status__in=TERMINAL_STATUSES)
    elif state == "history":
        runs = runs.filter(status__in=TERMINAL_STATUSES)
    elif state == "downloads":
        runs = runs.filter(kind="export")
    sources = {"task": runs}
    group_id = request.user.caiduser.workgroup_id
    if group_id and state != "downloads":
        sources["identification"] = models.IdentificationRunStatistic.objects.filter(workgroup_id=group_id)
        if request.user.is_staff:
            sources["outliers"] = models.IdentificationOutlierSuggestionResult.objects.filter(workgroup_id=group_id)
            sources["similar"] = models.IdentificationSimilarPairResult.objects.filter(workgroup_id=group_id)
    projections = []
    for kind, queryset in sources.items():
        active = {
            "identification": Q(finished_at__isnull=True),
            "outliers": Q(status__in=["pending", "processing"]),
            "similar": Q(status="processing") | Q(pixel_status="processing"),
        }.get(kind)
        if active is not None and state in {"active", "history"}:
            queryset = queryset.filter(active if state == "active" else ~active)
        projections.append(queryset.order_by().annotate(task_type=Value(kind)).values_list(
            "pk", "created_at", "task_type"
        ))

    index = projections[0].union(*projections[1:], all=True) if len(projections) > 1 else projections[0]
    page = Paginator(index.order_by("-created_at", "task_type", "-pk"), 25).get_page(request.GET.get("page"))
    page_rows = list(page)
    records = {kind: queryset.in_bulk([pk for pk, _, source in page_rows if source == kind])
               for kind, queryset in sources.items()}
    payloads = []
    for pk, _, kind in page_rows:
        run = records[kind].get(pk)
        if run is None:
            continue
        if kind == "task":
            payloads.append(_payload(_refresh(run)))
        else:
            payloads.append(_identification_task_payload(kind, run))
    schedule = models.ObservationEmbeddingSchedule.objects.filter(
        workgroup_id=request.user.caiduser.workgroup_id, task_id__isnull=False
    ).first()
    return render(
        request, "caidapp/tasks.html", {"runs": payloads, "page_obj": page, "schedule": schedule, "state": state}
    )


def _identification_task_payload(kind, run):
    """Link each identification job to its existing workgroup review page."""
    if kind == "identification":
        status = "running" if run.finished_at is None else ("succeeded" if run.status == "finished" else "failed")
        return {
            "operation_label": run.get_operation_display(),
            "scope_label": f"{run.image_number} images, {run.video_number} videos",
            "created_at": run.created_at,
            "status_label": STATUS_LABELS[status],
            "detail_url": reverse("caidapp:dash_identities"),
        }
    label, route = {
        "outliers": ("Identification outliers", "caidapp:identification_outlier_suggestions_result"),
        "similar": ("Identity merge suggestions", "caidapp:similar_identity_pairs_result"),
    }[kind]
    active = run.status in {"pending", "processing"} or getattr(run, "pixel_status", "") == "processing"
    status = "running" if active else ("succeeded" if run.status == "done" else "failed")
    return {
        "operation_label": label,
        "scope_label": "Workgroup",
        "created_at": run.created_at,
        "status_label": STATUS_LABELS[status],
        "detail_url": reverse(route, args=[run.pk]),
        "result_summary": run.message if status == "succeeded" else "",
    }


@login_required
@require_GET
def task_detail(request, run_uuid):
    """Render the durable result, independently of the browser that started it."""
    run = _refresh(get_object_or_404(_runs_for(request), uuid=run_uuid))
    payload = _payload(run, include_items=True, staff_links=request.user.is_staff)
    if run.kind in observation_comparisons.COMPARISON_LABELS:
        from .views import _identification_bbox, user_can_manage_identification

        rows = run.result.get("rows", [])
        if run.kind == "review_known":
            rows = sorted(rows, key=_comparison_margin, reverse=True)
        page = Paginator(rows, 25).get_page(request.GET.get("page"))
        saved = [item for row in page for item in [row["query"], *row["candidates"]]]
        live = models.AnimalObservation.objects.filter(
            mediafile__parent__owner__workgroup_id=run.workgroup_id,
            is_no_detection_placeholder=False,
        ).select_related("mediafile", "identity").in_bulk({item["id"] for item in saved})
        assigned = run.result.get("assigned", {})
        for item in saved:
            observation = live.get(item["id"])
            if observation is not None:
                item["media"] = observation.mediafile
                item["bbox"] = _identification_bbox(observation, observation.mediafile)
                item["current_identity"] = str(observation.identity) if observation.identity_id else ""
                expected = assigned.get(str(item["id"]), item)["version"]
                item["is_current"] = input_version(observation, observation.mediafile) == expected
        for row in page:
            query_id = assigned.get(str(row["query"]["id"]), {}).get("assigned_identity_id")
            for candidate in row["candidates"]:
                candidate_id = assigned.get(str(candidate["id"]), {}).get("assigned_identity_id")
                candidate["grouped"] = query_id is not None and query_id == candidate_id
        return render(request, "caidapp/observation_comparison.html", {
            "run": payload, "page_obj": page,
            "can_group": run.kind == "compare_unknown" and user_can_manage_identification(request.user),
        })
    return render(
        request,
        "caidapp/task_detail.html",
        {"run": payload, "status_url": payload["status_url"], "result_url": payload["result_url"]},
    )


@login_required
@require_GET
def task_status(request, run_uuid):
    """Expose state and item outcomes only through authorized public run UUIDs."""
    run = _refresh(get_object_or_404(_runs_for(request), uuid=run_uuid))
    return JsonResponse(_payload(run, include_items=True, staff_links=request.user.is_staff))


def _comparison_margin(row):
    scores = {candidate["role"]: candidate["score"] for candidate in row["candidates"]}
    return scores["other"] - scores["own"] if {"own", "other"}.issubset(scores) else float("-inf")


@login_required
@require_POST
def start_observation_comparison(request):
    from .views import user_can_manage_identification

    if not user_can_manage_identification(request.user):
        return HttpResponseForbidden()
    user = request.user.caiduser
    try:
        run = observation_comparisons.start(
            user.workgroup, user, request.POST.get("mode"),
            include_non_representative=request.POST.get("gallery") == "all_identified",
        )
    except ValidationError as exc:
        messages.error(request, exc.message)
        return redirect("caidapp:pre_identify")
    return redirect("caidapp:task_detail", run_uuid=run.uuid)


@login_required
@require_POST
def group_observation_pair(request, run_uuid):
    from .views import user_can_manage_identification

    if not user_can_manage_identification(request.user):
        return HttpResponseForbidden()
    run = get_object_or_404(_runs_for(request), uuid=run_uuid, kind="compare_unknown", status="succeeded")
    try:
        identity = observation_comparisons.group_pair(
            run, int(request.POST.get("query_id", "")), int(request.POST.get("candidate_id", "")),
            request.user.caiduser,
        )
    except (ValidationError, ValueError) as exc:
        messages.warning(request, exc.message if isinstance(exc, ValidationError) else "Select a valid pair.")
    else:
        messages.success(request, f"Pair assigned to {identity.name}.")
    return redirect("caidapp:task_detail", run_uuid=run.uuid)


@login_required
@require_GET
def task_download(request, run_uuid):
    """Serve an unexpired private export exclusively to the user who requested it."""
    run = get_object_or_404(models.TaskRun, uuid=run_uuid, kind="export", requested_by=request.user.caiduser)
    if run.status != "succeeded":
        raise Http404
    artifact = run.artifact
    if not artifact.expires_at or artifact.expires_at <= timezone.now():
        return HttpResponse("This export has expired. Please request a new export.", status=410)
    try:
        path = export_runs.artifact_path(artifact)
        stream = path.open("rb")
    except (ValidationError, OSError):
        return HttpResponse("The export file is no longer available. Please request a new export.", status=410)
    response = FileResponse(stream, as_attachment=True, filename=artifact.filename)
    response["Cache-Control"] = "private, no-store"
    response["X-Content-Type-Options"] = "nosniff"
    return response
