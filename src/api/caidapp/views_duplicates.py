"""Duplicate suggestion list, merge previews, and authenticated API endpoints."""

import json
import uuid

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core import signing
from django.core.exceptions import PermissionDenied
from django.core.paginator import Paginator
from django.db.models import Q
from django.http import Http404, HttpResponseBadRequest, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from . import duplicate_hashes as hashes
from . import duplicate_services as service
from . import models

KIND_LABELS = {
    "byte_exact": "Identical files",
    "pixel_exact": "Identical image pixels",
    "crop": "Likely crop / resized copy",
    "reference_exact": "Identical reference frames (videos may differ)",
}


def _scan(request, scan_uuid):
    user = request.user.caiduser
    scope = Q(workgroup__isnull=True, requested_by=user)
    if user.workgroup_id:
        scope |= Q(workgroup_id=user.workgroup_id)
    return get_object_or_404(models.DuplicateScan.objects.filter(scope), uuid=scan_uuid)


def _proposal(request, proposal_uuid):
    proposal = get_object_or_404(
        models.DuplicateProposal.objects.select_related("scan"),
        uuid=proposal_uuid,
    )
    _scan(request, proposal.scan.uuid)
    try:
        service.proposal_media(proposal, request.user.caiduser)
    except ValueError as exc:
        raise Http404 from exc
    return proposal


def _visible_proposals(request, scan, status):
    accessible = models.MediaFile.objects.for_user(request.user.caiduser).values("pk")
    if scan and not scan.workgroup_id:
        rows = scan.proposals.all()
    else:
        rows = models.DuplicateProposal.objects.filter(scan__workgroup_id=request.user.caiduser.workgroup_id)
        accessible = accessible.filter(
            parent__owner__workgroup_id=request.user.caiduser.workgroup_id, parent__import_finished=True
        )
    hidden = models.DuplicateProposalMember.objects.filter(proposal__in=rows, mediafile__isnull=False)
    hidden = hidden.exclude(mediafile_id__in=accessible).values("proposal_id")
    rows = rows.exclude(pk__in=hidden)
    if status in {"pending", "resolved", "dismissed", "stale"}:
        rows = rows.filter(status=status)
    return (
        rows.order_by("pk")
        .select_related("scan")
        .prefetch_related(
            "members__mediafile__parent",
            "members__mediafile__locality",
            "members__mediafile__sequence",
            "members__mediafile__observations__taxon",
            "members__mediafile__observations__identity",
            "members__mediafile__observations__identity",
            "members__mediafile__album_set",
            "members__mediafile__cover",
        )
    )


def _describe_proposals(page):
    for proposal in page:
        proposal.kind_label = KIND_LABELS.get(proposal.kind, proposal.kind)
        proposal.summary = service.proposal_summary(proposal)
        proposal.identity_merges = service.identity_merge_candidates(proposal)


def _proposal_page(rows, number):
    """Rank the whole visible catalogue before pagination, retaining only IDs between batches."""
    ranked = []
    for proposal in rows.iterator(chunk_size=100):
        summary = service.proposal_summary(proposal)
        count = summary["difference_count"]
        ranked.append((not summary["perfect"], count if count is not None else float("inf"), proposal.pk))
    ranked.sort()
    page = Paginator([pk for _, _, pk in ranked], 20).get_page(number)
    ids = list(page.object_list)
    by_id = {proposal.pk: proposal for proposal in rows.filter(pk__in=ids)}
    page.object_list = [by_id[pk] for pk in ids if pk in by_id]
    _describe_proposals(page)
    return page


def _processing_summary(total, ready, errors):
    """Build disjoint compact-ring segments, also for empty or concurrently changing imports."""
    ready = min(max(ready, 0), total)
    errors = min(max(errors, 0), total - ready)
    ready_percent = 100 * ready / total if total else 0
    error_percent = 100 * errors / total if total else 0
    return {
        "total": total, "ready": ready, "errors": errors, "pending": total - ready - errors,
        "ready_dash": f"{ready_percent:.3f} {100 - ready_percent:.3f}",
        "error_dash": f"{error_percent:.3f} {100 - error_percent:.3f}",
        "error_offset": f"{-ready_percent:.3f}",
        "label": f"{round(ready_percent)}%" if total else "—",
    }


@login_required
@require_GET
def duplicate_list(request):
    """Show the whole group's persistent catalogue, without per-file launch controls."""
    user = request.user.caiduser
    if not user.workgroup_id:
        raise PermissionDenied("Select a working group first")
    hashes.recover_group(user.workgroup_id)
    media = hashes.group_media(user.workgroup_id)
    scan = models.DuplicateScan.objects.filter(workgroup_id=user.workgroup_id).order_by("-pk").first()
    status = request.GET.get("status", "pending")
    page = _proposal_page(_visible_proposals(request, scan, status), request.GET.get("page"))
    schedule = models.DuplicateHashSchedule.objects.filter(workgroup_id=user.workgroup_id).first()
    batch = (
        schedule.active_scan.hash_batches.filter(status="queued").first()
        if schedule and schedule.active_scan_id
        else None
    )
    progress = hashes.batch_progress(batch) if batch else None
    if progress and not progress["active"]:
        progress = None
    total = media.count()
    hashed = media.filter(hash_status="ready", hash_version=hashes.HASH_VERSION).count()
    errors = media.filter(hash_status="error").count()
    return render(
        request,
        "caidapp/duplicates.html",
        {
            "workgroup": user.workgroup,
            "can_force": user.workgroup_admin,
            "scan": scan,
            "schedule": schedule,
            "worker_progress": progress,
            "delivery_message": batch.message if batch else "",
            "total": total,
            "hashed": hashed,
            "errors": errors,
            "summary_stats": _processing_summary(total, hashed, errors),
            "issues": media.exclude(hash_message="").order_by("pk")[:50],
            "page_obj": page,
            "status_filter": status,
            "pending": bool(schedule and (schedule.task_id or schedule.active_scan_id)),
            "bulk_allowed": True,
        },
    )


@login_required
@require_POST
def bulk_review(request):
    """Preview explicit group selections or perfect matches; only a signed confirmation deletes."""
    user = request.user.caiduser
    if not user.workgroup_id:
        raise PermissionDenied("Select a working group first")
    rows = _visible_proposals(request, None, "pending")
    try:
        if request.POST.get("action") == "apply":
            count = service.apply_bulk(user, request.POST.get("token", ""), rows.values_list("pk", flat=True))
            messages.success(request, f"Removed {count} duplicate media files. Retained files were not changed.")
            return redirect("caidapp:duplicates")
        perfect_only = request.POST.get("action") == "perfect"
        if not perfect_only:
            selected = {str(uuid.UUID(value)) for value in request.POST.getlist("proposals")}
            if not selected or len(selected) > service.MAX_BULK_PROPOSALS:
                raise ValueError(f"Select between 1 and {service.MAX_BULK_PROPOSALS} groups")
            rows = rows.filter(uuid__in=selected)
            if rows.count() != len(selected):
                raise ValueError("Some selected proposals are unavailable")
        preview = service.bulk_preview(rows.iterator(chunk_size=100), user, perfect_only=perfect_only)
        return render(request, "caidapp/duplicate_bulk_review.html", {**preview, "perfect_only": perfect_only})
    except (ValueError, TypeError, KeyError, signing.BadSignature, OSError) as exc:
        messages.error(request, str(exc))
        return redirect("caidapp:duplicates")


@login_required
@require_POST
def scan_start(request):
    """Request group-wide exact hashing; only group administrators can force a recalculation."""
    is_json = request.content_type == "application/json"
    try:
        data = json.loads(request.body) if is_json else request.POST
        if "mediafile_ids" in data or "include_crops" in data:
            raise ValueError("Exact duplicate detection always checks the whole workgroup")
        force = data.get("force", False)
        if not isinstance(force, bool):
            force = force == "on"
        schedule = hashes.request_hashes(request.user.caiduser, force=force)
    except (ValueError, TypeError, AttributeError, OSError) as exc:
        return JsonResponse({"error": str(exc)}, status=400) if is_json else HttpResponseBadRequest(str(exc))
    url = reverse("caidapp:duplicates")
    if is_json:
        return JsonResponse(
            {
                "url": url,
                "status": "scheduled" if schedule else "current",
                "workgroup_id": request.user.caiduser.workgroup_id,
            },
            status=202,
        )
    return redirect(url)


@login_required
@require_GET
def scan_detail(request, scan_uuid):
    """List groups and pairs, with pagination and durable status filters."""
    scan = _scan(request, scan_uuid)
    if scan.purpose == "exact":
        hashes.recover_group(scan.workgroup_id)
        scan.refresh_from_db()
    else:
        service.refresh_scan(scan)
    status = request.GET.get("status", "pending")
    proposals = _visible_proposals(request, scan, status)
    page = _proposal_page(proposals, request.GET.get("page"))
    accessible = set(models.MediaFile.objects.for_user(request.user.caiduser).values_list("pk", flat=True))
    return render(
        request,
        "caidapp/duplicate_scan.html",
        {
            "scan": scan,
            "page_obj": page,
            "status_filter": status,
            "bulk_allowed": bool(scan.workgroup_id),
            "issues": [
                item
                for item in scan.items
                if item["id"] in accessible and (item.get("error") or item.get("pixel_error"))
            ],
        },
    )


@login_required
@require_GET
def scan_status(request, scan_uuid):
    """Return a small API status without exposing filesystem paths or inaccessible items."""
    scan = _scan(request, scan_uuid)
    if scan.purpose == "exact":
        hashes.recover_group(scan.workgroup_id)
        scan.refresh_from_db()
    else:
        service.refresh_scan(scan)
    return JsonResponse(
        {
            "uuid": str(scan.uuid),
            "status": scan.status,
            "message": scan.message,
            "algorithm": scan.algorithm,
            "total": scan.total,
            "processed": scan.processed,
            "failed": scan.failed_count,
            "skipped": scan.skipped,
            "proposals": _visible_proposals(request, scan, "all").count(),
        }
    )


@login_required
@require_GET
def scan_matches(request, scan_uuid):
    """Return paginated authorized proposals with immutable IDs for deleted members."""
    scan = _scan(request, scan_uuid)
    if scan.purpose == "exact":
        hashes.recover_group(scan.workgroup_id)
        scan.refresh_from_db()
    else:
        service.refresh_scan(scan)
    rows = _visible_proposals(request, scan, request.GET.get("status", "pending"))
    if request.GET.get("type") in KIND_LABELS:
        rows = rows.filter(kind=request.GET["type"])
    page = _proposal_page(rows, request.GET.get("page"))
    return JsonResponse(
        {
            "count": page.paginator.count,
            "page": page.number,
            "results": [
                {
                    "uuid": str(row.uuid),
                    "kind": row.kind,
                    "status": row.status,
                    "metadata_difference_count": row.summary["difference_count"],
                    "perfect_metadata_match": row.summary["perfect"],
                    "suggested_keep_id": row.summary["keep_id"],
                    "evidence": row.evidence,
                    "members": [
                        {
                            "id": member.original_id,
                            "exists": member.mediafile_id is not None,
                            "name": member.snapshot.get("name", ""),
                        }
                        for member in row.members.all()
                    ],
                    "review_url": reverse("caidapp:duplicate_review", args=[row.uuid]),
                }
                for row in page
            ],
        }
    )


def _next_proposal(request, scan, kept_id=None):
    rows = _visible_proposals(request, scan, "pending")
    next_row = rows.filter(members__mediafile_id=kept_id).first() if kept_id else None
    if next_row is None:
        next_page = _proposal_page(rows, 1)
        next_row = next_page[0] if next_page else None
    if next_row:
        url = reverse("caidapp:duplicate_review", args=[next_row.uuid])
        if kept_id and next_row.members.filter(mediafile_id=kept_id).exists():
            url += f"?keep={kept_id}"
        return redirect(url)
    if scan.workgroup_id:
        return redirect("caidapp:duplicates")
    return redirect("caidapp:duplicate_scan", scan_uuid=scan.uuid)


@login_required
@require_http_methods(["GET", "POST"])
def review(request, proposal_uuid):
    """Review directional deletion with explicit retained-file edits and no observation transfers."""
    proposal = _proposal(request, proposal_uuid)
    is_json = request.content_type == "application/json"
    if request.method == "POST":
        try:
            data = json.loads(request.body) if is_json else request.POST
            if is_json and not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            if data.get("action") == "dismiss":
                service.dismiss_proposal(proposal.pk, request.user.caiduser)
                kept_id = None
            else:
                if "choices" in data or any(key.startswith("choice_") for key in data):
                    raise ValueError("Metadata merging is no longer used. Reload the deletion preview.")
                edits = (
                    data.get("edits", {})
                    if is_json
                    else {
                        "take": {
                            field.removeprefix("take_"): int(value)
                            for field, value in data.items()
                            if field.startswith("take_") and value
                        }
                    }
                )
                if not is_json and "note" in data:
                    edits["note"] = data["note"]
                kept_id = service.apply_merge(proposal.pk, request.user.caiduser, data.get("token", ""), edits)
            if is_json:
                return JsonResponse({"status": "ok", "kept_id": kept_id})
            messages.success(
                request,
                (
                    "Proposal dismissed."
                    if kept_id is None
                    else "Marked media files removed. Your selected edits to the retained media file were saved."
                ),
            )
            return _next_proposal(request, proposal.scan, kept_id)
        except (ValueError, TypeError, KeyError, signing.BadSignature, OSError) as exc:
            if is_json:
                return JsonResponse({"error": str(exc)}, status=409)
            messages.error(request, str(exc))
            return redirect(request.get_full_path())
    members = service.proposal_media(proposal, request.user.caiduser)
    plan, token, error = None, "", ""
    keep_id, remove_ids = None, []
    try:
        if proposal.status == "pending" and len(members) >= 2:
            preferred = service.preferred_media(members)
            keep_id = int(request.GET.get("keep", preferred.pk))
            remove_ids = (
                [int(key) for key in request.GET.getlist("remove")]
                if "subset" in request.GET
                else [next(media.pk for media in members if media.pk != keep_id)]
            )
            plan = service.build_plan(proposal, request.user.caiduser, keep_id, remove_ids)
            token = service.preview_token(proposal, request.user.caiduser, plan)
    except (ValueError, KeyError, OSError, StopIteration) as exc:
        error = str(exc)
    by_id = {media.pk: media for media in members}
    chosen = (
        ([by_id[key] for key in remove_ids if key in by_id] + [by_id[keep_id]]) if keep_id in by_id else members[:2]
    )
    cards = []
    for media in chosen:
        real = media.observations.filter(is_no_detection_placeholder=False).select_related("identity", "taxon")
        identities = ", ".join(sorted({str(obs.identity) for obs in real if obs.identity_id}))
        taxa = ", ".join(sorted({str(obs.taxon) for obs in real if obs.taxon_id}))
        differences = {warning["field"] for warning in plan["warnings"]} if plan else set()
        facts = []
        for field, label, value in [
            ("identities", "Identities", identities or "No assigned identity"),
            ("taxa", "Taxa", taxa or "No taxon"),
            ("captured_at", "Captured", media.captured_at),
            ("locality", "Locality", media.locality),
            ("location", "Location", media.location),
            ("sequence", "Sequence", media.sequence),
            ("note", "Note", media.note),
        ]:
            facts.append(
                {
                    "field": field,
                    "label": label,
                    "value": value,
                    "different": field in differences,
                    "options": (
                        [
                            option
                            for option in plan["transfer_options"]
                            if media.pk == keep_id and option["field"] == field
                        ]
                        if plan
                        else []
                    ),
                    "manual": field in {"identities", "taxa"} and field in differences,
                }
            )
        cards.append(
            {
                "media": media,
                "role": ("keep" if media.pk == keep_id else "delete") if plan else "candidate",
                "facts": facts,
            }
        )
    swap_url = ""
    if plan and len(remove_ids) == 1:
        swap_url = (
            reverse("caidapp:duplicate_review", args=[proposal.uuid])
            + f"?keep={remove_ids[0]}&subset=1&remove={keep_id}"
        )
    if request.GET.get("format") == "json":
        return JsonResponse(
            {
                "uuid": str(proposal.uuid),
                "status": proposal.status,
                "kind": proposal.kind,
                "error": error,
                "token": token,
                "plan": (
                    {key: plan[key] for key in ("keep_id", "remove_ids", "warnings", "transfer_options")}
                    if plan
                    else None
                ),
            }
        )
    return render(
        request,
        "caidapp/duplicate_review.html",
        {
            "proposal": proposal,
            "members": cards,
            "all_members": members,
            "identity_merges": service.identity_merge_candidates(proposal, chosen),
            "plan": plan,
            "token": token,
            "error": error,
            "swap_url": swap_url,
            "selected_keep": keep_id,
            "retained": by_id.get(keep_id),
            "selected_remove": remove_ids[0] if len(remove_ids) == 1 else None,
            "evidence_json": proposal.evidence,
            "kind_label": KIND_LABELS.get(proposal.kind, proposal.kind),
        },
    )
