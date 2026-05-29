from __future__ import annotations

from typing import Any

import plotly.express as px
import plotly.graph_objects as go
from django.db.models import Count, Q, QuerySet
from django.db.models.functions import TruncMonth
from django.utils import timezone

from .. import models


def _scope_mediafiles_queryset(caiduser: models.CaIDUser) -> QuerySet[models.MediaFile]:
    """Return mediafiles visible on the home dashboard for one user."""
    if caiduser.workgroup:
        return models.MediaFile.objects.filter(parent__owner__workgroup=caiduser.workgroup)
    return models.MediaFile.objects.filter(parent__owner=caiduser)


def _serialize_monthly_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized = []
    for row in rows:
        month = row.get("month")
        serialized.append(
            {
                **row,
                "month": month.isoformat() if month else None,
            }
        )
    return serialized


def build_home_dashboard_payload(mediafiles: QuerySet[models.MediaFile]) -> dict[str, Any]:
    """Build a serializable snapshot payload for the home dashboard."""
    mediafile_ids = mediafiles.values_list("id", flat=True)

    total_mediafiles = mediafiles.count()
    images_count = mediafiles.filter(media_type="image").count()
    videos_count = mediafiles.filter(media_type="video").count()
    verified_taxon_count = mediafiles.filter(taxon_verified=True).count()
    identified_count = mediafiles.filter(identity__isnull=False).count()

    media_type_counts = list(
        mediafiles.values("media_type").annotate(count=Count("id")).order_by("media_type")
    )
    monthly_uploads = list(
        mediafiles.annotate(month=TruncMonth("parent__uploaded_at"))
        .values("month")
        .annotate(count=Count("id"))
        .order_by("month")
    )
    monthly_by_type = list(
        mediafiles.annotate(month=TruncMonth("parent__uploaded_at"))
        .values("month", "media_type")
        .annotate(count=Count("id"))
        .order_by("month", "media_type")
    )
    top_taxons = list(
        models.AnimalObservation.objects.filter(mediafile_id__in=mediafile_ids, taxon__isnull=False)
        .exclude(
            Q(taxon__name="Animalia", taxon_verified=False) |
            Q(taxon__name="Not classified") |
            Q(taxon__name="Nothing")
        )
        .values("taxon__name")
        .annotate(count=Count("mediafile_id", distinct=True))
        .order_by("-count", "taxon__name")[:12]
    )

    return {
        "summary": {
            "total_mediafiles": total_mediafiles,
            "images_count": images_count,
            "videos_count": videos_count,
            "verified_taxon_count": verified_taxon_count,
            "unverified_taxon_count": max(total_mediafiles - verified_taxon_count, 0),
            "identified_count": identified_count,
            "unidentified_count": max(total_mediafiles - identified_count, 0),
        },
        "media_type_counts": media_type_counts,
        "monthly_uploads": _serialize_monthly_rows(monthly_uploads),
        "monthly_by_type": _serialize_monthly_rows(monthly_by_type),
        "top_taxons": top_taxons,
    }


def refresh_workgroup_home_dashboard_snapshot(workgroup: models.WorkGroup) -> models.HomeDashboardSnapshot:
    """Recompute and persist a home dashboard snapshot for one workgroup."""
    mediafiles = models.MediaFile.objects.filter(parent__owner__workgroup=workgroup)
    payload = build_home_dashboard_payload(mediafiles)
    snapshot, _ = models.HomeDashboardSnapshot.objects.update_or_create(
        workgroup=workgroup,
        defaults={"payload": payload},
    )
    return snapshot


def _card_figure_html(fig) -> str:
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=24, r=16, t=48, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        height=320,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _empty_chart_html(title: str, text: str) -> str:
    fig = go.Figure()
    fig.update_layout(title=title)
    fig.add_annotation(text=text, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _card_figure_html(fig)


def _render_media_type_chart(payload: dict[str, Any]) -> str:
    rows = payload["media_type_counts"]
    if not rows:
        return _empty_chart_html("Media Type Split", "No media files available yet.")
    fig = px.pie(
        rows,
        names="media_type",
        values="count",
        hole=0.55,
        title="Media Type Split",
        color="media_type",
        color_discrete_map={"image": "#2a9d8f", "video": "#e76f51"},
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _card_figure_html(fig)


def _render_monthly_uploads_chart(payload: dict[str, Any]) -> str:
    rows = payload["monthly_uploads"]
    if not rows:
        return _empty_chart_html("Uploads by Month", "No uploaded media files available yet.")
    fig = px.bar(
        rows,
        x="month",
        y="count",
        title="Uploads by Month",
        labels={"month": "Month", "count": "Media files"},
    )
    fig.update_traces(marker_color="#457b9d")
    return _card_figure_html(fig)


def _render_monthly_by_type_chart(payload: dict[str, Any]) -> str:
    rows = payload["monthly_by_type"]
    if not rows:
        return _empty_chart_html("Monthly Images vs Videos", "No uploaded media files available yet.")
    fig = px.bar(
        rows,
        x="month",
        y="count",
        color="media_type",
        barmode="stack",
        title="Monthly Images vs Videos",
        labels={"month": "Month", "count": "Media files", "media_type": "Type"},
        color_discrete_map={"image": "#2a9d8f", "video": "#e76f51"},
    )
    return _card_figure_html(fig)


def _render_top_taxons_chart(payload: dict[str, Any]) -> str:
    rows = payload["top_taxons"]
    if not rows:
        return _empty_chart_html("Top Taxa", "No taxon annotations available yet.")
    chart_rows = list(reversed(rows))
    fig = px.bar(
        chart_rows,
        x="count",
        y="taxon__name",
        orientation="h",
        title="Top Taxa by Media Files",
        labels={"count": "Media files", "taxon__name": "Taxon"},
    )
    fig.update_traces(marker_color="#6c8f3d")
    return _card_figure_html(fig)


def _render_quality_chart(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    rows = [
        {"group": "Verified taxon", "count": summary["verified_taxon_count"]},
        {"group": "Unverified taxon", "count": summary["unverified_taxon_count"]},
        {"group": "With identity", "count": summary["identified_count"]},
        {"group": "Without identity", "count": summary["unidentified_count"]},
    ]
    fig = px.bar(
        rows,
        x="group",
        y="count",
        color="group",
        title="Annotation Coverage",
        labels={"group": "", "count": "Media files"},
        color_discrete_sequence=["#588157", "#bc4749", "#4361ee", "#adb5bd"],
    )
    return _card_figure_html(fig)


def render_home_dashboard_context(
    caiduser: models.CaIDUser,
    *,
    prefer_snapshot: bool = True,
) -> dict[str, Any]:
    """Prepare rendered dashboard context for the home page."""
    snapshot = None
    payload = None
    snapshot_source = "live"

    if prefer_snapshot and caiduser.workgroup:
        snapshot = models.HomeDashboardSnapshot.objects.filter(workgroup=caiduser.workgroup).first()
        if snapshot:
            payload = snapshot.payload
            snapshot_source = "daily snapshot"

    if payload is None:
        payload = build_home_dashboard_payload(_scope_mediafiles_queryset(caiduser))

    return {
        "home_dashboard_updated_at": snapshot.updated_at if snapshot else timezone.now(),
        "home_dashboard_source": snapshot_source,
        "home_dashboard_charts": {
            "monthly_by_type_html": _render_monthly_by_type_chart(payload),
            "top_taxons_html": _render_top_taxons_chart(payload),
        },
    }
