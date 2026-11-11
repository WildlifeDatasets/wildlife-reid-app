"""Shared, safely escaped media-file metadata for hover previews and a copyable modal."""

import json

from django import template
from django.core.serializers.json import DjangoJSONEncoder
from django.urls import reverse
from django.utils import formats, timezone

register = template.Library()


@register.inclusion_tag("caidapp/includes/mediafile_info.html")
def mediafile_info(mediafile, request=None):
    """Render one metadata trigger, preserving privacy and reusing prefetched observations."""
    private = bool(request and getattr(request, "session", {}).get("private_mode"))
    cache = getattr(mediafile, "_info_payloads", {})
    if private not in cache:
        observations = mediafile.observations.all()
        if "observations" not in getattr(mediafile, "_prefetched_objects_cache", {}):
            observations = observations.select_related("taxon", "identity")
        real = [obs for obs in observations if not obs.is_no_detection_placeholder]
        taxa = ", ".join(sorted({str(obs.taxon) for obs in real if obs.taxon_id})) or "—"
        identities = ", ".join(sorted({str(obs.identity) for obs in real if obs.identity_id})) or "—"
        captured = mediafile.captured_at
        if captured and timezone.is_aware(captured):
            captured = timezone.localtime(captured)
        locality = mediafile.locality
        place = (locality.visible_name if private else locality.name) if locality else "—"
        rows = [
            {"label": label, "value": str(value) if value is not None and value != "" else "—"}
            for label, value in [
                ("Media file", mediafile.original_filename or mediafile.pk),
                ("Upload", mediafile.parent.name if mediafile.parent else None),
                ("ID", mediafile.pk),
                ("Type", mediafile.get_media_type_display()),
                ("Captured", formats.date_format(captured, "DATETIME_FORMAT") if captured else None),
                ("Taxa", taxa),
                ("Identities", identities),
                ("Locality", place),
                ("Coordinates", "Hidden in private mode" if private else mediafile.effective_location),
                ("Sequence", mediafile.sequence),
                ("Note", mediafile.note),
                ("Detected objects", len(real)),
            ]
        ]
        for obs in real:
            box = ", ".join(
                str(getattr(obs, key)) for key in ("bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height")
            )
            rows.append(
                {
                    "label": f"Observation #{obs.pk}",
                    "value": f"Taxon: {obs.taxon or '—'}; Identity: {obs.identity or '—'}; Bbox (cx, cy, w, h): {box}",
                }
            )
        metadata = (
            "Hidden in private mode"
            if private
            else json.dumps(mediafile.metadata_json or {}, cls=DjangoJSONEncoder, ensure_ascii=False, indent=2)
        )
        summary = "\n".join(f"{row['label']}: {row['value']}" for row in rows[:8])
        cache[private] = {
            "payload": json.dumps({"rows": rows, "metadata": metadata}, ensure_ascii=False),
            "summary": summary,
        }
        mediafile._info_payloads = cache
    return {
        **cache[private],
        "mediafile_id": mediafile.pk,
        "detail_url": reverse("caidapp:media_file_update", args=[mediafile.pk]),
    }
