"""Bounding-box unions for media previews without changing observation-level data."""

import math

from caidapp.models import AnimalObservation


def mediafile_bbox_unions(mediafile_ids, by_identity=False):
    """Return normalized center/size unions of valid image boxes, optionally grouped by identity.

    Video boxes lack a frame position and cannot safely be combined across time.
    Placeholders and boxes outside the media area do not contribute to a union.
    """
    bounds = {}
    observations = AnimalObservation.objects.filter(
        mediafile_id__in=mediafile_ids, mediafile__media_type="image", is_no_detection_placeholder=False
    ).values_list("mediafile_id", "identity_id", "bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height")
    for mediafile_id, identity_id, x, y, width, height in observations:
        if not all(value is not None and math.isfinite(value) for value in (x, y, width, height)):
            continue
        if width <= 0 or height <= 0:
            continue
        left, top = max(0, x - width / 2), max(0, y - height / 2)
        right, bottom = min(1, x + width / 2), min(1, y + height / 2)
        if left >= right or top >= bottom:
            continue
        key = (mediafile_id, identity_id) if by_identity else mediafile_id
        if key in bounds:
            old = bounds[key]
            bounds[key] = (min(left, old[0]), min(top, old[1]), max(right, old[2]), max(bottom, old[3]))
        else:
            bounds[key] = (left, top, right, bottom)
    return {
        key: [(left + right) / 2, (top + bottom) / 2, right - left, bottom - top]
        for key, (left, top, right, bottom) in bounds.items()
    }
