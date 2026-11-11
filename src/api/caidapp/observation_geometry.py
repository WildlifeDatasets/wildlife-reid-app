"""Shared query predicate for observations with usable bbox geometry."""

from django.db.models import Q

HAS_BBOX = Q(
    is_no_detection_placeholder=False,
    bbox_x_center__gte=0,
    bbox_x_center__lte=1,
    bbox_y_center__gte=0,
    bbox_y_center__lte=1,
    bbox_width__gt=0,
    bbox_width__lte=1,
    bbox_height__gt=0,
    bbox_height__lte=1,
)
