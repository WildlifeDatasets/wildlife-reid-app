import logging

from django import template

logger = logging.getLogger(__name__)
register = template.Library()


@register.filter
def attr(obj, attr_name):
    """Get attribute value by name from an object, return empty string if not found."""
    return getattr(obj, attr_name, "")
