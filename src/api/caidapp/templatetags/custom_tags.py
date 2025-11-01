import logging

from django import template

logger = logging.getLogger(__name__)
register = template.Library()


@register.filter
def attr(obj, attr_name):
    return getattr(obj, attr_name, "")
