import math

from django import template

register = template.Library()


@register.filter
def mul(value, arg):
    """Multiplies the value by the argument."""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return ""


@register.filter
def floor(value):
    """Floors the float to the nearest lower integer."""
    try:
        return math.floor(float(value))
    except (ValueError, TypeError):
        return ""


@register.filter
def chunk(queryset, size):
    """Split a queryset into chunks of the specified size."""
    for i in range(0, len(queryset), size):
        yield queryset[i:i+size]


@register.filter
def order_by(queryset, field):
    return queryset.order_by(field)