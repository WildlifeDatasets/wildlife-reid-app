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
