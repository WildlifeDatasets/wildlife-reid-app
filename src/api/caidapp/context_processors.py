from .models import NotificationRecipient


def notification_status(request):
    """Expose whether the signed-in user has an unread notification."""
    if not request.user.is_authenticated or not hasattr(request.user, "caiduser"):
        return {"has_unread_notifications": False}
    return {
        "has_unread_notifications": NotificationRecipient.objects.filter(
            user=request.user.caiduser,
            read=False,
        ).exists()
    }
