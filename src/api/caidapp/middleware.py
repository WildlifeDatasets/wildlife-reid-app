from django.contrib.auth import get_user_model
from django.conf import settings
from django.core.exceptions import TooManyFilesSent
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.urls import reverse_lazy
from django.utils.deprecation import MiddlewareMixin

User = get_user_model()

# class ImpersonateMiddleware(MiddlewareMixin):
#     def process_request(self, request):
#         if request.user.is_authenticated and request.user.is_superuser:
#             impersonate_user_id = request.session.get('impersonate_user_id')
#             if impersonate_user_id:
#                 try:
#                     user = User.objects.get(id=impersonate_user_id)
#                     request.user = user
#                 except User.DoesNotExist:
#                     pass


class ImpersonateMiddleware(MiddlewareMixin):
    def process_request(self, request):
        """Impersonate user."""
        if request.user.is_authenticated and request.user.is_superuser:
            impersonate_user_id = request.session.get("impersonate_user_id")
            if impersonate_user_id:
                try:
                    user = User.objects.get(id=impersonate_user_id)
                    request.user = user
                except User.DoesNotExist:
                    pass
        elif request.user.is_authenticated and "impersonate_user_id" in request.session:
            try:
                user = User.objects.get(id=request.session["impersonate_user_id"])
                request.user = user
            except User.DoesNotExist:
                pass


class UploadLimitExceededMiddleware(MiddlewareMixin):
    """Return a user-facing upload error when Django rejects too many multipart files."""

    def process_exception(self, request, exception):
        if not isinstance(exception, TooManyFilesSent):
            return None
        if request.path != str(reverse_lazy("caidapp:new_upload")):
            return None

        limit = settings.DATA_UPLOAD_MAX_NUMBER_FILES
        html = render_to_string(
            "caidapp/partial_message.html",
            {
                "headline": "Upload failed",
                "text": (
                    f"You selected more than {limit} individual files. "
                    "Please upload a ZIP archive instead, or split the upload into smaller batches."
                ),
                "next": reverse_lazy("caidapp:new_upload"),
                "next_text": "Back to upload",
            },
            request=request,
        )
        return JsonResponse({"ok": False, "html": html}, status=400)
