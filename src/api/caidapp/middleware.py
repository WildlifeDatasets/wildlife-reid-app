from django.contrib.auth import get_user_model
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
