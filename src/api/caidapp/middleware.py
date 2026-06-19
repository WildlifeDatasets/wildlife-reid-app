from django.contrib.auth import get_user_model
from django.conf import settings
from django.core.exceptions import TooManyFilesSent
from django.http import JsonResponse
from django.shortcuts import render
from django.template.loader import render_to_string
from django.urls import reverse_lazy
from django.utils.deprecation import MiddlewareMixin

User = get_user_model()


IDENTITY_WRITE_ROUTES = {
    "import_identities",
    "individual_identity_create",
    "individual_identity_update",
    "delete_individual_identity",
    "manual_identification",
    "manual_identification_mediafile",
    "get_individual_identity",
    "get_individual_identity_by_media_file",
    "set_individual_identity",
    "remove_foridentification",
    "merge_identities",
    "merge_identities_no_preview",
    "merge_selected_identities",
    "refresh_merge_identities_suggestions",
    "clear_identity_suggestions",
    "accept_identification_outlier_suggestion",
    "apply_identity_code_suggestion",
    "apply_selected_identity_code_suggestions",
    "toggle_identity_representative",
}

TAXON_WRITE_ROUTES = {
    "update_taxon",
    "add_taxon",
    "missing_taxon_annotation",
    "missing_taxon_annotation_in_uploadedarchive",
    "missing_taxon_annotation_for_mediafile",
    "verify_taxa",
    "taxons_on_page_are_overviewed",
    "confirm_prediction",
}

OTHER_WRITE_ROUTES = {
    "delete_mediafile",
    "observation_delete",
    "update_locality",
    "delete_locality",
    "new_album",
    "album_update",
    "delete_album",
    "new_upload",
    "upload_archive",
    "workgroup_invitation",
    "workgroup_invitation_accept",
    "workgroup_invitation_decline",
}

# Legacy actions that modify data directly from a link instead of using POST.
GET_MUTATION_ROUTES = {
    "delete_individual_identity",
    "set_individual_identity",
    "remove_foridentification",
    "merge_identities_no_preview",
    "apply_identity_code_suggestion",
    "toggle_identity_representative",
    "taxons_on_page_are_overviewed",
    "confirm_prediction",
    "delete_mediafile",
    "delete_album",
}


def _permission_denied_response(request, message):
    return render(
        request,
        "caidapp/edit_permission_denied.html",
        {"permission_message": message},
        status=403,
    )


def _posted_observation_changes(request):
    """Return protected areas changed by an observation inline formset."""
    total_forms = int(request.POST.get("observations-TOTAL_FORMS", 0) or 0)
    if not total_forms:
        return set()

    from .models import AnimalObservation

    ids = [request.POST.get(f"observations-{index}-id") for index in range(total_forms)]
    observations = {
        str(observation.pk): observation
        for observation in AnimalObservation.objects.filter(pk__in=[value for value in ids if value])
    }
    changed = set()
    for index in range(total_forms):
        prefix = f"observations-{index}-"
        observation = observations.get(str(request.POST.get(prefix + "id") or ""))
        posted_taxon = request.POST.get(prefix + "taxon") or ""
        posted_identity = request.POST.get(prefix + "identity") or ""
        posted_taxon_verified = prefix + "taxon_verified" in request.POST
        posted_representative = prefix + "identity_is_representative" in request.POST
        deleting = bool(request.POST.get(prefix + "DELETE"))

        if observation is None:
            if posted_taxon or posted_taxon_verified:
                changed.add("taxon")
            if posted_identity or posted_representative:
                changed.add("identity")
            continue

        if deleting:
            if observation.taxon_id or observation.taxon_verified:
                changed.add("taxon")
            if observation.identity_id or observation.identity_is_representative:
                changed.add("identity")
        if posted_taxon != (str(observation.taxon_id) if observation.taxon_id else ""):
            changed.add("taxon")
        if posted_taxon_verified != observation.taxon_verified:
            changed.add("taxon")
        if posted_identity != (str(observation.identity_id) if observation.identity_id else ""):
            changed.add("identity")
        if posted_representative != observation.identity_is_representative:
            changed.add("identity")
    return changed


class RecordEditPermissionMiddleware(MiddlewareMixin):
    """Enforce workgroup record-edit permissions before write views run."""

    unsafe_methods = {"POST", "PUT", "PATCH", "DELETE"}

    def process_view(self, request, view_func, view_args, view_kwargs):
        if not request.user.is_authenticated or not hasattr(request.user, "caiduser"):
            return None

        caiduser = request.user.caiduser
        route_name = request.resolver_match.url_name if request.resolver_match else None
        explicit_area = None
        if route_name in IDENTITY_WRITE_ROUTES:
            explicit_area = "identity"
        elif route_name in TAXON_WRITE_ROUTES:
            explicit_area = "taxon"
        elif route_name in OTHER_WRITE_ROUTES:
            explicit_area = "other"

        route_performs_write = request.method in self.unsafe_methods or route_name in GET_MUTATION_ROUTES
        if explicit_area and route_performs_write and not caiduser.can_edit(explicit_area):
            return _permission_denied_response(
                request,
                f"You do not have permission to edit {explicit_area} records.",
            )

        if request.method not in self.unsafe_methods:
            return None
        if caiduser.workgroup_admin:
            return None
        if caiduser.is_observer:
            return _permission_denied_response(request, "Read-only observers cannot modify records.")

        changed_areas = _posted_observation_changes(request)
        for area in changed_areas:
            if not caiduser.can_edit(area):
                return _permission_denied_response(
                    request,
                    f"You do not have permission to edit {area} records.",
                )

        if explicit_area is None and not caiduser.can_edit("other"):
            return _permission_denied_response(request, "You do not have permission to edit these records.")
        return None

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
