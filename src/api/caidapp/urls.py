from django.urls import include, path

# from rest_framework import routers
from django.views.generic import DetailView

from . import (
    models,
    views,
    views_admin,
    views_general,
    views_locality,
    views_mediafile,
    views_uploads,
)


def trigger_error(request):
    """Trigger an error for Sentry testing purposes."""
    division_by_zero = 1 / 0
    return division_by_zero


# router = routers.DefaultRouter()
# router.register(r"localities", views.LocalitiesViewSet)


app_name = "caidapp"
urlpatterns = [
    path("", views.login, name="index"),
    path("djangologin/", views.MyLoginView.as_view(), name="djangologin"),
    path("logout/", views.logout_view, name="logout_view"),
    path("user_settings/", views.CaIDUserSettingsView.as_view(), name="update_caiduser"),
    # path("rest_api/", include(router.urls)), # not used any more
    path("upload/", views.upload_archive, name="upload_archive"),
    # path("user_settings/", views.update_caiduser, name="update_caiduser"),
    path(
        "upload/contains_single_taxon/",
        views.upload_archive,
        {"contains_single_taxon": True},
        name="upload_archive_contains_single_taxon",
    ),
    path(
        "upload/contains_identities/",
        views.upload_archive,
        {"contains_identities": True, "contains_single_taxon": True},
        name="upload_archive_contains_identities",
    ),
    # path("login/", TemplateView.as_view(template_name="caidapp/login.html"), name="login"),
    # Uploads
    path("uploads/", views.uploads_species, name="uploads"),
    path("uploads_identities/", views.uploads_identities, name="uploads_identities"),
    path("uploads_known_identities/", views.uploads_known_identities, name="uploads_known_identities"),
    path(
        "uploadedarchive_mediafiles/<int:uploadedarchive_id>",
        # views.uploadedarchive_mediafiles,
        views.media_files_update,
        name="uploadedarchive_mediafiles",
    ),
    path(
        "uploadedarchive_detail/<int:uploadedarchive_id>",
        # views.uploadedarchive_mediafiles,
        views_uploads.uploadedarchive_detail,
        name="uploadedarchive_detail",
    ),
    # path("media_files/", views.media_files, name="media_files"),
    path(
        "<int:uploadedarchive_id>/delete_upload/<str:next_page>/",
        views.delete_upload,
        name="delete_upload",
    ),
    path(
        "update_uploadedarchive/<int:uploadedarchive_id>/",
        views.update_uploadedarchive,
        name="update_uploadedarchive",
    ),
    path(
        "<int:uploadedarchive_id>/run_processing/",
        views.run_taxon_classification,
        name="run_processing",
    ),
    path(
        "<int:uploadedarchive_id>/run_taxon_classification_force_init/",
        views.run_taxon_classification_force_init,
        name="run_taxon_classification_force_init",
    ),
    path("refresh_data/", views.refresh_data, name="refresh_data"),
    # Localities
    path("manage_localities/", views_locality.manage_localities, name="manage_localities"),
    path("delete_locality/<int:locality_id>/", views_locality.delete_locality, name="delete_locality"),
    path(
        "update_locality/<int:locality_id>/",
        views_locality.update_locality,
        name="update_locality",
    ),
    path(
        "update_locality/",
        views_locality.update_locality,
        name="update_locality",
    ),
    path("taxon/<int:taxon_id>", views.media_files_update, name="taxon"),
    # Media Files
    path("media_files/", views.media_files_update, name="media_files"),
    path(
        "mediafile/<int:pk>/update/",
        views_mediafile.MediaFileUpdateView.as_view(),
        name="media_file_update",
    ),
    # path(
    #     "media_file_update/<int:media_file_id>/",
    #     views_mediafile.media_file_update,
    #     name="media_file_update",
    # ),
    path(
        "media_files/locality/<str:locality_hash>",
        views.media_files_update,
        name="media_files_locality",
    ),
    path(
        "representative_mediafiles/",
        lambda request: views.media_files_update(request, identity_is_representative=True),
        name="representative_mediafiles",
    ),
    path(
        "individual_identity_mediafiles/<int:individual_identity_id>",
        views.media_files_update,
        name="individual_identity_mediafiles",
    ),
    path(
        "delete_mediafile/<int:mediafile_id>/",
        views.delete_mediafile,
        name="delete_mediafile",
    ),
    path(
        "observation/<int:pk>/delete/",
        views_mediafile.ObservationDeleteView.as_view(),
        name="observation_delete",
    ),
    # Albums
    path("albums/", views.albums, name="albums"),
    path("album/<str:album_hash>", views.media_files_update, name="album"),
    path("album_update/<str:album_hash>/", views.album_update, name="album_update"),
    path("delete_album/<str:album_hash>/", views.delete_album, name="delete_album"),
    path("new_album/", views.new_album, name="new_album"),
    # Identity
    path("identities/export_csv", views.export_identities_csv, name="export_identities_csv"),
    path("identities/export_xlsx", views.export_identities_xlsx, name="export_identities_xlsx"),
    path("identities/import", views.import_identities_view, name="import_identities"),
    path(
        "individual_identity_create/",
        views.individual_identity_create,
        name="individual_identity_create",
    ),
    path(
        "individual_identity_create/media_file/<int:media_file_id>",
        views.individual_identity_create,
        name="individual_identity_create",
    ),
    # path(
    #     "individual_identity_update/<int:individual_identity_id>",
    #     views.individual_identity_update,
    #     name="individual_identity_update",
    # ),
    path(
        "individual_identity_update/<int:pk>",
        views.IndividualIdentityUpdateView.as_view(),
        name="individual_identity_update",
    ),
    path(
        "individual_identity/share/<str:identity_hash>",
        views.shared_individual_identity_view,
        name="shared_individual_identity",
    ),
    path(
        "delete_individual_identity/<int:individual_identity_id>",
        views.delete_individual_identity,
        name="delete_individual_identity",
    ),
    # path(
    #     "individual_identities/",
    #     views.individual_identities,
    #     name="individual_identities",
    # ),
    path(
        "individual_identities/",
        views.IdentityListView.as_view(),
        name="individual_identities",
    ),
    path("dash_identities/", views.dash_identities, name="dash_identities"),
    path("init_identification/", views.init_identification_view, name="init_identification"),
    path("train_identification/", views.train_identification, name="train_identification"),
    path("stop_init_identification/", views.stop_init_identification, name="stop_init_identification"),
    path(
        "run_identification/<int:uploadedarchive_id>/",
        views.run_identification_view,
        name="run_identification",
    ),
    path(
        "run_identification_on_unindentified/",
        views.run_identification_on_unidentified,
        name="run_identification_on_unidentified",
    ),
    path(
        "get_individual_identity_zoomed/<int:foridentification_id>/<int:reid_suggestion_id>",
        views.get_individual_identity_zoomed,
        name="get_individual_identity_zoomed",
    ),
    path(
        "get_individual_identity_zoomed_by_identity/<int:foridentification_id>/<int:identity_id>",
        views.get_individual_identity_zoomed_by_identity,
        name="get_individual_identity_zoomed_by_identity",
    ),
    path(
        "get_individual_identity_zoomed_paired_points/<int:foridentification_id>/<int:reid_suggestion_id>",
        views.get_individual_identity_zoomed_paired_points,
        name="get_individual_identity_zoomed_paired_points",
    ),
    path(
        "get_individual_identity/",
        views.get_individual_identity_from_foridentification,
        name="get_individual_identity",
    ),
    path(
        "get_individual_identity/<int:foridentification_id>",
        views.get_individual_identity_from_foridentification,
        name="get_individual_identity",
    ),
    path(
        "get_individual_identity/media_file/<int:media_file_id>",
        views.get_individual_identity_from_foridentification,
        name="get_individual_identity_by_media_file",
    ),
    path(
        "remove_foridentification/<int:foridentification_id>",
        views.remove_foridentification,
        name="remove_foridentification",
    ),
    path(
        "set_individual_identity/" + "<int:mediafiles_for_identification_id>/<int:individual_identity_id>",
        views.set_individual_identity,
        name="set_individual_identity",
    ),
    path(
        "not_identified_mediafiles/",
        views.not_identified_mediafiles,
        name="not_identified_mediafiles",
    ),
    path("show_log/", views.show_log, name="show_log"),
    path("show_taxons/", views.show_taxons, name="show_taxons"),
    # path(
    #     "workgroup_update/<str:workgroup_hash>/",
    #     views.workgroup_update,
    #     name="workgroup_update",
    # ),
    path(
        "missing_taxon_annotation",
        views_mediafile.start_missing_taxon_annotation,
        name="missing_taxon_annotation",
    ),
    path(
        "missing_taxon_annotation/uploaded_archive/<int:uploaded_archive_id>",
        views_mediafile.start_missing_taxon_annotation,
        name="missing_taxon_annotation_in_uploadedarchive",
    ),
    path(
        "missing_taxon_annotation_for_mediafile/mediafile/<int:pk>",
        # views_mediafile.missing_taxon_annotation_for_mediafile,
        views_mediafile.MediaFileGetMissingTaxonView.as_view(),
        name="missing_taxon_annotation_for_mediafile",
    ),
    # path(
    #     "missing_taxon_annotation_for_mf/mediafile/<int:mediafile_id>/uploaded_archive/<int:uploaded_archive_id>",
    #     views_mediafile.missing_taxon_annotation_for_mediafile,
    #     name="missing_taxon_annotation_for_mediafile_in_uploadedarchive",
    # ),
    path("sample_data/", views.sample_data, name="sample_data"),
    path("cloud_import_preview/", views.cloud_import_preview_view, name="cloud_import_preview"),
    path("do_cloud_import/", views.do_cloud_import_view, name="do_cloud_import"),
    path(
        "do_cloud_import_single_taxon/",
        views.do_cloud_import_view_single_taxon,
        name="do_cloud_import_single_taxon",
    ),
    path(
        "do_cloud_import_single_taxon_known_identities/",
        views.do_cloud_import_view_single_taxon_known_identities,
        name="do_cloud_import_single_taxon_known_identities",
    ),
    path("break_cloud_import/", views.break_cloud_import_view, name="break_cloud_import"),
    path(
        "download_csv_for_mediafiles/",
        views.download_csv_for_mediafiles_view,
        name="download_csv_for_mediafiles",
    ),
    path(
        "download_zip_for_mediafiles/",
        views.download_zip_for_mediafiles_view,
        name="download_zip_for_mediafiles",
    ),
    path(
        "download_zip_for_mediafiles/<int:uploadedarchive_id>/",
        views.download_zip_for_mediafiles_view,
        name="download_zip_for_mediafiles",
    ),
    path(
        "download_xlsx_for_mediafiles/",
        views.download_xlsx_for_mediafiles_view,
        name="download_xlsx_for_mediafiles",
    ),
    path(
        "download_uploadedarchive_csv/<int:uploadedarchive_id>",
        views.download_csv_for_mediafiles_view,
        name="download_uploadedarchive_csv",
    ),
    path(
        "download_uploadedarchive_xlsx/<int:uploadedarchive_id>",
        views.download_xlsx_for_mediafiles_view,
        name="download_uploadedarchive_xlsx",
    ),
    path(
        "download_xlsx_for_mediafiles_NDOP/",
        views.download_xlsx_for_mediafiles_view_NDOP,
        name="download_xlsx_for_mediafiles_NDOP",
    ),
    path("check_zip_status/<str:task_id>/", views.check_zip_status_view, name="check_zip_status"),
    path("mediafiles_stats/", views.mediafiles_stats_view, name="mediafiles_stats"),
    path(
        "change_mediafiles_datetime",
        views.change_mediafiles_datetime,
        name="change_mediafiles_datetime",
    ),
    path(
        "select_taxon_for_identification/<int:uploadedarchive_id>/",
        views.select_taxon_for_identification,
        name="select_taxon_for_identification",
    ),
    # path("localities/", views_locality.localities_view, name="localities"),
    path("localities/", views_locality.LocalityListView.as_view(), name="localities"),
    path("localities/export/", views_locality.export_localities_view, name="export_localities"),
    path(
        "localities/export_xls/",
        views_locality.export_localities_view_xls,
        name="export_localities_xls",
    ),
    path("localities/import/", views_locality.import_localities_view, name="import_localities"),
    path(
        "localities/checks/<str:locality_hash>/",
        views_locality.uploads_of_locality,
        name="uploads_of_locality",
    ),
    path(
        "localities/download_records_csv/<str:locality_hash>/",
        views_locality.download_records_from_locality_csv_view,
        name="download_records_from_locality_csv",
    ),
    path(
        "localities/download_records_xls/<str:locality_hash>/",
        views_locality.download_records_from_locality_xls_view,
        name="download_records_from_locality_xls",
    ),
    path(
        "uploaded_archives/set_sort_by/<str:sort_by>/",
        views.set_sort_uploaded_archives_by,
        name="set_sort_uploaded_archives_by",
    ),
    path(
        "uploaded_archives/set_item_number/<int:item_number>/",
        views.set_item_number_uploaded_archives,
        name="set_item_number_uploaded_archives",
    ),
    path(
        "set_item_number/<str:name_plural>/<int:item_number>/",
        views_general.set_item_number_anything,
        name="set_item_number_anything",
    ),
    path(
        "set_sort_by/<str:name_plural>/<str:sort_by>/",
        views_general.set_sort_anything_by,
        name="set_sort_anything_by",
    ),
    # path(
    #     "identities/set_sort_by/<str:sort_by>/",
    #     views.set_sort_uploaded_archives_by,
    #     name="set_sort_uploaded_archives_by",
    # ),
    path("stream_video/<int:mediafile_id>/", views_mediafile.stream_video, name="stream_video"),
    path("mediafiles/verify_taxa/", views_mediafile.verify_taxa_view, name="verify_taxa"),
    path(
        "mediafiles/verify_taxa/uploaded_archive/<int:uploaded_archive_id>",
        views_mediafile.verify_taxa_view,
        name="verify_taxa",
    ),
    path(
        "mediafiles/taxons_on_page_are_overviewed/",
        views_mediafile.taxons_on_page_are_verified,
        name="taxons_on_page_are_overviewed",
    ),
    path(
        "mediafiles/set_mediafiles_order_by/<str:order_by>/",
        views_mediafile.set_mediafiles_order_by,
        name="set_mediafiles_order_by",
    ),
    path(
        "mediafiles/set_mediafiles_records_per_page/<int:records_per_page>/",
        views_mediafile.set_mediafiles_records_per_page,
        name="set_mediafiles_records_per_page",
    ),
    path("impersonate/", views.impersonate_user, name="impersonate_user"),
    path("stop-impersonation/", views.stop_impersonation, name="stop_impersonation"),
    path("switch_private_mode/", views.switch_private_mode, name="switch_private_mode"),
    path("update_taxon/<int:taxon_id>/", views.update_taxon, name="update_taxon"),
    path("add_taxon/", views.update_taxon, name="add_taxon"),
    path(
        "confirm_prediction/<int:mediafile_id>",
        views_mediafile.confirm_prediction,
        name="confirm_prediction",
    ),
    path("taxon_processing", views_uploads.taxon_processing, name="taxon_processing"),
    path("check_dates/", views_uploads.camera_trap_check_dates_view, name="check_dates"),
    path(
        "check_dates/<int:year>/",
        views_uploads.camera_trap_check_dates_view,
        name="check_dates_with_year",
    ),
    path("check_date/<str:date>/", views_uploads.camera_trap_check_date_view, name="check_date"),
    path("check_date/", views_uploads.camera_trap_check_date_view, name="check_date_empty"),
    # urls.py
    path("users_stats/", views.ImageUploadGraphView.as_view(), name="users_stats"),
    path("select_reid_model/", views.select_reid_model, name="select_reid_model"),
    path(
        "merge_identities/<int:individual_identity_from_id>/<int:individual_identity_to_id>/",
        views.MergeIdentitiesWithPreview.as_view(),
        name="merge_identities",
    ),
    path(
        "merge_identities_no_preview/<int:individual_identity_from_id>/<int:individual_identity_to_id>/",
        views.MergeIdentitiesNoPreview.as_view(),
        name="merge_identities_no_preview",
    ),
    path(
        "merge_identities/<int:individual_identity1_id>/",
        views.select_second_id_for_identification_merge,
        name="merge_identities",
    ),
    path(
        "suggest_merge_identities/",
        views.suggest_merge_identities_view,
        name="suggest_merge_identities",
    ),
    path(
        "update_uploaded_archive_with_spreadsheet/<int:uploaded_archive_id>/",
        views.UpdateUploadedArchiveBySpreadsheetFile.as_view(),
        name="update_uploaded_archive_with_spreadsheet",
    ),
    path("pygwalker/", include("djangoaddicts.pygwalker.urls")),
    path("pygwalker_mediafiles/", views.MyPygWalkerView.as_view(), name="pygwalker_mediafiles"),
    path(
        "pygwalker_localities/",
        views.PygWalkerLocalitiesView.as_view(),
        name="pygwalker_localities",
    ),
    # path("generic/locality/", ListView.as_view(model=models.Locality), name="generic_localities"),
    path("generic/locality/", views_locality.LocalityListView.as_view(), name="generic_locality_list"),
    path(
        "generic/locality/<int:pk>/",
        DetailView.as_view(model=models.Locality),
        name="generic_locality_detail",
    ),
    path(
        "suggest_merge_localities/",
        views_locality.suggest_merge_localities_view,
        name="suggest_merge_localities",
    ),
    path(
        "refresh_merge_localities_suggestions/",
        views_locality.refresh_merge_localities_suggestions,
        name="refresh_merge_localities_suggestions",
    ),
    path(
        "merge_localities/<int:locality_from_id>/<int:locality_to_id>/",
        views_locality.merge_localities_view,
        name="merge_localities",
    ),
    path(
        "merge_selected_identities/",
        views.merge_selected_identities_view,
        name="merge_selected_identities",
    ),
    path("do_admin_stuff/<str:process_name>/", views_admin.do_admin_stuff, name="do_admin_stuff"),
    path(
        "show_identity_code_suggestions",
        views.show_identity_code_suggestions,
        name="show_identity_code_suggestions",
    ),
    path(
        "apply_identity_code_suggestion/<int:identity_id>/",
        views.apply_identity_code_suggestion,
        name="apply_identity_code_suggestion",
    ),
    # path("uploads_status_api/<bool:species>/", views.uploads_status_api, name="uploads_status_api"),
    path("uploads_status_api/<str:group>/", views.uploads_status_api, name="uploads_status_api"),
    path("sentry-debug/", trigger_error),
    path("home/", views.home_view, name="home"),
    path("wellcome/", views.WellcomeView.as_view(), name="wellcome"),
    path("pre_identify/", views.pre_identify_view, name="pre_identify"),
    path(
        "assign_unidentified_to_identification/",
        views.assign_unidentified_to_identification_view,
        name="assign_unidentified_to_identification",
    ),
    path(
        "ajax/identity-card/<int:foridentification_id>/<int:identity_id>/",
        views.get_individual_identity_remaining_card_content,
        name="ajax_identity_card",
    ),
    path(
        "mediafile/<int:mediafile_id>/toggle-representative/",
        views.toggle_identity_representative,
        name="toggle_identity_representative",
    ),
    path("workgroups/<int:pk>/update/", views.WorkgroupUpdateView.as_view(), name="workgroup-update"),
    # notifications
    path("notifications/create/", views.NotificationCreateView.as_view(), name="notification-create"),
    path("notifications/", views.NotificationListView.as_view(), name="notifications"),
    path(
        "notifications/<int:pk>/",
        views.NotificationDetailView.as_view(),
        name="notification-detail",
    ),
    path(
        "notifications/<int:pk>/update/",
        views.NotificationUpdateView.as_view(),
        name="notification-update",
    ),
    path(
        "notifications/<int:pk>/delete/",
        views.NotificationDeleteView.as_view(),
        name="notification-delete",
    ),
]

# if settings.DEBUG:
