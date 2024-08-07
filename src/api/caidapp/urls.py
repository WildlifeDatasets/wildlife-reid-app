from django.urls import path

from . import views_location
from . import views
from . import views_mediafile
from . import views_uploads

app_name = "caidapp"
urlpatterns = [
    path("", views.login, name="index"),
    path("upload/", views.upload_archive, name="upload_archive"),
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
    path("logout/", views.logout_view, name="logout_view"),
    path("uploads/", views.uploads_species, name="uploads"),
    path("uploads_identities/", views.uploads_identities, name="uploads_identities"),
    path(
        "uploadedarchive_detail/<int:uploadedarchive_id>",
        # views.uploadedarchive_detail,
        views.media_files_update,
        name="uploadedarchive_detail",
    ),
    path(
        "download_uploadedarchive_csv/<int:uploadedarchive_id>",
        views.download_uploadedarchive_csv,
        name="download_uploadedarchive_csv",
    ),
    # path("media_files/", views.media_files, name="media_files"),
    path("media_files/", views.media_files_update, name="media_files"),
    path(
        "delete_mediafile/<int:mediafile_id>/",
        views.delete_mediafile,
        name="delete_mediafile",
    ),
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
    path("refresh_data/", views.refresh_data, name="refresh_data"
    ),
    path("djangologin/", views.MyLoginView.as_view(), name="djangologin"),
    path(
        "media_file_update/<int:media_file_id>/",
        views.media_file_update,
        name="media_file_update",
    ),
    path("manage_locations/", views_location.manage_locations, name="manage_locations"),
    path("delete_location/<int:location_id>/", views_location.delete_location, name="delete_location"),
    path(
        "update_location/<int:location_id>/",
        views_location.update_location,
        name="update_location",
    ),
    path("albums/", views.albums, name="albums"),
    path("album/<str:album_hash>", views.media_files_update, name="album"),
    path("taxon/<int:taxon_id>", views.media_files_update, name="taxon"),
    path("media_files/location/<str:location_hash>", views.media_files_update, name="media_files_location"),
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
    path("album_update/<str:album_hash>/", views.album_update, name="album_update"),
    path("delete_album/<str:album_hash>/", views.delete_album, name="delete_album"),
    path("new_album/", views.new_album, name="new_album"),
    path(
        "new_individual_identity/",
        views.new_individual_identity,
        name="new_individual_identity",
    ),
    path(
        "update_individual_identity/<int:individual_identity_id>",
        views.update_individual_identity,
        name="update_individual_identity",
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
    path(
        "individual_identities/",
        views.individual_identities,
        name="individual_identities",
    ),
    path("init_identification/", views.init_identification, name="init_identification"),
    path(
        "run_identification/<int:uploadedarchive_id>/",
        views.run_identification,
        name="run_identification",
    ),
    path(
        "run_identification_on_unindentified/",
        views.run_identification_on_unidentified,
        name="run_identification_on_unidentified",
    ),
    path(
        "get_individual_identity_zoomed/<int:foridentification_id>/<int:top_id>",
        views.get_individual_identity_zoomed,
        name="get_individual_identity_zoomed",
    ),
    path(
        "get_individual_identity_zoomed_paired_points/<int:foridentification_id>/<int:top_id>",
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
        "set_individual_identity/"
        + "<int:mediafiles_for_identification_id>/<int:individual_identity_id>",
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
    path(
        "workgroup_update/<str:workgroup_hash>/",
        views.workgroup_update,
        name="workgroup_update",
    ),
    path(
        "manual_taxon_classification_on_non_classified",
        views_mediafile.manual_taxon_classification_on_non_classified,
        name="manual_taxon_classification_on_non_classified",
    ),
    path("sample_data/", views.sample_data, name="sample_data"),
    path("cloud_import_preview/", views.cloud_import_preview_view, name="cloud_import_preview"),
    path("do_cloud_import/", views.do_cloud_import_view, name="do_cloud_import"),
    path("break_cloud_import/", views.break_cloud_import_view, name="break_cloud_import"),
    path("download_csv_for_mediafiles/", views.download_csv_for_mediafiles_view, name="download_csv_for_mediafiles"),
    path("download_zip_for_mediafiles/", views.download_zip_for_mediafiles_view, name="download_zip_for_mediafiles"),
    path("download_xlsx_for_mediafiles/", views.download_xlsx_for_mediafiles_view, name="download_xlsx_for_mediafiles"),
    path("mediafiles_stats/", views.mediafiles_stats_view, name="mediafiles_stats"),
    path("select_taxon_for_identification/<int:uploadedarchive_id>/", views.select_taxon_for_identification, name="select_taxon_for_identification"),
    path("locations/", views.locations_view, name="locations"),
    path("locations/export/", views_location.export_locations_view, name="export_locations"),
    path("locations/export_xls/", views_location.export_locations_view_xls, name="export_locations_xls"),
    path("locations/import/", views_location.import_locations_view, name="import_locations"),
    path("locations/checks/<str:location_hash>/", views_location.uploads_of_location, name="uploads_of_location"),
    path("locations/download_records_csv/<str:location_hash>/", views_location.download_records_from_location_csv_view, name="download_records_from_location_csv"),
    path("locations/download_records_xls/<str:location_hash>/", views_location.download_records_from_location_xls_view, name="download_records_from_location_xls"),
    path("uploaded_archives/set_sort_by/<str:sort_by>/", views.set_sort_uploaded_archives_by, name="set_sort_uploaded_archives_by"),
    path("uploaded_archives/set_item_number/<int:item_number>/", views.set_item_number_uploaded_archives, name="set_item_number_uploaded_archives"),
    path("stream_video/<int:mediafile_id>/", views_mediafile.stream_video, name="stream_video"),
    path("mediafiles/overview_taxons/", views_mediafile.overview_taxons, name="overview_taxons"),
    path("mediafiles/overview_taxons/uploaded_archive/<int:uploaded_archive_id>", views_mediafile.overview_taxons, name="overview_taxons"),
    path("mediafiles/taxons_on_page_are_overviewed/", views_mediafile.taxons_on_page_are_overviewed, name="taxons_on_page_are_overviewed"),
    path("mediafiles/set_mediafiles_order_by/<str:order_by>/", views_mediafile.set_mediafiles_order_by, name="set_mediafiles_order_by"),
    path("mediafiles/set_mediafiles_records_per_page/<int:records_per_page>/", views_mediafile.set_mediafiles_records_per_page, name="set_mediafiles_records_per_page"),

    path('impersonate/', views.impersonate_user, name='impersonate_user'),
    path('stop-impersonation/', views.stop_impersonation, name='stop_impersonation'),
    path('switch_private_mode/', views.switch_private_mode, name='switch_private_mode'),
    path('update_taxon/<int:taxon_id>/', views.update_taxon, name='update_taxon'),
    path('add_taxon/', views.update_taxon, name='add_taxon'),
    path('confirm_prediction/<int:mediafile_id>', views_mediafile.confirm_prediction, name='confirm_prediction'),
    path('taxon_processing',views_uploads.taxon_processing, name='taxon_processing'),
]

