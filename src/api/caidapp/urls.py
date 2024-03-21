from django.urls import path

from . import views

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
        views.uploadedarchive_detail,
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
        "<int:uploadedarchive_id>/run_processing/",
        views.run_taxon_classification,
        name="run_processing",
    ),
    path("djangologin/", views.MyLoginView.as_view(), name="djangologin"),
    path(
        "media_file_update/<int:media_file_id>/",
        views.media_file_update,
        name="media_file_update",
    ),
    path("manage_locations/", views.manage_locations, name="manage_locations"),
    path(
        "update_location/<int:location_id>/",
        views.update_location,
        name="update_location",
    ),
    path("albums/", views.albums, name="albums"),
    path("album/<str:album_hash>", views.media_files_update, name="album"),
    path("taxon/<int:taxon_id>", views.media_files_update, name="taxon"),
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
        views.manual_taxon_classification_on_non_classified,
        name="manual_taxon_classification_on_non_classified",
    ),
    path("sample_data/", views.sample_data, name="sample_data"),
]
