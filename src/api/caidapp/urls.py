from django.urls import path

from . import views

app_name = "caidapp"
urlpatterns = [
    path("", views.login, name="index"),
    path("upload/", views.model_form_upload, name="model_form_upload"),
    # path("login/", TemplateView.as_view(template_name="caidapp/login.html"), name="login"),
    path("logout/", views.logout_view, name="logout_view"),
    path("uploads/", views.uploads, name="uploads"),
    path(
        "uploadedarchive_detail/<int:uploadedarchive_id>",
        views.uploadedarchive_detail,
        name="uploadedarchive_detail",
    ),
    # path("media_files/", views.media_files, name="media_files"),
    path("media_files/", views.media_files_update, name="media_files"),
    path("delete_mediafile/<int:mediafile_id>/", views.delete_mediafile, name="delete_mediafile"),
    path("<int:uploadedarchive_id>/delete_upload/", views.delete_upload, name="delete_upload"),
    path("<int:uploadedarchive_id>/run_processing/", views.run_processing, name="run_processing"),
    path("djangologin/", views.MyLoginView.as_view(), name="djangologin"),
    path(
        "media_file_update/<int:media_file_id>/", views.media_file_update, name="media_file_update"
    ),
    path("manage_locations/", views.manage_locations, name="manage_locations"),
    path("albums/", views.albums, name="albums"),
    path("album/<str:album_hash>", views.media_files_update, name="album"),
    path(
        "individual_identity_mediafiles/<int:individual_identity_id>",
        views.media_files_update,
        name="individual_identity_mediafiles",
    ),
    path("album_update/<str:album_hash>/", views.album_update, name="album_update"),
    path("delete_album/<str:album_hash>/", views.delete_album, name="delete_album"),
    path("new_album/", views.new_album, name="new_album"),
    path("new_individual_identity/", views.new_individual_identity, name="new_individual_identity"),
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
    path("individual_identities/", views.individual_identities, name="individual_identities"),
    path("init_identification/", views.init_identification, name="init_identification"),
    path("run_identification/<int:uploadedarchive_id>/", views.run_identification, name="run_identification"),
    path("get_individual_identity/", views.get_individual_identity, name="get_individual_identity"),
    path("workgroup_update/<str:workgroup_hash>/", views.workgroup_update, name="workgroup_update"),
]
