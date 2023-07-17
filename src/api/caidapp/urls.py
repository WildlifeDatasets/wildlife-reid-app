from django.urls import path
from django.views.generic import TemplateView

from . import views

app_name = "caidapp"
urlpatterns = [
    # path('', views.index, name='index'),
    path("upload/", views.model_form_upload, name="model_form_upload"),
    path("login/", TemplateView.as_view(template_name="caidapp/login.html"), name="login"),
    path("logout/", views.logout_view, name="logout_view"),
    path("uploads/", views.uploads, name="uploads"),
    path(
        "uploadedarchive_detail/<int:uploadedarchive_id>/",
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
]
