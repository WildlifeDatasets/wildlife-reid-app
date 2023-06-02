from django.urls import path
from django.views.generic import TemplateView

from . import views

app_name = "caidapp"
urlpatterns = [
    # path('', views.index, name='index'),
    path("upload/", views.model_form_upload, name="model_form_upload"),
    path("login/", TemplateView.as_view(template_name="caidapp/login.html"), name="login"),  # <--I
    path("logout/", views.logout_view, name="logout_view"),
    path("uploads/", views.uploads, name="uploads"),  # <--I
    path("media_files/<int:uploadedarchive_id>/", views.media_files, name="media_files"),  # <--I
    path("delete_mediafile/<int:mediafile_id>/", views.delete_mediafile, name="delete_mediafile"),
    path("<int:uploadedarchive_id>/delete_upload/", views.delete_upload, name="delete_upload"),
    path('djangologin/', views.MyLoginView.as_view(), name='djangologin'),
]
