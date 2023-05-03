from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from django.views.generic import TemplateView  # <--

from . import views

app_name = "cidapp"
urlpatterns = [
    # path('', views.index, name='index'),
    path("upload/", views.model_form_upload, name="model_form_upload"),
    path("login/", TemplateView.as_view(template_name="cidapp/login.html"), name="login"),  # <--I
    path("logout/", views.logout_view, name="logout_view"),
    path("uploads/", views.uploads, name="uploads"),  # <--I
    path("<int:uploadedarchive_id>/delete_upload/", views.delete_upload, name="delete_upload"),
]
