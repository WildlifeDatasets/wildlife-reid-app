"""
URL configuration for CarnivoreIDApp project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView  # TemplateView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("caidapp/", include("caidapp.urls")),
    path("accounts/", include("allauth.urls")),  # <--
    path("accounts/", include("allauth.socialaccount.urls")),  # <--
    path("", RedirectView.as_view(url="/caidapp/")),  # <--I
    # path("", TemplateView.as_view(template_name="caidapp/login.html"), name="login"),  # <--I
]

if settings.DEBUG_TOOLBAR:
    import debug_toolbar

    # urlpatterns = [
    #     *urlpatterns,
    # ] + debug_toolbar.debug_toolbar_urls()
    urlpatterns = [
        # path("__debug__/", include(debug_toolbar.urls)),
        path("__debug__/", include("debug_toolbar.urls")),
    ] + urlpatterns
    print(f"{urlpatterns=}")

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
# print (static(settings.MEDIA_URL))
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
