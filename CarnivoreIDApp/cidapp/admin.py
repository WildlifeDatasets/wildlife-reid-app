from django.contrib import admin

# Register your models here.

from .models import (
    CIDUser,
    UploadedArchive
)


admin.site.register(CIDUser)
admin.site.register(UploadedArchive)
