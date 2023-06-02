from django.contrib import admin

from .models import CIDUser, UploadedArchive, MediaFile

# Register your models here.
admin.site.register(CIDUser)
admin.site.register(UploadedArchive)
admin.site.register(MediaFile)
