from django.contrib import admin

from .models import CIDUser, Location, MediaFile, Taxon, UploadedArchive, Album, AlbumShareRoleType

# Register your models here.
admin.site.register(CIDUser)
admin.site.register(UploadedArchive)
admin.site.register(MediaFile)
admin.site.register(Taxon)
admin.site.register(Location)
admin.site.register(Album)
admin.site.register(AlbumShareRoleType)
