from django.contrib import admin

from . import models
# Register your models here.
admin.site.register(models.CIDUser)
admin.site.register(models.UploadedArchive)
admin.site.register(models.MediaFile)
admin.site.register(models.Taxon)
admin.site.register(models.Location)
admin.site.register(models.Album)
admin.site.register(models.AlbumShareRoleType)
admin.site.register(models.IndividualIdentity)
admin.site.register(models.MediafilesForIdentification)
admin.site.register(models.WorkGroup)
