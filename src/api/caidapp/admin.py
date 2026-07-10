from django.contrib import admin
from django.db.migrations.recorder import MigrationRecorder

from . import models

# Register your models here.


@admin.register(models.IdentificationModel)
class IdentificationModelAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "public",
        "workgroup",
        "model_path",
        "base_model_path",
        "source_identification_model",
    )
    list_filter = ("public", "workgroup")
    search_fields = ("name", "model_path", "base_model_path", "description")


admin.site.register(models.CaIDUser)
admin.site.register(models.UploadedArchive)
admin.site.register(models.MediaFile)
admin.site.register(models.Taxon)
admin.site.register(models.Locality)
admin.site.register(models.Album)
admin.site.register(models.AlbumShareRoleType)
admin.site.register(models.IndividualIdentity)
admin.site.register(models.MediafilesForIdentification)
admin.site.register(models.WorkGroup)
admin.site.register(models.ArchiveCollection)
admin.site.register(MigrationRecorder.Migration)
admin.site.register(models.Sequence)
admin.site.register(models.MediafileIdentificationSuggestion)
admin.site.register(models.Area)
admin.site.register(models.Notification)
admin.site.register(models.AnimalObservation)
admin.site.register(models.MergeIdentitySuggestionResult)
admin.site.register(models.IdentificationOutlierSuggestionResult)
admin.site.register(models.WorkGroupInvitation)
admin.site.register(models.IdentificationRunStatistic)
