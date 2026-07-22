from dataclasses import dataclass
from pathlib import Path

from django.contrib import admin
from django.contrib import messages
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db.migrations.recorder import MigrationRecorder
from django.utils.html import format_html

from . import models

# Register your models here.


@dataclass(frozen=True)
class IdentificationModelFileCheck:
    """Result of checking the local checkpoint used by an identification model."""

    status: str
    detail: str


def check_identification_model_file(model: models.IdentificationModel) -> IdentificationModelFileCheck:
    """Check model configuration and, for local models, its checkpoint file.

    Hub and timm model sources intentionally have no local checkpoint to verify.
    """
    try:
        model.clean()
        checkpoint_path = model.get_runtime_checkpoint_path()
        model.get_runtime_model_source()
    except (ObjectDoesNotExist, ValidationError) as exc:
        return IdentificationModelFileCheck("invalid", f"Neplatná konfigurace: {exc}")

    if not checkpoint_path:
        return IdentificationModelFileCheck("external", "Vzdálený model – lokální .pth soubor se nekontroluje.")

    path = Path(checkpoint_path).expanduser()
    try:
        if path.is_file():
            return IdentificationModelFileCheck("ok", f"Soubor existuje: {path}")
        if path.exists():
            return IdentificationModelFileCheck("invalid", f"Cesta není soubor: {path}")
    except OSError as exc:
        return IdentificationModelFileCheck("invalid", f"Soubor nelze ověřit ({exc}): {path}")
    return IdentificationModelFileCheck("invalid", f"Soubor neexistuje: {path}")


@admin.register(models.IdentificationModel)
class IdentificationModelAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "public",
        "workgroup",
        "model_path",
        "base_model_path",
        "source_identification_model",
        "checkpoint_status",
    )
    list_filter = ("public", "workgroup")
    search_fields = ("name", "model_path", "base_model_path", "description")
    actions = ("verify_selected_model_files",)

    @admin.display(description="Checkpoint")
    def checkpoint_status(self, obj):
        result = check_identification_model_file(obj)
        colors = {"ok": "green", "external": "#666", "invalid": "#b91c1c"}
        labels = {"ok": "OK", "external": "Vzdálený", "invalid": "CHYBA"}
        return format_html(
            '<span style="color: {}; font-weight: 600" title="{}">{}</span>',
            colors[result.status],
            result.detail,
            labels[result.status],
        )

    @admin.action(description="Ověřit soubory vybraných identifikačních modelů")
    def verify_selected_model_files(self, request, queryset):
        results = [(model, check_identification_model_file(model)) for model in queryset]
        invalid = [(model, result) for model, result in results if result.status == "invalid"]
        ok_count = sum(result.status == "ok" for _, result in results)
        external_count = sum(result.status == "external" for _, result in results)

        if not results:
            self.message_user(request, "Nevybrali jste žádný model.", messages.WARNING)
            return

        summary = (
            f"Ověřeno modelů: {len(results)}. Lokální soubor existuje: {ok_count}; "
            f"vzdálený zdroj: {external_count}; neplatný záznam: {len(invalid)}."
        )
        self.message_user(
            request,
            summary,
            messages.ERROR if invalid else messages.SUCCESS,
        )
        if invalid:
            details = "; ".join(
                f"#{model.pk} {model.name}: {result.detail}" for model, result in invalid
            )
            self.message_user(
                request,
                "Doporučení: označte tyto neplatné záznamy a použijte standardní akci "
                f"„Delete selected Identification models“ (s potvrzením). {details}",
                messages.WARNING,
            )


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
