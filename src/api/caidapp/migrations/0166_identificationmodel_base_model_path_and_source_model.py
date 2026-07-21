from django.db import migrations, models


LEGACY_TRAINING_BASE_MODEL_SOURCE = "hf-hub:BVRA/MegaDescriptor-T-224"
CHECKPOINT_SUFFIXES = (".pth", ".pt", ".ckpt", ".bin", ".safetensors")


def backfill_identification_model_base_path(apps, schema_editor):
    IdentificationModel = apps.get_model("caidapp", "IdentificationModel")
    for identification_model in IdentificationModel.objects.all():
        model_path = (identification_model.model_path or "").strip()
        if identification_model.base_model_path:
            continue
        if not model_path:
            continue
        if model_path.startswith("file:") or model_path.startswith("/") or model_path.startswith("~/") or model_path.endswith(
            CHECKPOINT_SUFFIXES
        ):
            identification_model.base_model_path = LEGACY_TRAINING_BASE_MODEL_SOURCE
            identification_model.save(update_fields=["base_model_path"])


class Migration(migrations.Migration):

    dependencies = [
        ("caidapp", "0165_identificationrunstatistic"),
    ]

    operations = [
        migrations.AlterField(
            model_name="identificationmodel",
            name="name",
            field=models.CharField(max_length=120),
        ),
        migrations.AlterField(
            model_name="identificationmodel",
            name="model_path",
            field=models.CharField(
                blank=True,
                default="",
                help_text=(
                    "Primary model source. Supported values include hf-hub:owner/model, timm:model_name, "
                    "or a local checkpoint path such as file:/shared_data/media/.../model.pth."
                ),
                max_length=512,
            ),
        ),
        migrations.AddField(
            model_name="identificationmodel",
            name="base_model_path",
            field=models.CharField(
                blank=True,
                default="",
                help_text=(
                    "Optional base model source used when model_path points to a local checkpoint. "
                    "Use hf-hub:... or timm:... here."
                ),
                max_length=512,
            ),
        ),
        migrations.AddField(
            model_name="identificationmodel",
            name="source_identification_model",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=models.SET_NULL,
                related_name="trained_models",
                to="caidapp.identificationmodel",
            ),
        ),
        migrations.RunPython(
            backfill_identification_model_base_path,
            migrations.RunPython.noop,
        ),
    ]
