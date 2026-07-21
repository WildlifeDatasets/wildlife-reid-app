from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("caidapp", "0166_identificationmodel_base_model_path_and_source_model"),
    ]

    operations = [
        migrations.AddField(
            model_name="workgroup",
            name="identification_initialized_model",
            field=models.ForeignKey(
                blank=True,
                help_text=(
                    "Identification model used by the last successfully completed "
                    "re-identification initialization."
                ),
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="initialized_workgroups",
                to="caidapp.identificationmodel",
            ),
        ),
    ]
