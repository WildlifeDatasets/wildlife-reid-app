import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("caidapp", "0175_identificationsimilarpairresult")]

    operations = [
        migrations.CreateModel(
            name="IdentificationDashboardSnapshot",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("payload", models.JSONField(default=dict)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "workgroup",
                    models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to="caidapp.workgroup"),
                ),
            ],
        ),
    ]
