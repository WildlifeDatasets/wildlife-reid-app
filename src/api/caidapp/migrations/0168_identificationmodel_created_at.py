from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("caidapp", "0167_workgroup_identification_initialized_model"),
    ]

    operations = [
        migrations.AddField(
            model_name="identificationmodel",
            name="created_at",
            field=models.DateTimeField(auto_now_add=True, blank=True, null=True),
        ),
    ]
