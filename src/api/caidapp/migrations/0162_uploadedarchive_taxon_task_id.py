from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("caidapp", "0161_caiduser_edit_permissions"),
    ]

    operations = [
        migrations.AddField(
            model_name="uploadedarchive",
            name="taxon_task_id",
            field=models.CharField(blank=True, default="", max_length=255),
        ),
    ]
