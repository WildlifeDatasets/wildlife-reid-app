from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("caidapp", "0153_backfill_uploadedarchive_is_for_identification"),
    ]

    operations = [
        migrations.AddField(
            model_name="uploadedarchive",
            name="import_log",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AddField(
            model_name="uploadedarchive",
            name="import_mapping",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
