from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("caidapp", "0184_incremental_duplicate_hashes")]

    operations = [
        migrations.AddField(
            model_name="identificationsimilarpairresult",
            name="pixel_pairs",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="identificationsimilarpairresult",
            name="pixel_status",
            field=models.CharField(blank=True, default="", max_length=16),
        ),
        migrations.AddField(
            model_name="identificationsimilarpairresult",
            name="pixel_message",
            field=models.TextField(blank=True, default=""),
        ),
    ]
