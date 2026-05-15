from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("caidapp", "0156_backfill_mediafile_locality_from_uploadedarchive"),
    ]

    operations = [
        migrations.AddField(
            model_name="caiduser",
            name="show_base_between_regular_uploads",
            field=models.BooleanField(default=False),
        ),
    ]
