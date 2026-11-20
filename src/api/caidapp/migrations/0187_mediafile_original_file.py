from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("caidapp", "0186_added_media_paths")]

    operations = [
        migrations.AddField(
            model_name="mediafile",
            name="original_file",
            field=models.CharField(blank=True, default="", editable=False, max_length=500),
        ),
    ]
