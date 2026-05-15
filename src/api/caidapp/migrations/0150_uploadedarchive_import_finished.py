from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("caidapp", "0149_mediafile_location"),
    ]

    operations = [
        migrations.AddField(
            model_name="uploadedarchive",
            name="import_finished",
            field=models.BooleanField(default=False, verbose_name="Taxon import finished"),
        ),
    ]
