from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("caidapp", "0154_uploadedarchive_import_log_import_mapping"),
    ]

    operations = [
        migrations.AddField(
            model_name="uploadedarchive",
            name="path_structure_regex",
            field=models.TextField(blank=True, default=""),
        ),
    ]
