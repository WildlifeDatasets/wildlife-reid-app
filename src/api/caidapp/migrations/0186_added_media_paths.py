from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("caidapp", "0185_similar_pair_pixel_matches")]

    operations = [
        migrations.AddField(
            model_name="mediafile",
            name="added_directory",
            field=models.CharField(blank=True, editable=False, max_length=40, null=True),
        ),
        migrations.AddField(
            model_name="mediafile",
            name="addition_key",
            field=models.CharField(blank=True, editable=False, max_length=255, null=True),
        ),
        migrations.AddConstraint(
            model_name="mediafile",
            constraint=models.UniqueConstraint(
                fields=("parent", "added_directory"), name="unique_added_media_directory"
            ),
        ),
        migrations.AddConstraint(
            model_name="mediafile",
            constraint=models.UniqueConstraint(fields=("parent", "addition_key"), name="unique_media_addition_key"),
        ),
    ]
