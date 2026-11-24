from django.db import migrations, models
from django.db.models import deletion


def clear_pending_reviews(apps, schema_editor):
    """Discard the previous media-file queue before requiring an observation."""
    reviews = apps.get_model("caidapp", "MediafilesForIdentification")
    reviews.objects.using(schema_editor.connection.alias).all().delete()
    if schema_editor.connection.vendor == "postgresql":
        # Flush deferred foreign-key checks before altering the affected tables.
        schema_editor.execute("SET CONSTRAINTS ALL IMMEDIATE")


class Migration(migrations.Migration):
    dependencies = [("caidapp", "0187_mediafile_original_file")]

    operations = [
        migrations.RunPython(clear_pending_reviews, migrations.RunPython.noop),
        migrations.AddField(
            model_name="mediafilesforidentification",
            name="observation",
            field=models.OneToOneField(
                on_delete=deletion.CASCADE, related_name="identity_review", to="caidapp.animalobservation"
            ),
        ),
        migrations.RemoveField(model_name="mediafilesforidentification", name="mediafile"),
        migrations.AddField(
            model_name="mediafilesforidentification",
            name="input_version",
            field=models.CharField(blank=True, max_length=64),
        ),
        migrations.AddField(
            model_name="mediafilesforidentification",
            name="model_version",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="mediafilesforidentification",
            name="provenance",
            field=models.CharField(default="reid", max_length=20),
        ),
        migrations.AddField(
            model_name="mediafilesforidentification",
            name="generated_at",
            field=models.DateTimeField(auto_now=True),
        ),
        migrations.AddField(
            model_name="mediafilesforidentification",
            name="run",
            field=models.ForeignKey(
                blank=True, null=True, on_delete=deletion.SET_NULL, to="caidapp.identificationrunstatistic"
            ),
        ),
        migrations.AddField(
            model_name="mediafileidentificationsuggestion",
            name="reference_observation",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=deletion.CASCADE,
                related_name="identity_references",
                to="caidapp.animalobservation",
            ),
        ),
    ]
