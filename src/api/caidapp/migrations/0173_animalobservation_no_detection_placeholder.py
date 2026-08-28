from django.db import migrations, models
from django.db.models import Q


def create_missing_observations(apps, schema_editor):
    MediaFile = apps.get_model("caidapp", "MediaFile")
    AnimalObservation = apps.get_model("caidapp", "AnimalObservation")
    database = schema_editor.connection.alias

    missing_ids = MediaFile.objects.using(database).filter(observations__isnull=True).values_list("id", flat=True).iterator()
    batch = []
    for mediafile_id in missing_ids:
        batch.append(
            AnimalObservation(
                mediafile_id=mediafile_id,
                is_no_detection_placeholder=True,
            )
        )
        if len(batch) >= 1000:
            AnimalObservation.objects.using(database).bulk_create(batch)
            batch = []
    if batch:
        AnimalObservation.objects.using(database).bulk_create(batch)


def remove_placeholders(apps, schema_editor):
    AnimalObservation = apps.get_model("caidapp", "AnimalObservation")
    database = schema_editor.connection.alias
    AnimalObservation.objects.using(database).filter(is_no_detection_placeholder=True).delete()


class Migration(migrations.Migration):
    # PostgreSQL cannot create the partial unique index while FK trigger
    # events from the placeholder backfill are still pending in the same
    # transaction. Commit the schema change and backfill before adding it.
    atomic = False

    dependencies = [
        ("caidapp", "0172_remove_mediafile_animal_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="animalobservation",
            name="is_no_detection_placeholder",
            field=models.BooleanField(
                default=False,
                editable=False,
                help_text="Temporary observation representing a media file where no object was detected.",
            ),
        ),
        migrations.RunPython(create_missing_observations, remove_placeholders),
        migrations.AddConstraint(
            model_name="animalobservation",
            constraint=models.UniqueConstraint(
                condition=Q(is_no_detection_placeholder=True),
                fields=("mediafile",),
                name="unique_no_detection_placeholder_per_mediafile",
            ),
        ),
    ]
