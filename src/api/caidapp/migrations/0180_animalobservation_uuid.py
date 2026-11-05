import uuid

from django.db import migrations, models


def populate_uuids(apps, schema_editor):
    """Assign a different public UUID to every existing observation."""
    observation = apps.get_model("caidapp", "AnimalObservation")
    pending = []
    for row in observation.objects.using(schema_editor.connection.alias).all().iterator(chunk_size=1000):
        row.uuid = uuid.uuid4()
        pending.append(row)
        if len(pending) == 1000:
            observation.objects.using(schema_editor.connection.alias).bulk_update(pending, ["uuid"], batch_size=500)
            pending = []
    if pending:
        observation.objects.using(schema_editor.connection.alias).bulk_update(pending, ["uuid"], batch_size=500)


class Migration(migrations.Migration):
    dependencies = [("caidapp", "0179_observation_embedding_schedule")]

    operations = [
        migrations.AddField(
            model_name="animalobservation", name="uuid", field=models.UUIDField(null=True, editable=False)
        ),
        migrations.RunPython(populate_uuids, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="animalobservation",
            name="uuid",
            field=models.UUIDField(default=uuid.uuid4, unique=True, editable=False),
        ),
    ]
