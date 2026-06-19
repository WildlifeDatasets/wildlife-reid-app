from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("caidapp", "0160_notification_internal_link")]

    operations = [
        migrations.AddField(
            model_name="caiduser",
            name="can_edit_taxon_data",
            field=models.BooleanField(default=True),
        ),
        migrations.AddField(
            model_name="caiduser",
            name="can_edit_identity_data",
            field=models.BooleanField(default=True),
        ),
        migrations.AddField(
            model_name="caiduser",
            name="can_edit_other_records",
            field=models.BooleanField(default=True),
        ),
        migrations.AddField(
            model_name="caiduser",
            name="is_observer",
            field=models.BooleanField(default=False),
        ),
    ]
