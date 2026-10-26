from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("caidapp", "0176_identificationdashboardsnapshot")]

    operations = [
        migrations.AddField(
            model_name="identificationsimilarpairresult",
            name="name_pairs",
            field=models.JSONField(blank=True, default=list),
        ),
    ]
