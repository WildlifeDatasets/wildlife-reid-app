from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("caidapp", "0159_homedashboardsnapshot")]

    operations = [
        migrations.AddField(
            model_name="notification",
            name="link_label",
            field=models.CharField(blank=True, default="", max_length=100),
        ),
        migrations.AddField(
            model_name="notification",
            name="link_url_kwargs",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="notification",
            name="link_url_name",
            field=models.CharField(blank=True, default="", max_length=255),
        ),
    ]
