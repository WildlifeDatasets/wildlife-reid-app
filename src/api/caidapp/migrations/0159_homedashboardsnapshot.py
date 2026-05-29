from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("caidapp", "0158_alter_mediafile_locality"),
    ]

    operations = [
        migrations.CreateModel(
            name="HomeDashboardSnapshot",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("payload", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "workgroup",
                    models.OneToOneField(blank=True, null=True, on_delete=models.deletion.CASCADE, to="caidapp.workgroup"),
                ),
            ],
        ),
    ]
