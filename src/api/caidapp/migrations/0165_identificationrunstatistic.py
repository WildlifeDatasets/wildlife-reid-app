from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("caidapp", "0164_workgroup_identity_merge_distinguishing_regex"),
    ]

    operations = [
        migrations.CreateModel(
            name="IdentificationRunStatistic",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
                ("duration_seconds", models.FloatField(blank=True, null=True)),
                (
                    "operation",
                    models.CharField(
                        choices=[("init", "Init identification"), ("identify", "Identify")],
                        max_length=32,
                    ),
                ),
                ("image_number", models.PositiveIntegerField(default=0)),
                ("video_number", models.PositiveIntegerField(default=0)),
                ("task_id", models.CharField(blank=True, default="", max_length=255)),
                ("status", models.CharField(blank=True, default="started", max_length=64)),
                (
                    "workgroup",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="identification_run_statistics",
                        to="caidapp.workgroup",
                    ),
                ),
            ],
            options={
                "ordering": ("-created_at", "-id"),
            },
        ),
    ]
