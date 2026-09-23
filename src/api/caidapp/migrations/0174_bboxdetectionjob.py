from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [("caidapp", "0173_animalobservation_no_detection_placeholder")]
    operations = [
        migrations.CreateModel(
            name="BboxDetectionJob",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("batch", models.UUIDField(db_index=True)),
                ("purpose", models.CharField(default="replace", max_length=16)),
                ("status", models.CharField(default="queued", max_length=16)),
                ("task_id", models.CharField(blank=True, max_length=64)),
                ("options", models.JSONField(default=dict)),
                ("snapshot", models.JSONField(default=dict)),
                ("result", models.JSONField(default=dict)),
                ("message", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
                ("requested_by", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to="caidapp.caiduser")),
                ("mediafile", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to="caidapp.mediafile")),
            ],
            options={
                "constraints": [
                    models.UniqueConstraint(fields=("batch", "mediafile"), name="unique_bbox_batch_mediafile")
                ]
            },
        ),
    ]
