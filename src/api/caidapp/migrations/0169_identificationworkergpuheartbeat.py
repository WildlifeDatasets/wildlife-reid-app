from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("caidapp", "0168_identificationmodel_created_at"),
    ]

    operations = [
        migrations.CreateModel(
            name="IdentificationWorkerGpuHeartbeat",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("recorded_at", models.DateTimeField(auto_now_add=True)),
                ("available", models.BooleanField(default=False)),
                ("device", models.CharField(default="cuda:0", max_length=64)),
                ("device_name", models.CharField(blank=True, default="", max_length=255)),
                ("free_memory_gb", models.FloatField(blank=True, null=True)),
                ("total_memory_gb", models.FloatField(blank=True, null=True)),
                ("error_message", models.TextField(blank=True, default="")),
            ],
            options={"ordering": ("-recorded_at", "-id")},
        ),
        migrations.AddIndex(
            model_name="identificationworkergpuheartbeat",
            index=models.Index(fields=["-recorded_at"], name="gpu_heartbeat_time_idx"),
        ),
    ]
