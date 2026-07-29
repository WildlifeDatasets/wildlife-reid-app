from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [("caidapp", "0169_identificationworkergpuheartbeat")]

    operations = [
        migrations.CreateModel(
            name="ObservationImport",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("source_filename", models.CharField(max_length=255)),
                ("stored_file", models.CharField(max_length=512)),
                ("task_id", models.CharField(blank=True, default="", max_length=255)),
                ("status", models.CharField(choices=[("QUEUED", "Queued"), ("PROCESSING", "Processing"), ("SUCCEEDED", "Succeeded"), ("FAILED", "Failed")], default="QUEUED", max_length=16)),
                ("create_missing_localities", models.BooleanField(default=False)),
                ("create_missing_identities", models.BooleanField(default=False)),
                ("created_count", models.PositiveIntegerField(default=0)),
                ("updated_count", models.PositiveIntegerField(default=0)),
                ("error_message", models.TextField(blank=True, default="")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
                ("caiduser", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="observation_imports", to="caidapp.caiduser")),
            ],
            options={"ordering": ("-created_at", "-id")},
        ),
    ]
