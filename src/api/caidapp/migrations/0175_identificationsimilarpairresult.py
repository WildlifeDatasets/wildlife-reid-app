from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [("caidapp", "0174_bboxdetectionjob")]

    operations = [
        migrations.CreateModel(
            name="IdentificationSimilarPairResult",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
                ("status", models.CharField(default="processing", max_length=32)),
                ("message", models.TextField(blank=True, default="")),
                ("task_id", models.CharField(blank=True, default="", max_length=255)),
                ("input_csv_filename", models.CharField(blank=True, default="", max_length=255)),
                ("output_csv_filename", models.CharField(blank=True, default="", max_length=255)),
                ("mega_pairs", models.JSONField(blank=True, default=list)),
                ("local_pairs", models.JSONField(blank=True, default=list)),
                ("workgroup", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="similar_pair_results", to="caidapp.workgroup")),
            ],
            options={"ordering": ("-created_at", "-id")},
        ),
    ]
