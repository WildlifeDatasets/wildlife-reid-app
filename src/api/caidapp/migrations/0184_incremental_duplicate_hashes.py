import uuid

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("caidapp", "0183_duplicate_scans"),
    ]

    operations = [
        migrations.AddField(
            model_name="duplicateproposal",
            name="fingerprint",
            field=models.CharField(blank=True, max_length=64, null=True, unique=True),
        ),
        migrations.AddField(
            model_name="duplicatescan",
            name="failed_count",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="duplicatescan",
            name="processed",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="duplicatescan",
            name="purpose",
            field=models.CharField(default="legacy", max_length=16),
        ),
        migrations.AddField(
            model_name="duplicatescan",
            name="skipped",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="duplicatescan",
            name="total",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="duplicatescan",
            name="workgroup",
            field=models.ForeignKey(
                blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to="caidapp.workgroup"
            ),
        ),
        migrations.AddField(
            model_name="mediafile",
            name="file_sha256",
            field=models.CharField(blank=True, db_index=True, default="", editable=False, max_length=64),
        ),
        migrations.AddField(
            model_name="mediafile",
            name="hash_computed_at",
            field=models.DateTimeField(blank=True, editable=False, null=True),
        ),
        migrations.AddField(
            model_name="mediafile",
            name="hash_details",
            field=models.JSONField(default=dict, editable=False),
        ),
        migrations.AddField(
            model_name="mediafile",
            name="hash_message",
            field=models.TextField(blank=True, default="", editable=False),
        ),
        migrations.AddField(
            model_name="mediafile",
            name="hash_revision",
            field=models.PositiveIntegerField(default=0, editable=False),
        ),
        migrations.AddField(
            model_name="mediafile",
            name="hash_status",
            field=models.CharField(blank=True, default="", editable=False, max_length=16),
        ),
        migrations.AddField(
            model_name="mediafile",
            name="hash_version",
            field=models.CharField(blank=True, default="", editable=False, max_length=32),
        ),
        migrations.AddField(
            model_name="mediafile",
            name="pixel_sha256",
            field=models.CharField(blank=True, db_index=True, default="", editable=False, max_length=64),
        ),
        migrations.AddField(
            model_name="mediafile",
            name="reference_sha256",
            field=models.CharField(blank=True, db_index=True, default="", editable=False, max_length=64),
        ),
        migrations.AlterField(
            model_name="duplicatescan",
            name="requested_by",
            field=models.ForeignKey(
                blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to="caidapp.caiduser"
            ),
        ),
        migrations.CreateModel(
            name="DuplicateHashBatch",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("uuid", models.UUIDField(default=uuid.uuid4, editable=False, unique=True)),
                ("status", models.CharField(default="pending", max_length=16)),
                ("items", models.JSONField(default=list)),
                ("message", models.TextField(blank=True)),
                ("dispatched_at", models.DateTimeField(blank=True, null=True)),
                (
                    "scan",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="hash_batches",
                        to="caidapp.duplicatescan",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="DuplicateHashSchedule",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("task_id", models.CharField(blank=True, max_length=64, null=True)),
                ("eta", models.DateTimeField(blank=True, null=True)),
                ("dirty", models.BooleanField(default=False)),
                ("message", models.TextField(blank=True)),
                (
                    "active_scan",
                    models.ForeignKey(
                        blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to="caidapp.duplicatescan"
                    ),
                ),
                (
                    "workgroup",
                    models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to="caidapp.workgroup"),
                ),
            ],
        ),
        migrations.CreateModel(
            name="DuplicateIdentitySuggestion",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("active", models.BooleanField(default=True)),
                ("evidence", models.JSONField(default=list)),
                (
                    "identity_a",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="duplicate_candidates_a",
                        to="caidapp.individualidentity",
                    ),
                ),
                (
                    "identity_b",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="duplicate_candidates_b",
                        to="caidapp.individualidentity",
                    ),
                ),
                ("workgroup", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to="caidapp.workgroup")),
            ],
            options={
                "constraints": [
                    models.UniqueConstraint(
                        fields=("workgroup", "identity_a", "identity_b"), name="unique_duplicate_identity_pair"
                    )
                ],
            },
        ),
    ]
