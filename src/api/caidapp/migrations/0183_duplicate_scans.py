import uuid

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("caidapp", "0182_export_artifact"),
    ]

    operations = [
        migrations.CreateModel(
            name="DuplicateFileCleanup",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("storage_name", models.CharField(max_length=500)),
                ("status", models.CharField(default="pending", max_length=16)),
                ("message", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name="DuplicateScan",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("request_key", models.UUIDField(default=uuid.uuid4)),
                ("uuid", models.UUIDField(default=uuid.uuid4, editable=False, unique=True)),
                ("status", models.CharField(default="queued", max_length=16)),
                ("task_id", models.CharField(blank=True, max_length=64)),
                ("include_crops", models.BooleanField(default=True)),
                ("items", models.JSONField(default=list)),
                ("algorithm", models.CharField(blank=True, max_length=64)),
                ("message", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
                ("requested_by", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to="caidapp.caiduser")),
            ],
        ),
        migrations.CreateModel(
            name="DuplicateProposal",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("uuid", models.UUIDField(default=uuid.uuid4, editable=False, unique=True)),
                ("kind", models.CharField(max_length=16)),
                ("status", models.CharField(db_index=True, default="pending", max_length=16)),
                ("evidence", models.JSONField(default=dict)),
                ("decision", models.JSONField(default=dict)),
                ("reviewed_at", models.DateTimeField(blank=True, null=True)),
                (
                    "scan",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="proposals",
                        to="caidapp.duplicatescan",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="DuplicateProposalMember",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("original_id", models.BigIntegerField()),
                ("snapshot", models.JSONField(default=dict)),
                ("position", models.PositiveIntegerField(default=0)),
                (
                    "mediafile",
                    models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to="caidapp.mediafile"),
                ),
                (
                    "proposal",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="members",
                        to="caidapp.duplicateproposal",
                    ),
                ),
            ],
            options={
                "ordering": ["position", "pk"],
                "constraints": [
                    models.UniqueConstraint(fields=("proposal", "original_id"), name="unique_duplicate_member")
                ],
            },
        ),
        migrations.AddConstraint(
            model_name="duplicatescan",
            constraint=models.UniqueConstraint(fields=("requested_by", "request_key"), name="unique_duplicate_request"),
        ),
    ]
