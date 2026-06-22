import django.db.models.deletion
from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [
        ("caidapp", "0162_uploadedarchive_taxon_task_id"),
    ]

    operations = [
        migrations.CreateModel(
            name="MergeIdentitySuggestionExclusion",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "identity_a",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="merge_suggestion_exclusions_as_a",
                        to="caidapp.individualidentity",
                    ),
                ),
                (
                    "identity_b",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="merge_suggestion_exclusions_as_b",
                        to="caidapp.individualidentity",
                    ),
                ),
                (
                    "workgroup",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="merge_suggestion_exclusions",
                        to="caidapp.workgroup",
                    ),
                ),
            ],
            options={
                "constraints": [
                    models.UniqueConstraint(
                        fields=("workgroup", "identity_a", "identity_b"),
                        name="unique_merge_identity_suggestion_exclusion",
                    )
                ],
            },
        ),
    ]
