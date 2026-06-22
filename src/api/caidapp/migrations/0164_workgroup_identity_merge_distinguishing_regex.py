from django.db import migrations, models

import caidapp.models


class Migration(migrations.Migration):
    dependencies = [
        ("caidapp", "0163_mergeidentitysuggestionexclusion"),
    ]

    operations = [
        migrations.AddField(
            model_name="workgroup",
            name="identity_merge_distinguishing_regex",
            field=models.CharField(
                blank=True,
                default="(?i)(?<![a-z0-9])juv[._ -]*(\\d{2,4})[._ -]+(\\d+)(?!\\d)",
                help_text=(
                    "Regex whose match or capture groups distinguish identities during merge suggestions. "
                    "If both names match but the extracted values differ, the pair is excluded."
                ),
                max_length=256,
                validators=[caidapp.models.validate_identity_merge_distinguishing_regex],
            ),
        ),
    ]
