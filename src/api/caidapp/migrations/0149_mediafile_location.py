import location_field.models.plain
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("caidapp", "0148_bboxsequence"),
    ]

    operations = [
        migrations.AddField(
            model_name="mediafile",
            name="location",
            field=location_field.models.plain.PlainLocationField(blank=True, max_length=63, null=True),
        ),
    ]
