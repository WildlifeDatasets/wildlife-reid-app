from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("caidapp", "0170_observationimport"),
    ]

    operations = [
        migrations.AlterModelOptions(
            name="mediafile",
            options={"ordering": ["captured_at", "id"]},
        ),
    ]
