# Generated manually to keep linked mediafiles when deleting a locality.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("caidapp", "0157_caiduser_show_base_between_regular_uploads"),
    ]

    operations = [
        migrations.AlterField(
            model_name="mediafile",
            name="locality",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=models.SET_NULL,
                related_name="mediafiles",
                to="caidapp.locality",
            ),
        ),
    ]
