from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("caidapp", "0190_observation_embedding_generation")]

    operations = [
        migrations.AlterField(
            model_name="taskrun",
            name="workgroup",
            field=models.ForeignKey(blank=True, null=True, on_delete=models.CASCADE, to="caidapp.workgroup"),
        ),
    ]
