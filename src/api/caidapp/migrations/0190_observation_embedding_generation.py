from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("caidapp", "0189_shared_observation_features")]

    operations = [
        migrations.AddField(
            model_name="observationembeddingschedule",
            name="generation",
            field=models.PositiveBigIntegerField(default=0),
        ),
    ]
