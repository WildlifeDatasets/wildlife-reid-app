from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("caidapp", "0188_observation_identity_reviews")]
    operations = [
        migrations.AddField(
            model_name="observationembedding", name="local_features", field=models.JSONField(blank=True, null=True)
        ),
        migrations.AddField(
            model_name="observationembedding", name="feature_image", field=models.CharField(blank=True, max_length=500)
        ),
        migrations.AddField(
            model_name="mediafileidentificationsuggestion", name="feature_images",
            field=models.JSONField(blank=True, default=list),
        ),
    ]
