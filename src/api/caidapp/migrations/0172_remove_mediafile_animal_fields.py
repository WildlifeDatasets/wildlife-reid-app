from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("caidapp", "0171_alter_mediafile_ordering"),
    ]

    operations = [
        migrations.RemoveField(model_name="mediafile", name="animal_number"),
        migrations.RemoveField(model_name="mediafile", name="identity"),
        migrations.RemoveField(model_name="mediafile", name="identity_is_representative"),
        migrations.RemoveField(model_name="mediafile", name="orientation"),
        migrations.RemoveField(model_name="mediafile", name="predicted_taxon"),
        migrations.RemoveField(model_name="mediafile", name="predicted_taxon_confidence"),
        migrations.RemoveField(model_name="mediafile", name="taxon"),
        migrations.RemoveField(model_name="mediafile", name="taxon_verified"),
        migrations.RemoveField(model_name="mediafile", name="taxon_verified_at"),
    ]
