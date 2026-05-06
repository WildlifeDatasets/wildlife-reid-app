from django.db import migrations


def set_is_for_identification_from_archive_flags(apps, schema_editor):
    UploadedArchive = apps.get_model("caidapp", "UploadedArchive")
    UploadedArchive.objects.filter(contains_identities=True).update(is_for_identification=True)
    UploadedArchive.objects.filter(contains_single_taxon=True).update(is_for_identification=True)
    UploadedArchive.objects.filter(taxon_for_identification__isnull=False).update(is_for_identification=True)


class Migration(migrations.Migration):

    dependencies = [
        ("caidapp", "0152_remove_uploadedarchive_mediafiles_imported_and_more"),
    ]

    operations = [
        migrations.RunPython(
            set_is_for_identification_from_archive_flags,
            migrations.RunPython.noop,
        ),
    ]
