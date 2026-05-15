from django.db import migrations


def backfill_mediafile_locality_from_uploadedarchive(apps, schema_editor):
    UploadedArchive = apps.get_model("caidapp", "UploadedArchive")
    MediaFile = apps.get_model("caidapp", "MediaFile")

    for uploaded_archive in UploadedArchive.objects.exclude(locality_at_upload_object__isnull=True).iterator():
        MediaFile.objects.filter(parent=uploaded_archive, locality__isnull=True).update(
            locality_id=uploaded_archive.locality_at_upload_object_id
        )


class Migration(migrations.Migration):

    dependencies = [
        ("caidapp", "0155_uploadedarchive_path_structure_regex"),
    ]

    operations = [
        migrations.RunPython(
            backfill_mediafile_locality_from_uploadedarchive,
            migrations.RunPython.noop,
        ),
    ]
