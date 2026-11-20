from zipfile import BadZipFile, ZipFile

from caidapp.models import UploadedArchive
from caidapp.services.original_media import (
    archive_entries,
    has_source_zip_times,
    register_existing_added_original,
    restore_media_original,
)
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction


class Command(BaseCommand):
    help = "Restore and verify originals for registered media while retaining the source ZIPs."

    def add_arguments(self, parser):
        """Allow operators to select uploads or resume by upload ID."""
        parser.add_argument("--upload-id", type=int, action="append")
        parser.add_argument("--after-id", type=int, default=0)

    def handle(self, *args, **options):
        """Restore originals for completed uploads, reporting entries that need attention."""
        ids = UploadedArchive.objects.filter(import_finished=True, pk__gt=options["after_id"])
        if options["upload_id"]:
            ids = ids.filter(pk__in=options["upload_id"])
        failures = 0
        for upload_id in ids.order_by("pk").values_list("pk", flat=True).iterator():
            with transaction.atomic():
                upload = UploadedArchive.objects.select_for_update().get(pk=upload_id)
                if upload.taxon_status == "TAIP" or not upload.import_finished:
                    self.stderr.write(f"Upload {upload_id}: skipped while processing.")
                    failures += 1
                    continue
                failures += self._restore_upload(upload)
        if failures:
            raise CommandError(f"{failures} media files or uploads need attention. All input ZIPs were retained.")

    def _restore_upload(self, upload):
        failures = 0
        verified = 0
        mediafiles = upload.mediafile_set.select_for_update()
        for media in mediafiles.filter(added_directory__isnull=False).iterator():
            try:
                register_existing_added_original(media)
                verified += 1
            except (OSError, ValidationError) as exc:
                self.stderr.write(f"Upload {upload.pk}, media {media.pk}: {exc}")
                failures += 1

        archived = mediafiles.filter(added_directory__isnull=True)
        if archived.exists():
            try:
                preserve_zip_times = has_source_zip_times(upload)
                with upload.archivefile.open("rb") as stream, ZipFile(stream) as source:
                    entries = archive_entries(source)
                    for media in archived.iterator():
                        try:
                            restore_media_original(
                                media, stream, source, entries, preserve_zip_times=preserve_zip_times
                            )
                            verified += 1
                        except (BadZipFile, OSError, RuntimeError, ValidationError, ValueError) as exc:
                            self.stderr.write(f"Upload {upload.pk}, media {media.pk}: {exc}")
                            failures += 1
            except (BadZipFile, OSError, ValidationError, ValueError) as exc:
                self.stderr.write(f"Upload {upload.pk}: {exc}")
                failures += 1
        self.stdout.write(f"Upload {upload.pk}: {verified} originals verified, {failures} problems.")
        return failures
