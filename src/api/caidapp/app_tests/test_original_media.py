import json
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path
from struct import pack
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4
from zipfile import ZipFile, ZipInfo

from caidapp.app_tests.factories import UploadedArchiveFactory
from caidapp.models import MediaFile
from caidapp.tasks import create_original_media_zip
from caidapp.upload_services import build_upload_zip
from caidapp.zip_media_times import read_zip_file_times
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse


class OriginalMediaTest(TestCase):
    def test_extended_creation_time_can_appear_only_in_local_zip_header(self):
        """Read Info-ZIP creation time even when the central entry omits it."""
        info = ZipInfo("a.jpg", date_time=(2020, 3, 4, 9, 0, 0))
        info.header_offset = 0
        info.extra = pack("<HHBi", 0x5455, 5, 0b101, 1_600_000_000)
        extended = pack("<HHBii", 0x5455, 9, 0b101, 1_600_000_000, 1_599_999_999)
        stream = BytesIO(pack("<4s22xHH", b"PK\x03\x04", 0, len(extended)) + extended)
        times = read_zip_file_times(stream, info)
        self.assertEqual(times["created"]["value"], "2020-09-13T12:26:39+00:00")
        self.assertEqual(times["modified"]["value"], "2020-09-13T12:26:40+00:00")
        self.assertEqual(times["zip_dos_modified"], "2020-03-04T09:00:00")

    def test_restore_repack_and_export_preserve_original_and_timestamps(self):
        """Verify a restored original, exclude deleted entries and retain NTFS and DOS times."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            media_root = root / "media"
            media_root.mkdir()
            source_path = media_root / "source.zip"
            content = b"camera bytes and EXIF"
            entry = ZipInfo("camera-a/IMG_0001.JPG", date_time=(2020, 3, 4, 9, 0, 0))
            modified = 132278688000000000
            created = modified - 10_000_000
            entry.extra = pack("<HHIHHQQQ", 0x000A, 32, 0, 1, 24, modified, 0, created)
            with ZipFile(source_path, "w") as archive:
                archive.writestr(entry, content)
                archive.writestr("camera-b/IMG_0001.JPG", b"previously deleted")

            with self.settings(MEDIA_ROOT=media_root):
                upload = UploadedArchiveFactory(archivefile="source.zip")
                captured_at = datetime(2019, 1, 1, tzinfo=timezone.utc)
                media = MediaFile.objects.create(
                    parent=upload, original_filename=entry.filename, captured_at=captured_at
                )
                for _ in range(2):
                    call_command("restore_media_originals", upload_id=[upload.pk], stdout=StringIO())
                media.refresh_from_db()
                original = root / "media_originals" / media.original_file
                self.assertEqual(original.read_bytes(), content)
                self.assertEqual(media.metadata_json["original_file_sha256"], sha256(content).hexdigest())
                self.assertEqual(media.metadata_json["original_file_times"]["created"]["filetime"], created)
                self.assertEqual(media.metadata_json["original_file_times"]["modified"]["filetime"], modified)
                self.assertEqual(media.captured_at, captured_at)
                self.assertEqual(upload.mediafile_set.count(), 1)
                self.assertTrue(source_path.is_file())
                modified_at = datetime.fromisoformat(media.metadata_json["original_file_times"]["modified"]["value"])
                self.assertAlmostEqual(original.stat().st_mtime, modified_at.timestamp(), places=3)

                output = root / "export.zip"
                create_original_media_zip(upload.pk, output)
                with output.open("rb") as stream, ZipFile(stream) as archive:
                    self.assertEqual(archive.read(entry.filename), content)
                    self.assertNotIn("camera-b/IMG_0001.JPG", archive.namelist())
                    times = read_zip_file_times(stream, archive.getinfo(entry.filename))
                    self.assertEqual(times, media.metadata_json["original_file_times"])
                    self.assertEqual(json.loads(archive.read("_original_media.json"))[0]["file_times"], times)

                repacked = build_upload_zip(
                    [SimpleUploadedFile("source.zip", source_path.read_bytes())],
                    spreadsheet_file=SimpleUploadedFile("metadata.csv", b"original_path\ncamera-a/IMG_0001.JPG\n"),
                )
                with ZipFile(repacked.file) as archive:
                    self.assertEqual(read_zip_file_times(repacked.file, archive.getinfo(entry.filename)), times)

                with self.captureOnCommitCallbacks(execute=True):
                    media.delete()
                    self.assertTrue(original.is_file())
                self.assertFalse(original.exists())
                self.assertTrue(source_path.is_file())

    def test_original_export_requires_staff_and_uses_private_export_job(self):
        """Prevent a regular upload owner from requesting the admin export."""
        upload = UploadedArchiveFactory()
        MediaFile.objects.create(parent=upload, original_filename="a.jpg", original_file=f"{upload.pk}/a.jpg")
        user = upload.owner.user
        self.client.force_login(user)
        url = reverse("caidapp:export_original_media", args=[upload.pk])
        self.assertEqual(self.client.post(url).status_code, 403)
        user.is_staff = True
        user.save(update_fields=["is_staff"])
        with patch("caidapp.export_runs.start_export") as start:
            start.return_value.uuid = uuid4()
            response = self.client.post(url)
        self.assertRedirects(
            response, reverse("caidapp:task_detail", args=[start.return_value.uuid]), fetch_redirect_response=False
        )
        self.assertEqual(start.call_args.args[1:4], ("create_original_media_zip", [upload.pk, None], 1))
