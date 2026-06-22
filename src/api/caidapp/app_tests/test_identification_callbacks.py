from django.test import TestCase

from caidapp import tasks
from caidapp.app_tests.factories import CaidUserFactory, UploadedArchiveFactory


class IdentificationCallbackTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory()
        self.workgroup = self.caiduser.workgroup
        self.archive = UploadedArchiveFactory(
            owner=self.caiduser,
            identification_status="IAIP",
        )

    def test_bulk_failure_saves_error_on_uploaded_archive(self):
        tasks.identify_bulk_on_success.run(
            {"status": "ERROR", "error": "Worker failed"},
            workgroup_id=self.workgroup.id,
            uploaded_archive_ids=[self.archive.id],
        )

        self.archive.refresh_from_db()
        self.assertEqual(self.archive.identification_status, "F")
        self.assertEqual(self.archive.status_message, "Identification failed. Worker failed")

    def test_missing_worker_status_saves_diagnostic_message(self):
        tasks.identify_bulk_on_success.run(
            {},
            workgroup_id=self.workgroup.id,
            uploaded_archive_ids=[self.archive.id],
        )

        self.archive.refresh_from_db()
        self.assertEqual(self.archive.identification_status, "U")
        self.assertIn("missing 'status' field", self.archive.status_message)
