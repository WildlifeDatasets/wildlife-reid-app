from caidapp import tasks
from caidapp.app_tests.factories import CaidUserFactory, UploadedArchiveFactory
from django.test import TestCase


class IdentificationCallbackTest(TestCase):
    def setUp(self):
        """Create an archive for identification callback status tests."""
        self.caiduser = CaidUserFactory()
        self.workgroup = self.caiduser.workgroup
        self.archive = UploadedArchiveFactory(
            owner=self.caiduser,
            identification_status="IAIP",
        )

    def test_bulk_failure_saves_error_on_uploaded_archive(self):
        """Persist worker failure status and diagnostic text on the uploaded archive."""
        tasks.identify_bulk_on_success.run(
            {"status": "ERROR", "error": "Worker failed"},
            workgroup_id=self.workgroup.id,
            uploaded_archive_ids=[self.archive.id],
        )

        self.archive.refresh_from_db()
        self.assertEqual(self.archive.identification_status, "F")
        self.assertEqual(self.archive.status_message, "Identification failed. Worker failed")

    def test_missing_worker_status_saves_diagnostic_message(self):
        """Record an unknown state with a diagnostic when the worker omits its status field."""
        tasks.identify_bulk_on_success.run(
            {},
            workgroup_id=self.workgroup.id,
            uploaded_archive_ids=[self.archive.id],
        )

        self.archive.refresh_from_db()
        self.assertEqual(self.archive.identification_status, "U")
        self.assertIn("missing 'status' field", self.archive.status_message)
