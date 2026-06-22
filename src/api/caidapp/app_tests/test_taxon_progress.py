from pathlib import Path
from unittest.mock import Mock, patch

from django.test import TestCase
from django.template.loader import render_to_string
from django.urls import reverse

from caidapp import tasks
from caidapp.app_tests.factories import CaidUserFactory, UploadedArchiveFactory


class TaxonProgressApiTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory()
        self.client.force_login(self.caiduser.user)
        self.archive = UploadedArchiveFactory(
            owner=self.caiduser,
            taxon_status="TAIP",
            taxon_task_id="taxon-task-1",
        )
        self.url = reverse("caidapp:uploads_status_api", args=["species"])

    @patch("caidapp.views.AsyncResult")
    def test_returns_worker_progress(self, async_result):
        async_result.return_value = Mock(
            state="PROGRESS",
            info={"percent": 54, "stage": "detection", "message": "Detecting animals"},
        )

        response = self.client.get(self.url, {"ids": str(self.archive.id)})

        self.assertEqual(response.status_code, 200)
        progress = response.json()["archives"][0]["progress"]
        self.assertEqual(progress["percent"], 54)
        self.assertEqual(progress["stage"], "detection")
        async_result.assert_called_once_with("taxon-task-1")

    @patch("caidapp.views.AsyncResult")
    def test_successful_worker_waiting_for_import_reports_99_percent(self, async_result):
        async_result.return_value = Mock(state="SUCCESS", info={"status": "DONE"})

        response = self.client.get(self.url, {"ids": str(self.archive.id)})

        progress = response.json()["archives"][0]["progress"]
        self.assertEqual(progress["percent"], 99)
        self.assertEqual(progress["stage"], "import_results")

    @patch("caidapp.views.AsyncResult", side_effect=RuntimeError("Redis unavailable"))
    def test_backend_error_keeps_status_endpoint_available(self, async_result):
        response = self.client.get(self.url, {"ids": str(self.archive.id)})

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["archives"][0]["progress"])

    @patch("caidapp.views.AsyncResult")
    def test_requested_ids_limit_backend_lookups(self, async_result):
        other = UploadedArchiveFactory(
            owner=self.caiduser,
            taxon_status="TAIP",
            taxon_task_id="taxon-task-2",
        )
        async_result.return_value = Mock(state="PENDING", info=None)

        response = self.client.get(self.url, {"ids": str(self.archive.id)})

        self.assertEqual([item["id"] for item in response.json()["archives"]], [self.archive.id])
        async_result.assert_called_once_with("taxon-task-1")
        self.assertNotEqual(other.id, self.archive.id)


class TaxonTaskDispatchTest(TestCase):
    @patch("caidapp.tasks.update_metadata_csv_by_uploaded_archive")
    @patch("caidapp.tasks._run_taxon_classification_init")
    @patch("caidapp.tasks._run_taxon_classification_init_message")
    @patch("caidapp.tasks.signature")
    def test_dispatch_stores_worker_task_id(
        self,
        signature_mock,
        init_message,
        init,
        update_metadata,
    ):
        archive = UploadedArchiveFactory()
        init.return_value = (Path("images.zip"), Path("output"), Path("metadata.csv"))
        signature_mock.return_value.apply_async.return_value = Mock(id="worker-task-id", task_id="worker-task-id")

        tasks.run_species_prediction_async(archive)

        archive.refresh_from_db()
        self.assertEqual(archive.taxon_task_id, "worker-task-id")


class TaxonProgressTemplateTest(TestCase):
    def test_processing_archive_renders_progress_indicator(self):
        archive = UploadedArchiveFactory(taxon_status="TAIP", status_message="Processing upload")

        html = render_to_string("caidapp/upload_progress.html", {"uploadedarchive": archive})

        self.assertIn(f'id="progress-{archive.id}"', html)
        self.assertIn(f'data-archive-id="{archive.id}"', html)
        self.assertNotIn(" d-none", html)
