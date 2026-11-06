"""Verify deferred private downloads and export lifetime independently of the browser."""

from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from caidapp import models
from caidapp.export_runs import artifact_path, cleanup_exports, execute_export, start_export
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from .factories import CaidUserFactory


class ExportRunTests(TestCase):
    def setUp(self):
        """Use isolated private storage and authenticate an export's owner."""
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        override = self.settings(PRIVATE_DATA_PATH=directory.name)
        override.enable()
        self.addCleanup(override.disable)
        self.user = CaidUserFactory()
        self.client.force_login(self.user.user)

    def submit(self):
        """Create a run without a broker, using the real submission and persistence code."""
        with patch("caidapp.export_runs.signature") as publish:
            with self.captureOnCommitCallbacks(execute=True):
                run = start_export(
                    self.user,
                    "create_mediafiles_zip",
                    [self.user.hash, [], "unused"],
                    2,
                    filename="mediafiles.zip",
                    label="Media files export",
                )
        return run, publish.call_args.kwargs["args"]

    def complete(self):
        """Execute the real ZIP writer and durable wrapper."""
        run, args = self.submit()
        execute_export(*args)
        run.refresh_from_db()
        return run

    def test_export_survives_leaving_page_and_can_be_downloaded_repeatedly(self):
        """Durable history offers the same artifact without re-running the export."""
        run = self.complete()
        self.assertEqual(run.status, "succeeded")
        self.assertGreater(run.artifact.size, 0)
        self.assertGreater(run.artifact.expires_at, timezone.now() + timedelta(days=6))
        page = self.client.get(reverse("caidapp:task_list"), {"state": "downloads"})
        self.assertContains(page, "mediafiles.zip")
        url = reverse("caidapp:task_download", kwargs={"run_uuid": run.uuid})
        self.assertContains(page, url)
        for _ in range(2):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(b"".join(response.streaming_content).startswith(b"PK"))
            response.close()
        detail = self.client.get(reverse("caidapp:task_detail", kwargs={"run_uuid": run.uuid}))
        self.assertContains(detail, 'id="export-download"')
        self.assertNotContains(detail, 'id="embedding-retry"')

    def test_another_workgroup_member_cannot_access_export(self):
        """Export ownership is narrower than shared embedding-run visibility."""
        run = self.complete()
        colleague = CaidUserFactory()
        colleague.workgroup = self.user.workgroup
        colleague.save(update_fields=["workgroup"])
        self.client.force_login(colleague.user)
        for route in ("task_detail", "task_status", "task_result", "task_download"):
            self.assertEqual(
                self.client.get(reverse(f"caidapp:{route}", kwargs={"run_uuid": run.uuid})).status_code, 404
            )
        self.assertEqual(
            self.client.get(reverse("caidapp:check_zip_status", args=[run.prepare_task_id])).status_code, 404
        )
        self.assertNotContains(self.client.get(reverse("caidapp:task_list")), "mediafiles.zip")

    def test_legacy_polling_uses_authorized_download_endpoint(self):
        """Automatic download and delayed download resolve to the identical private artifact."""
        run = self.complete()
        data = self.client.get(reverse("caidapp:check_zip_status", args=[run.prepare_task_id])).json()
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["download_url"], reverse("caidapp:task_download", kwargs={"run_uuid": run.uuid}))

    def test_expiration_and_cleanup_preserve_successful_history(self):
        """Expiry denies downloads immediately; cleanup only removes the file."""
        run = self.complete()
        path = artifact_path(run.artifact)
        models.ExportArtifact.objects.filter(run=run).update(expires_at=timezone.now() - timedelta(seconds=1))
        self.assertEqual(
            self.client.get(reverse("caidapp:task_download", kwargs={"run_uuid": run.uuid})).status_code, 410
        )
        self.assertEqual(cleanup_exports(), 1)
        self.assertFalse(path.exists())
        self.assertEqual(cleanup_exports(), 0)
        run.refresh_from_db()
        self.assertEqual(run.status, "succeeded")
        payload = self.client.get(reverse("caidapp:task_status", kwargs={"run_uuid": run.uuid})).json()
        self.assertIsNone(payload["download_url"])
        self.assertIn("no longer available", payload["result_summary"])

    def test_missing_file_does_not_change_completion_and_redelivery_does_not_recreate_it(self):
        """A completed export remains historical even when its artifact has disappeared."""
        run, args = self.submit()
        execute_export(*args)
        run.refresh_from_db()
        artifact_path(run.artifact).unlink()
        execute_export(*args)
        self.assertEqual(
            self.client.get(reverse("caidapp:task_download", kwargs={"run_uuid": run.uuid})).status_code, 410
        )
        run.refresh_from_db()
        self.assertEqual(run.status, "succeeded")

    def test_worker_error_is_recorded_without_exposing_traceback(self):
        """A failed export has a durable failure rather than a broken download link."""
        run, args = self.submit()
        with patch("caidapp.tasks.create_mediafiles_zip.run", side_effect=OSError("secret path")):
            with self.assertRaises(OSError):
                execute_export(*args)
        run.refresh_from_db()
        self.assertEqual(run.status, "failed")
        self.assertNotIn("secret", run.message)
        self.assertIsNone(run.artifact.expires_at)

    def test_publish_failure_remains_visible(self):
        """A broker outage leaves a failed task in history instead of losing the request."""
        with patch("caidapp.export_runs.signature", side_effect=OSError("broker unavailable")):
            with self.captureOnCommitCallbacks(execute=True):
                run = start_export(
                    self.user,
                    "create_mediafiles_zip",
                    [self.user.hash, [], "unused"],
                    2,
                    filename="mediafiles.zip",
                    label="Export",
                )
        run.refresh_from_db()
        self.assertEqual(run.status, "failed")

    def test_download_and_cleanup_reject_an_outside_path(self):
        """A corrupt artifact reference cannot expose or delete another file."""
        run = self.complete()
        outside = Path(run.artifact.path).parents[2] / "keep.txt"
        outside.write_text("private data")
        models.ExportArtifact.objects.filter(run=run).update(path=str(outside))
        self.assertEqual(
            self.client.get(reverse("caidapp:task_download", kwargs={"run_uuid": run.uuid})).status_code, 410
        )
        models.ExportArtifact.objects.filter(run=run).update(expires_at=timezone.now() - timedelta(days=1))
        self.assertEqual(cleanup_exports(), 0)
        self.assertEqual(outside.read_text(), "private data")
