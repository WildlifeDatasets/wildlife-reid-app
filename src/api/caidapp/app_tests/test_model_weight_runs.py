"""Verify administrator access and reconciliation for shared model downloads."""

from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from .. import model_weight_runs, models, tasks
from .factories import CaidUserFactory


class ModelWeightRunTests(TestCase):
    def setUp(self):
        """Create an installation administrator without a workgroup."""
        self.admin = CaidUserFactory(workgroup=None)
        self.admin.user.is_staff = self.admin.user.is_superuser = True
        self.admin.user.save(update_fields=["is_staff", "is_superuser"])
        self.client.force_login(self.admin.user)
        self.start_url = reverse("caidapp:download_model_weights")

    def test_start_is_post_only_and_reuses_the_active_run(self):
        """Publish each worker job once with deduplicated downloadable model sources."""
        installed_sources = {model.get_runtime_model_source() for model in models.IdentificationModel.objects.all()}
        base = models.IdentificationModel.objects.create(name="Base", model_path="timm:resnet18")
        models.IdentificationModel.objects.create(
            name="Trained", model_path="file:/models/trained.pth", source_identification_model=base
        )
        self.assertEqual(self.client.get(self.start_url).status_code, 405)
        with patch.object(model_weight_runs.app, "send_task") as send, self.captureOnCommitCallbacks(execute=True):
            first = self.client.post(self.start_url)
            second = self.client.post(self.start_url)
        run = models.TaskRun.objects.get(kind=model_weight_runs.KIND)
        url = reverse("caidapp:task_detail", args=[run.uuid])
        self.assertEqual(first.url, url)
        self.assertEqual(second.url, url)
        self.assertIsNone(run.workgroup_id)
        self.assertEqual(send.call_count, 2)
        self.assertEqual(send.call_args_list[0].kwargs["queue"], "taxon_worker")
        identification = send.call_args_list[1].kwargs
        self.assertEqual(identification["queue"], "identification_worker")
        self.assertEqual(identification["kwargs"]["model_sources"], sorted(installed_sources | {"resnet18"}))

    def test_publication_failure_still_allows_the_other_worker_to_run(self):
        """Save the failed submission and continue tracking the worker that received its task."""
        with (
            patch.object(model_weight_runs.app, "send_task", side_effect=[OSError("broker unavailable"), None]),
            self.assertLogs(model_weight_runs.logger, level="ERROR"),
            self.captureOnCommitCallbacks(execute=True),
        ):
            run = model_weight_runs.start(self.admin)
        run.refresh_from_db()
        self.assertEqual(run.result["taxon"]["status"], "failed")
        self.assertEqual(run.result["identification"]["status"], "queued")
        self.assertEqual(run.request_key, model_weight_runs.ACTIVE_KEY)

    def test_worker_progress_and_outcome_are_visible_and_persisted(self):
        """Reconcile a mixed result without requiring an open browser."""
        run = model_weight_runs.start(self.admin)
        taxon_id = run.result["taxon"]["task_id"]
        worker_meta = {
            "status": "PROGRESS",
            "result": {"current": "SAM3", "done": 1, "total": 5},
        }
        status_url = reverse("caidapp:task_status", args=[run.uuid])
        with patch.object(model_weight_runs.app.backend, "get_task_meta", return_value=worker_meta):
            payload = self.client.get(status_url).json()
        self.assertEqual(payload["status"], "running")
        self.assertIn("SAM3 (2 of 5)", payload["steps"][0]["message"])

        def finish(task_id):
            failures = ["SAM3"] if task_id == taxon_id else []
            return {"status": "SUCCESS", "result": {"total": 5, "failures": failures}}

        with patch.object(model_weight_runs.app.backend, "get_task_meta", side_effect=finish):
            tasks.maintain_model_weight_download.run()
        run.refresh_from_db()
        self.assertEqual(run.status, "partial")
        self.assertIsNone(run.request_key)
        with patch.object(model_weight_runs.app.backend, "get_task_meta") as backend:
            payload = self.client.get(status_url).json()
            backend.assert_not_called()
        self.assertTrue(payload["is_terminal"])
        self.assertIn("Failed: SAM3", payload["steps"][0]["message"])
        self.assertContains(self.client.get(reverse("caidapp:task_detail", args=[run.uuid])), "Failed: SAM3")
        self.assertContains(self.client.get(reverse("caidapp:task_list")), "Download model weights")

    def test_staff_without_installation_admin_access_cannot_start_or_read(self):
        """Keep system-wide tasks out of the regular task views."""
        run = model_weight_runs.start(self.admin)
        staff = CaidUserFactory()
        staff.user.is_staff = True
        staff.user.save(update_fields=["is_staff"])
        self.client.force_login(staff.user)
        self.assertEqual(self.client.post(self.start_url).status_code, 403)
        for route in ("task_detail", "task_status", "task_result"):
            self.assertEqual(self.client.get(reverse(f"caidapp:{route}", args=[run.uuid])).status_code, 404)
        self.assertNotContains(self.client.get(reverse("caidapp:task_list")), "Download model weights")
