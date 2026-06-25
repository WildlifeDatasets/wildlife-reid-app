from unittest.mock import Mock, patch

from django.test import TestCase
from django.urls import reverse

from caidapp import models, tasks, views
from caidapp.app_tests.factories import CaidUserFactory, MediaFileFactory, UploadedArchiveFactory


class IdentificationProgressApiTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory()
        self.client.force_login(self.caiduser.user)
        self.workgroup = self.caiduser.workgroup
        self.url = reverse("caidapp:identification_progress_api")

    @patch("caidapp.views.AsyncResult")
    def test_returns_reid_worker_progress(self, async_result):
        self.workgroup.identification_reid_status = "Processing"
        self.workgroup.identification_reid_message = "Running identification"
        self.workgroup.identification_scheduled_run_task_id = "identify-task-1"
        self.workgroup.save()
        async_result.return_value = Mock(
            state="PROGRESS",
            info={"percent": 41, "stage": "identify", "message": "Comparing images"},
        )

        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        payload = response.json()["run"]
        self.assertEqual(payload["progress"]["percent"], 41)
        self.assertEqual(payload["progress"]["stage"], "identify")
        async_result.assert_called_once_with("identify-task-1")

    @patch("caidapp.views.AsyncResult", side_effect=RuntimeError("Redis unavailable"))
    def test_backend_error_keeps_progress_endpoint_available(self, async_result):
        self.workgroup.identification_init_status = "Processing"
        self.workgroup.identification_init_message = "Initializing"
        self.workgroup.identification_scheduled_init_task_id = "init-task-1"
        self.workgroup.save()

        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["init"]["progress"]["stage"], "starting")

    def test_dash_identification_progress_uses_compact_circular_indicators(self):
        self.workgroup.identification_init_status = "Processing"
        self.workgroup.identification_init_message = "Encoding references"
        self.workgroup.save()

        response = self.client.get(reverse("caidapp:dash_identities"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "ident-progress-dot")
        self.assertContains(response, "--ident-progress")
        self.assertContains(response, "data-operation=\"init\"")
        self.assertNotContains(response, "Identification progress")
        self.assertNotContains(response, "ident-progress-bar")


class IdentificationRunStatisticTest(TestCase):
    def test_finish_identification_run_statistic_sets_duration(self):
        workgroup = CaidUserFactory().workgroup
        statistic = tasks.create_identification_run_statistic(
            workgroup=workgroup,
            operation="identify",
            image_number=2,
            video_number=1,
        )

        tasks.finish_identification_run_statistic(statistic.id, "finished", "task-1")

        statistic.refresh_from_db()
        self.assertEqual(statistic.status, "finished")
        self.assertEqual(statistic.task_id, "task-1")
        self.assertIsNotNone(statistic.finished_at)
        self.assertGreaterEqual(statistic.duration_seconds, 0)

    @patch("caidapp.views.signature")
    def test_bulk_identification_dispatch_creates_statistic(self, signature_mock):
        caiduser = CaidUserFactory()
        workgroup = caiduser.workgroup
        workgroup.check_taxon_before_identification = False
        workgroup.identification_model = models.IdentificationModel.objects.create(
            name="Test model",
            model_path="model-path",
            workgroup=workgroup,
        )
        workgroup.save()
        archive = UploadedArchiveFactory(owner=caiduser, is_for_identification=True, import_finished=True)
        MediaFileFactory(parent=archive, media_type="image", identity=None)
        MediaFileFactory(parent=archive, media_type="video", identity=None)
        signature_mock.return_value.apply_async.return_value = Mock(id="identify-task-1")

        started = views.run_identification_bulk(workgroup, uploaded_archives=[archive])

        self.assertTrue(started)
        statistic = models.IdentificationRunStatistic.objects.get(task_id="identify-task-1")
        self.assertEqual(statistic.operation, "identify")
        self.assertEqual(statistic.image_number, 1)
        self.assertEqual(statistic.video_number, 1)
