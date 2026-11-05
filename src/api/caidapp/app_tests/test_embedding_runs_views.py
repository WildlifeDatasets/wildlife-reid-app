"""Exercise embedding task scope, permissions and model-change integration."""

import uuid
from tempfile import TemporaryDirectory
from unittest.mock import patch

from caidapp import models
from django.contrib.messages import get_messages
from django.test import Client, TestCase
from django.urls import reverse

from .factories import AlbumFactory, CaidUserFactory, MediaFileFactory, UploadedArchiveFactory


class EmbeddingRunViewsTests(TestCase):
    def setUp(self):
        """Create two independently selectable observations in an isolated media file."""
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        settings = self.settings(MEDIA_ROOT=directory.name)
        settings.enable()
        self.addCleanup(settings.disable)
        self.member = CaidUserFactory()
        self.workgroup = self.member.workgroup
        self.model = models.IdentificationModel.objects.create(
            name="Embedding model", public=True, model_path="hf-hub:example/embedding-model"
        )
        models.WorkGroup.objects.filter(pk=self.workgroup.pk).update(identification_model=self.model)
        self.workgroup.refresh_from_db()
        self.mediafile = MediaFileFactory(parent=UploadedArchiveFactory(owner=self.member))
        self.first = self.mediafile.observations.first()
        self.first.is_no_detection_placeholder = False
        self.first.save()
        self.second = models.AnimalObservation.objects.create(mediafile=self.mediafile)
        self.client.force_login(self.member.user)
        self.start_url = reverse("caidapp:embedding_run_start")

    def start(self, **overrides):
        """Submit an explicit observation selection with a fresh double-click key."""
        data = {
            "scope": "observations",
            "observation_ids": [self.first.pk],
            "request_key": str(uuid.uuid4()),
        }
        data.update(overrides)
        return self.client.post(self.start_url, data, HTTP_ACCEPT="application/json")

    def test_single_observation_selection_preserves_sibling(self):
        """Starting from a grouped media file must not silently select its other observation."""
        response = self.start()
        self.assertEqual(response.status_code, 202)
        run = models.TaskRun.objects.get(uuid=response.json()["uuid"])
        self.assertEqual(list(run.items.values_list("observation_id_snapshot", flat=True)), [self.first.pk])
        self.assertEqual(run.workgroup_id, self.workgroup.pk)
        self.assertEqual(run.requested_by_id, self.member.pk)
        self.assertIn("detail_url", response.json())
        self.assertIn("status_url", response.json())

    def test_failed_run_can_be_retried_from_its_detail_with_csrf(self):
        """The result page supplies a CSRF token and explicit remaining observation IDs."""
        response = self.start()
        run = models.TaskRun.objects.get(uuid=response.json()["uuid"])
        run.status = "failed"
        run.save(update_fields=["status"])
        run.items.update(status="failed")
        client = Client(enforce_csrf_checks=True)
        client.force_login(self.member.user)
        detail = client.get(reverse("caidapp:task_detail", kwargs={"run_uuid": run.uuid}))
        self.assertContains(detail, 'name="csrfmiddlewaretoken"')
        payload = client.get(reverse("caidapp:task_status", kwargs={"run_uuid": run.uuid})).json()
        self.assertEqual(payload["retry_ids"], [self.first.pk])
        retry = client.post(
            self.start_url,
            {
                "scope": "observations",
                "observation_ids": payload["retry_ids"],
                "request_key": str(uuid.uuid4()),
                "csrfmiddlewaretoken": client.cookies["csrftoken"].value,
            },
        )
        self.assertEqual(retry.status_code, 202)
        self.assertNotEqual(retry.json()["uuid"], str(run.uuid))

    def test_mediafile_scope_expands_all_observations(self):
        """An explicit media-file action includes both observations belonging to that file."""
        response = self.start(scope="mediafiles", observation_ids=[], mediafile_ids=[self.mediafile.pk])
        self.assertEqual(response.status_code, 202)
        run = models.TaskRun.objects.get(uuid=response.json()["uuid"])
        self.assertEqual(
            set(run.items.values_list("observation_id_snapshot", flat=True)), {self.first.pk, self.second.pk}
        )

    def test_double_click_key_reuses_run_but_fresh_key_allows_another_run(self):
        """Suppress a repeated request without forbidding an intentional independent computation."""
        request_key = str(uuid.uuid4())
        first = self.start(request_key=request_key)
        duplicate = self.start(request_key=request_key)
        another = self.start()
        self.assertEqual([first.status_code, duplicate.status_code, another.status_code], [202, 202, 202])
        self.assertEqual(first.json()["uuid"], duplicate.json()["uuid"])
        self.assertNotEqual(first.json()["uuid"], another.json()["uuid"])
        self.assertEqual(models.TaskRun.objects.count(), 2)

    def test_foreign_observation_rejects_entire_selection(self):
        """A mixed selection must not create a partial run containing unauthorized observations."""
        foreign = MediaFileFactory().observations.first()
        response = self.start(observation_ids=[self.first.pk, foreign.pk])
        self.assertEqual(response.status_code, 403)
        self.assertFalse(models.TaskRun.objects.exists())

    def test_album_membership_does_not_authorize_foreign_mediafile_compute(self):
        """Seeing a foreign file through an album does not grant workgroup computation rights."""
        foreign = MediaFileFactory()
        album = AlbumFactory(owner=self.member)
        album.mediafiles.add(foreign)
        response = self.start(scope="mediafiles", observation_ids=[], mediafile_ids=[foreign.pk])
        self.assertEqual(response.status_code, 403)
        self.assertFalse(models.TaskRun.objects.exists())

    def test_manual_start_requires_post_authentication_and_csrf(self):
        """GET, unauthenticated and cross-site requests cannot enqueue work."""
        self.assertEqual(self.client.get(self.start_url).status_code, 405)
        self.client.logout()
        self.assertEqual(self.start().status_code, 302)
        protected = Client(enforce_csrf_checks=True)
        protected.force_login(self.member.user)
        response = protected.post(self.start_url, {"scope": "observations", "observation_ids": [self.first.pk]})
        self.assertEqual(response.status_code, 403)
        self.assertFalse(models.TaskRun.objects.exists())

    def test_invalid_scope_and_missing_selection_return_readable_error(self):
        """Malformed input is rejected before a durable run or task is created."""
        for data in ({"scope": "all"}, {"scope": "observations"}, {"scope": "observations", "observation_ids": ["x"]}):
            with self.subTest(data=data):
                response = self.client.post(self.start_url, data)
                self.assertEqual(response.status_code, 400)
                self.assertTrue(response.json()["message"])
        self.assertFalse(models.TaskRun.objects.exists())

    def test_workgroup_member_sees_automatic_run_but_foreign_member_cannot(self):
        """Persisted system runs remain visible to the group after reload, without cross-group leakage."""
        run = models.TaskRun.objects.create(workgroup=self.workgroup, trigger="automatic", status="succeeded")
        colleague = CaidUserFactory(workgroup=self.workgroup)
        # New CaIDUsers receive a personal group from the post-save signal.
        colleague.workgroup = self.workgroup
        colleague.save(update_fields=["workgroup"])
        self.client.force_login(colleague.user)
        for name in ("task_detail", "task_status", "task_result"):
            with self.subTest(name=name):
                response = self.client.get(reverse(f"caidapp:{name}", kwargs={"run_uuid": run.uuid}))
                self.assertEqual(response.status_code, 200)
        other = CaidUserFactory()
        self.client.force_login(other.user)
        for name in ("task_detail", "task_status", "task_result"):
            with self.subTest(name=name):
                response = self.client.get(reverse(f"caidapp:{name}", kwargs={"run_uuid": run.uuid}))
                self.assertEqual(response.status_code, 404)

    def test_list_does_not_include_another_workgroups_runs(self):
        """The task list must not expose another group's run even when its UUID is known."""
        local = models.TaskRun.objects.create(workgroup=self.workgroup, status="succeeded")
        foreign = models.TaskRun.objects.create(workgroup=CaidUserFactory().workgroup, status="succeeded")
        response = self.client.get(reverse("caidapp:task_list"), {"state": "history"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, str(local.uuid))
        self.assertNotContains(response, str(foreign.uuid))

    def test_model_switch_schedules_embedding_scan_and_identification_init_after_commit(self):
        """Use the single workgroup model selection to schedule both independent pipelines."""
        self.member.workgroup_admin = True
        self.member.save(update_fields=["workgroup_admin"])
        replacement = models.IdentificationModel.objects.create(
            name="Replacement encoder", public=True, model_path="hf-hub:example/replacement"
        )
        data = {
            "name": self.workgroup.name,
            "sequence_time_limit": self.workgroup.sequence_time_limit,
            "identity_code_regex": self.workgroup.identity_code_regex,
            "identity_merge_distinguishing_regex": self.workgroup.identity_merge_distinguishing_regex,
            "identification_model": replacement.pk,
            "detection_model_path": self.workgroup.detection_model_path,
            "detection_model_architecture": self.workgroup.detection_model_architecture,
        }
        with patch("caidapp.tasks.schedule_init_identification_for_workgroup") as initialize:
            with patch("caidapp.observation_embeddings._publish_preparation") as publish:
                with self.captureOnCommitCallbacks(execute=True):
                    response = self.client.post(reverse("caidapp:workgroup-update", args=[self.workgroup.pk]), data)
                    initialize.assert_not_called()
                    publish.assert_not_called()
        self.assertEqual(response.status_code, 302)
        self.workgroup.refresh_from_db()
        self.assertEqual(self.workgroup.identification_model_id, replacement.pk)
        initialize.assert_called_once()
        publish.assert_called_once()
        schedule = models.ObservationEmbeddingSchedule.objects.get(workgroup=self.workgroup)
        self.assertIsNotNone(schedule.task_id)
        self.assertIsNotNone(schedule.eta)
        notices = " ".join(str(message) for message in get_messages(response.wsgi_request)).lower()
        self.assertIn("embedding", notices)
        self.assertIn("initializ", notices)
