"""Cover the ordering and durable workgroup scope of identification refreshes."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from caidapp import identification_refresh as refresh, models
from caidapp.observation_embeddings import (
    prepare_scheduled_embeddings, reset_workgroup_embeddings, schedule_embeddings, store_embeddings,
)
from caidapp.services.identification_coverage import workgroup_coverage
from caidapp.services.identity_review import input_version, model_version

from .factories import (
    AnimalObservationFactory, CaidUserFactory, IndividualIdentityFactory, MediaFileFactory, UploadedArchiveFactory,
)


class IdentificationRefreshTest(TestCase):
    def setUp(self):
        self.member = CaidUserFactory(admin=True)
        self.client.force_login(self.member.user)
        self.group = self.member.workgroup
        media = MediaFileFactory(parent=UploadedArchiveFactory(owner=self.member))
        self.observation = AnimalObservationFactory(
            mediafile=media, bbox_x_center=0.5, bbox_y_center=0.5, bbox_width=0.2, bbox_height=0.2,
        )

    def test_refresh_compares_identified_observations_with_other_known_observations(self):
        identity = IndividualIdentityFactory(owner_workgroup=self.group)
        self.observation.identity = identity
        self.observation.save(update_fields=["identity"])
        AnimalObservationFactory(mediafile=self.observation.mediafile, identity=identity)
        with patch.object(refresh.observation_comparisons, "start") as submit:
            refresh._launch("known", self.group)
        submit.assert_called_once_with(
            self.group, None, "review_known", include_non_representative=True, defer_submission=True,
        )

    def test_coverage_counts_observations_once_and_excludes_stale_comparisons(self):
        other = AnimalObservationFactory(mediafile=self.observation.mediafile)
        identity = IndividualIdentityFactory(owner_workgroup=self.group)
        for observation in (self.observation, other):
            observation.identity = identity
            observation.save(update_fields=["identity"])
        for dimension in (1, 2):
            profile = models.EmbeddingProfile.objects.create(
                dimension=dimension, specification={"dimension": dimension},
            )
            self.observation.embeddings.create(
                profile=profile, input_fingerprint="input", input_specification={}, vector=[1] * dimension,
            )
        comparison = {
            "query": {"id": self.observation.pk, "version": input_version(self.observation, self.observation.mediafile)},
            "candidates": [{"id": other.pk, "version": input_version(other, other.mediafile)}],
        }
        models.TaskRun.objects.create(
            workgroup=self.group, kind="review_known", status="succeeded", finished_at=timezone.now(),
            model_snapshot={**model_version(self.group.identification_model), "generation": 0},
            result={"rows": [comparison]},
        )
        response = self.client.get(reverse("caidapp:identification_information"))
        self.assertContains(response, "Observation coverage")
        counts = response.context["coverage"]
        self.assertEqual((counts["images"], counts["observations"], counts["embedded"]), (1, 2, 1))
        self.assertEqual((counts["known_comparisons"], counts["with_suggestions"]), (1, 1))
        other.orientation = "L"
        other.save(update_fields=["orientation"])
        self.assertEqual(workgroup_coverage(self.group)["known_comparisons"], 0)

    def test_refresh_prepares_features_before_launching_analysis(self):
        run = refresh.request_refresh(self.group.pk, self.member)
        with patch.object(refresh, "_launch", wraps=refresh._launch) as launch:
            refresh.advance(run.pk)
            child = models.TaskRun.objects.get(kind="embedding")
            refresh.advance(run.pk)
            self.assertEqual([call.args[0] for call in launch.call_args_list], ["features"])
            child.status = "failed"
            child.message = "Computing image features failed. The worker ran out of memory."
            child.save(update_fields=["status", "message"])
            refresh.advance(run.pk)
        run.refresh_from_db()
        self.assertEqual(run.status, "failed")
        self.assertEqual(run.result["steps"]["identify"]["status"], "skipped")
        self.assertEqual(run.result["steps"]["features"]["record_id"], child.pk)
        self.assertEqual(run.result["steps"]["features"]["message"], child.message)

    def test_followup_timer_waits_and_creates_one_run_after_completion(self):
        first = refresh.request_refresh(self.group.pk, self.member)
        schedule_embeddings(self.group.pk, delay_minutes=0)
        self.assertEqual(refresh.request_refresh(self.group.pk, self.member).pk, first.pk)
        timer = models.ObservationEmbeddingSchedule.objects.get(workgroup=self.group).task_id
        self.assertEqual(prepare_scheduled_embeddings(self.group.pk, timer), {"status": "waiting"})
        models.TaskRun.objects.filter(pk=first.pk).update(status="succeeded", finished_at=timezone.now())
        self.assertEqual(prepare_scheduled_embeddings(self.group.pk, timer)["status"], "queued")
        self.assertEqual(prepare_scheduled_embeddings(self.group.pk, timer)["status"], "superseded")
        self.assertEqual(models.TaskRun.objects.filter(workgroup=self.group, kind=refresh.KIND).count(), 2)

    def test_refresh_runs_each_analysis_and_records_partial_completion(self):
        run = refresh.request_refresh(self.group.pk, self.member)
        records = [Mock(pk=pk) for pk in range(1, 7)]
        with (
            patch.object(refresh, "_launch", side_effect=records) as launch,
            patch.object(refresh, "_child_status", side_effect=["succeeded"] * 4 + ["failed", "succeeded"]),
        ):
            for _ in range(7):
                refresh.advance(run.pk)
        run.refresh_from_db()
        self.assertEqual([call.args[0] for call in launch.call_args_list], list(refresh.STEPS))
        self.assertEqual(run.status, "partial")
        self.assertEqual(run.result["steps"]["outliers"]["status"], "failed")
        self.assertEqual(run.result["steps"]["merges"]["status"], "succeeded")

    def test_menu_opens_observable_refresh_and_reset_stops_followup(self):
        dash = self.client.get(reverse("caidapp:dash_identities"))
        self.assertContains(dash, "Refresh identification suggestions")
        self.assertContains(dash, "Reset image features")
        self.assertContains(dash, 'aria-label="Tasks"')
        response = self.client.post(reverse("caidapp:manage_workgroup_embeddings"), {"action": "refresh"})
        run = models.TaskRun.objects.get(workgroup=self.group, kind=refresh.KIND)
        self.assertRedirects(response, reverse("caidapp:task_detail", args=[run.uuid]))
        self.assertContains(self.client.get(reverse("caidapp:task_detail", args=[run.uuid])), "Find possible identity merges")
        self.assertContains(self.client.get(reverse("caidapp:task_list"), {"state": "active"}), "Refresh identification suggestions")
        schedule_embeddings(self.group.pk, delay_minutes=0)
        reset_workgroup_embeddings(self.group.pk)
        run.refresh_from_db()
        self.assertEqual(run.status, "failed")
        self.assertIsNone(models.ObservationEmbeddingSchedule.objects.get(workgroup=self.group).task_id)

    def test_dashboard_can_stop_current_refresh_and_start_another_without_reset(self):
        actions = reverse("caidapp:manage_workgroup_embeddings")
        dash = reverse("caidapp:dash_identities")
        self.assertRedirects(self.client.post(actions, {"action": "refresh", "dashboard": "yes"}), dash)
        run = refresh.active_refresh(self.group.pk)
        refresh.advance(run.pk)
        child = models.TaskRun.objects.get(kind="embedding")
        profile = models.EmbeddingProfile.objects.create(dimension=1, specification={"dimension": 1})
        embedding = self.observation.embeddings.create(
            profile=profile, input_fingerprint="saved", input_specification={}, vector=[1],
        )
        schedule_embeddings(self.group.pk, delay_minutes=0)
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "input.json"
            child.manifest_path = str(manifest)
            child.status = "running"
            child.compute_task_id = "feature-job"
            child.save(update_fields=["manifest_path", "status", "compute_task_id"])
            progress = {"label": "Segmenting observations", "completed": 12, "total": 30}
            with patch.object(refresh, "AsyncResult", return_value=Mock(state="PROGRESS", info=progress)):
                status = self.client.get(reverse("caidapp:identification_progress_api"), {"scope": "refresh"})
            self.assertEqual(status.json()["refresh"]["step"], 1)
            self.assertEqual(status.json()["refresh"]["feature_progress"], progress)
            with self.captureOnCommitCallbacks(execute=True):
                self.assertRedirects(self.client.post(actions, {"action": "stop_refresh", "run_uuid": run.uuid}), dash)
            self.assertTrue(manifest.with_name("cancelled").is_file())
            self.assertEqual(store_embeddings({}, manifest, run_id=child.pk), {"cancelled": 1})
        run.refresh_from_db()
        self.assertEqual(run.status, "cancelled")
        self.assertEqual(run.result["steps"]["features"]["status"], "cancelled")
        self.assertEqual(run.result["steps"]["identify"]["status"], "skipped")
        self.assertTrue(models.ObservationEmbedding.objects.filter(pk=embedding.pk).exists())
        self.assertIsNone(models.ObservationEmbeddingSchedule.objects.get(workgroup=self.group).task_id)
        refresh.advance(run.pk)
        self.assertRedirects(self.client.post(actions, {"action": "refresh", "dashboard": "yes"}), dash)
        self.assertNotEqual(refresh.active_refresh(self.group.pk).pk, run.pk)
        self.client.post(actions, {"action": "stop_refresh", "run_uuid": run.uuid})
        self.assertIsNotNone(refresh.active_refresh(self.group.pk))

