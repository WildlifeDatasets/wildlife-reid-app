"""Cover the ordering and durable workgroup scope of identification refreshes."""

from unittest.mock import Mock, patch

from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from caidapp import identification_refresh as refresh, models
from caidapp.observation_embeddings import prepare_scheduled_embeddings, reset_workgroup_embeddings, schedule_embeddings

from .factories import AnimalObservationFactory, CaidUserFactory, MediaFileFactory, UploadedArchiveFactory


class IdentificationRefreshTest(TestCase):
    def setUp(self):
        self.member = CaidUserFactory(admin=True)
        self.client.force_login(self.member.user)
        self.group = self.member.workgroup
        media = MediaFileFactory(parent=UploadedArchiveFactory(owner=self.member))
        AnimalObservationFactory(
            mediafile=media, bbox_x_center=0.5, bbox_y_center=0.5, bbox_width=0.2, bbox_height=0.2,
        )

    def test_refresh_prepares_features_before_launching_analysis(self):
        run = refresh.request_refresh(self.group.pk, self.member)
        with patch.object(refresh, "_launch", wraps=refresh._launch) as launch:
            refresh.advance(run.pk)
            child = models.TaskRun.objects.get(kind="embedding")
            refresh.advance(run.pk)
            self.assertEqual([call.args[0] for call in launch.call_args_list], ["features"])
            child.status = "failed"
            child.save(update_fields=["status"])
            refresh.advance(run.pk)
        run.refresh_from_db()
        self.assertEqual(run.status, "failed")
        self.assertEqual(run.result["steps"]["identify"]["status"], "skipped")
        self.assertEqual(run.result["steps"]["features"]["record_id"], child.pk)

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
        records = [Mock(pk=pk) for pk in range(1, 6)]
        with (
            patch.object(refresh, "_launch", side_effect=records) as launch,
            patch.object(refresh, "_child_status", side_effect=["succeeded"] * 3 + ["failed", "succeeded"]),
        ):
            for _ in range(6):
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

