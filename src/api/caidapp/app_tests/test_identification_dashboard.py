from datetime import timedelta
from unittest.mock import call, patch

from django.core.management import call_command
from django.db import connection
from django.test import TestCase
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils import timezone

from caidapp import models, tasks
from caidapp.celery import app, _on_worker_ready
from celery.beat import ScheduleEntry
from caidapp.app_tests.factories import CaidUserFactory, IndividualIdentityFactory, MediaFileFactory, UploadedArchiveFactory
from caidapp.services.identification_dashboard import (
    identification_dashboard_context,
    refresh_identification_dashboard_snapshot,
)


class IdentificationDashboardTest(TestCase):
    def setUp(self):
        self.user = CaidUserFactory()
        self.workgroup = self.user.workgroup
        self.client.force_login(self.user.user)

    def test_snapshot_counts_mediafiles_once_with_multiple_observations(self):
        taxon = models.get_taxon("Lynx lynx")
        archive = UploadedArchiveFactory(owner=self.user, taxon_for_identification=taxon)
        mediafile = MediaFileFactory(parent=archive)
        identity = IndividualIdentityFactory(owner_workgroup=self.workgroup)
        for _ in range(2):
            models.AnimalObservation.objects.create(mediafile=mediafile, identity=identity)
        snapshot = refresh_identification_dashboard_snapshot(self.workgroup)
        self.assertEqual(snapshot.payload["identification_stats"]["mediafiles"], 1)
        self.assertEqual(snapshot.payload["identification_stats"]["known_identity"], 1)
        self.assertEqual(snapshot.payload["identification_stats"]["missing_identity"], 0)

    def test_old_snapshot_is_read_without_recomputing(self):
        snapshot = refresh_identification_dashboard_snapshot(self.workgroup)
        old_time = timezone.now() - timedelta(hours=5)
        models.IdentificationDashboardSnapshot.objects.filter(pk=snapshot.pk).update(updated_at=old_time)
        IndividualIdentityFactory(owner_workgroup=self.workgroup)
        with patch("caidapp.services.identification_dashboard.refresh_identification_dashboard_snapshot") as refresh:
            with self.assertNumQueries(1):
                context = identification_dashboard_context(self.workgroup)
        refresh.assert_not_called()
        self.assertTrue(context["identification_dashboard_stale"])
        self.assertEqual(context["identification_stats"]["identities"], 0)

    def test_snapshots_are_scoped_to_workgroups(self):
        other = CaidUserFactory()
        IndividualIdentityFactory(owner_workgroup=other.workgroup)
        call_command("refresh_identification_dashboard_stats")
        self.assertEqual(identification_dashboard_context(self.workgroup)["identification_stats"]["identities"], 0)
        self.assertEqual(identification_dashboard_context(other.workgroup)["identification_stats"]["identities"], 1)

    def test_dashboard_uses_snapshot_but_task_status_is_live(self):
        refresh_identification_dashboard_snapshot(self.workgroup)
        self.workgroup.identification_init_status = "Processing"
        self.workgroup.save()
        with patch("caidapp.models.CaIDUser.number_of_identities", side_effect=AssertionError("live count")):
            with patch("caidapp.services.identification_dashboard.build_next_steps", side_effect=AssertionError("live steps")):
                response = self.client.get(reverse("caidapp:dash_identities"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Statistics updated at")
        self.assertContains(response, "fa-spinner fa-spin")

    def test_refresh_replaces_snapshot_after_changes(self):
        refresh_identification_dashboard_snapshot(self.workgroup)
        IndividualIdentityFactory(owner_workgroup=self.workgroup)
        call_command("refresh_identification_dashboard_stats", workgroup_id=self.workgroup.pk)
        self.assertEqual(identification_dashboard_context(self.workgroup)["identification_stats"]["identities"], 1)

    def test_dashboard_query_count_does_not_grow_with_uploads_and_identities(self):
        refresh_identification_dashboard_snapshot(self.workgroup)
        url = reverse("caidapp:dash_identities")
        with CaptureQueriesContext(connection) as initial:
            self.client.get(url)
        UploadedArchiveFactory.create_batch(10, owner=self.user)
        IndividualIdentityFactory.create_batch(10, owner_workgroup=self.workgroup)
        with CaptureQueriesContext(connection) as larger:
            self.client.get(url)
        self.assertEqual(len(initial), len(larger))

    def test_startup_dispatch_skips_fresh_groups_and_includes_stale_or_missing_groups(self):
        snapshot = refresh_identification_dashboard_snapshot(self.workgroup)
        with patch.object(tasks.refresh_workgroup_identification_dashboard, "apply_async") as dispatch:
            tasks.refresh_identification_dashboard_stats.run(due_only=True)
            self.assertNotIn(call(args=[self.workgroup.pk], queue="celery"), dispatch.call_args_list)
            models.IdentificationDashboardSnapshot.objects.filter(pk=snapshot.pk).update(
                updated_at=timezone.now() - timedelta(hours=4)
            )
            dispatch.reset_mock()
            tasks.refresh_identification_dashboard_stats.run(due_only=True)
            dispatch.assert_any_call(args=[self.workgroup.pk], queue="celery")
            models.IdentificationDashboardSnapshot.objects.filter(pk=snapshot.pk).delete()
            dispatch.reset_mock()
            tasks.refresh_identification_dashboard_stats.run(due_only=True)
            dispatch.assert_any_call(args=[self.workgroup.pk], queue="celery")

    def test_periodic_refresh_includes_fresh_groups(self):
        refresh_identification_dashboard_snapshot(self.workgroup)
        with patch.object(tasks.refresh_workgroup_identification_dashboard, "apply_async") as dispatch:
            tasks.refresh_identification_dashboard_stats.run()
        dispatch.assert_any_call(args=[self.workgroup.pk], queue="celery")

    def test_worker_task_refreshes_snapshot(self):
        refresh_identification_dashboard_snapshot(self.workgroup)
        IndividualIdentityFactory(owner_workgroup=self.workgroup)
        tasks.refresh_workgroup_identification_dashboard.run(self.workgroup.pk)
        self.assertEqual(identification_dashboard_context(self.workgroup)["identification_stats"]["identities"], 1)

    def test_worker_task_retries_failure_and_preserves_existing_snapshot(self):
        snapshot = refresh_identification_dashboard_snapshot(self.workgroup)
        with patch(
            "caidapp.services.identification_dashboard.refresh_identification_dashboard_snapshot",
            side_effect=RuntimeError("database unavailable"),
        ) as refresh:
            result = tasks.refresh_workgroup_identification_dashboard.apply(args=[self.workgroup.pk], throw=False)
        self.assertTrue(result.failed())
        self.assertEqual(refresh.call_count, 4)  # Initial attempt plus three retries.
        snapshot.refresh_from_db()
        self.assertEqual(snapshot.payload["identification_stats"]["identities"], 0)

    def test_beat_schedule_becomes_due_after_three_hours(self):
        config = app.conf.beat_schedule["refresh-identification-dashboard-every-three-hours"]
        entry = ScheduleEntry(name="dashboard", app=app, last_run_at=timezone.now() - timedelta(hours=2), **config)
        self.assertFalse(entry.is_due().is_due)
        entry.last_run_at = timezone.now() - timedelta(hours=3, seconds=1)
        self.assertTrue(entry.is_due().is_due)
        self.assertEqual(entry.task, tasks.refresh_identification_dashboard_stats.name)
        self.assertEqual(entry.options["queue"], "celery")

    def test_worker_startup_enqueues_bootstrap_refresh(self):
        with patch.object(app, "send_task") as send:
            _on_worker_ready()
        send.assert_called_once_with(
            tasks.refresh_identification_dashboard_stats.name,
            kwargs={"due_only": True}, queue="celery", expires=3 * 60 * 60,
        )
