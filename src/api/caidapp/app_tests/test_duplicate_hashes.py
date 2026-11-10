"""Incremental workgroup hashing, automatic scheduling, and exact-image identity evidence."""

import json
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from caidapp import duplicate_hashes as service
from caidapp import duplicate_services, models, tasks
from django.core.exceptions import PermissionDenied
from django.core.files.base import ContentFile
from django.db import transaction
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .factories import (
    AnimalObservationFactory,
    CaidUserFactory,
    IndividualIdentityFactory,
    MediaFileFactory,
    UploadedArchiveFactory,
)


def worker_result(items, root):
    """Exercise the actual isolated worker algorithm, including canonical pixel hashes."""
    worker = Path(__file__).resolve().parents[3] / "taxon_worker"
    with patch.object(sys, "path", [str(worker), *sys.path]):
        from duplicates.fingerprints import hash_files

    return hash_files(items, root)


class _HashTestSetup:
    def setUp(self):
        """Keep sources, durable results, and broker calls isolated from the running application."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        settings = override_settings(
            MEDIA_ROOT=self.root, DUPLICATE_SCAN_ROOT=self.root / "private", DUPLICATE_HASH_BATCH_SIZE=2
        )
        settings.enable()
        self.addCleanup(settings.disable)
        for target in ("_publish_timer", "_dispatch_batch", "_queue_next_batch"):
            mocked = patch(f"caidapp.duplicate_hashes.{target}")
            mocked.start()
            self.addCleanup(mocked.stop)
        self.archive = UploadedArchiveFactory()
        self.user = self.archive.owner
        self.group = self.user.workgroup
        models.WorkGroup.objects.filter(pk=self.group.pk).update(identification_model=None)
        self.group.refresh_from_db()
        self.client.force_login(self.user.user)

    def media(self, **kwargs):
        """Create a source in the current working group."""
        return MediaFileFactory(parent=self.archive, **kwargs)

    def start(self):
        """Claim a workgroup check immediately, without a live broker or ReID model."""
        schedule = service.schedule_hashes(self.group.pk, immediate=True)
        service.prepare_hashes(self.group.pk, schedule.task_id)
        return models.DuplicateScan.objects.filter(workgroup=self.group).latest("pk")

    def finish(self, scan):
        """Run every reserved batch through the real worker and callback."""
        while scan.hash_batches.filter(status__in=["pending", "queued"]).exists():
            service.dispatch_next_batch(scan.pk)
            batch = scan.hash_batches.get(status="queued")
            result = worker_result(batch.items, self.root)
            directory = service.batch_directory(batch)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
            with self.captureOnCommitCallbacks(execute=True):
                service.finish_batch(batch.pk)
        scan.refresh_from_db()


class WorkgroupExactHashesTest(_HashTestSetup, TestCase):
    def test_hashes_only_missing_and_compares_with_previous_runs(self):
        """A later import matches a previously hashed image without sending that image to the worker again."""
        first = self.media()
        run = self.start()
        self.assertEqual(run.total, 1)
        self.finish(run)
        first.refresh_from_db()
        self.assertTrue(first.pixel_sha256)
        self.assertEqual(first.hash_version, service.HASH_VERSION)
        second = self.media()
        run = self.start()
        self.assertEqual(run.total, 1)
        self.assertEqual(run.hash_batches.get().items[0]["id"], second.pk)
        self.finish(run)
        proposal = models.DuplicateProposal.objects.get(status="pending")
        self.assertEqual(proposal.kind, "pixel_exact")
        self.assertEqual(set(proposal.members.values_list("mediafile_id", flat=True)), {first.pk, second.pk})

    def test_repeated_check_reuses_hashes_and_proposals(self):
        """Ordinary checks do not rehash ready sources or duplicate catalogue entries."""
        self.media()
        self.media()
        run = self.start()
        self.finish(run)
        first_id = models.DuplicateProposal.objects.get().pk
        run_again = self.start()
        self.assertEqual(run_again.pk, run.pk)
        self.assertEqual(models.DuplicateProposal.objects.get().pk, first_id)
        self.assertEqual(models.DuplicateHashBatch.objects.count(), 1)

    def test_more_than_100_files_are_processed_in_bounded_batches(self):
        """Batch size limits worker memory, never the total working-group scope."""
        for _ in range(103):
            self.media()
        with override_settings(DUPLICATE_HASH_BATCH_SIZE=32):
            run = self.start()
        self.assertEqual(run.total, 103)
        self.assertEqual(run.hash_batches.count(), 4)
        self.assertLessEqual(max(len(batch.items) for batch in run.hash_batches.all()), 32)
        self.finish(run)
        self.assertEqual(run.processed, 103)
        self.assertEqual(models.DuplicateProposal.objects.get(status="pending").members.count(), 103)

    def test_admin_force_rehashes_all_but_member_cannot(self):
        """Force permission is enforced by the service as well as hidden in the GUI."""
        media = self.media()
        self.finish(self.start())
        member = CaidUserFactory(workgroup=self.group, workgroup_admin=False)
        member.workgroup, member.workgroup_admin = self.group, False
        member.save(update_fields=["workgroup", "workgroup_admin"])
        with self.assertRaises(PermissionDenied):
            service.request_hashes(member, force=True)
        self.user.workgroup_admin = True
        self.user.save(update_fields=["workgroup_admin"])
        before = models.MediaFile.objects.get(pk=media.pk).hash_revision
        schedule = service.request_hashes(self.user, force=True)
        media.refresh_from_db()
        self.assertEqual(media.hash_revision, before + 1)
        self.assertEqual(media.pixel_sha256, "")
        service.prepare_hashes(self.group.pk, schedule.task_id)
        run = models.DuplicateScan.objects.filter(workgroup=self.group).latest("pk")
        self.assertEqual(run.total, 1)

    def test_late_callback_cannot_restore_replaced_source(self):
        """A generation change invalidates already-produced results and requests another pass."""
        media = self.media()
        run = self.start()
        service.dispatch_next_batch(run.pk)
        batch = run.hash_batches.get()
        result = worker_result(batch.items, self.root)
        media.mediafile.save("replacement.jpg", ContentFile(b"changed source"))
        directory = service.batch_directory(batch)
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
        service.finish_batch(batch.pk)
        media.refresh_from_db()
        run.refresh_from_db()
        self.assertEqual(media.file_sha256, "")
        self.assertEqual(run.skipped, 1)
        self.assertTrue(models.DuplicateHashSchedule.objects.get(workgroup=self.group).dirty)

    def test_force_during_active_run_rejects_old_results(self):
        """Forced recalculation is queued behind active work instead of cancelling another workflow."""
        media = self.media()
        run = self.start()
        self.user.workgroup_admin = True
        service.request_hashes(self.user, force=True)
        self.finish(run)
        media.refresh_from_db()
        self.assertEqual(media.file_sha256, "")
        self.assertEqual(run.skipped, 1)

    def test_completion_callback_is_idempotent(self):
        """A repeated callback cannot count the same batch twice or create extra proposals."""
        self.media()
        run = self.start()
        self.finish(run)
        service.finish_batch(run.hash_batches.get().pk)
        run.refresh_from_db()
        self.assertEqual(run.processed, 1)

    def test_broken_source_does_not_block_other_files(self):
        """Errors are durable and retried later, while valid sources finish normally."""
        broken, valid = self.media(), self.media()
        (self.root / broken.mediafile.name).unlink()
        run = self.start()
        self.finish(run)
        broken.refresh_from_db()
        valid.refresh_from_db()
        self.assertEqual(broken.hash_status, "error")
        self.assertEqual(valid.hash_status, "ready")
        self.assertEqual(run.failed_count, 1)
        self.assertEqual(self.start().pk, run.pk)
        models.MediaFile.objects.filter(pk=broken.pk).update(hash_computed_at=timezone.now() - timedelta(hours=7))
        retry = self.start()
        self.assertEqual(retry.total, 1)

    def test_one_timer_for_repeated_changes_without_reid(self):
        """The timer coalesces uploads even when no identification model is configured."""
        self.media()
        first = service.schedule_hashes(self.group.pk)
        second = service.schedule_hashes(self.group.pk)
        self.assertEqual(first.task_id, second.task_id)
        self.assertIsNone(self.group.identification_model_id)
        self.assertEqual(models.DuplicateHashSchedule.objects.count(), 1)

    def test_completed_import_automatically_schedules_after_commit(self):
        """No manual user action is needed after an upload becomes available."""
        self.archive.import_finished = False
        self.archive.save(update_fields=["import_finished"])
        self.media()
        with self.captureOnCommitCallbacks(execute=True):
            self.archive.import_finished = True
            self.archive.save(update_fields=["import_finished"])
        self.assertTrue(models.DuplicateHashSchedule.objects.get(workgroup=self.group).task_id)

    def test_rolled_back_import_never_schedules(self):
        """A transaction rollback cannot publish hashing for uncommitted sources."""
        with self.captureOnCommitCallbacks(execute=True):
            with self.assertRaises(RuntimeError), transaction.atomic():
                self.media()
                raise RuntimeError("rollback")
        self.assertFalse(models.DuplicateHashSchedule.objects.filter(workgroup=self.group).exists())

    def test_metadata_edits_preserve_cached_hashes(self):
        """Stale model instances saving notes cannot overwrite newer worker hash values."""
        media = self.media()
        self.finish(self.start())
        media.note = "reviewed later"
        media.save()
        media.refresh_from_db()
        self.assertTrue(media.pixel_sha256)
        self.assertEqual(media.hash_status, "ready")

    def test_outdated_timer_does_not_start_another_run(self):
        """Superseded and early timers cannot duplicate an active scan."""
        self.media()
        first = service.schedule_hashes(self.group.pk)
        second = service.schedule_hashes(self.group.pk, immediate=True)
        self.assertEqual(service.prepare_hashes(self.group.pk, first.task_id)["status"], "superseded")
        service.prepare_hashes(self.group.pk, second.task_id)
        self.assertEqual(models.DuplicateScan.objects.filter(workgroup=self.group).count(), 1)

    def test_video_reference_match_is_not_whole_video_equality(self):
        """Different video bytes with the same reference frame produce a read-only reference proposal."""
        a, b = self.media(media_type="video"), self.media(media_type="video")
        a.mediafile.save("a.mp4", ContentFile(b"video A"))
        b.mediafile.save("b.mp4", ContentFile(b"video B"))
        self.finish(self.start())
        a.refresh_from_db()
        b.refresh_from_db()
        self.assertNotEqual(a.file_sha256, b.file_sha256)
        self.assertEqual(a.reference_sha256, b.reference_sha256)
        proposal = models.DuplicateProposal.objects.get(status="pending")
        self.assertEqual(proposal.kind, "reference_exact")
        with self.assertRaisesMessage(ValueError, "do not establish"):
            duplicate_services.build_plan(proposal, self.user, a.pk)

    def test_workgroup_scope_is_shared_but_never_crosses_membership(self):
        """Members share catalogue results, but files in another group never reach the worker."""
        self.media()
        self.media()
        foreign = MediaFileFactory()
        run = self.start()
        self.assertEqual(run.total, 2)
        self.finish(run)
        foreign.refresh_from_db()
        self.assertEqual(foreign.file_sha256, "")
        member = CaidUserFactory(workgroup=self.group)
        member.workgroup = self.group
        member.save(update_fields=["workgroup"])
        self.client.force_login(member.user)
        response = self.client.get(reverse("caidapp:duplicate_scan_matches", args=[run.uuid]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 1)
        self.client.force_login(foreign.parent.owner.user)
        self.assertEqual(self.client.get(reverse("caidapp:duplicate_scan_status", args=[run.uuid])).status_code, 404)

    def test_ui_has_only_group_controls_and_admin_force(self):
        """The ordinary UX has no file-ID inputs, crop launch controls, or force action for members."""
        self.media()
        self.user.workgroup_admin = False
        self.user.save(update_fields=["workgroup_admin"])
        response = self.client.get(reverse("caidapp:duplicates"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Check working group now")
        self.assertNotContains(response, 'name="mediafile_ids"')
        self.assertNotContains(response, 'name="include_crops"')
        self.assertNotContains(response, "Recalculate all fingerprints")
        response = self.client.post(reverse("caidapp:duplicate_scan_start"), {"force": "on"})
        self.assertEqual(response.status_code, 403)

    def test_new_finished_source_automatically_schedules(self):
        """New media in an already-finished import are discovered without any user action."""
        with self.captureOnCommitCallbacks(execute=True):
            self.media()
        self.assertTrue(models.DuplicateHashSchedule.objects.get(workgroup=self.group).task_id)

    def test_lost_callback_recovers_from_shared_result(self):
        """Redis/callback loss does not lose a worker's durable output."""
        media = self.media()
        scan = self.start()
        service.dispatch_next_batch(scan.pk)
        batch = scan.hash_batches.get()
        directory = service.batch_directory(batch)
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps(worker_result(batch.items, self.root)), encoding="utf-8")
        with self.captureOnCommitCallbacks(execute=True):
            service.recover_group(self.group.pk)
        media.refresh_from_db()
        scan.refresh_from_db()
        self.assertEqual(media.hash_status, "ready")
        self.assertEqual(scan.status, "succeeded")

    def test_failed_batch_does_not_block_later_batches(self):
        """A worker failure leaves failed-source state while other bounded batches can finish."""
        for _ in range(3):
            self.media()
        scan = self.start()
        service.dispatch_next_batch(scan.pk)
        first = scan.hash_batches.get(status="queued")
        with self.captureOnCommitCallbacks(execute=True):
            service.fail_batch(first.pk)
        self.finish(scan)
        self.assertEqual(scan.failed_count, 2)
        self.assertEqual(scan.processed, 3)
        self.assertEqual(service.group_media(self.group.pk).filter(hash_status="ready").count(), 1)
        service.finish_batch(first.pk)
        scan.refresh_from_db()
        self.assertEqual(scan.processed, 3)

    def test_dismissed_unchanged_group_does_not_reappear(self):
        """Automatic catalogue refresh preserves reviewed dismissals for the same exact group."""
        self.media()
        self.media()
        self.finish(self.start())
        proposal = models.DuplicateProposal.objects.get()
        duplicate_services.dismiss_proposal(proposal.pk, self.user)
        self.start()
        self.assertEqual(models.DuplicateProposal.objects.count(), 1)
        proposal.refresh_from_db()
        self.assertEqual(proposal.status, "dismissed")

    def test_hash_version_change_recomputes_cached_sources(self):
        """Decoder/canonicalization upgrades can backfill without a user forcing every group."""
        media = self.media()
        self.finish(self.start())
        models.MediaFile.objects.filter(pk=media.pk).update(hash_version="previous-version")
        scan = self.start()
        self.assertEqual(scan.total, 1)
        self.finish(scan)
        media.refresh_from_db()
        self.assertEqual(media.hash_version, service.HASH_VERSION)

    def test_expired_worker_batch_releases_the_group(self):
        """A missing worker eventually fails its batch rather than leaving the group stuck forever."""
        self.media()
        scan = self.start()
        service.dispatch_next_batch(scan.pk)
        scan.hash_batches.update(dispatched_at=timezone.now() - timedelta(hours=3))
        with self.captureOnCommitCallbacks(execute=True):
            service.recover_group(self.group.pk)
        scan.refresh_from_db()
        self.assertEqual(scan.status, "succeeded")
        self.assertEqual(scan.failed_count, 1)
        self.assertIsNone(models.DuplicateHashSchedule.objects.get(workgroup=self.group).active_scan_id)

    def test_lost_timer_is_republished(self):
        """A timer lost after publication must not reserve the group forever."""
        self.media()
        original = service.schedule_hashes(self.group.pk)
        models.DuplicateHashSchedule.objects.filter(workgroup=self.group).update(
            eta=timezone.now() - timedelta(minutes=10),
        )
        service.recover_group(self.group.pk)
        replacement = models.DuplicateHashSchedule.objects.get(workgroup=self.group)
        self.assertTrue(replacement.task_id)
        self.assertNotEqual(original.task_id, replacement.task_id)

    def test_broker_restart_republishes_unacknowledged_batch(self):
        """A durable queued row with a lost broker message is retried without waiting two hours."""
        self.media()
        scan = self.start()
        service.dispatch_next_batch(scan.pk)
        batch = scan.hash_batches.get()
        original_uuid = batch.uuid
        scan.hash_batches.update(dispatched_at=timezone.now() - timedelta(minutes=5))
        with patch.object(service, "_dispatch_batch") as publish:
            with self.captureOnCommitCallbacks(execute=True):
                service.recover_group(self.group.pk)
            publish.assert_called_once_with(batch.pk)
            with self.captureOnCommitCallbacks(execute=True):
                service.recover_group(self.group.pk)
            publish.assert_called_once()
        batch.refresh_from_db()
        scan.refresh_from_db()
        self.assertEqual(batch.uuid, original_uuid)
        self.assertEqual(batch.status, "queued")
        self.assertEqual(scan.processed, 0)
        self.finish(scan)
        self.assertEqual(scan.processed, 1)

    def test_recent_worker_heartbeat_prevents_duplicate_delivery(self):
        """A long-running first file must not look like a lost queue message."""
        self.media()
        scan = self.start()
        service.dispatch_next_batch(scan.pk)
        batch = scan.hash_batches.get()
        scan.hash_batches.update(dispatched_at=timezone.now() - timedelta(minutes=5))
        directory = service.batch_directory(batch)
        directory.mkdir(parents=True)
        (directory / "progress.json").write_text(
            json.dumps(
                {
                    "updated_at": timezone.now().timestamp(),
                    "completed": 0,
                }
            ),
            encoding="utf-8",
        )
        with patch.object(service, "_dispatch_batch") as publish:
            with self.captureOnCommitCallbacks(execute=True):
                service.recover_group(self.group.pk)
            publish.assert_not_called()

    def test_check_button_recovers_existing_lost_delivery(self):
        """Check now repairs a stuck active run instead of only setting its dirty flag."""
        self.media()
        scan = self.start()
        service.dispatch_next_batch(scan.pk)
        batch = scan.hash_batches.get()
        scan.hash_batches.update(dispatched_at=timezone.now() - timedelta(minutes=5))
        with patch.object(service, "_dispatch_batch") as publish:
            with self.captureOnCommitCallbacks(execute=True):
                response = self.client.post(reverse("caidapp:duplicate_scan_start"))
            self.assertEqual(response.status_code, 302)
            publish.assert_called_once_with(batch.pk)

    def test_ui_distinguishes_waiting_for_delivery_from_processing(self):
        """A queued database row is not proof that a worker has started hashing."""
        self.media()
        scan = self.start()
        service.dispatch_next_batch(scan.pk)
        response = self.client.get(reverse("caidapp:duplicates"))
        self.assertContains(response, "Waiting for the duplicate worker to accept the batch")
        batch = scan.hash_batches.get()
        directory = service.batch_directory(batch)
        directory.mkdir(parents=True)
        (directory / "progress.json").write_text(
            json.dumps(
                {
                    "updated_at": timezone.now().timestamp(),
                    "completed": 0,
                }
            ),
            encoding="utf-8",
        )
        response = self.client.get(reverse("caidapp:duplicates"))
        self.assertContains(response, "Worker processing batch")


class ExactIdentityEvidenceTest(_HashTestSetup, TestCase):
    def test_pixel_method_uses_shared_merge_review_without_identification(self):
        """Reuse ready hashes in a saved merge result with previews, exclusions, and stable review indices."""
        _, _, first, second = self.annotated_copies()
        self.finish(self.start())
        self.user.user.is_staff = True
        self.user.user.save(update_fields=["is_staff"])
        with patch("caidapp.tasks.prepare_pixel_identity_pairs.apply_async") as pixel_queue, patch(
            "caidapp.tasks.prepare_similar_identity_pairs.apply_async"
        ) as visual_queue:
            response = self.client.post(reverse("caidapp:run_similar_identity_pairs"), {"method": "pixel"})
        result = models.IdentificationSimilarPairResult.objects.get()
        self.assertRedirects(
            response, reverse("caidapp:similar_identity_pairs_result", args=[result.pk]), fetch_redirect_response=False
        )
        pixel_queue.assert_called_once_with(args=[result.pk, True], queue="celery", retry=False)
        visual_queue.assert_not_called()
        with patch("caidapp.tasks.signature") as gpu:
            tasks.prepare_pixel_identity_pairs(result.pk, pixel_only=True)
        gpu.assert_not_called()
        result.refresh_from_db()
        self.assertEqual((result.status, result.pixel_status), ("done", "done"))
        self.assertEqual(len(result.pixel_pairs), 1)
        self.assertEqual(
            {result.pixel_pairs[0]["class_id_a"], result.pixel_pairs[0]["class_id_b"]},
            {first.identity_id, second.identity_id},
        )
        self.assertEqual(result.ordered_pairs(), result.pixel_pairs)
        response = self.client.get(
            reverse("caidapp:similar_identity_pairs_result", args=[result.pk]), {"layout": "rows"}
        )
        self.assertContains(response, "Identical image pixels")
        self.assertContains(response, "Merge preview")
        self.assertContains(response, "Matching samples")
        self.assertContains(response, f"review_result={result.pk}")
        self.assertNotContains(response, "Prepare identification references")
        self.assertEqual(response.context["pixel_cards"][0]["review_index"], 0)
        models.MergeIdentitySuggestionExclusion.objects.create(
            workgroup=self.group,
            identity_a_id=min(first.identity_id, second.identity_id),
            identity_b_id=max(first.identity_id, second.identity_id),
        )
        response = self.client.get(reverse("caidapp:similar_identity_pairs_result", args=[result.pk]))
        self.assertEqual(response.context["pixel_cards"], [])
        retry = models.IdentificationSimilarPairResult.objects.create(workgroup=self.group, pixel_status="processing")
        tasks.prepare_pixel_identity_pairs(retry.pk, pixel_only=True)
        retry.refresh_from_db()
        self.assertEqual(retry.pixel_pairs, [])

    def test_pixel_method_waits_for_missing_hashes_then_finishes_with_available_results(self):
        """Prepare missing fingerprints once and retry outside the request without launching any GPU work."""
        self.annotated_copies()
        result = models.IdentificationSimilarPairResult.objects.create(workgroup=self.group, pixel_status="processing")
        with patch("caidapp.tasks.prepare_pixel_identity_pairs.apply_async") as retry:
            tasks.prepare_pixel_identity_pairs(result.pk, pixel_only=True)
        result.refresh_from_db()
        self.assertEqual(result.pixel_status, "processing")
        self.assertIn("Waiting for fingerprints for 2 media files", result.pixel_message)
        retry.assert_called_once_with(args=[result.pk, True, 1], countdown=15, queue="celery")
        self.finish(self.start())
        with patch.object(service, "schedule_hashes") as schedule:
            tasks.prepare_pixel_identity_pairs(result.pk, pixel_only=True, attempt=1)
        schedule.assert_not_called()
        result.refresh_from_db()
        self.assertEqual(result.pixel_status, "done")
        self.assertEqual(len(result.pixel_pairs), 1)

    def test_visual_failure_does_not_discard_completed_pixel_results(self):
        """Retain exact-image suggestions when the independent legacy visual search fails."""
        self.annotated_copies()
        self.finish(self.start())
        result = models.IdentificationSimilarPairResult.objects.create(
            workgroup=self.group, pixel_status="processing", status="error", message="No reference embeddings"
        )
        tasks.prepare_pixel_identity_pairs(result.pk)
        result.refresh_from_db()
        self.assertEqual(result.status, "error")
        self.assertEqual(len(result.pixel_pairs), 1)

    def test_unavailable_hashes_are_reported_without_waiting_forever(self):
        """Finish with an explicit incomplete-result message for recently failed source inputs."""
        media = self.media()
        models.MediaFile.objects.filter(pk=media.pk).update(
            hash_status="error", hash_version=service.HASH_VERSION, hash_computed_at=timezone.now()
        )
        result = models.IdentificationSimilarPairResult.objects.create(workgroup=self.group, pixel_status="processing")
        tasks.prepare_pixel_identity_pairs(result.pk, pixel_only=True)
        result.refresh_from_db()
        self.assertEqual(result.pixel_status, "done")
        self.assertIn("1 media files could not be fingerprinted", result.pixel_message)
        self.assertEqual(result.pixel_pairs, [])

    def test_missing_worker_eventually_reports_actionable_error(self):
        """Bound waiting for a duplicate worker and preserve independent visual results."""
        self.media()
        result = models.IdentificationSimilarPairResult.objects.create(
            workgroup=self.group, pixel_status="processing", status="done", mega_pairs=[{"similarity": 0.9}]
        )
        tasks.prepare_pixel_identity_pairs(result.pk, attempt=240)
        result.refresh_from_db()
        self.assertEqual(result.pixel_status, "error")
        self.assertIn("Check the duplicate worker", result.pixel_message)
        self.assertEqual(result.status, "done")
        self.assertEqual(result.mega_pairs, [{"similarity": 0.9}])

    def annotated_copies(self, same_box=True):
        """Attach two different identities to observations in identical image copies."""
        a, b = self.media(), self.media()
        first = IndividualIdentityFactory(owner_workgroup=self.group)
        second = IndividualIdentityFactory(owner_workgroup=self.group)
        box = dict(bbox_x_center=0.3, bbox_y_center=0.5, bbox_width=0.2, bbox_height=0.2)
        obs_a = AnimalObservationFactory(mediafile=a, identity=first, **box)
        if not same_box:
            box["bbox_x_center"] = 0.7
        obs_b = AnimalObservationFactory(mediafile=b, identity=second, **box)
        return a, b, obs_a, obs_b

    def test_matching_objects_suggest_identity_merge(self):
        """Exact copies plus matching observation locations provide a separate merge-suggestion category."""
        _, _, first, second = self.annotated_copies()
        self.finish(self.start())
        suggestion = models.DuplicateIdentitySuggestion.objects.get(active=True)
        self.assertEqual({suggestion.identity_a_id, suggestion.identity_b_id}, {first.identity_id, second.identity_id})
        response = self.client.get(reverse("caidapp:duplicate_identity_suggestions"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Review identity merge")

    def test_different_objects_never_suggest_identity_merge(self):
        """Co-occurrence in a shared image does not establish global identity."""
        self.annotated_copies(same_box=False)
        self.finish(self.start())
        self.assertFalse(models.DuplicateIdentitySuggestion.objects.filter(active=True).exists())

    def test_annotation_changes_refresh_without_rehashing(self):
        """Identity changes use cached image hashes and remove obsolete candidate pairs."""
        _, _, first, second = self.annotated_copies()
        run = self.start()
        self.finish(run)
        second.identity_id = first.identity_id
        second.save(update_fields=["identity"])
        self.start()
        self.assertEqual(models.DuplicateScan.objects.filter(workgroup=self.group).count(), 1)
        self.assertFalse(models.DuplicateIdentitySuggestion.objects.filter(active=True).exists())

    def test_shared_exclusion_is_respected(self):
        """Pairs rejected in existing identity suggestions are also excluded from this new category."""
        _, _, first, second = self.annotated_copies()
        ids = sorted([first.identity_id, second.identity_id])
        models.MergeIdentitySuggestionExclusion.objects.create(
            workgroup=self.group, identity_a_id=ids[0], identity_b_id=ids[1]
        )
        self.finish(self.start())
        self.assertFalse(models.DuplicateIdentitySuggestion.objects.filter(active=True).exists())
