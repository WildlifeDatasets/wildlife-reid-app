"""Explicit bulk confirmations, metadata completeness, authorization and rollback."""

import time
from datetime import timedelta
from unittest.mock import patch

from caidapp import duplicate_jobs as jobs
from caidapp import duplicate_services as service
from caidapp import models
from django.core import signing
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from . import test_duplicates as fixtures
from .factories import AnimalObservationFactory, MediaFileFactory, TaxonFactory


class DuplicateJobTest(TestCase):
    setUp = fixtures.DuplicateReviewTest.setUp
    media = fixtures.DuplicateReviewTest.media
    proposal = fixtures.DuplicateReviewTest.proposal

    def group(self):
        """Create one exact group using real, stable source files."""
        proposal = self.proposal([self.media(), self.media()])
        proposal.scan.workgroup = self.user.workgroup
        proposal.scan.save(update_fields=["workgroup"])
        return proposal

    def ready_preview(self):
        """Prepare one preview in the worker, independently of its status-page request."""
        proposal = self.group()
        run = jobs.start_preview(self.user, [str(proposal.uuid)])
        jobs.process(run.pk)
        run.refresh_from_db()
        self.assertEqual(run.status, "succeeded", run.message)
        return run

    def test_http_enqueue_and_status_never_read_source_files(self):
        """A slow storage read cannot hold the initial POST or polling GET open."""
        proposal = self.group()
        with patch("caidapp.duplicate_services.source_hash", side_effect=AssertionError("HTTP read source")):
            response = self.client.post(
                reverse("caidapp:duplicate_bulk_review"), {"action": "selected", "proposals": [proposal.uuid]}
            )
            self.assertEqual(response.status_code, 302)
            self.assertEqual(self.client.get(response.url).status_code, 200)
        run = models.TaskRun.objects.get(kind="duplicate_preview")
        self.assertEqual(run.status, "queued")
        jobs.process(run.pk)
        with patch("caidapp.duplicate_services.source_hash", side_effect=AssertionError("HTTP rehashed preview")):
            result = self.client.get(response.url)
            self.assertContains(result, 'id="duplicate-confirm-list"')
            self.assertNotContains(result, 'class="card mb-3"')

    def test_confirmation_is_fast_and_repeat_submission_or_delivery_is_idempotent(self):
        """Only the worker deletes, and one accepted confirmation has one durable result."""
        preview = self.ready_preview()
        payload = {"action": "apply", "token": preview.result["token"]}
        with patch("caidapp.duplicate_services.apply_bulk", side_effect=AssertionError("HTTP deleted files")):
            first = self.client.post(reverse("caidapp:duplicate_bulk_review"), payload)
            second = self.client.post(reverse("caidapp:duplicate_bulk_review"), payload)
        self.assertEqual(first.url, second.url)
        run = models.TaskRun.objects.get(kind="duplicate_removal")
        self.assertEqual(models.MediaFile.objects.count(), 2)
        with patch("caidapp.duplicate_services.apply_bulk", wraps=service.apply_bulk) as apply:
            jobs.process(run.pk)
            jobs.process(run.pk)
        self.assertEqual(apply.call_count, 1)
        run.refresh_from_db()
        self.assertEqual(run.status, "succeeded")
        self.assertEqual(run.result["removed"], 1)

    def test_accepted_job_does_not_expire_while_waiting_in_queue(self):
        """An accepted confirmation remains authorized, but stale unaccepted previews still expire."""
        preview = self.ready_preview()
        run = jobs.start_removal(self.user, preview.result["token"])
        future = time.time() + 3600
        with patch("django.core.signing.time.time", return_value=future):
            with self.assertRaises(signing.SignatureExpired):
                signing.loads(preview.result["token"], salt="duplicate-bulk-v1", max_age=1800)
            jobs.process(run.pk)
        run.refresh_from_db()
        self.assertEqual(run.status, "succeeded", run.message)

    def test_job_page_is_private_and_worker_rechecks_group_membership(self):
        """Another user cannot view the job, and changed access prevents worker execution."""
        proposal = self.group()
        run = jobs.start_preview(self.user, [str(proposal.uuid)])
        other = MediaFileFactory().parent.owner
        self.client.force_login(other.user)
        self.assertEqual(self.client.get(reverse("caidapp:duplicate_job", args=[run.uuid])).status_code, 404)
        models.CaIDUser.objects.filter(pk=self.user.pk).update(workgroup=other.workgroup)
        jobs.process(run.pk)
        run.refresh_from_db()
        self.assertEqual(run.status, "failed")
        self.assertIn("Access", run.message)
        self.assertEqual(models.MediaFile.objects.count(), 3)

    def test_preview_signing_happens_after_all_source_reads(self):
        """Preparation time does not consume the user's confirmation expiry window."""
        proposals = [self.group(), self.group()]
        calls = []
        hash_source, sign = service.source_hash, service.preview_token

        def read(*args):
            calls.append("read")
            return hash_source(*args)

        def token(*args):
            calls.append("sign")
            return sign(*args)

        with patch("caidapp.duplicate_services.source_hash", side_effect=read), patch(
            "caidapp.duplicate_services.preview_token", side_effect=token
        ):
            service.bulk_preview(proposals, self.user)
        self.assertEqual(calls[-2:], ["sign", "sign"])
        self.assertNotIn("sign", calls[:-2])

    def test_delivery_failure_is_durable_and_retryable_without_deletion(self):
        """A broker failure reports an outcome instead of losing the requested job."""
        proposal = self.group()
        with patch("caidapp.duplicate_jobs.signature") as publish:
            publish.return_value.apply_async.side_effect = OSError("broker unavailable")
            with self.captureOnCommitCallbacks(execute=True):
                run = jobs.start_preview(self.user, [str(proposal.uuid)])
        run.refresh_from_db()
        self.assertEqual(run.status, "failed")
        self.assertIsNone(run.started_at)
        self.assertEqual(models.MediaFile.objects.count(), 2)
        with patch("caidapp.duplicate_jobs.signature") as publish:
            jobs.retry_delivery(run)
            self.assertEqual(publish.call_count, 1)
        run.refresh_from_db()
        self.assertEqual(run.status, "queued")

    def test_lost_queued_delivery_is_republished_once(self):
        """Recovery is throttled and reuses the durable job rather than creating another."""
        proposal = self.group()
        run = jobs.start_preview(self.user, [str(proposal.uuid)])
        models.TaskRun.objects.filter(pk=run.pk).update(updated_at=timezone.now() - timedelta(minutes=5))
        with patch("caidapp.duplicate_jobs._publish") as publish:
            jobs.retry_delivery(run)
            jobs.retry_delivery(run)
        publish.assert_called_once_with(run.pk)


class DuplicateBulkTest(TestCase):
    setUp = fixtures.DuplicateReviewTest.setUp
    media = fixtures.DuplicateReviewTest.media
    proposal = fixtures.DuplicateReviewTest.proposal

    def test_runtime_reference_ids_do_not_block_perfect_match(self):
        """Sequence-local positions and cached ReID observation references are not source facts."""
        a = self.media(
            metadata_json={"sequence_number": 9, "reid_observation_id": 194279, "datetime": "2025-09-05 23:52:41"}
        )
        b = self.media(
            metadata_json={"sequence_number": 6, "reid_observation_id": 194460, "datetime": "2025-09-05 23:52:41"}
        )
        proposal = self.group_proposal([a, b])
        summary = service.proposal_summary(proposal)
        self.assertTrue(summary["perfect"])
        self.assertEqual(summary["total_difference_count"], 0)
        self.assertEqual(summary["extra_differences"], [])
        a.refresh_from_db()
        self.assertEqual(a.metadata_json["reid_observation_id"], 194279)

    def test_strict_differences_have_specific_labels_and_values(self):
        """Explain verification/geometry/EXIF differences in both catalogue and signed plan."""
        a, b = self.media(), self.media(metadata_json={"exif": {"EXIF:ISO": 640}})
        taxon = TaxonFactory()
        AnimalObservationFactory(mediafile=a, taxon=taxon, taxon_verified=True)
        AnimalObservationFactory(mediafile=b, taxon=taxon, taxon_verified=False)
        proposal = self.group_proposal([a, b])
        summary = service.proposal_summary(proposal)
        labels = [detail["label"] for detail in summary["extra_differences"]]
        self.assertTrue(any("EXIF:ISO" in label or "Source metadata" in label for label in labels))
        self.assertIn("Taxon verification differs", labels)
        self.assertEqual(summary["total_difference_count"], 2)
        plan = service.build_plan(proposal, self.user, a.pk)
        self.assertEqual(plan["total_difference_count"], summary["total_difference_count"])
        self.assertTrue(any(detail["field"] == "taxon_verified" for detail in plan["extra_differences"]))
        preview = self.preview([proposal])
        self.assertContains(preview, "Taxon verification differs")
        self.assertNotContains(preview, "Other metadata or observations differ")

    def test_exif_difference_names_the_exact_changed_key(self):
        """A meaningful nested source field can be located using the metadata popup."""
        proposal = self.group_proposal(
            [
                self.media(metadata_json={"exif": {"EXIF:ISO": 320}}),
                self.media(metadata_json={"exif": {"EXIF:ISO": 640}}),
            ]
        )
        summary = service.proposal_summary(proposal)
        self.assertIn("exif.EXIF:ISO", summary["extra_differences"][0]["label"])

    def group_proposal(self, mediafiles, kind="pixel_exact"):
        """Make an authorized durable group suggestion using real source bytes."""
        proposal = self.proposal(mediafiles, kind=kind)
        proposal.scan.workgroup = self.user.workgroup
        proposal.scan.save(update_fields=["workgroup"])
        return proposal

    def preview(self, proposals):
        """Request selected whole-group deletion without applying it."""
        response = self.client.post(
            reverse("caidapp:duplicate_bulk_review"),
            {"action": "selected", "proposals": [str(p.uuid) for p in proposals]},
        )
        return self.finish_job(response)

    def finish_job(self, response):
        """Simulate Celery execution after a short HTTP enqueue request."""
        if response.status_code == 302 and "/duplicates/jobs/" in response.url:
            run = models.TaskRun.objects.get(uuid=response.url.rstrip("/").split("/")[-1])
            jobs.process(run.pk)
            return self.client.get(response.url)
        return response

    def confirm(self, response):
        """Confirm exactly the server-signed preview."""
        with patch("caidapp.tasks.cleanup_duplicate_files.delay"):
            result = self.client.post(
                reverse("caidapp:duplicate_bulk_review"), {"action": "apply", "token": response.context["token"]}
            )
        return self.finish_job(result)

    def test_difference_count_counts_fields_not_pairs_and_does_not_hash_sources(self):
        """A triplicate with three different notes has one differing field."""
        proposal = self.group_proposal([self.media(note=value) for value in ["a", "b", "c"]])
        with patch("caidapp.duplicate_services.source_hash", side_effect=AssertionError("Unexpected hashing")):
            summary = service.proposal_summary(proposal)
        self.assertEqual(summary["difference_count"], 1)
        self.assertFalse(summary["perfect"])
        self.assertEqual(sum(m["role"] == "delete" for m in summary["members"]), 2)

    def test_hidden_annotation_and_metadata_differences_are_not_perfect(self):
        """Zero visible fields never authorizes perfect-only deletion of different object data."""
        for difference in ["bbox", "verified", "metadata", "album", "cover"]:
            with self.subTest(difference=difference):
                a, b = self.media(), self.media()
                taxon = TaxonFactory()
                AnimalObservationFactory(mediafile=a, taxon=taxon)
                observation = AnimalObservationFactory(mediafile=b, taxon=taxon)
                if difference == "bbox":
                    observation.bbox_x_center = 0.3
                    observation.save(update_fields=["bbox_x_center"])
                elif difference == "verified":
                    observation.taxon_verified = False
                    observation.save(update_fields=["taxon_verified"])
                elif difference == "metadata":
                    b.metadata_json = {"camera": "extra"}
                    b.save(update_fields=["metadata_json"])
                elif difference == "album":
                    album = models.Album.objects.create(name="Source only", owner=self.user)
                    album.mediafiles.add(b)
                else:
                    album = models.Album.objects.create(name="Source cover", owner=self.user, cover=b)
                    album.mediafiles.add(a, b)
                summary = service.proposal_summary(self.group_proposal([a, b]))
                self.assertEqual(summary["difference_count"], 0)
                self.assertFalse(summary["perfect"])

    def test_runtime_metadata_is_included_in_perfect_preview_and_confirmation(self):
        """Reimports differ in provenance only; list, count and signed preview must agree."""
        a = self.media(
            metadata_json={
                "full_image_path": "/processing/first/image.webp",
                "exif": "{'SourceFile': '/tmp/first/image.jpg', 'File:FileModifyDate': 'first', 'EXIF:Make': 'Apple'}",
            }
        )
        b = self.media(
            metadata_json={
                "full_image_path": "/processing/second/image.webp",
                "exif": (
                    "{'SourceFile': '/tmp/second/image.jpg', 'File:FileModifyDate': 'second', " "'EXIF:Make': 'Apple'}"
                ),
            }
        )
        proposal = self.group_proposal([a, b])
        summary = service.proposal_summary(proposal)
        self.assertEqual(summary["difference_count"], 0)
        self.assertTrue(summary["perfect"])
        kept_id = summary["keep_id"]
        before = models.MediaFile.objects.filter(pk=kept_id).values().get()
        response = self.client.post(reverse("caidapp:duplicate_bulk_review"), {"action": "perfect"})
        response = self.finish_job(response)
        self.assertEqual(response.context["remove_count"], 1)
        self.assertTrue(response.context["previews"][0]["summary"]["perfect"])
        self.assertNotContains(response, "Other metadata or observations differ")
        self.confirm(response)
        self.assertEqual(models.MediaFile.objects.filter(pk=kept_id).values().get(), before)
        self.assertEqual(models.MediaFile.objects.filter(pk__in=[a.pk, b.pk]).count(), 1)

    def test_meaningful_exif_change_still_blocks_perfect_action(self):
        """Ignoring runtime provenance must not ignore capture-time facts."""
        a = self.media(metadata_json={"exif": "{'EXIF:DateTimeOriginal': '2026:09:13 21:56:16'}"})
        b = self.media(metadata_json={"exif": "{'EXIF:DateTimeOriginal': '2026:09:14 21:56:16'}"})
        proposal = self.group_proposal([a, b])
        self.assertFalse(service.proposal_summary(proposal)["perfect"])
        with self.assertRaisesRegex(ValueError, "No eligible duplicate groups"):
            service.bulk_preview([proposal], self.user, perfect_only=True)

    def test_preview_then_confirm_triplicate(self):
        """Preview changes nothing; confirmation removes both red files and preserves the green one."""
        a, b, c = [self.media() for _ in range(3)]
        proposal = self.group_proposal([a, b, c])
        before = models.MediaFile.objects.filter(pk=a.pk).values().get()
        preview = self.preview([proposal])
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.context["remove_count"], 2)
        self.assertEqual(models.MediaFile.objects.filter(pk__in=[a.pk, b.pk, c.pk]).count(), 3)
        self.confirm(preview)
        self.assertEqual(models.MediaFile.objects.filter(pk=a.pk).values().get(), before)
        self.assertFalse(models.MediaFile.objects.filter(pk__in=[b.pk, c.pk]).exists())
        proposal.refresh_from_db()
        self.assertEqual(proposal.status, "resolved")

    def test_stale_second_group_rolls_back_entire_batch(self):
        """No first-group deletion or cleanup intent may survive a later conflict."""
        files = [self.media() for _ in range(4)]
        first = self.group_proposal(files[:2])
        second = self.group_proposal(files[2:])
        preview = self.preview([first, second])
        models.MediaFile.objects.filter(pk=files[-1].pk).update(note="Changed after preview")
        self.confirm(preview)
        self.assertEqual(models.MediaFile.objects.filter(pk__in=[m.pk for m in files]).count(), 4)
        self.assertFalse(models.DuplicateFileCleanup.objects.exists())
        first.refresh_from_db()
        self.assertEqual(first.status, "pending")

    def test_perfect_action_skips_imperfect_and_video_reference_groups(self):
        """Perfect selection examines the group catalogue without deleting or including reference-only matches."""
        perfect = self.group_proposal([self.media(), self.media()])
        self.group_proposal([self.media(note="first"), self.media(note="second")])
        self.group_proposal([self.media(), self.media()], kind="reference_exact")
        response = self.client.post(reverse("caidapp:duplicate_bulk_review"), {"action": "perfect"})
        response = self.finish_job(response)
        self.assertEqual([row["proposal"].pk for row in response.context["previews"]], [perfect.pk])
        self.assertEqual(response.context["skipped"], 2)
        self.assertEqual(models.MediaFile.objects.count(), 6)

    def test_foreign_selection_and_tampered_confirmation_cannot_delete(self):
        """Both preview scope and signed confirmation are authorization boundaries."""
        foreign = MediaFileFactory()
        hidden = self.group_proposal([self.media(), foreign])
        response = self.preview([hidden])
        self.assertEqual(response.status_code, 302)
        response = self.client.post(reverse("caidapp:duplicate_bulk_review"), {"action": "apply", "token": "forged"})
        self.assertEqual(response.status_code, 302)
        self.assertTrue(models.MediaFile.objects.filter(pk=foreign.pk).exists())

    def test_overlapping_proposals_are_not_double_deleted(self):
        """One source may appear in at most one reviewed group within a batch."""
        a, b, c = self.media(), self.media(), self.media()
        response = self.preview([self.group_proposal([a, b]), self.group_proposal([b, c])])
        self.assertEqual(len(response.context["previews"]), 1)
        self.assertEqual(response.context["skipped"], 1)

    def test_batch_limit_is_reported_in_preview(self):
        """Bound the transaction and explicitly report groups remaining beyond the batch."""
        proposals = [self.group_proposal([self.media(), self.media()]) for _ in range(2)]
        with patch("caidapp.duplicate_services.MAX_BULK_PROPOSALS", 1):
            preview = service.bulk_preview(proposals, self.user, perfect_only=True)
        self.assertEqual(len(preview["previews"]), 1)
        self.assertTrue(preview["limited"])

    def test_album_cover_changes_after_preview_reject_deletion(self):
        """Changing an external reference invalidates the reviewed deletion snapshot."""
        a, b = self.media(), self.media()
        proposal = self.group_proposal([a, b])
        preview = self.preview([proposal])
        models.Album.objects.create(name="New cover", owner=self.user, cover=b)
        self.confirm(preview)
        self.assertEqual(models.MediaFile.objects.filter(pk__in=[a.pk, b.pk]).count(), 2)

    def test_list_displays_roles_upload_and_info_without_visible_ids(self):
        """Catalogue display agrees with score-based preselection and exposes copyable metadata."""
        self.group_proposal([self.media(), self.media(note="richer")])
        with patch("caidapp.duplicate_hashes.recover_group"):
            response = self.client.get(reverse("caidapp:duplicates"))
        self.assertContains(response, 'class="mediafile-info-trigger"')
        self.assertContains(response, f"Upload: {self.archive.name}")
        self.assertContains(response, "text-success")
        self.assertContains(response, 'name="proposals"')
        self.assertNotContains(response, "(#")

    def test_catalogue_sorts_perfect_then_fewest_differing_fields(self):
        """Perfect matches lead even when they were created later than conflicting groups."""
        high = self.group_proposal([self.media(), self.media(note="Different", location="50,14")])
        medium = self.group_proposal([self.media(), self.media(note="Different")])
        perfect = self.group_proposal([self.media(), self.media()])
        hidden = self.group_proposal([self.media(), self.media(metadata_json={"extra": 1})])
        with patch("caidapp.duplicate_hashes.recover_group"):
            response = self.client.get(reverse("caidapp:duplicates"))
        self.assertEqual([row.pk for row in response.context["page_obj"]], [perfect.pk, medium.pk, hidden.pk, high.pk])

    def test_perfect_match_moves_to_first_page_and_api_uses_same_order(self):
        """Sort before slicing pages, with stable order among equally matching groups."""
        earlier = [self.group_proposal([self.media(), self.media(note="Different")]) for _ in range(21)]
        perfect = self.group_proposal([self.media(), self.media()])
        with patch("caidapp.duplicate_hashes.recover_group"):
            first = self.client.get(reverse("caidapp:duplicates"))
            second = self.client.get(reverse("caidapp:duplicates"), {"page": 2})
        self.assertEqual(first.context["page_obj"][0].pk, perfect.pk)
        self.assertEqual([row.pk for row in second.context["page_obj"]], [earlier[-2].pk, earlier[-1].pk])
        with patch("caidapp.duplicate_services.refresh_scan"):
            api = self.client.get(reverse("caidapp:duplicate_scan_matches", args=[perfect.scan.uuid]))
        self.assertEqual(api.json()["results"][0]["uuid"], str(perfect.uuid))

    def test_show_next_follows_match_order(self):
        """Individual review advances to the next perfect match before older conflicting groups."""
        self.group_proposal([self.media(), self.media(note="Different")])
        current = self.group_proposal([self.media(), self.media()])
        following = self.group_proposal([self.media(), self.media()])
        url = reverse("caidapp:duplicate_review", args=[current.uuid])
        preview = self.client.get(url)
        with patch("caidapp.tasks.cleanup_duplicate_files.delay"):
            result = self.client.post(url, {"token": preview.context["token"]})
        self.assertEqual(result.url, reverse("caidapp:duplicate_review", args=[following.uuid]))
