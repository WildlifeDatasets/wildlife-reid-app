"""Missing-bbox batches preserve observation identity and exact selection scope."""

import tempfile
from datetime import timedelta
from unittest.mock import patch

from caidapp import bbox_services as service
from caidapp import models, tasks
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .factories import AnimalObservationFactory, CaidUserFactory, IndividualIdentityFactory, MediaFileFactory


class FillMissingBboxTest(TestCase):
    def setUp(self):
        """Create one accessible image with a placeholder in temporary storage."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        settings = override_settings(MEDIA_ROOT=directory.name)
        settings.enable()
        self.addCleanup(settings.disable)
        self.mediafile = MediaFileFactory()
        self.user = self.mediafile.parent.owner
        self.client.force_login(self.user.user)

    def target(self, **kwargs):
        """Add a real object observation without geometry."""
        return AnimalObservationFactory(mediafile=self.mediafile, **kwargs)

    def job(self, ids=None):
        """Queue a fill job with explicit observation IDs, or all missing boxes."""
        return service.create_jobs(
            self.user,
            models.MediaFile.objects.filter(pk=self.mediafile.pk),
            {**service.DEFAULT_OPTIONS, "unmatched": "keep"},
            purpose="fill",
            observation_ids=ids,
        )[1][0]

    def result(self, *boxes):
        """Return a valid detection-only worker result."""
        return {"status": "ok", "detector": "sam3", "detections": [{"bbox": b, "confidence": 0.9} for b in boxes]}

    def test_single_missing_object_keeps_id_taxon_identity_and_sibling_bbox(self):
        """Reserve detections of existing objects and fill the original selected row."""
        identity = IndividualIdentityFactory(owner_workgroup=self.user.workgroup)
        target = self.target(identity=identity, orientation="L")
        sibling = self.target(bbox_x_center=0.75, bbox_y_center=0.75, bbox_width=0.4, bbox_height=0.4)
        sibling_before = models.AnimalObservation.objects.filter(pk=sibling.pk).values().get()
        taxon_id = target.taxon_id
        job = self.job([target.pk])
        result = self.result([0.55, 0.55, 0.95, 0.95], [0.1, 0.1, 0.3, 0.4])
        self.assertEqual(service.finish_job(job.pk, result), "succeeded")
        target.refresh_from_db()
        self.assertEqual((target.identity_id, target.taxon_id, target.orientation), (identity.pk, taxon_id, "L"))
        self.assertAlmostEqual(target.bbox_width, 0.2)
        self.assertEqual(models.AnimalObservation.objects.filter(pk=sibling.pk).values().get(), sibling_before)
        self.assertEqual(self.mediafile.observations.count(), 2)
        rows = list(self.mediafile.observations.values())
        service.finish_job(job.pk, result)
        self.assertEqual(list(self.mediafile.observations.values()), rows)
        self.mediafile.refresh_from_db()
        self.assertTrue(self.mediafile.metadata_json["detection_crop_stale"])

    def test_multiple_candidates_require_review_and_reuse_saved_proposals_after_fifteen_minutes(self):
        """Do not guess between detections; detail suggestions can retrieve the retained result."""
        target = self.target()
        original = list(self.mediafile.observations.values())
        job = self.job([target.pk])
        result = self.result([0, 0, 0.3, 0.3], [0.6, 0.6, 1, 1])
        self.assertEqual(service.finish_job(job.pk, result), "review")
        self.assertEqual(list(self.mediafile.observations.values()), original)
        models.BboxDetectionJob.objects.filter(pk=job.pk).update(created_at=timezone.now() - timedelta(hours=1))
        response = self.client.post(reverse("caidapp:bbox_proposal", args=[self.mediafile.pk]) + f"?bbox_job={job.pk}")
        proposal = self.client.get(response.json()["url"]).json()
        self.assertEqual(proposal["status"], "succeeded")
        self.assertEqual(proposal["detections"], result["detections"])
        self.assertEqual(models.BboxDetectionJob.objects.count(), 1)
        status = self.client.get(reverse("caidapp:bbox_detection_status", args=[job.batch]))
        self.assertContains(status, "Needs assignment")
        self.assertContains(status, f"?bbox_job={job.pk}")

    def test_unselected_missing_object_prevents_automatic_assignment(self):
        """One selected row does not establish which of two objects was detected."""
        target = self.target()
        self.target()
        original = list(self.mediafile.observations.values())
        self.assertEqual(service.finish_job(self.job([target.pk]).pk, self.result([0, 0, 1, 1])), "review")
        self.assertEqual(list(self.mediafile.observations.values()), original)

    def test_multiple_targets_and_equal_number_of_detections_still_need_assignment(self):
        """Equal counts cannot establish an identity-to-geometry mapping."""
        targets = [self.target(), self.target()]
        self.assertEqual(
            service.finish_job(self.job([o.pk for o in targets]).pk, self.result([0, 0, 0.3, 0.3], [0.6, 0.6, 1, 1])),
            "review",
        )
        self.assertFalse(self.mediafile.observations.exclude(bbox_x_center=None).exists())

    def test_placeholder_reused_and_additional_objects_created_without_annotations(self):
        """A placeholder is not a reviewed object and can seed new detections."""
        placeholder = self.mediafile.observations.get()
        service.finish_job(self.job([placeholder.pk]).pk, self.result([0, 0, 0.3, 0.3], [0.6, 0.6, 1, 1]))
        placeholder.refresh_from_db()
        self.assertFalse(placeholder.is_no_detection_placeholder)
        self.assertEqual(self.mediafile.observations.count(), 2)
        self.assertFalse(self.mediafile.observations.exclude(identity=None, taxon=None).exists())

    def test_empty_legacy_mediafile_gets_detections_or_a_placeholder(self):
        """Sequences can repair files that have no observation rows at all."""
        for detections in (self.result(), self.result([0, 0, 1, 1])):
            self.mediafile.observations.all().delete()
            self.assertEqual(service.finish_job(self.job().pk, detections), "succeeded")
            self.assertEqual(self.mediafile.observations.count(), 1)
            self.assertEqual(
                self.mediafile.observations.get().is_no_detection_placeholder, not bool(detections["detections"])
            )

    def test_no_detection_and_worker_failure_preserve_existing_annotations(self):
        """Missing detections are not a request to delete or clear observations."""
        target = self.target()
        original = list(self.mediafile.observations.values())
        for result in (self.result(), {"status": "error", "message": "SAM3 unavailable"}):
            service.finish_job(self.job([target.pk]).pk, result)
            self.assertEqual(list(self.mediafile.observations.values()), original)

    def test_concurrent_edit_and_revoked_access_reject_results(self):
        """Fill jobs retain the shared snapshot and access checks."""
        target = self.target()
        job = self.job([target.pk])
        models.AnimalObservation.objects.filter(pk=target.pk).update(orientation="R")
        self.assertEqual(service.finish_job(job.pk, self.result([0, 0, 1, 1])), "conflict")
        job = self.job([target.pk])
        with patch.object(models.MediaFile.objects, "for_user", return_value=models.MediaFile.objects.none()):
            self.assertEqual(service.finish_job(job.pk, self.result([0, 0, 1, 1])), "conflict")

    def test_mid_write_failure_rolls_back_placeholder_and_new_observations(self):
        """A failed multi-object fill cannot leave a partially applied result."""
        original = list(self.mediafile.observations.values())
        job = self.job()
        with patch.object(service, "_invalidate_reid", side_effect=RuntimeError("storage failed")):
            self.assertEqual(
                tasks.finish_bbox_detection(self.result([0, 0, 0.3, 0.3], [0.6, 0.6, 1, 1]), job.pk), "failed"
            )
        self.assertEqual(list(self.mediafile.observations.values()), original)

    def test_review_result_is_not_reused_after_annotations_change_or_by_another_user(self):
        """Retained proposals remain tied to their exact source, annotations and requester."""
        target = self.target()
        job = self.job([target.pk])
        service.finish_job(job.pk, self.result([0, 0, 0.3, 0.3], [0.6, 0.6, 1, 1]))
        models.AnimalObservation.objects.filter(pk=target.pk).update(orientation="R")
        url = reverse("caidapp:bbox_proposal", args=[self.mediafile.pk]) + f"?bbox_job={job.pk}"
        response = self.client.post(url)
        self.assertNotEqual(response.json()["url"], reverse("caidapp:bbox_proposal_status", args=[job.pk]))
        self.client.force_login(CaidUserFactory().user)
        self.assertEqual(self.client.post(url).status_code, 404)
        self.assertEqual(self.client.get(reverse("caidapp:bbox_proposal_status", args=[job.pk])).status_code, 404)

    def confirmation(self, **post):
        """Prepare an observation-scoped fill confirmation."""
        return self.client.post(reverse("caidapp:observations"), {"btnFillMissingBboxes": "", **post})

    def start(self, confirmation):
        """Submit the signed scope without exposing reconciliation policies."""
        return self.client.post(
            reverse("caidapp:bbox_fill_missing_start"),
            {
                "selection": confirmation.context["form"].initial["selection"],
                "detector": "sam3",
                "confidence": 0.5,
            },
        )

    def test_filtered_selection_dispatches_once_per_mediafile_and_submission_is_idempotent(self):
        """Select across pages while keeping exact IDs and excluding existing bboxes."""
        targets = [self.target(), self.target()]
        boxed = self.target(bbox_x_center=0.5, bbox_y_center=0.5, bbox_width=0.2, bbox_height=0.2)
        confirmation = self.client.post(
            reverse("caidapp:observations") + "?has_bbox=false",
            {
                "select_all_filtered": "on",
                "btnFillMissingBboxes": "",
            },
        )
        self.assertEqual(confirmation.context["observation_count"], 2)
        self.assertEqual(confirmation.context["mediafile_count"], 1)
        with self.captureOnCommitCallbacks(execute=True), patch.object(service, "dispatch_job") as dispatch:
            self.assertEqual(self.start(confirmation).status_code, 302)
        dispatch.assert_called_once()
        job = models.BboxDetectionJob.objects.get()
        self.assertEqual(job.options["observation_ids"], [o.pk for o in targets])
        self.assertNotIn(boxed.pk, job.options["observation_ids"])
        self.assertEqual(job.options["detector"], "sam3")
        self.assertEqual(self.start(confirmation).status_code, 302)
        self.assertEqual(models.BboxDetectionJob.objects.count(), 1)

    def test_selection_preserves_unselected_sibling_and_changed_box_before_start_is_skipped(self):
        """Signed confirmation cannot expand to unselected observations."""
        target, sibling = self.target(), self.target()
        confirmation = self.confirmation(selected_observation_ids=[target.pk])
        self.start(confirmation)
        self.assertEqual(models.BboxDetectionJob.objects.get().options["observation_ids"], [target.pk])
        self.assertNotIn(sibling.pk, models.BboxDetectionJob.objects.get().options["observation_ids"])
        models.BboxDetectionJob.objects.all().delete()
        target.set_bbox_from_xyxy(0, 0, 1, 1, 1, 1)
        target.save()
        self.assertEqual(self.start(confirmation).status_code, 400)
        self.assertFalse(models.BboxDetectionJob.objects.exists())

    def test_confirmation_is_user_bound_and_tampering_is_rejected(self):
        """Neither token tampering nor changing the logged-in user grants access."""
        confirmation = self.confirmation(selected_observation_ids=[self.target().pk])
        self.client.force_login(CaidUserFactory().user)
        self.assertEqual(self.start(confirmation).status_code, 400)
        self.assertEqual(
            self.client.post(reverse("caidapp:bbox_fill_missing_start"), {"selection": "invalid"}).status_code, 400
        )
        self.assertEqual(self.client.get(reverse("caidapp:bbox_fill_missing_start")).status_code, 405)

    def test_sequences_can_select_empty_files_and_skip_videos_and_fully_boxed_files(self):
        """Media-file scope includes empty legacy files but never dispatches videos."""
        self.mediafile.observations.all().delete()
        video = MediaFileFactory(parent=self.mediafile.parent, media_type="video")
        boxed = MediaFileFactory(parent=self.mediafile.parent)
        AnimalObservationFactory(mediafile=boxed, bbox_x_center=0.5, bbox_y_center=0.5, bbox_width=0.2, bbox_height=0.2)
        confirmation = self.client.post(
            reverse("caidapp:sequences"),
            {
                "btnFillMissingBboxes": "",
                "selected_mediafile_ids": [self.mediafile.pk, video.pk, boxed.pk],
            },
        )
        self.assertEqual(confirmation.context["mediafile_count"], 1)
        self.assertEqual(confirmation.context["empty_count"], 1)
        self.assertEqual(confirmation.context["video_count"], 1)
        self.assertEqual(self.start(confirmation).status_code, 302)
        self.assertEqual(models.BboxDetectionJob.objects.get().mediafile_id, self.mediafile.pk)
