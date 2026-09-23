import tempfile
import uuid
from unittest.mock import patch

from django.core import signing
from django.test import TestCase, override_settings
from django.urls import reverse

from caidapp import bbox_services as service, models, tasks
from .factories import (
    AnimalObservationFactory,
    CaidUserFactory,
    IndividualIdentityFactory,
    MediaFileFactory,
    SequenceFactory,
)


class BboxRedetectionTest(TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings_override = override_settings(MEDIA_ROOT=self.directory.name)
        self.settings_override.enable()
        self.addCleanup(self.settings_override.disable)
        self.mediafile = MediaFileFactory()
        self.user = self.mediafile.parent.owner
        self.client.force_login(self.user.user)

    def observation(self, box, **kwargs):
        observation = AnimalObservationFactory(mediafile=self.mediafile, **kwargs)
        observation.set_bbox_from_xyxy(*box, 1, 1)
        observation.save()
        return observation

    def job(self, **options):
        return service.create_jobs(
            self.user,
            models.MediaFile.objects.filter(pk=self.mediafile.pk),
            {**service.DEFAULT_OPTIONS, **options},
        )[1][0]

    def result(self, *boxes):
        return {"status": "ok", "detector": "sam3", "detections": [{"bbox": b, "confidence": 0.9} for b in boxes]}

    def test_reordered_detections_preserve_ids_and_annotations(self):
        identity = IndividualIdentityFactory(owner_workgroup=self.user.workgroup)
        left = self.observation([0.1, 0.1, 0.3, 0.4], identity=identity, taxon_verified=True)
        right = self.observation([0.6, 0.2, 0.9, 0.8])
        original_taxon = left.taxon_id
        job = self.job()
        self.assertEqual(
            service.finish_job(job.pk, self.result([0.61, 0.2, 0.91, 0.8], [0.11, 0.1, 0.31, 0.4])), "succeeded"
        )
        left.refresh_from_db()
        right.refresh_from_db()
        self.assertAlmostEqual(left.bbox_x_center, 0.21)
        self.assertAlmostEqual(right.bbox_x_center, 0.76)
        self.assertEqual(left.identity_id, identity.pk)
        self.assertEqual(left.taxon_id, original_taxon)
        self.assertTrue(left.taxon_verified)
        self.assertEqual(self.mediafile.observations.count(), 2)

    def test_placeholder_reused_and_excess_detections_created_without_identity(self):
        placeholder_id = self.mediafile.observations.get().pk
        job = self.job()
        service.finish_job(job.pk, self.result([0, 0, 0.3, 0.3], [0.6, 0.6, 1, 1]))
        self.assertEqual(self.mediafile.observations.count(), 2)
        self.assertFalse(self.mediafile.observations.get(pk=placeholder_id).is_no_detection_placeholder)
        self.assertFalse(self.mediafile.observations.exclude(identity=None).exists())

    def test_fewer_detections_clear_only_unmatched_bbox_and_keep_identity(self):
        left = self.observation([0, 0, 0.3, 0.3])
        identity = IndividualIdentityFactory(owner_workgroup=self.user.workgroup)
        right = self.observation([0.6, 0.6, 1, 1], identity=identity)
        job = self.job()
        service.finish_job(job.pk, self.result([0, 0, 0.3, 0.3]))
        right.refresh_from_db()
        self.assertIsNone(right.bbox_width)
        self.assertEqual(right.identity_id, identity.pk)
        self.assertFalse(right.is_no_detection_placeholder)
        self.assertTrue(self.mediafile.observations.filter(pk=left.pk).exists())

    def test_no_overlap_does_not_transfer_identity(self):
        old = self.observation([0, 0, 0.2, 0.2], identity=IndividualIdentityFactory())
        service.finish_job(self.job().pk, self.result([0.8, 0.8, 1, 1]))
        old.refresh_from_db()
        self.assertIsNone(old.bbox_width)
        self.assertIsNone(self.mediafile.observations.exclude(pk=old.pk).get().identity_id)

    def test_unmatched_policies_and_already_bboxless_observations(self):
        old = self.observation([0, 0, 0.2, 0.2])
        bboxless = AnimalObservationFactory(mediafile=self.mediafile)
        models.AnimalObservation.objects.filter(pk=bboxless.pk).update(**dict.fromkeys(service.BBOX_FIELDS))
        service.finish_job(self.job(unmatched="keep", create_new=False).pk, self.result([0.8, 0.8, 1, 1]))
        old.refresh_from_db()
        self.assertAlmostEqual(old.bbox_width, 0.2)
        self.assertEqual(self.mediafile.observations.count(), 2)
        service.finish_job(self.job(unmatched="delete").pk, self.result())
        self.assertFalse(self.mediafile.observations.filter(pk=old.pk).exists())
        self.assertTrue(self.mediafile.observations.filter(pk=bboxless.pk).exists())

    def test_delete_last_observation_restores_placeholder(self):
        self.observation([0, 0, 1, 1])
        service.finish_job(self.job(unmatched="delete").pk, self.result())
        self.assertTrue(self.mediafile.observations.get().is_no_detection_placeholder)

    def test_failure_and_malformed_results_do_not_clear_boxes(self):
        self.observation([0, 0, 1, 1])
        original = list(self.mediafile.observations.values())
        for result in ({"status": "error", "message": "SAM3 unavailable"}, self.result([0, 0, float("nan"), 1])):
            self.assertEqual(service.finish_job(self.job().pk, result), "failed")
            self.assertEqual(list(self.mediafile.observations.values()), original)

    def test_changed_observation_without_updated_at_causes_conflict(self):
        observation = self.observation([0, 0, 1, 1])
        job = self.job()
        models.AnimalObservation.objects.filter(pk=observation.pk).update(bbox_width=0.4)
        self.assertEqual(service.finish_job(job.pk, self.result()), "conflict")
        observation.refresh_from_db()
        self.assertEqual(observation.bbox_width, 0.4)

    def test_missing_source_at_callback_causes_conflict(self):
        observation = self.observation([0, 0, 1, 1])
        job = self.job()
        with patch.object(service, "observation_snapshot", side_effect=FileNotFoundError):
            self.assertEqual(service.finish_job(job.pk, self.result()), "conflict")
        observation.refresh_from_db()
        self.assertEqual(observation.bbox_width, 1)

    def test_result_delivery_is_idempotent(self):
        job = self.job()
        result = self.result([0, 0, 0.3, 0.3], [0.6, 0.6, 1, 1])
        service.finish_job(job.pk, result)
        rows = list(self.mediafile.observations.values())
        service.finish_job(job.pk, result)
        self.assertEqual(rows, list(self.mediafile.observations.values()))

    def test_mid_write_failure_rolls_back_every_change(self):
        old = self.observation([0, 0, 0.3, 0.3])
        original = list(self.mediafile.observations.values())
        job = self.job()
        with patch.object(service, "_invalidate_reid", side_effect=RuntimeError("storage failure")):
            self.assertEqual(
                tasks.finish_bbox_detection(self.result([0, 0, 0.3, 0.3], [0.6, 0.6, 1, 1]), job.pk), "failed"
            )
        self.assertEqual(original, list(self.mediafile.observations.values()))

    def test_weak_pairs_are_forbidden_before_assignment(self):
        # Unconstrained assignment would choose .9 + .19 and then discard .19,
        # losing the better valid .55 + .5 assignment.
        with patch.object(service, "iou", side_effect=[0.9, 0.55, 0.5, 0.19]):
            matches = service.match_boxes([[0] * 4, [0] * 4], [[0] * 4, [0] * 4], 0.2)
        self.assertEqual([(r, c) for r, c, _ in matches], [(0, 1), (1, 0)])

    def test_proposals_are_cached_and_do_not_write_observations(self):
        original = list(self.mediafile.observations.values())
        url = reverse("caidapp:bbox_proposal", args=[self.mediafile.pk])
        first = self.client.post(url).json()
        self.assertEqual(first, self.client.post(url).json())
        job = models.BboxDetectionJob.objects.get()
        service.finish_job(job.pk, self.result([0, 0, 1, 1]))
        self.assertEqual(self.client.get(first["url"]).json()["status"], "succeeded")
        self.assertEqual(original, list(self.mediafile.observations.values()))

    def test_proposal_requires_access_and_post_and_rejects_video(self):
        url = reverse("caidapp:bbox_proposal", args=[self.mediafile.pk])
        self.assertEqual(self.client.get(url).status_code, 405)
        other = MediaFileFactory()
        self.assertEqual(self.client.post(reverse("caidapp:bbox_proposal", args=[other.pk])).status_code, 404)
        self.mediafile.media_type = "video"
        self.mediafile.save()
        self.assertEqual(self.client.post(url).status_code, 404)

    def test_observation_selection_expands_to_all_siblings_and_deduplicates(self):
        first = self.observation([0, 0, 0.3, 0.3])
        second = self.observation([0.6, 0.6, 1, 1])
        response = self.client.post(
            reverse("caidapp:observations"),
            {
                "btnRedetectBboxes": "",
                "selected_observation_ids": [first.pk],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["mediafile_count"], 1)
        self.assertEqual(response.context["observation_count"], 2)
        selection = response.context["form"].initial["selection"]
        start = {**service.DEFAULT_OPTIONS, "selection": selection}
        with self.captureOnCommitCallbacks(execute=True), patch.object(service, "dispatch_job") as dispatch:
            self.assertEqual(self.client.post(reverse("caidapp:bbox_redetection_start"), start).status_code, 302)
        dispatch.assert_called_once()
        self.client.post(reverse("caidapp:bbox_redetection_start"), start)
        self.assertEqual(models.BboxDetectionJob.objects.count(), 1)

    def test_other_user_cannot_use_selection_or_read_results(self):
        job = self.job()
        self.client.force_login(CaidUserFactory().user)
        self.assertEqual(self.client.get(reverse("caidapp:bbox_detection_status", args=[job.batch])).status_code, 404)
        token = signing.dumps(
            {"ids": [self.mediafile.pk], "user": self.user.user_id, "batch": str(uuid.uuid4())}, salt="bbox-selection"
        )
        self.assertEqual(
            self.client.post(
                reverse("caidapp:bbox_redetection_start"), {**service.DEFAULT_OPTIONS, "selection": token}
            ).status_code,
            400,
        )

    def test_sequences_action_prepares_confirmation(self):
        sequence = SequenceFactory(uploaded_archive=self.mediafile.parent)
        self.mediafile.sequence = sequence
        self.mediafile.save()
        response = self.client.post(
            reverse("caidapp:sequences"), {"btnRedetectBboxes": "", "selected_mediafile_ids": [self.mediafile.pk]}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["mediafile_count"], 1)
