from io import StringIO

from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import IntegrityError, transaction
from django.test import TestCase

from caidapp import models, tasks

from .factories import AnimalObservationFactory, MediaFileFactory, TaxonFactory


class ObservationInvariantTest(TestCase):
    def test_new_mediafile_has_one_no_detection_placeholder(self):
        mediafile = MediaFileFactory()

        observations = list(mediafile.observations.all())

        self.assertEqual(len(observations), 1)
        self.assertTrue(observations[0].is_no_detection_placeholder)
        self.assertFalse(observations[0].has_object_data)

    def test_real_observation_removes_placeholder(self):
        mediafile = MediaFileFactory()
        placeholder_id = mediafile.observations.get().id

        observation = AnimalObservationFactory(mediafile=mediafile)

        self.assertEqual(list(mediafile.observations.values_list("id", flat=True)), [observation.id])
        self.assertNotEqual(observation.id, placeholder_id)
        self.assertFalse(observation.is_no_detection_placeholder)

    def test_adding_object_data_promotes_placeholder(self):
        mediafile = MediaFileFactory()
        observation = mediafile.observations.get()
        observation.taxon = TaxonFactory()

        observation.save(update_fields=["taxon"])

        observation.refresh_from_db()
        self.assertFalse(observation.is_no_detection_placeholder)

    def test_deleting_last_observation_restores_placeholder(self):
        mediafile = MediaFileFactory(taxon=TaxonFactory())
        observation = mediafile.observations.get()

        with self.captureOnCommitCallbacks(execute=True):
            observation.delete()

        replacement = mediafile.observations.get()
        self.assertNotEqual(replacement.id, observation.id)
        self.assertTrue(replacement.is_no_detection_placeholder)

    def test_mediafile_cascade_delete_does_not_recreate_child(self):
        mediafile = MediaFileFactory()
        mediafile_id = mediafile.id

        with self.captureOnCommitCallbacks(execute=True):
            mediafile.delete()

        self.assertFalse(models.MediaFile.objects.filter(pk=mediafile_id).exists())
        self.assertFalse(models.AnimalObservation.objects.filter(mediafile_id=mediafile_id).exists())

    def test_only_one_placeholder_is_allowed_per_mediafile(self):
        mediafile = MediaFileFactory()

        with self.assertRaises(IntegrityError), transaction.atomic():
            models.AnimalObservation.objects.create(
                mediafile=mediafile,
                is_no_detection_placeholder=True,
            )

    def test_two_detections_reuse_placeholder_then_create_second_observation(self):
        mediafile = MediaFileFactory()
        placeholder_id = mediafile.observations.get().id
        taxon = TaxonFactory()
        detection_results = [
            {
                "bbox": [10, 20, 30, 60],
                "size": [100, 200],
                "orientation": "right",
                "orientation_score": 0.9,
            },
            {
                "bbox": [100, 10, 180, 90],
                "size": [100, 200],
                "orientation": "unknown",
                "orientation_score": 0.9,
            },
        ]

        found = tasks._sync_observations_from_detection_results(
            mediafile,
            detection_results,
            taxon=taxon,
            predicted_taxon=taxon,
            predicted_taxon_confidence=0.8,
            identity=None,
            identity_is_representative=False,
        )

        observations = list(mediafile.observations.order_by("id"))
        self.assertTrue(found)
        self.assertEqual(len(observations), 2)
        self.assertEqual(observations[0].id, placeholder_id)
        self.assertFalse(observations[0].is_no_detection_placeholder)
        self.assertAlmostEqual(observations[0].bbox_x_center, 0.1)
        self.assertAlmostEqual(observations[1].bbox_x_center, 0.7)

    def test_empty_detection_result_restores_single_placeholder(self):
        mediafile = MediaFileFactory()
        taxon = TaxonFactory()
        tasks._sync_observations_from_detection_results(
            mediafile,
            [
                {
                    "bbox": [10, 20, 30, 60],
                    "size": [100, 200],
                    "orientation": "right",
                    "orientation_score": 0.9,
                }
            ],
            taxon=taxon,
            predicted_taxon=taxon,
            predicted_taxon_confidence=0.8,
            identity=None,
            identity_is_representative=False,
        )

        found = tasks._sync_observations_from_detection_results(
            mediafile,
            [],
            taxon=taxon,
            predicted_taxon=taxon,
            predicted_taxon_confidence=0.8,
            identity=None,
            identity_is_representative=False,
        )

        observation = mediafile.observations.get()
        self.assertFalse(found)
        self.assertTrue(observation.is_no_detection_placeholder)
        self.assertIsNone(observation.taxon_id)
        self.assertIsNone(observation.bbox_x_center)

    def test_detection_sync_rolls_back_all_rows_if_a_later_detection_is_invalid(self):
        mediafile = MediaFileFactory()
        first = AnimalObservationFactory(mediafile=mediafile, bbox_x_center=0.2)
        second = AnimalObservationFactory(mediafile=mediafile, bbox_x_center=0.8)
        original_rows = list(
            mediafile.observations.order_by("id").values_list("id", "bbox_x_center")
        )
        taxon = TaxonFactory()
        detection_results = [
            {
                "bbox": [10, 20, 30, 60],
                "size": [100, 200],
                "orientation": "right",
                "orientation_score": 0.9,
            },
            {
                "size": [100, 200],
                "orientation": "left",
                "orientation_score": 0.9,
            },
        ]

        with self.assertRaises(KeyError):
            tasks._sync_observations_from_detection_results(
                mediafile,
                detection_results,
                taxon=taxon,
                predicted_taxon=taxon,
                predicted_taxon_confidence=0.8,
                identity=None,
                identity_is_representative=False,
            )

        self.assertEqual(
            list(mediafile.observations.order_by("id").values_list("id", "bbox_x_center")),
            original_rows,
        )
        self.assertEqual({first.id, second.id}, {row[0] for row in original_rows})

    def test_audit_is_read_only_by_default_and_repair_is_explicit(self):
        mediafile = MediaFileFactory()
        mediafile.observations.all().delete()
        output = StringIO()

        with self.assertRaises(CommandError):
            call_command("audit_mediafile_observations", stdout=output)
        self.assertFalse(mediafile.observations.exists())

        call_command("audit_mediafile_observations", "--repair", stdout=output)

        self.assertTrue(mediafile.observations.get().is_no_detection_placeholder)
        call_command("audit_mediafile_observations", stdout=output)

    def test_audit_repairs_malformed_and_conflicting_placeholders(self):
        malformed_mediafile = MediaFileFactory()
        malformed_mediafile.observations.update(predicted_taxon_confidence=0.42)
        conflicting_mediafile = MediaFileFactory()
        models.AnimalObservation.objects.bulk_create(
            [models.AnimalObservation(mediafile=conflicting_mediafile)]
        )
        output = StringIO()

        with self.assertRaises(CommandError):
            call_command("audit_mediafile_observations", stdout=output)

        call_command("audit_mediafile_observations", "--repair", stdout=output)

        malformed = malformed_mediafile.observations.get()
        self.assertFalse(malformed.is_no_detection_placeholder)
        self.assertEqual(conflicting_mediafile.observations.count(), 1)
        self.assertFalse(conflicting_mediafile.observations.get().is_no_detection_placeholder)
        call_command("audit_mediafile_observations", stdout=output)

    def test_audit_preserves_data_on_placeholder_that_is_also_conflicting(self):
        mediafile = MediaFileFactory()
        placeholder = mediafile.observations.get()
        taxon = TaxonFactory()
        models.AnimalObservation.objects.filter(pk=placeholder.pk).update(
            taxon=taxon,
            bbox_x_center=0.4,
        )
        models.AnimalObservation.objects.bulk_create(
            [models.AnimalObservation(mediafile=mediafile)]
        )
        output = StringIO()

        call_command("audit_mediafile_observations", "--repair", stdout=output)

        placeholder.refresh_from_db()
        self.assertFalse(placeholder.is_no_detection_placeholder)
        self.assertEqual(placeholder.taxon_id, taxon.id)
        self.assertEqual(placeholder.bbox_x_center, 0.4)
        self.assertEqual(mediafile.observations.count(), 2)
        call_command("audit_mediafile_observations", stdout=output)
