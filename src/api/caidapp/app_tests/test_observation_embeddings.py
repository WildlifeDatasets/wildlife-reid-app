"""Regression tests for observation-level embedding versions and worker callbacks."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.test import TestCase

from caidapp.models import AnimalObservation, EmbeddingProfile, ObservationEmbedding
from caidapp.observation_embeddings import enqueue_embeddings, fingerprint, snapshot, store_embeddings

from .factories import MediaFileFactory


class ObservationEmbeddingTests(TestCase):
    def setUp(self):
        """Use isolated media files and two observations sharing the same media file."""
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        override = self.settings(MEDIA_ROOT=self.directory.name)
        override.enable()
        self.addCleanup(override.disable)
        self.mediafile = MediaFileFactory()
        self.workgroup = self.mediafile.parent.owner.workgroup
        self.first = self.mediafile.observations.first()
        self.first.is_no_detection_placeholder = False
        self.first.save()
        self.second = AnimalObservation.objects.create(mediafile=self.mediafile)
        self.spec = {"dimension": 2, "model_source": "test-model", "model_state_sha256": "weights-v1"}

    def result(self, observations, spec=None):
        """Construct a worker result from exactly the submitted input snapshots."""
        inputs = [snapshot(observation) for observation in observations]
        manifest = {
            "workgroup_id": self.workgroup.pk,
            "model_source": "test-model",
            "observations": inputs,
        }
        path = Path(self.directory.name) / "input.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        output_path = path.with_name("output.json")
        output_path.write_text(
            json.dumps(
                {
                    "profile": spec or self.spec,
                    "embeddings": [
                        {
                            "observation_id": row["observation_id"],
                            "input_fingerprint": row["input_fingerprint"],
                            "vector": [0.1, 0.9],
                        }
                        for row in inputs
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {"output_path": str(output_path)}, str(path)

    def test_multiple_observations_and_idempotent_callback(self):
        """Keep observations distinct and tolerate repeated Celery delivery."""
        output, manifest = self.result([self.first, self.second])
        self.assertEqual(store_embeddings(output, manifest)["stored"], 2)
        self.assertEqual(store_embeddings(output, manifest)["stored"], 0)
        self.assertEqual(ObservationEmbedding.objects.count(), 2)
        existing = ObservationEmbedding.objects.first()
        with self.assertRaises(IntegrityError), transaction.atomic():
            ObservationEmbedding.objects.create(
                observation=existing.observation,
                profile=existing.profile,
                input_fingerprint=existing.input_fingerprint,
                input_specification=existing.input_specification,
                vector=existing.vector,
            )

    def test_profiles_coexist_and_cannot_be_edited(self):
        """Changing encoder weights creates a separate space and preserves old vectors."""
        store_embeddings(*self.result([self.first]))
        new_spec = {**self.spec, "model_state_sha256": "weights-v2"}
        store_embeddings(*self.result([self.first], new_spec))
        self.assertEqual(self.first.embeddings.count(), 2)
        profile = EmbeddingProfile.objects.first()
        profile.specification = new_spec
        with self.assertRaises(ValidationError):
            profile.save()

    def test_bbox_change_during_computation_discards_result(self):
        """Do not apply a vector computed for an obsolete crop."""
        output, manifest = self.result([self.first])
        AnimalObservation.objects.filter(pk=self.first.pk).update(
            bbox_x_center=0.5, bbox_y_center=0.5, bbox_width=0.2, bbox_height=0.2
        )
        self.assertEqual(store_embeddings(output, manifest)["stale"], 1)
        self.assertFalse(ObservationEmbedding.objects.exists())

    def test_input_versions_retain_history_without_identity_invalidation(self):
        """BBox and source changes invalidate inputs, while labels do not."""
        store_embeddings(*self.result([self.first]))
        original = snapshot(self.first)["input_fingerprint"]
        self.first.identity_is_representative = True
        self.assertEqual(snapshot(self.first)["input_fingerprint"], original)
        self.first.bbox_x_center = self.first.bbox_y_center = 0.5
        self.first.bbox_width = self.first.bbox_height = 0.2
        self.first.save()
        store_embeddings(*self.result([self.first]))
        self.assertEqual(self.first.embeddings.count(), 2)
        before = snapshot(self.first)["input_fingerprint"]
        with open(self.mediafile.image_file.path, "ab") as stream:
            stream.write(b"changed")
        self.assertNotEqual(snapshot(self.first)["input_fingerprint"], before)

    def test_placeholder_and_video_are_rejected(self):
        """Avoid treating an absent detection or an unlocated video bbox as an object crop."""
        self.first.is_no_detection_placeholder = True
        with self.assertRaises(ValidationError):
            snapshot(self.first)
        self.first.is_no_detection_placeholder = False
        self.mediafile.media_type = "video"
        self.first.mediafile = self.mediafile
        with self.assertRaises(ValidationError):
            snapshot(self.first)

    def test_enqueue_filters_placeholders_and_scopes_workgroup(self):
        """Send unlabelled observations with cache keys, never another workgroup's data."""
        store_embeddings(*self.result([self.first]))
        foreign = MediaFileFactory().observations.first()
        with self.assertRaises(ValidationError):
            enqueue_embeddings(self.workgroup.pk, [foreign.pk])
        with patch("caidapp.observation_embeddings.signature") as task:
            enqueue_embeddings(self.workgroup.pk)
        manifest_path = task.call_args_list[0].kwargs["kwargs"]["manifest_path"]
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.assertEqual({row["observation_id"] for row in manifest["observations"]}, {self.first.pk, self.second.pk})
        self.assertIn(fingerprint(self.spec), manifest["cached"])

    def test_invalid_vectors_roll_back_callback(self):
        """Reject malformed or non-finite vectors before persisting any results."""
        for vector in ([1.0], [float("nan"), 1.0], [0.0, 0.0], [True, 1.0]):
            output, manifest = self.result([self.first])
            path = Path(output["output_path"])
            data = json.loads(path.read_text(encoding="utf-8"))
            data["embeddings"][0]["vector"] = vector
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValidationError):
                store_embeddings(output, manifest)
            self.assertFalse(ObservationEmbedding.objects.exists())

    def test_deleted_observation_is_discarded(self):
        """A late callback cannot recreate an observation deleted during inference."""
        output, manifest = self.result([self.first])
        self.first.delete()
        self.assertEqual(store_embeddings(output, manifest)["stale"], 1)
