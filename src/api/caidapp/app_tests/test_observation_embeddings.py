"""Regression tests for observation-level embedding versions and worker callbacks."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.test import TestCase

from caidapp.models import (
    AnimalObservation,
    EmbeddingProfile,
    MediafileIdentificationSuggestion,
    MediafilesForIdentification,
    ObservationEmbedding,
)
from caidapp.observation_embeddings import (
    embedding_generation,
    enqueue_embeddings,
    fingerprint,
    reset_workgroup_embeddings,
    snapshot,
    store_embeddings,
)
from caidapp.services.shared_identification import export_match_inputs, prepare_feature_manifest, prepare_identification

from .factories import IndividualIdentityFactory, MediaFileFactory


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

    def test_shared_features_reuse_by_bbox_across_query_and_reference_roles(self):
        """Store both feature types and choose the role at match time."""
        Path(self.mediafile.image_file.path).unlink()
        source = self.mediafile.mediafile.path
        for observation, x in ((self.first, 0.25), (self.second, 0.75)):
            observation.bbox_x_center = x
            observation.bbox_y_center = 0.5
            observation.bbox_width = observation.bbox_height = 0.2
            observation.save()
        rows = [snapshot(self.first), snapshot(self.second)]
        self.assertEqual([row["image_path"] for row in rows], [source, source])
        manifest = {
            "workgroup_id": self.workgroup.pk,
            "model_source": "test-model",
            "weights_path": "",
            "observations": rows,
            "query_ids": [self.first.pk],
            "reference_ids": [self.second.pk],
        }
        prepare_feature_manifest(manifest)
        self.assertNotEqual(rows[0]["feature_image"], rows[1]["feature_image"])
        path = Path(self.directory.name) / "input.json"
        output = path.with_name("output.json")
        payload = []
        for row in rows:
            image = Path(row["feature_image"])
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"fixture")
            payload.append({
                "observation_id": row["observation_id"],
                "input_fingerprint": row["input_fingerprint"],
                "feature_image": str(image),
                "vector": [0.1, 0.9],
                "local_features": {"keypoints": [[1, 2]], "descriptors": [[0.3, 0.5]], "image_size": [8, 8]},
            })
        path.write_text(json.dumps(manifest), encoding="utf-8")
        spec = {"model_source": "test-model", "dimension": 2, "feature_config": manifest["feature_config"]}
        output.write_text(json.dumps({"profile": spec, "embeddings": payload}), encoding="utf-8")
        result = store_embeddings({"output_path": str(output)}, path)
        self.assertEqual(result["stored"], 2)
        submitted = path.with_name("queries.csv")
        model = {"model_source": "test-model", "weights_path": ""}
        submitted.write_text(f"observation_id\n{self.first.pk}\n{self.second.pk}\n", encoding="utf-8")
        comparison = Path(prepare_identification(self.workgroup.pk, submitted, model, "compare_unknown"))
        export_match_inputs(comparison, result["profile_uuid"])
        for name in ("queries.jsonl", "references.jsonl"):
            samples = [json.loads(line) for line in comparison.with_name(name).read_text().splitlines()]
            self.assertEqual([sample["observation_id"] for sample in samples], [self.first.pk, self.second.pk])
            self.assertTrue(all(sample["embedding"][1] is None for sample in samples))

        self.second.identity = IndividualIdentityFactory(owner_workgroup=self.workgroup)
        self.second.identity_is_representative = True
        self.second.taxon = self.workgroup.default_taxon_for_identification
        self.second.save()
        submitted.write_text(f"observation_id\n{self.first.pk}\n", encoding="utf-8")
        match = Path(prepare_identification(
            self.workgroup.pk, submitted, model, "identify"
        ))
        export_match_inputs(match, result["profile_uuid"])
        reference = json.loads(match.with_name("references.jsonl").read_text(encoding="utf-8"))
        query = json.loads(match.with_name("queries.jsonl").read_text(encoding="utf-8"))
        self.assertEqual((query["observation_id"], reference["observation_id"]), (self.first.pk, self.second.pk))
        self.assertEqual((query["original_path"], reference["original_path"]), (source, source))
        self.assertEqual(reference["embedding"][1]["descriptors"], [[0.3, 0.5]])
        self.second.identity_is_representative = False
        self.second.save(update_fields=["identity_is_representative"])
        self.assertRaises(ValidationError, prepare_identification, self.workgroup.pk, submitted, model, "identify")
        expanded = prepare_identification(
            self.workgroup.pk, submitted, model, "identify", include_non_representative=True,
        )
        export_match_inputs(expanded, result["profile_uuid"])
        prepare_feature_manifest(manifest)
        self.assertCountEqual(manifest["cached"][fingerprint(spec)], [
            f"{row['observation_id']}:{row['input_fingerprint']}" for row in rows
        ])

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

    def test_changed_input_removes_features_and_suggestions_but_label_changes_do_not(self):
        """BBox and source changes invalidate inputs, while labels do not."""
        store_embeddings(*self.result([self.first]))
        original = snapshot(self.first)["input_fingerprint"]
        self.first.identity_is_representative = True
        self.first.save(update_fields=["identity_is_representative"])
        self.assertEqual(snapshot(self.first)["input_fingerprint"], original)
        self.assertEqual(self.first.embeddings.count(), 1)
        review = MediafilesForIdentification.objects.create(observation=self.second)
        MediafileIdentificationSuggestion.objects.create(for_identification=review, reference_observation=self.first)
        self.first.bbox_x_center = self.first.bbox_y_center = 0.5
        self.first.bbox_width = self.first.bbox_height = 0.2
        self.first.save()
        self.assertFalse(self.first.embeddings.exists())
        self.assertFalse(MediafilesForIdentification.objects.filter(pk=review.pk).exists())
        store_embeddings(*self.result([self.first]))
        self.assertEqual(self.first.embeddings.count(), 1)
        before = snapshot(self.first)["input_fingerprint"]
        with open(self.mediafile.image_file.path, "ab") as stream:
            stream.write(b"changed")
        self.assertNotEqual(snapshot(self.first)["input_fingerprint"], before)

    def test_reset_clears_only_its_workgroup_and_rejects_in_flight_results(self):
        """A reset removes saved features and pending reviews and rejects old output."""
        other_media = MediaFileFactory()
        other = AnimalObservation.objects.create(mediafile=other_media)
        own = store_embeddings(*self.result([self.first]))
        foreign_output, foreign_manifest = self.result([other])
        path = Path(foreign_manifest)
        payload = json.loads(path.read_text())
        payload["workgroup_id"] = other_media.parent.owner.workgroup_id
        path.write_text(json.dumps(payload))
        store_embeddings(foreign_output, foreign_manifest)
        self.assertEqual(own["stored"], 1)
        self.assertTrue(other.embeddings.exists())
        review = MediafilesForIdentification.objects.create(observation=self.first)
        crop = Path(self.directory.name) / "identification_feature_images" / "saved.png"
        crop.parent.mkdir()
        crop.touch()
        self.first.embeddings.update(
            local_features={"keypoints": [], "descriptors": [], "image_size": [1, 1]},
            feature_image=crop.relative_to(self.directory.name).as_posix(),
        )
        output, manifest = self.result([self.first])
        with self.captureOnCommitCallbacks(execute=True):
            self.assertEqual(reset_workgroup_embeddings(self.workgroup.pk), 1)
        self.assertEqual(store_embeddings(output, manifest)["stale"], 1)
        self.assertFalse(crop.exists())
        self.assertFalse(MediafilesForIdentification.objects.filter(pk=review.pk).exists())
        self.assertFalse(self.first.embeddings.exists())
        self.assertTrue(other.embeddings.exists())
        path = Path(manifest)
        payload = json.loads(path.read_text())
        payload["generation"] = embedding_generation(self.workgroup.pk)
        path.write_text(json.dumps(payload))
        self.assertEqual(store_embeddings(output, manifest)["stored"], 1)

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
        self.assertEqual(manifest["cached"], {})
        self.assertIn("feature_config", manifest)

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
