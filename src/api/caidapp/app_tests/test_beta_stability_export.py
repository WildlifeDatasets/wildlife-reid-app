"""Verify the portable dataset format, exact selections and access checks."""

import json
import pickle
import zipfile
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from caidapp.beta_stability_export import BetaExportError, export_beta_stability
from caidapp.models import AnimalObservation, EmbeddingProfile, ObservationEmbedding
from caidapp.observation_embeddings import snapshot
from caidapp.tasks import create_beta_stability_export
from caidapp.views_beta_stability import BETA_STABILITY_SELECTION_KEY
from django.core.exceptions import ValidationError
from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse
from PIL import Image

from .factories import MediaFileFactory


class BetaStabilityExportTests(TestCase):
    def setUp(self):
        """Create two observations in one image and two independent embedding spaces."""
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        override = self.settings(MEDIA_ROOT=self.directory.name, PRIVATE_DATA_PATH=self.directory.name)
        override.enable()
        self.addCleanup(override.disable)
        self.mediafile = MediaFileFactory()
        self.user = self.mediafile.parent.owner
        self.first = self.mediafile.observations.first()
        self.first.bbox_x_center = self.first.bbox_y_center = 0.5
        self.first.bbox_width = self.first.bbox_height = 0.4
        self.first.save()
        self.second = AnimalObservation.objects.create(mediafile=self.mediafile)
        self.profiles = [
            EmbeddingProfile.objects.create(dimension=2, specification={"dimension": 2, "model_source": name})
            for name in ("test-model-a", "test-model-b")
        ]
        for observation in (self.first, self.second):
            for index, profile in enumerate(self.profiles):
                current = snapshot(observation)
                ObservationEmbedding.objects.create(
                    observation=observation,
                    profile=profile,
                    input_fingerprint=current["input_fingerprint"],
                    input_specification=current["input_specification"],
                    vector=[float(index + 1), float(observation.pk)],
                )
        self.queryset = AnimalObservation.objects.filter(pk__in=[self.first.pk, self.second.pk])
        self.output = Path(self.directory.name) / "export.zip"

    def export(self, queryset=None, **kwargs):
        """Write a package with the production service."""
        return export_beta_stability(
            self.queryset if queryset is None else queryset, self.profiles, self.output, **kwargs
        )

    def login(self):
        """Select the observations through the existing web download session."""
        self.client.force_login(self.user.user)
        session = self.client.session
        session[BETA_STABILITY_SELECTION_KEY] = [self.first.pk, self.second.pk]
        session.save()

    def test_zip_contains_aligned_profiles_pixel_boxes_and_one_copy_of_source(self):
        """Keep multiple observations per image and align each matrix with public UUIDs."""
        manifest = self.export(include_images=True)
        with zipfile.ZipFile(self.output) as archive:
            annotations = json.loads(archive.read("annotations.json"))
            self.assertEqual(len(annotations["images"]), 1)
            self.assertEqual(len(annotations["annotations"]), 2)
            image = annotations["images"][0]
            self.assertEqual(len([name for name in archive.namelist() if name.startswith("images/")]), 1)
            self.assertEqual(archive.read(image["file_name"]), Path(self.mediafile.image_file.path).read_bytes())
            with Image.open(self.mediafile.image_file.path) as source:
                left, top, right, bottom = self.first.get_bbox_xyxy(*source.size)
            self.assertEqual(annotations["annotations"][0]["bbox"], [left, top, right - left, bottom - top])
            for index, profile in enumerate(manifest["profiles"]):
                matrix, ids = pickle.loads(archive.read(profile["file"]))
                self.assertEqual(ids, [str(self.first.uuid), str(self.second.uuid)])
                self.assertEqual(matrix.dtype, np.dtype("float32"))
                self.assertEqual(matrix.shape, (2, 2))
                np.testing.assert_array_equal(matrix[:, 0], [index + 1, index + 1])
            config = json.loads(archive.read("config.yaml"))
            self.assertIsNone(config["data"]["name_keys"])
            self.assertIn("no_human", config["edge_weights"]["verifier_names"])
            self.assertNotIn("identity", annotations["annotations"][0])
            self.assertNotIn("individuals", annotations)
            self.assertTrue(all(not name.startswith("/") and ".." not in name for name in archive.namelist()))

    def test_embedding_only_export_preserves_image_metadata_without_payload(self):
        """Internal exports can be clustered without copying images."""
        self.export()
        with zipfile.ZipFile(self.output) as archive:
            self.assertFalse(any(name.startswith("images/") for name in archive.namelist()))
            self.assertEqual(len(json.loads(archive.read("annotations.json"))["images"]), 1)

    def test_default_export_rejects_stale_embeddings(self):
        """A changed crop must never silently reuse a historical vector."""
        AnimalObservation.objects.filter(pk=self.first.pk).update(bbox_width=0.2)
        with self.assertRaises(BetaExportError) as caught:
            self.export()
        self.assertEqual(caught.exception.report["skipped"][0]["reason"], "missing_current_embedding")
        self.assertFalse(self.output.exists())

    def test_partial_export_uses_intersection_and_records_missing_profiles(self):
        """Opt-in partial export applies identical row selection to every matrix."""
        ObservationEmbedding.objects.filter(observation=self.first, profile=self.profiles[1]).delete()
        manifest = self.export(allow_partial=True)
        self.assertEqual(manifest["observation_uuids"], [str(self.second.uuid)])
        self.assertEqual(manifest["report"]["skipped"][0]["profile_uuids"], [str(self.profiles[1].uuid)])
        with zipfile.ZipFile(self.output) as archive:
            for profile in manifest["profiles"]:
                matrix, ids = pickle.loads(archive.read(profile["file"]))
                self.assertEqual(ids, [str(self.second.uuid)])
                self.assertEqual(matrix.shape, (1, 2))

    def test_video_placeholder_and_unfinished_upload_are_reported(self):
        """Skip unlocated video observations and non-object placeholder rows explicitly."""
        video = MediaFileFactory(parent=self.mediafile.parent, media_type="video")
        video_observation = AnimalObservation.objects.create(mediafile=video)
        placeholder = MediaFileFactory(parent=self.mediafile.parent).observations.first()
        unfinished = MediaFileFactory().observations.first()
        unfinished.is_no_detection_placeholder = False
        unfinished.save()
        unfinished.mediafile.parent.import_finished = False
        unfinished.mediafile.parent.save(update_fields=["import_finished"])
        queryset = AnimalObservation.objects.filter(
            pk__in=[self.first.pk, video_observation.pk, placeholder.pk, unfinished.pk]
        )
        report = self.export(queryset)["report"]
        self.assertEqual(report["exported"], 1)
        self.assertEqual(
            {row["reason"] for row in report["skipped"]},
            {
                "video_frame_not_located",
                "no_detection_placeholder",
                "unfinished_upload",
            },
        )

    def test_invalid_stored_vector_is_rejected(self):
        """A malformed database vector must not create a usable-looking artifact."""
        ObservationEmbedding.objects.filter(observation=self.first).update(vector=[0, 0])
        with self.assertRaises(ValidationError):
            self.export()
        self.assertFalse(self.output.exists())

    def test_concurrent_crop_change_fails_without_publishing_partial_zip(self):
        """Revalidate inputs after packaging, keeping failed exports unpublished."""
        calls = 0

        def changing_snapshot(observation):
            nonlocal calls
            calls += 1
            if calls == 3:
                AnimalObservation.objects.filter(pk=self.first.pk).update(bbox_width=0.2)
                observation.refresh_from_db()
            return snapshot(observation)

        with patch("caidapp.beta_stability_export.snapshot", side_effect=changing_snapshot):
            with self.assertRaises(ValidationError):
                self.export(include_images=True)
        self.assertFalse(self.output.exists())

    def test_selection_never_expands_to_sibling_observations(self):
        """Selecting one observation in an image exports exactly one annotation."""
        manifest = self.export(AnimalObservation.objects.filter(pk=self.second.pk))
        self.assertEqual(manifest["observation_uuids"], [str(self.second.uuid)])

    def test_command_runs_same_export(self):
        """Administrative backfills can create the same portable package."""
        output = StringIO()
        call_command(
            "export_beta_stability",
            workgroup=self.user.workgroup_id,
            profile=[str(self.profiles[0].uuid)],
            output=str(self.output),
            stdout=output,
        )
        self.assertIn("Exported 2 observations", output.getvalue())

    def test_web_configuration_and_enqueue_use_exact_selection(self):
        """Expose profile choice in the web workflow and queue the saved observation IDs."""
        self.login()
        response = self.client.get(reverse("caidapp:beta_stability_export"))
        self.assertContains(response, str(self.profiles[0].uuid))
        with patch("caidapp.export_runs.signature") as publish:
            with self.captureOnCommitCallbacks(execute=True):
                response = self.client.get(
                    reverse("caidapp:beta_stability_export"),
                    {"download": "1", "profile": str(self.profiles[0].uuid), "include_images": "1"},
                )
        run_id, task_name, args, options = publish.call_args.kwargs["args"]
        self.assertEqual(task_name, "create_beta_stability_export")
        self.assertEqual(args[1], [self.first.pk, self.second.pk])
        self.assertTrue(options["include_images"])
        self.assertIn("detail_url", response.json())
        self.assertTrue(response.json()["task_id"])

    def test_tracked_beta_export_writes_a_private_downloadable_artifact(self):
        """The unchanged beta encoder export also supports the new protected destination."""
        from caidapp.export_runs import execute_export
        from caidapp.models import TaskRun

        self.login()
        with patch("caidapp.export_runs.signature") as publish:
            with self.captureOnCommitCallbacks(execute=True):
                response = self.client.get(
                    reverse("caidapp:beta_stability_export"),
                    {
                        "download": "1",
                        "profile": str(self.profiles[0].uuid),
                    },
                )
        self.assertEqual(response.status_code, 200)
        execute_export(*publish.call_args.kwargs["args"])
        run = TaskRun.objects.get(kind="export")
        self.assertEqual(run.status, "succeeded")
        self.assertIn("export_downloads", run.artifact.path)
        with zipfile.ZipFile(run.artifact.path) as archive:
            self.assertEqual(
                json.loads(archive.read("manifest.json"))["observation_uuids"],
                [str(self.first.uuid), str(self.second.uuid)],
            )
        download = self.client.get(reverse("caidapp:task_download", kwargs={"run_uuid": run.uuid}))
        self.assertEqual(download.status_code, 200)
        download.close()

    def test_web_rejects_foreign_observations_and_unavailable_profiles(self):
        """Session IDs and profile UUIDs must not bypass workgroup access checks."""
        self.login()
        foreign = MediaFileFactory().observations.first()
        session = self.client.session
        session[BETA_STABILITY_SELECTION_KEY] = [foreign.pk]
        session.save()
        response = self.client.get(
            reverse("caidapp:beta_stability_export"), {"download": "1", "profile": str(self.profiles[0].uuid)}
        )
        self.assertEqual(response.status_code, 400)
        self.login()
        unreferenced = EmbeddingProfile.objects.create(
            dimension=2, specification={"dimension": 2, "model_source": "foreign"}
        )
        response = self.client.get(
            reverse("caidapp:beta_stability_export"), {"download": "1", "profile": str(unreferenced.uuid)}
        )
        self.assertEqual(response.status_code, 400)

    def test_background_task_rechecks_permission_and_destination(self):
        """A queued job cannot export inputs that the requesting user no longer owns."""
        with self.assertRaises(ValidationError):
            create_beta_stability_export(self.user.pk, [self.first.pk], [str(self.profiles[0].uuid)], str(self.output))
        foreign = MediaFileFactory().observations.first()
        destination = Path(self.directory.name) / "users" / self.user.hash / "beta_stability.zip"
        with self.assertRaises(ValidationError):
            create_beta_stability_export(self.user.pk, [foreign.pk], [str(self.profiles[0].uuid)], str(destination))
        result = create_beta_stability_export(
            self.user.pk, [self.first.pk], [str(self.profiles[0].uuid)], str(destination)
        )
        self.assertEqual(result, str(destination))
        with zipfile.ZipFile(destination) as archive:
            self.assertEqual(json.loads(archive.read("manifest.json"))["observation_uuids"], [str(self.first.uuid)])

    def test_filter_capture_keeps_observation_scope(self):
        """Capturing a filtered download scope must not add other objects from the same image."""
        self.login()
        self.client.post(
            reverse("caidapp:prepare_observation_download"), {"selected_observation_ids": [str(self.first.pk)]}
        )
        self.assertEqual(self.client.session[BETA_STABILITY_SELECTION_KEY], [self.first.pk])

    def test_filter_without_explicit_selection_keeps_observation_scope(self):
        """A filtered export must not expand to every observation in matching media files."""
        self.login()
        self.first.orientation = "L"
        self.first.save(update_fields=["orientation"])
        self.client.get(reverse("caidapp:prepare_observation_download"), {"orientation": "L"})
        self.assertEqual(self.client.session[BETA_STABILITY_SELECTION_KEY], [self.first.pk])

    def test_existing_destination_is_preserved(self):
        """A second export must not replace a previously created artifact."""
        self.output.write_bytes(b"previous export")
        with self.assertRaises(ValidationError):
            self.export()
        self.assertEqual(self.output.read_bytes(), b"previous export")

    def test_empty_export_is_rejected_with_report(self):
        """Do not hand beta-stability an empty dataset that its loader cannot use."""
        with self.assertRaises(BetaExportError) as caught:
            self.export(AnimalObservation.objects.none())
        self.assertEqual(caught.exception.report["exported"], 0)
        self.assertFalse(self.output.exists())

    def test_export_error_survives_worker_serialization(self):
        """Failed background jobs must retain a readable message and skipped-input report."""
        error = BetaExportError("Missing current embeddings", {"exported": 0, "skipped": []})
        restored = pickle.loads(pickle.dumps(error))
        self.assertEqual(restored.messages, error.messages)
        self.assertEqual(restored.report, error.report)

    def test_observation_uuid_is_unique_and_stable_after_save(self):
        """Public IDs survive observation edits and distinguish observations sharing one file."""
        original = self.first.uuid
        self.first.orientation = "L"
        self.first.save()
        self.first.refresh_from_db()
        self.assertEqual(self.first.uuid, original)
        self.assertNotEqual(self.first.uuid, self.second.uuid)
