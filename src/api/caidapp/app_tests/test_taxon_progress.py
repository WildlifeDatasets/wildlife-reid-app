from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from zipfile import ZipFile, ZipInfo

import pandas as pd
from caidapp import tasks
from caidapp.app_tests.factories import CaidUserFactory, UploadedArchiveFactory
from django.template.loader import render_to_string
from django.test import TestCase
from django.urls import reverse


class TaxonProgressApiTest(TestCase):
    def setUp(self):
        """Prepare an authenticated archive owner and a scheduled taxon task for progress queries."""
        self.caiduser = CaidUserFactory()
        self.client.force_login(self.caiduser.user)
        self.archive = UploadedArchiveFactory(
            owner=self.caiduser,
            taxon_status="TAIP",
            taxon_task_id="taxon-task-1",
        )
        self.url = reverse("caidapp:uploads_status_api", args=["species"])

    @patch("caidapp.views.AsyncResult")
    def test_returns_worker_progress(self, async_result):
        """Return the taxon worker's detection stage and 54 percent progress."""
        async_result.return_value = Mock(
            state="PROGRESS",
            info={"percent": 54, "stage": "detection", "message": "Detecting animals"},
        )

        response = self.client.get(self.url, {"ids": str(self.archive.id)})

        self.assertEqual(response.status_code, 200)
        progress = response.json()["archives"][0]["progress"]
        self.assertEqual(progress["percent"], 54)
        self.assertEqual(progress["stage"], "detection")
        async_result.assert_called_once_with("taxon-task-1")

    @patch("caidapp.views.AsyncResult")
    def test_successful_worker_waiting_for_import_reports_99_percent(self, async_result):
        """Report 99 percent while a successful worker result is still awaiting import."""
        async_result.return_value = Mock(state="SUCCESS", info={"status": "DONE"})

        response = self.client.get(self.url, {"ids": str(self.archive.id)})

        progress = response.json()["archives"][0]["progress"]
        self.assertEqual(progress["percent"], 99)
        self.assertEqual(progress["stage"], "import_results")

    @patch("caidapp.views.AsyncResult", side_effect=RuntimeError("Redis unavailable"))
    def test_backend_error_keeps_status_endpoint_available(self, async_result):
        """Keep the status response available with null progress when the result backend fails."""
        response = self.client.get(self.url, {"ids": str(self.archive.id)})

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["archives"][0]["progress"])

    @patch("caidapp.views.AsyncResult")
    def test_requested_ids_limit_backend_lookups(self, async_result):
        """Look up task results only for archive IDs requested by the client."""
        other = UploadedArchiveFactory(
            owner=self.caiduser,
            taxon_status="TAIP",
            taxon_task_id="taxon-task-2",
        )
        async_result.return_value = Mock(state="PENDING", info=None)

        response = self.client.get(self.url, {"ids": str(self.archive.id)})

        self.assertEqual([item["id"] for item in response.json()["archives"]], [self.archive.id])
        async_result.assert_called_once_with("taxon-task-1")
        self.assertNotEqual(other.id, self.archive.id)


class TaxonTaskDispatchTest(TestCase):
    @patch("caidapp.tasks.update_metadata_csv_by_uploaded_archive")
    @patch("caidapp.tasks._run_taxon_classification_init_message")
    @patch("caidapp.tasks.signature")
    def test_dispatch_stores_worker_task_id(
        self,
        signature_mock,
        init_message,
        update_metadata,
    ):
        """Persist the preparation ID and dispatch the ZIP to preparation only."""
        archive = UploadedArchiveFactory()

        tasks.run_species_prediction_async(archive)

        archive.refresh_from_db()
        dispatch = signature_mock.return_value.apply_async.call_args.kwargs
        self.assertEqual(archive.taxon_task_id, dispatch["task_id"])
        self.assertEqual(signature_mock.call_args.args, ("prepare_taxon_archive",))

    @patch("caidapp.tasks.signature")
    @patch("caidapp.tasks.make_thumbnail_for_uploaded_archive")
    def test_prepared_archive_registers_media_before_inference_and_applies_result_by_id(self, thumbnail, signature):
        """Exercise the API stages of the archive handoff with a prepared image manifest."""
        with TemporaryDirectory() as root, self.settings(MEDIA_ROOT=Path(root) / "media"):
            media_root = Path(root) / "media"
            output_dir = media_root / "output"
            output_dir.mkdir(parents=True)
            with ZipFile(media_root / "source.zip", "w") as source:
                source.writestr(ZipInfo("camera/a.jpg", date_time=(2020, 3, 4, 9, 0, 0)), b"original JPG")
            archive = UploadedArchiveFactory(
                outputdir=str(output_dir), taxon_status="TAIP", taxon_task_id="prepare",
                import_finished=False, archivefile="source.zip",
            )
            manifest_path = output_dir / "prepared.csv"
            metadata_file = output_dir / "metadata.csv"
            prepared_image = output_dir / "images" / "a.webp"
            pd.DataFrame([{
                "image_path": "a.webp",
                "absolute_media_path": str(prepared_image),
                "datetime": "2026-11-19T14:32:08Z",
                "datetime_source": "File system",
                "media_type": "image",
                "original_path": "camera/a.jpg",
            }]).to_csv(manifest_path)
            tasks.register_prepared_taxon_archive(
                archive.pk, "prepare", str(manifest_path), str(output_dir / "images.zip"), str(metadata_file), False,
            )

            media = archive.mediafile_set.get()
            placeholder_id = media.observations.get().pk
            self.assertTrue(media.observations.get().is_no_detection_placeholder)
            prepared = pd.read_csv(manifest_path, index_col=0)
            self.assertEqual(prepared["media_id"].tolist(), [media.pk])
            self.assertEqual(prepared["datetime"].tolist(), ["2020-03-04T09:00:00"])
            self.assertIn("assumed", prepared.loc[0, "datetime_source"])
            original = Path(root) / "media_originals" / media.original_file
            self.assertEqual(original.read_bytes(), b"original JPG")
            self.assertEqual(signature.call_args.args, ("predict_prepared_media",))
            self.assertNotIn("input_archive_file", signature.call_args.kwargs["kwargs"]["archive"])

            archive.refresh_from_db()
            prepared["predicted_category"] = "Canis lupus"
            prepared["detection_results"] = repr([
                {"bbox": [1, 1, 7, 7], "size": [8, 8], "orientation": "unknown", "orientation_score": 0},
            ])
            prepared.to_csv(metadata_file)
            with patch("caidapp.models.MediaFile.make_thumbnail_for_mediafile_if_necessary"):
                tasks.on_success_predict_taxon(
                    {"status": "DONE"}, uploaded_archive_id=archive.pk, zip_file="output/images.zip",
                    csv_file="output/metadata.csv", expected_taxon_task_id=archive.taxon_task_id,
                )
            archive.refresh_from_db()
            self.assertEqual(archive.taxon_status, "TAID", archive.status_message)
            self.assertEqual(archive.mediafile_set.count(), 1)
            media.refresh_from_db()
            self.assertEqual(media.captured_at.year, 2020)
            self.assertEqual(media.metadata_json["original_file_times"]["modified"]["source"], "zip_dos")
            observation = media.observations.get()
            self.assertEqual(observation.pk, placeholder_id)
            self.assertEqual(observation.taxon.name, "Canis lupus")


class TaxonProgressTemplateTest(TestCase):
    def test_processing_archive_renders_progress_indicator(self):
        """Render a visible progress ring associated with the processing archive ID."""
        archive = UploadedArchiveFactory(taxon_status="TAIP", status_message="Processing upload")

        html = render_to_string("caidapp/upload_progress.html", {"uploadedarchive": archive})

        self.assertIn(f'id="progress-{archive.id}"', html)
        self.assertIn(f'data-archive-id="{archive.id}"', html)
        self.assertIn('class="upload-progress-ring"', html)
        self.assertIn('class="upload-progress-value"', html)
        self.assertNotIn(" d-none", html)


class ResetTaxonProcessingTest(TestCase):
    def test_staff_can_reset_only_the_requested_run(self):
        """Protect the action, mark the worker task and ignore its subsequent callbacks."""
        with TemporaryDirectory() as root, self.settings(MEDIA_ROOT=root):
            owner = CaidUserFactory()
            archive = UploadedArchiveFactory(
                owner=owner, outputdir=str(Path(root) / "output"), taxon_status="TAIP", taxon_task_id="active",
            )
            url = reverse("caidapp:reset_taxon_processing", args=[archive.pk])
            self.client.force_login(owner.user)
            self.assertEqual(self.client.post(url, {"task_id": "active"}).status_code, 403)
            owner.user.is_staff = True
            owner.user.save(update_fields=["is_staff"])
            self.assertEqual(self.client.get(url).status_code, 405)
            self.assertEqual(self.client.post(url, {"task_id": "stale"}).status_code, 302)
            archive.refresh_from_db()
            self.assertEqual(archive.taxon_task_id, "active")

            with patch("caidapp.tasks.current_app.control.revoke") as revoke:
                with self.captureOnCommitCallbacks(execute=True):
                    self.assertEqual(self.client.post(url, {"task_id": "active"}).status_code, 302)
                revoke.assert_called_once_with("active")
            archive.refresh_from_db()
            marker = Path(archive.outputdir) / f".taxon-cancelled-{sha256(b'active').hexdigest()}"
            self.assertTrue(marker.is_file())
            self.assertEqual(archive.taxon_status, "F")
            self.assertEqual(archive.taxon_task_id, "")
            self.assertTrue(archive.import_finished)

            archive.taxon_status = "TAIP"
            archive.taxon_task_id = "replacement"
            archive.save(update_fields=["taxon_status", "taxon_task_id"])
            tasks.on_success_predict_taxon(
                {"status": "DONE"}, uploaded_archive_id=archive.pk, zip_file="", csv_file="",
                expected_taxon_task_id="active",
            )
            archive.refresh_from_db()
            self.assertEqual(archive.taxon_status, "TAIP")
            self.assertEqual(archive.taxon_task_id, "replacement")
