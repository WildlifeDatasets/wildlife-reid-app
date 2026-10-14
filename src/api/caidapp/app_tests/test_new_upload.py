import json
import os
import shutil
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from caidapp import models
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings, tag
from django.urls import reverse

from .factories import CaidUserFactory

WRAP_TEST_DATA_DIR = os.getenv("WRAP_TEST_DATA_DIR")


class NewUploadViewTest(TestCase):
    def setUp(self):
        """Prepare temporary media storage and an authenticated user for upload requests."""
        self.media_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.media_root)
        self.override = override_settings(MEDIA_ROOT=self.media_root)
        self.override.enable()
        self.addCleanup(self.override.disable)

        self.caiduser = CaidUserFactory(admin=True)
        self.user = self.caiduser.user
        self.client.login(username=self.user.username, password="test123")

    def _set_capabilities(
        self,
        *,
        show_taxon_classification=True,
        show_reid=True,
        show_base_dataset=False,
        show_base_between_regular_uploads=False,
        workgroup_admin=True,
        is_staff=None,
    ):
        self.caiduser.show_taxon_classification = show_taxon_classification
        self.caiduser.show_reid = show_reid
        self.caiduser.show_base_dataset = show_base_dataset
        self.caiduser.show_base_between_regular_uploads = show_base_between_regular_uploads
        self.caiduser.workgroup_admin = workgroup_admin
        self.caiduser.save()
        if is_staff is not None:
            self.user.is_staff = is_staff
            self.user.save()

    def _post_upload(self, extra_data=None, files=None):
        data = {
            "locality_at_upload": "Brdy",
            "upload_target": "taxon_processing",
            "directory_structure": "",
            "ml_consent": "on",
            "upload_files": files
            or [
                SimpleUploadedFile("first.jpg", b"fake image 1", content_type="image/jpeg"),
                SimpleUploadedFile("second.jpg", b"fake image 2", content_type="image/jpeg"),
            ],
        }
        if extra_data:
            data.update(extra_data)
        return self.client.post(reverse("caidapp:new_upload"), data)

    def _make_xlsx_file(self, filename, records):
        buffer = BytesIO()
        pd.DataFrame(records).to_excel(buffer, index=False)
        buffer.seek(0)
        return SimpleUploadedFile(
            filename,
            buffer.read(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    def test_get_requires_upload_capability(self):
        """Deny upload form access to users without an enabled upload workflow."""
        regular_caiduser = CaidUserFactory()
        regular_caiduser.workgroup_admin = False
        regular_caiduser.show_taxon_classification = False
        regular_caiduser.show_reid = False
        regular_caiduser.save()
        regular_caiduser.user.is_staff = False
        regular_caiduser.user.save()
        self.client.logout()
        self.client.force_login(regular_caiduser.user)

        response = self.client.get(reverse("caidapp:new_upload"))

        self.assertEqual(response.status_code, 403)

    def test_get_renders_for_workgroup_admin(self):
        """Show workflow routing and representative-dataset choices to a workgroup admin."""
        self._set_capabilities(show_taxon_classification=True, show_reid=True, show_base_dataset=True)

        response = self.client.get(reverse("caidapp:new_upload"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Upload")
        self.assertNotContains(response, "New Upload")
        self.assertNotContains(response, "This is a beta upload flow")
        self.assertContains(response, "Where should this upload go?")
        self.assertContains(response, "the upload contains identified individuals and the images are representative")

    def test_taxon_only_user_does_not_see_processing_choices(self):
        """Hide workflow and base-dataset choices when only taxon processing is available."""
        self._set_capabilities(show_taxon_classification=True, show_reid=False)

        response = self.client.get(reverse("caidapp:new_upload"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Where should this upload go?")
        self.assertNotContains(response, "base dataset")

    def test_reid_only_non_admin_does_not_see_base_dataset_choice(self):
        """Hide workflow and base-dataset choices from a regular ReID-only member."""
        self._set_capabilities(show_taxon_classification=False, show_reid=True, workgroup_admin=False)

        response = self.client.get(reverse("caidapp:new_upload"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Where should this upload go?")
        self.assertNotContains(response, "base dataset")

    def test_reid_only_admin_with_dataset_access_sees_base_dataset_choice(self):
        """Offer a base-dataset choice to a ReID-only admin with dataset access."""
        self._set_capabilities(show_taxon_classification=False, show_reid=True, show_base_dataset=True)

        response = self.client.get(reverse("caidapp:new_upload"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Where should this upload go?")
        self.assertContains(response, "base dataset")

    def test_reid_only_admin_with_base_between_regular_uploads_sees_base_dataset_choice(self):
        """Offer the base-dataset choice even when the existing base appears among regular uploads."""
        self._set_capabilities(
            show_taxon_classification=False,
            show_reid=True,
            show_base_between_regular_uploads=True,
        )

        response = self.client.get(reverse("caidapp:new_upload"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Where should this upload go?")
        self.assertContains(response, "base dataset")

    def test_get_accepts_identification_query_defaults(self):
        """Initialize the identification target and representative-data flag from query parameters."""
        self._set_capabilities(show_taxon_classification=True, show_reid=True, show_base_dataset=True)

        response = self.client.get(
            reverse("caidapp:new_upload"),
            {"upload_target": "identification", "contains_identities": "1"},
        )

        self.assertEqual(response.status_code, 200)
        form = response.context["form"]
        self.assertEqual(form["upload_target"].value(), "identification")
        self.assertEqual(form["contains_identities"].value(), True)

    def test_metadata_section_is_present_before_files_are_selected(self):
        """Keep the metadata section visible before the user selects upload files."""
        response = self.client.get(reverse("caidapp:new_upload"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'id="new-upload-metadata-content"')
        self.assertNotContains(response, 'id="new-upload-metadata-content" class="d-none"')

    @patch("caidapp.views.run_species_prediction_async")
    def test_multiple_media_files_are_packed_into_zip(self, run_processing):
        """Bundle two images into a timestamp-named ZIP and dispatch processing once."""
        response = self._post_upload()

        self.assertEqual(response.status_code, 200)
        uploaded_archive = models.UploadedArchive.objects.get()
        self.assertTrue(uploaded_archive.archivefile.name.endswith(".zip"))
        self.assertRegex(uploaded_archive.name, r"^upload_\d{8}-\d{6}$")
        with zipfile.ZipFile(uploaded_archive.archivefile.path) as archive:
            self.assertEqual(sorted(archive.namelist()), ["first.jpg", "second.jpg"])
        self.assertEqual(uploaded_archive.files_at_upload, 2)
        run_processing.assert_called_once_with(uploaded_archive, extract_identites=False)

    @patch("caidapp.views.run_species_prediction_async")
    def test_more_than_django_default_file_count_can_be_uploaded(self, run_processing):
        """Accept 101 individual files when the configured limit exceeds Django's default."""
        files = [
            SimpleUploadedFile(f"image_{index:03d}.jpg", b"fake image", content_type="image/jpeg")
            for index in range(101)
        ]

        response = self._post_upload(files=files)

        self.assertEqual(response.status_code, 200)
        uploaded_archive = models.UploadedArchive.objects.get()
        self.assertEqual(uploaded_archive.files_at_upload, 101)
        run_processing.assert_called_once_with(uploaded_archive, extract_identites=False)

    @override_settings(DATA_UPLOAD_MAX_NUMBER_FILES=100)
    @patch("caidapp.views.run_species_prediction_async")
    def test_too_many_individual_files_returns_zip_guidance(self, run_processing):
        """Reject excessive file counts with ZIP guidance before creating an archive or dispatching work."""
        files = [
            SimpleUploadedFile(f"image_{index:03d}.jpg", b"fake image", content_type="image/jpeg")
            for index in range(101)
        ]

        response = self._post_upload(files=files)

        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertFalse(payload["ok"])
        self.assertIn("more than 100 individual files", payload["html"])
        self.assertIn("ZIP archive", payload["html"])
        self.assertEqual(models.UploadedArchive.objects.count(), 0)
        run_processing.assert_not_called()

    @patch("caidapp.views.run_species_prediction_async")
    def test_spreadsheet_is_stored_inside_generated_zip(self, run_processing):
        """Preserve an uploaded CSV and its detected columns inside the generated archive."""
        spreadsheet = SimpleUploadedFile(
            "metadata.csv",
            b"original path,taxon\nfirst.jpg,Lynx\n",
            content_type="text/csv",
        )

        response = self._post_upload(
            files=[
                SimpleUploadedFile("first.jpg", b"fake image 1", content_type="image/jpeg"),
                spreadsheet,
            ]
        )

        self.assertEqual(response.status_code, 200)
        uploaded_archive = models.UploadedArchive.objects.get()
        with zipfile.ZipFile(uploaded_archive.archivefile.path) as archive:
            self.assertIn("metadata.csv", archive.namelist())
        self.assertRegex(uploaded_archive.name, r"^upload_\d{8}-\d{6}$")
        self.assertEqual(
            uploaded_archive.import_mapping["spreadsheet"]["normalized_columns"], ["original_path", "taxon"]
        )
        run_processing.assert_called_once()

    @patch("caidapp.views.run_species_prediction_async")
    def test_xlsx_spreadsheet_mapping_creates_normalized_csv_in_generated_zip(self, run_processing):
        """Include the original XLSX and a mapped normalized CSV in the generated ZIP."""
        spreadsheet = self._make_xlsx_file(
            "metadata.xlsx",
            [{"image_name": "first.jpg", "animal_id": "Charles"}],
        )

        response = self._post_upload(
            extra_data={
                "spreadsheet_column_mapping": json.dumps({"original_path": "image_name", "unique_name": "animal_id"})
            },
            files=[
                SimpleUploadedFile("first.jpg", b"fake image 1", content_type="image/jpeg"),
                spreadsheet,
            ],
        )

        self.assertEqual(response.status_code, 200)
        uploaded_archive = models.UploadedArchive.objects.get()
        with zipfile.ZipFile(uploaded_archive.archivefile.path) as archive:
            self.assertIn("metadata.xlsx", archive.namelist())
            self.assertIn("mediafile.post_update.csv", archive.namelist())
            normalized_csv = archive.read("mediafile.post_update.csv").decode("utf-8-sig")
        self.assertIn("original_path,unique_name", normalized_csv)
        self.assertIn("first.jpg,Charles", normalized_csv)
        run_processing.assert_called_once()

    @patch("caidapp.views.run_species_prediction_async")
    def test_identity_spreadsheet_mapping_is_normalized_to_unique_name(self, run_processing):
        """Normalize a mapped identity column to the import format's unique_name column."""
        spreadsheet = self._make_xlsx_file(
            "metadata.xlsx",
            [{"image_name": "first.jpg", "animal_id": "Charles"}],
        )

        response = self._post_upload(
            extra_data={
                "spreadsheet_column_mapping": json.dumps({"original_path": "image_name", "identity": "animal_id"}),
            },
            files=[spreadsheet, SimpleUploadedFile("first.jpg", b"fake image", content_type="image/jpeg")],
        )

        self.assertEqual(response.status_code, 200)
        uploaded_archive = models.UploadedArchive.objects.get()
        with zipfile.ZipFile(uploaded_archive.archivefile.path) as archive:
            normalized_csv = archive.read("mediafile.post_update.csv").decode("utf-8-sig")
        self.assertIn("original_path,unique_name", normalized_csv)
        self.assertIn("first.jpg,Charles", normalized_csv)
        run_processing.assert_called_once()

    @patch("caidapp.views.run_species_prediction_async")
    def test_spreadsheet_path_adjustment_can_prepend_missing_prefix(self, run_processing):
        """Prepend the configured path prefix to normalized spreadsheet media paths."""
        spreadsheet = self._make_xlsx_file(
            "metadata.xlsx",
            [{"image_name": "first.jpg", "animal_id": "Charles"}],
        )

        response = self._post_upload(
            extra_data={
                "spreadsheet_column_mapping": json.dumps({"original_path": "image_name", "unique_name": "animal_id"}),
                "spreadsheet_path_adjustment": json.dumps({"remove_prefix": "", "add_prefix": "Brdy/Lynx/"}),
            },
            files=[
                SimpleUploadedFile("Brdy/Lynx/first.jpg", b"fake image 1", content_type="image/jpeg"),
                spreadsheet,
            ],
        )

        self.assertEqual(response.status_code, 200)
        uploaded_archive = models.UploadedArchive.objects.get()
        self.assertEqual(
            uploaded_archive.import_mapping["spreadsheet"]["path_adjustment"],
            {"remove_prefix": "", "add_prefix": "Brdy/Lynx/"},
        )
        with zipfile.ZipFile(uploaded_archive.archivefile.path) as archive:
            normalized_csv = archive.read("mediafile.post_update.csv").decode("utf-8-sig")
        self.assertIn("Brdy/Lynx/first.jpg,Charles", normalized_csv)
        run_processing.assert_called_once()

    @patch("caidapp.views.run_species_prediction_async")
    def test_spreadsheet_path_adjustment_can_remove_extra_prefix(self, run_processing):
        """Remove an extra spreadsheet path prefix before matching uploaded media files."""
        spreadsheet = self._make_xlsx_file(
            "metadata.xlsx",
            [{"image_name": "dataset/Brdy/Lynx/first.jpg", "animal_id": "Charles"}],
        )

        response = self._post_upload(
            extra_data={
                "spreadsheet_column_mapping": json.dumps({"original_path": "image_name", "unique_name": "animal_id"}),
                "spreadsheet_path_adjustment": json.dumps({"remove_prefix": "dataset/", "add_prefix": ""}),
            },
            files=[
                SimpleUploadedFile("Brdy/Lynx/first.jpg", b"fake image 1", content_type="image/jpeg"),
                spreadsheet,
            ],
        )

        self.assertEqual(response.status_code, 200)
        uploaded_archive = models.UploadedArchive.objects.get()
        self.assertEqual(
            uploaded_archive.import_mapping["spreadsheet"]["path_adjustment"],
            {"remove_prefix": "dataset/", "add_prefix": ""},
        )
        with zipfile.ZipFile(uploaded_archive.archivefile.path) as archive:
            normalized_csv = archive.read("mediafile.post_update.csv").decode("utf-8-sig")
        self.assertIn("Brdy/Lynx/first.jpg,Charles", normalized_csv)
        self.assertNotIn("dataset/Brdy/Lynx/first.jpg,Charles", normalized_csv)
        run_processing.assert_called_once()

    @patch("caidapp.views.run_species_prediction_async")
    def test_zip_internal_xlsx_mapping_creates_normalized_csv_in_archive(self, run_processing):
        """Normalize a spreadsheet already inside the uploaded ZIP and retain the original workbook."""
        spreadsheet_buffer = BytesIO()
        pd.DataFrame([{"file_ref": "Brdy/Lynx/first.jpg", "animal_id": "Charles"}]).to_excel(
            spreadsheet_buffer,
            index=False,
        )
        spreadsheet_buffer.seek(0)

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as archive:
            archive.writestr("Brdy/Lynx/first.jpg", b"fake image")
            archive.writestr("metadata.xlsx", spreadsheet_buffer.read())
        zip_buffer.seek(0)

        response = self._post_upload(
            extra_data={
                "spreadsheet_column_mapping": json.dumps({"original_path": "file_ref", "unique_name": "animal_id"})
            },
            files=[SimpleUploadedFile("Brdy_2026-05-01.zip", zip_buffer.read(), content_type="application/zip")],
        )

        self.assertEqual(response.status_code, 200)
        uploaded_archive = models.UploadedArchive.objects.get()
        with zipfile.ZipFile(uploaded_archive.archivefile.path) as archive:
            self.assertIn("metadata.xlsx", archive.namelist())
            self.assertIn("mediafile.post_update.csv", archive.namelist())
            normalized_csv = archive.read("mediafile.post_update.csv").decode("utf-8-sig")
        self.assertIn("original_path,unique_name", normalized_csv)
        self.assertIn("Brdy/Lynx/first.jpg,Charles", normalized_csv)
        self.assertEqual(
            uploaded_archive.import_mapping["spreadsheet"]["normalized_csv_filename"],
            "mediafile.post_update.csv",
        )
        run_processing.assert_called_once()

    @patch("caidapp.views.run_species_prediction_async")
    def test_relative_paths_and_directory_mapping_are_stored(self, run_processing):
        """Store relative archive paths and derive a named-group regex from directory roles."""
        manifest = [
            {
                "index": 0,
                "filename": "first.jpg",
                "relative_path": "2026-05-01/Brdy/Lynx/A12/first.jpg",
            }
        ]

        response = self._post_upload(
            extra_data={
                "directory_structure": "{check_date}/{locality}/{taxon}/{identity}",
                "directory_mapping": json.dumps({"check_date": 0, "locality": 1, "taxon": 2, "identity": 3}),
                "path_regex": (
                    r"(?P<check_date>\d{4}-\d{2}-\d{2})/(?P<locality>[^/]+)/" r"(?P<taxon>[^/]+)/(?P<identity>[^/]+)/.*"
                ),
                "upload_relative_paths": json.dumps(manifest),
            },
            files=[SimpleUploadedFile("first.jpg", b"fake image", content_type="image/jpeg")],
        )

        self.assertEqual(response.status_code, 200)
        uploaded_archive = models.UploadedArchive.objects.get()
        with zipfile.ZipFile(uploaded_archive.archivefile.path) as archive:
            self.assertEqual(archive.namelist(), ["2026-05-01/Brdy/Lynx/A12/first.jpg"])
        self.assertEqual(uploaded_archive.name, "2026-05-01")
        self.assertEqual(
            uploaded_archive.import_mapping["directory_structure"],
            "{check_date}/{locality}/{taxon}/{identity}",
        )
        self.assertEqual(uploaded_archive.import_mapping["path_source"], "relative_path")
        self.assertEqual(uploaded_archive.import_mapping["directory_mapping"]["taxon"], 2)
        self.assertEqual(uploaded_archive.import_mapping["directory_mapping"]["identity"], 3)
        self.assertIn("(?P<taxon>", uploaded_archive.path_structure_regex)
        self.assertEqual(uploaded_archive.import_mapping["path_regex"], uploaded_archive.path_structure_regex)
        run_processing.assert_called_once()

    @patch("caidapp.views.run_species_prediction_async")
    def test_path_regex_python_raw_string_wrapper_is_accepted(self, run_processing):
        """Accept a pasted Python raw-string wrapper and store only the regex content."""
        response = self._post_upload(
            extra_data={
                "path_regex": r'r"^(?P<taxon>[^/]+)/(?P<unique_name>[^/]+)/[^/]+$"',
            },
            files=[
                SimpleUploadedFile("Lynx/Charles/first.jpg", b"fake image", content_type="image/jpeg"),
            ],
        )

        self.assertEqual(response.status_code, 200)
        uploaded_archive = models.UploadedArchive.objects.get()
        self.assertEqual(uploaded_archive.path_structure_regex, r"^(?P<taxon>[^/]+)/(?P<unique_name>[^/]+)/[^/]+$")
        self.assertEqual(uploaded_archive.import_mapping["path_regex"], uploaded_archive.path_structure_regex)
        run_processing.assert_called_once()

    @patch("caidapp.views.run_species_prediction_async")
    def test_single_directory_layer_can_be_mapped_to_taxon(self, run_processing):
        """Map a single directory level to taxon and use that directory as the archive name."""
        manifest = [
            {
                "index": 0,
                "filename": "first.jpg",
                "relative_path": "Lynx/first.jpg",
            }
        ]

        response = self._post_upload(
            extra_data={
                "directory_structure": "{taxon}",
                "directory_mapping": json.dumps({"taxon": 0}),
                "upload_relative_paths": json.dumps(manifest),
            },
            files=[SimpleUploadedFile("first.jpg", b"fake image", content_type="image/jpeg")],
        )

        self.assertEqual(response.status_code, 200)
        uploaded_archive = models.UploadedArchive.objects.get()
        self.assertEqual(uploaded_archive.import_mapping["directory_structure"], "{taxon}")
        self.assertEqual(uploaded_archive.import_mapping["directory_mapping"], {"taxon": 0})
        self.assertEqual(uploaded_archive.path_structure_regex, r"^(?P<taxon>[^/]+)/[^/]+$")
        self.assertEqual(uploaded_archive.import_mapping["path_regex"], uploaded_archive.path_structure_regex)
        self.assertEqual(uploaded_archive.name, "Lynx")
        run_processing.assert_called_once()

    @patch("caidapp.views.run_species_prediction_async")
    def test_directory_mapping_is_allowed_for_zip_internal_paths(self, run_processing):
        """Apply directory-role mapping to paths inside an uploaded ZIP."""
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as archive:
            archive.writestr("Brdy/Lynx/first.jpg", b"fake image")
        zip_buffer.seek(0)

        response = self._post_upload(
            extra_data={
                "directory_structure": "{locality}/{taxon}",
                "directory_mapping": json.dumps({"locality": 0, "taxon": 1}),
            },
            files=[SimpleUploadedFile("Brdy_2026-05-01.zip", zip_buffer.read(), content_type="application/zip")],
        )

        self.assertEqual(response.status_code, 200)
        uploaded_archive = models.UploadedArchive.objects.get()
        self.assertEqual(uploaded_archive.name, "Brdy_2026-05-01")
        self.assertEqual(uploaded_archive.import_mapping["path_source"], "archive_path")
        self.assertEqual(uploaded_archive.import_mapping["directory_mapping"], {"locality": 0, "taxon": 1})
        self.assertEqual(uploaded_archive.path_structure_regex, r"^(?P<locality>[^/]+)/(?P<taxon>[^/]+)/[^/]+$")
        self.assertEqual(uploaded_archive.import_mapping["path_regex"], uploaded_archive.path_structure_regex)
        run_processing.assert_called_once()

    @patch("caidapp.views.run_species_prediction_async")
    @tag("long")
    def test_user_zip_upload_stores_inner_directory_locality_mapping(self, run_processing):
        """Store a locality mapping from inner ZIP directories and count its 11 images."""
        if not WRAP_TEST_DATA_DIR:
            self.skipTest("WRAP_TEST_DATA_DIR is not configured")
        zip_path = Path(WRAP_TEST_DATA_DIR) / "2021-05-06_Tri_lokality_XYZ.zip"
        if not zip_path.exists():
            self.skipTest(f"Missing test ZIP: {zip_path}")

        response = self._post_upload(
            extra_data={
                "directory_structure": "*/{locality}",
                "directory_mapping": json.dumps({"locality": 1}),
            },
            files=[SimpleUploadedFile(zip_path.name, zip_path.read_bytes(), content_type="application/zip")],
        )

        self.assertEqual(response.status_code, 200)
        uploaded_archive = models.UploadedArchive.objects.get()
        self.assertEqual(uploaded_archive.import_mapping["path_source"], "archive_path")
        self.assertEqual(uploaded_archive.import_mapping["directory_mapping"], {"locality": 1})
        self.assertEqual(uploaded_archive.files_at_upload, 11)
        self.assertEqual(uploaded_archive.images_at_upload, 11)
        run_processing.assert_called_once_with(uploaded_archive, extract_identites=False)

    @patch("caidapp.views.run_species_prediction_async")
    def test_single_taxon_metadata_is_stored(self, run_processing):
        """Persist representative single-taxon upload flags and request identity extraction."""
        self._set_capabilities(show_taxon_classification=True, show_reid=True, show_base_dataset=True)

        response = self._post_upload(
            extra_data={
                "upload_target": "identification",
                "contains_identities": "on",
            }
        )

        self.assertEqual(response.status_code, 200)
        uploaded_archive = models.UploadedArchive.objects.get()
        self.assertTrue(uploaded_archive.contains_single_taxon)
        self.assertTrue(uploaded_archive.contains_identities)
        self.assertTrue(uploaded_archive.is_for_identification)
        self.assertIsNone(uploaded_archive.taxon_for_identification)
        run_processing.assert_called_once_with(uploaded_archive, extract_identites=True)

    @patch("caidapp.views.run_species_prediction_async")
    def test_taxon_only_upload_stays_in_taxon_processing(self, run_processing):
        """Keep taxon-only uploads outside the identification workflow."""
        self._set_capabilities(show_taxon_classification=True, show_reid=False)

        response = self._post_upload(
            extra_data={
                "upload_target": "identification",
                "contains_identities": "on",
            }
        )

        self.assertEqual(response.status_code, 200)
        uploaded_archive = models.UploadedArchive.objects.get()
        self.assertFalse(uploaded_archive.contains_single_taxon)
        self.assertFalse(uploaded_archive.contains_identities)
        self.assertFalse(uploaded_archive.is_for_identification)
        run_processing.assert_called_once_with(uploaded_archive, extract_identites=False)

    @patch("caidapp.views.run_species_prediction_async")
    def test_reid_only_non_admin_upload_forces_single_taxon_without_base_dataset(self, run_processing):
        """Force a regular ReID-only upload to single-taxon mode without representative identities."""
        self._set_capabilities(show_taxon_classification=False, show_reid=True, workgroup_admin=False)

        response = self._post_upload()

        self.assertEqual(response.status_code, 200)
        uploaded_archive = models.UploadedArchive.objects.get()
        self.assertTrue(uploaded_archive.contains_single_taxon)
        self.assertFalse(uploaded_archive.contains_identities)
        self.assertTrue(uploaded_archive.is_for_identification)
        run_processing.assert_called_once_with(uploaded_archive, extract_identites=False)

    @patch("caidapp.views.run_species_prediction_async")
    def test_reid_only_admin_can_mark_upload_as_base_dataset(self, run_processing):
        """Allow an authorized ReID admin to mark an upload as a representative base dataset."""
        self._set_capabilities(show_taxon_classification=False, show_reid=True, show_base_dataset=True)

        response = self._post_upload(extra_data={"contains_identities": "on"})

        self.assertEqual(response.status_code, 200)
        uploaded_archive = models.UploadedArchive.objects.get()
        self.assertTrue(uploaded_archive.contains_single_taxon)
        self.assertTrue(uploaded_archive.contains_identities)
        self.assertTrue(uploaded_archive.is_for_identification)
        run_processing.assert_called_once_with(uploaded_archive, extract_identites=True)

    @patch("caidapp.views.run_species_prediction_async")
    def test_directory_mapping_without_relative_paths_blocks_processing(self, run_processing):
        """Reject directory mapping without usable relative paths before storing or processing an archive."""
        response = self._post_upload(extra_data={"directory_structure": "{check_date}/{locality}/{taxon}/{identity}"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(models.UploadedArchive.objects.count(), 0)
        run_processing.assert_not_called()
