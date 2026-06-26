import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from django.conf import settings
from django.test import TestCase, override_settings

from caidapp import tasks
from caidapp.app_tests.factories import (
    AnimalObservationFactory,
    CaidUserFactory,
    IndividualIdentityFactory,
    LocalityFactory,
    MediaFileFactory,
    UploadedArchiveFactory,
)


class MediaFileLocationFallbackTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory(admin=True)

    def test_effective_location_prefers_mediafile_location(self):
        locality = LocalityFactory(owner=self.caiduser, location="50.1,14.4")
        mediafile = MediaFileFactory(parent__owner=self.caiduser, locality=locality, location="49.9,14.2")

        self.assertEqual(str(mediafile.effective_location), "49.9,14.2")
        self.assertEqual(mediafile.effective_location_source, "mediafile")

    def test_effective_location_falls_back_to_locality(self):
        locality = LocalityFactory(owner=self.caiduser, location="50.1,14.4")
        mediafile = MediaFileFactory(parent__owner=self.caiduser, locality=locality, location=None)

        self.assertEqual(str(mediafile.effective_location), "50.1,14.4")
        self.assertEqual(mediafile.effective_location_source, "locality")

    def test_prepare_dataframe_for_identification_uses_effective_location(self):
        locality = LocalityFactory(owner=self.caiduser, location="50.1,14.4")
        mediafile = MediaFileFactory(parent__owner=self.caiduser, locality=locality, location="49.9,14.2")

        csv_data = tasks._prepare_dataframe_for_identification([mediafile])

        self.assertEqual(csv_data["locality_coordinates"][0], "49.9,14.2")

    def test_prepare_dataframe_for_identification_uses_static_thumbnail_for_video(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        static_thumbnail_relpath = "output/test-video/static_thumbnails/clip.webp"
        static_thumbnail_path = Path(settings.MEDIA_ROOT) / static_thumbnail_relpath
        static_thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        static_thumbnail_path.write_bytes(b"fake webp")
        mediafile = MediaFileFactory(
            parent=archive,
            media_type="video",
            image_file="output/test-video/videos/clip.mp4",
            static_thumbnail=static_thumbnail_relpath,
        )

        csv_data = tasks._prepare_dataframe_for_identification([mediafile])

        self.assertEqual(
            csv_data["image_path"][0],
            str(static_thumbnail_path),
        )

    def test_prepare_dataframe_for_identification_falls_back_to_static_thumbnail_for_missing_image(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        static_thumbnail_relpath = "output/test-image/static_thumbnails/first.webp"
        static_thumbnail_path = Path(settings.MEDIA_ROOT) / static_thumbnail_relpath
        static_thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        static_thumbnail_path.write_bytes(b"fake webp")
        mediafile = MediaFileFactory(
            parent=archive,
            media_type="image",
            image_file="output/test-image/images/missing.webp",
            static_thumbnail=static_thumbnail_relpath,
        )

        csv_data = tasks._prepare_dataframe_for_identification([mediafile])

        self.assertEqual(csv_data["image_path"][0], str(static_thumbnail_path))

    def test_prepare_dataframe_for_identification_skips_mediafile_with_missing_image_sources(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(
            parent=archive,
            media_type="image",
            image_file="output/test-image/images/missing.webp",
            static_thumbnail="output/test-image/static_thumbnails/missing.webp",
            preview="output/test-image/previews/missing.webp",
            thumbnail="output/test-image/thumbnails/missing.webp",
        )

        csv_data = tasks._prepare_dataframe_for_identification([mediafile])

        self.assertEqual(csv_data["image_path"], [])
        self.assertEqual(csv_data["mediafile_id"], [])

    def test_prepare_dataframe_for_identification_uses_representative_observation_identity(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Observed identity")
        mediafile = MediaFileFactory(
            parent=archive,
            identity=None,
            identity_is_representative=False,
        )
        AnimalObservationFactory(
            mediafile=mediafile,
            identity=identity,
            identity_is_representative=True,
        )

        csv_data = tasks._prepare_dataframe_for_identification([mediafile])

        self.assertEqual(csv_data["class_id"][0], identity.id)
        self.assertEqual(csv_data["label"][0], identity.name)

    def test_create_dataframe_from_mediafiles_exports_identity_codes_and_effective_location(self):
        locality = LocalityFactory(owner=self.caiduser, location="50.1,14.4")
        identity = IndividualIdentityFactory(
            owner_workgroup=self.caiduser.workgroup,
            name="Charles",
            code="B75",
            juv_code="J12",
        )
        mediafile = MediaFileFactory(
            parent__owner=self.caiduser,
            locality=locality,
            location="49.9,14.2",
            identity=identity,
            original_filename="Brdy/Charles/first.jpg",
        )

        df = tasks.create_dataframe_from_mediafiles([mediafile])

        self.assertEqual(df.loc[0, "unique_name"], "Charles")
        self.assertEqual(df.loc[0, "code"], "B75")
        self.assertEqual(df.loc[0, "juv_code"], "J12")
        self.assertEqual(df.loc[0, "latitude"], "49.9")
        self.assertEqual(df.loc[0, "longitude"], "14.2")
        self.assertEqual(df.loc[0, "locality coordinates"], "50.1,14.4")
        self.assertEqual(df.loc[0, "original_path"], "Brdy/Charles/first.jpg")

    def test_create_dataframe_from_mediafiles_falls_back_to_locality_location(self):
        locality = LocalityFactory(owner=self.caiduser, location="50.1,14.4")
        mediafile = MediaFileFactory(parent__owner=self.caiduser, locality=locality, location=None)

        df = tasks.create_dataframe_from_mediafiles([mediafile])

        self.assertEqual(df.loc[0, "latitude"], "50.1")
        self.assertEqual(df.loc[0, "longitude"], "14.4")

    def test_create_dataframe_from_mediafiles_does_not_export_stale_unique_name_from_metadata_json(self):
        mediafile = MediaFileFactory(
            parent__owner=self.caiduser,
            identity=None,
            metadata_json={
                "predicted_category": "Lynx",
                "unique_name": "Lenka_0b550fe649092e358b34f12705fc7c7c8b72cff50b87299fd0b46226f8b2b18.jpg",
            },
        )

        df = tasks.create_dataframe_from_mediafiles([mediafile])

        self.assertNotIn("unique_name", df.columns)


class UploadedArchiveLocalityCompatibilityTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory(admin=True)

    def test_uploaded_archive_localities_are_derived_from_mediafiles(self):
        uploaded_archive = UploadedArchiveFactory(owner=self.caiduser)
        locality_a = LocalityFactory(owner=self.caiduser, name="Xandovice")
        locality_b = LocalityFactory(owner=self.caiduser, name="Ypovice")
        MediaFileFactory(parent=uploaded_archive, locality=locality_b)
        MediaFileFactory(parent=uploaded_archive, locality=locality_a)

        self.assertEqual(
            list(uploaded_archive.localities().values_list("name", flat=True)),
            ["Xandovice", "Ypovice"],
        )
        self.assertEqual(uploaded_archive.locality.name, "Xandovice")
        self.assertEqual(uploaded_archive.localities_display, "Xandovice, Ypovice")

    def test_uploaded_archive_locality_falls_back_to_legacy_field(self):
        locality = LocalityFactory(owner=self.caiduser, name="Legacy Place")
        uploaded_archive = UploadedArchiveFactory(owner=self.caiduser, locality_at_upload_object=locality)

        self.assertEqual(uploaded_archive.locality, locality)
        self.assertEqual(
            list(uploaded_archive.localities().values_list("name", flat=True)),
            ["Legacy Place"],
        )


class LocalityCoverTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory(admin=True)

    def test_cover_falls_back_to_first_mediafile(self):
        locality = LocalityFactory(owner=self.caiduser, name="Cover Meadow")
        self.assertIsNone(locality.cover)

        first_mediafile = MediaFileFactory(parent__owner=self.caiduser, locality=locality)

        self.assertEqual(locality.cover, first_mediafile)


class SpreadsheetMetadataImportTest(TestCase):
    def setUp(self):
        self.media_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.media_root)
        self.override = override_settings(MEDIA_ROOT=self.media_root)
        self.override.enable()
        self.addCleanup(self.override.disable)
        self.caiduser = CaidUserFactory(admin=True)

    def test_metadata_row_assigns_identity_codes_and_mediafile_location_outside_identification_mode(self):
        uploaded_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            contains_identities=False,
            contains_single_taxon=False,
            is_for_identification=False,
        )
        output_dir = Path(settings.MEDIA_ROOT) / uploaded_archive.outputdir
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        source_dir = Path(settings.MEDIA_ROOT) / "source_media"
        source_dir.mkdir(parents=True, exist_ok=True)

        image_path = output_dir / "images" / "first.jpg"
        image_path.write_bytes(b"fake image")
        source_media_path = source_dir / "first.jpg"
        source_media_path.write_bytes(b"fake original image")

        row_data = {
            "image_path": "first.jpg",
            "absolute_media_path": str(source_media_path),
            "datetime": "2026-05-15T10:30:00",
            "predicted_category": "Lynx",
            "media_type": "image",
            "original_path": "Brdy/Charles/first.jpg",
            "unique_name": "Charles",
            "code": "B75",
            "juv_code": "J12",
            "locality_name": "Forest Edge",
            "latitude": "49.1234",
            "longitude": "13.4564",
        }
        df = pd.DataFrame([row_data])

        with patch("caidapp.models.MediaFile.make_thumbnail_for_mediafile_if_necessary", autospec=True, return_value=None):
            status = tasks._update_database_by_one_row_of_metadata(
                df=df,
                index=0,
                row=df.iloc[0],
                create_missing=True,
                extract_identites=False,
                locality=None,
                output_dir=output_dir,
                thumbnail_width=400,
                uploaded_archive=uploaded_archive,
            )

        mediafile = uploaded_archive.mediafile_set.get()
        self.assertEqual(status, "created and not updated by user")
        self.assertIsNotNone(mediafile.identity)
        self.assertEqual(mediafile.identity.name, "Charles")
        self.assertEqual(mediafile.identity.code, "B75")
        self.assertEqual(mediafile.identity.juv_code, "J12")
        self.assertEqual(str(mediafile.location), "49.123,13.456")
        self.assertEqual(mediafile.locality.name, "Forest Edge")

    def test_metadata_import_uses_prepared_media_variants_without_regenerating(self):
        uploaded_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            contains_identities=False,
            contains_single_taxon=False,
            is_for_identification=False,
        )
        output_dir = Path(settings.MEDIA_ROOT) / uploaded_archive.outputdir
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "images" / "first.webp"
        image_path.write_bytes(b"fake image")
        for relative_path in ("previews/first.webp", "thumbnails/first.webp", "static_thumbnails/first.webp"):
            path = output_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"prepared")

        row_data = {
            "image_path": "first.webp",
            "absolute_media_path": str(image_path),
            "datetime": "2026-05-15T10:30:00",
            "predicted_category": "Lynx",
            "media_type": "image",
            "preview_path": "previews/first.webp",
            "thumbnail_path": "thumbnails/first.webp",
            "static_thumbnail_path": "static_thumbnails/first.webp",
        }
        df = pd.DataFrame([row_data])

        with patch("caidapp.models.MediaFile.make_thumbnail_for_mediafile_if_necessary", autospec=True) as make_variants:
            status = tasks._update_database_by_one_row_of_metadata(
                df=df,
                index=0,
                row=df.iloc[0],
                create_missing=True,
                extract_identites=False,
                locality=None,
                output_dir=output_dir,
                thumbnail_width=400,
                uploaded_archive=uploaded_archive,
            )

        mediafile = uploaded_archive.mediafile_set.get()
        self.assertEqual(status, "created and not updated by user")
        self.assertEqual(mediafile.preview.name, f"{uploaded_archive.outputdir}/previews/first.webp")
        self.assertEqual(mediafile.thumbnail.name, f"{uploaded_archive.outputdir}/thumbnails/first.webp")
        self.assertEqual(
            mediafile.static_thumbnail.name,
            f"{uploaded_archive.outputdir}/static_thumbnails/first.webp",
        )
        make_variants.assert_not_called()
