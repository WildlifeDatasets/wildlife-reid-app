from django.test import TestCase

from caidapp import tasks
from caidapp.app_tests.factories import CaidUserFactory, LocalityFactory, MediaFileFactory, UploadedArchiveFactory


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
