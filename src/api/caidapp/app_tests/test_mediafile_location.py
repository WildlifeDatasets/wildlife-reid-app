from django.test import TestCase

from caidapp import tasks
from caidapp.app_tests.factories import CaidUserFactory, LocalityFactory, MediaFileFactory


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
