"""Guard list pages against database queries repeated for every visible card."""

from caidapp.app_tests.factories import (
    AnimalObservationFactory,
    CaidUserFactory,
    MediaFileFactory,
    SequenceFactory,
    TaxonFactory,
    UploadedArchiveFactory,
)
from caidapp.models import MediaFile, MediafilesForIdentification
from django.db import connection
from django.test import SimpleTestCase, TestCase
from django.test.utils import CaptureQueriesContext
from django.urls import reverse


class CardImageTest(SimpleTestCase):
    def test_image_uses_small_thumbnail_before_preview_or_full_image(self):
        """Older imports may have a thumbnail without a separate static derivative."""
        mediafile = MediaFile(
            media_type="image", thumbnail="small.webp", preview="large.webp", image_file="original.jpg"
        )
        self.assertEqual(mediafile.card_image.name, "small.webp")

    def test_video_prefers_static_frame_to_animated_thumbnail(self):
        """Do not automatically download motion when the static derivative is absent."""
        mediafile = MediaFile(media_type="video", thumbnail="motion.webp", image_file="frame.jpg")
        self.assertEqual(mediafile.card_image.name, "frame.jpg")


class ReviewListQueryTest(TestCase):
    def setUp(self):
        """Use shared context so adding cards introduces no new filter choices."""
        self.user = CaidUserFactory()
        self.client.force_login(self.user.user)
        self.archive = UploadedArchiveFactory(owner=self.user)
        self.sequence = SequenceFactory(uploaded_archive=self.archive)
        self.taxon = TaxonFactory()

    def add_mediafiles(self, count):
        """Include predictions and editor names that used to trigger per-card queries."""
        for _ in range(count):
            mediafile = MediaFileFactory(parent=self.archive, sequence=self.sequence)
            AnimalObservationFactory(mediafile=mediafile, taxon=None, predicted_taxon=self.taxon, updated_by=self.user)

    def assert_query_count_stays_bounded(self, view):
        """Rendering additional cards must reuse bulk-loaded related objects."""
        self.add_mediafiles(2)
        url = reverse(f"caidapp:{view}")
        with CaptureQueriesContext(connection) as small:
            response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.add_mediafiles(10)
        with CaptureQueriesContext(connection) as large:
            response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertLessEqual(len(large), len(small), [query["sql"] for query in large])
        return response

    def test_mediafile_cards_do_not_add_per_card_queries(self):
        """Sequence sizes, identification state and predictions are loaded in bulk."""
        response = self.assert_query_count_stays_bounded("media_files")
        self.assertContains(response, "12 media files in sequence")

    def test_observation_cards_do_not_add_per_card_queries(self):
        """Showing the editor's name must not fetch a user for each observation."""
        self.assert_query_count_stays_bounded("observations")

    def test_mediafile_identification_link_matches_queue_membership(self):
        """Bulk-loaded identification state must preserve each card's action."""
        self.add_mediafiles(2)
        queued, other = MediaFile.objects.filter(parent=self.archive).order_by("pk")
        MediafilesForIdentification.objects.create(mediafile=queued)
        response = self.client.get(reverse("caidapp:media_files"))
        self.assertContains(response, reverse("caidapp:get_individual_identity_by_media_file", args=[queued.pk]))
        self.assertNotContains(response, reverse("caidapp:get_individual_identity_by_media_file", args=[other.pk]))
