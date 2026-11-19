"""Guard list pages against database queries repeated for every visible card."""

from caidapp import models, views_uploads
from caidapp.app_tests.factories import (
    AnimalObservationFactory,
    CaidUserFactory,
    IndividualIdentityFactory,
    LocalityFactory,
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


class CameraTrapCheckQueryTest(TestCase):
    def setUp(self):
        """Render completed uploads with shared locality and taxon data."""
        self.user = CaidUserFactory()
        self.client.force_login(self.user.user)
        self.locality = LocalityFactory(owner=self.user)
        self.taxon = TaxonFactory()

    def add_archives(self, count):
        """Use verified observations so status refreshes need no writes."""
        for _ in range(count):
            archive = UploadedArchiveFactory(owner=self.user)
            mediafile = MediaFileFactory(parent=archive, locality=self.locality)
            AnimalObservationFactory(mediafile=mediafile, taxon=self.taxon, taxon_verified=True)

    def test_upload_cards_do_not_add_per_card_queries(self):
        """An entire page must load card summaries with a bounded query count."""
        self.add_archives(2)
        url = reverse("caidapp:uploads")
        with CaptureQueriesContext(connection) as small:
            response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.add_archives(10)
        with CaptureQueriesContext(connection) as large:
            response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertLessEqual(len(large), len(small), f"2 cards: {len(small)}, 12 cards: {len(large)}")

    def test_bulk_summaries_match_observation_based_counts(self):
        """Multiple observations and placeholders must preserve existing summary semantics."""
        archive = UploadedArchiveFactory(owner=self.user, taxon_for_identification=self.taxon)
        identity = IndividualIdentityFactory(owner_workgroup=self.user.workgroup)
        mediafile = MediaFileFactory(parent=archive, locality=self.locality)
        AnimalObservationFactory(mediafile=mediafile, taxon=self.taxon, identity=identity)
        AnimalObservationFactory(mediafile=mediafile, taxon=self.taxon, identity=identity, taxon_verified=False)
        MediaFileFactory(parent=archive)  # No-detection placeholder, no identity.
        unknown = MediaFileFactory(parent=archive)
        AnimalObservationFactory(mediafile=unknown, taxon=TaxonFactory(name="Animalia"), taxon_verified=False)
        empty = UploadedArchiveFactory(owner=self.user, locality_at_upload_object=self.locality)
        single_taxon = UploadedArchiveFactory(owner=self.user, contains_single_taxon=True)
        MediaFileFactory(parent=single_taxon)
        importing = UploadedArchiveFactory(owner=self.user, taxon_status="TAIP", mediafiles_at_upload=42)
        methods = (
            "count_of_mediafiles",
            "count_of_identities",
            "count_of_mediafiles_with_taxon_for_identification",
            "count_of_mediafiles_with_missing_taxon",
            "count_of_mediafiles_with_unverified_taxon",
            "localities_display",
            "locality",
        )

        def summary(obj):
            values = [getattr(obj, name) for name in methods]
            return [value() if callable(value) else value for value in values]

        archives = [archive, empty, single_taxon, importing]
        expected = [summary(obj) for obj in archives]
        self.assertEqual(expected[0][:5], [3, 1, 1, 2, 3])
        views_uploads.prepare_upload_card_summaries(archives, self.user)
        with self.assertNumQueries(0):
            self.assertEqual([summary(obj) for obj in archives], expected)

    def test_aggregate_sorting_and_pagination(self):
        """Only displayed uploads need summaries, including when sorting by aggregate values."""
        self.add_archives(2)
        extra = models.UploadedArchive.objects.filter(owner=self.user).first()
        MediaFileFactory(parent=extra, taxon=self.taxon, taxon_verified=True)
        session = self.client.session
        session["item_number_uploaded_archives"] = 1
        for order in ("-mediafile_count", "-mediafile_count_with_taxon", "earliest_mediafile_captured_at"):
            with self.subTest(order=order):
                models.UploadedArchive.objects.filter(owner=self.user).update(taxon_for_identification=self.taxon)
                session["sort_uploaded_archives_by"] = order
                session.save()
                response = self.client.get(reverse("caidapp:uploads"))
                self.assertEqual(response.status_code, 200)
                page = response.context["page_obj"]
                self.assertEqual(page.paginator.count, 2)
                self.assertEqual(len(page), 1)
                if order.startswith("-mediafile_count"):
                    self.assertEqual(page[0].pk, extra.pk)
