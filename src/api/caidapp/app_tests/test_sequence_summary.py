"""Regression coverage for sequence sizes, ordering, and individual estimates."""

from types import SimpleNamespace

from caidapp.models import Sequence
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from .factories import (
    AnimalObservationFactory,
    CaidUserFactory,
    IndividualIdentityFactory,
    LocalityFactory,
    MediaFileFactory,
    SequenceFactory,
    TaxonFactory,
    UploadedArchiveFactory,
)


class SequenceIndividualsTest(SimpleTestCase):
    """Check conservative estimates without requiring persisted media files."""

    def summary(self, *files):
        """Evaluate the model property with in-memory related collections."""
        sequence = SimpleNamespace(mediafile_set=SimpleNamespace(all=lambda: files))
        return Sequence.individuals_summary.fget(sequence)

    def mediafile(self, *identities, media_type="image", placeholder=False):
        """Build observations including explicit no-detection placeholders."""
        observations = [
            SimpleNamespace(identity_id=identity, is_no_detection_placeholder=placeholder) for identity in identities
        ]
        return SimpleNamespace(media_type=media_type, observations=SimpleNamespace(all=lambda: observations))

    def test_known_identities_are_distinct_across_files(self):
        """Repeated sightings of one identity must not increase the count."""
        summary = self.summary(self.mediafile(1, 2), self.mediafile(2, 3))
        self.assertEqual(summary["minimum"], 3)
        self.assertTrue(summary["is_exact"])

    def test_unknown_sightings_produce_a_lower_bound(self):
        """Unknown objects may be known individuals seen in other images."""
        summary = self.summary(self.mediafile(1), self.mediafile(2), self.mediafile(None, None, None))
        self.assertEqual(summary["minimum"], 3)
        self.assertEqual(summary["known"], 2)
        self.assertEqual(summary["unidentified_observations"], 3)
        self.assertFalse(summary["is_exact"])

    def test_video_observations_are_not_simultaneous(self):
        """Three unlocalized video detections can all show the same individual."""
        summary = self.summary(self.mediafile(None, None, None, media_type="video"))
        self.assertEqual(summary["minimum"], 1)
        self.assertFalse(summary["is_exact"])

    def test_placeholders_do_not_count_as_individuals(self):
        """No-detection placeholders contribute neither identities nor unknown objects."""
        summary = self.summary(self.mediafile(None, placeholder=True))
        self.assertEqual(summary["minimum"], 0)
        self.assertEqual(summary["unidentified_observations"], 0)


class SequenceSizeViewTest(TestCase):
    """Exercise complete-sequence counts through the actual listing and filter."""

    def setUp(self):
        """Create two sequences whose local IDs and file counts sort differently."""
        self.owner = CaidUserFactory()
        self.client.force_login(self.owner.user)
        archive = UploadedArchiveFactory(owner=self.owner)
        self.small = SequenceFactory(uploaded_archive=archive, local_id=20)
        self.large = SequenceFactory(uploaded_archive=archive, local_id=10)
        for sequence, filenames in ((self.small, ["match.jpg"]), (self.large, ["match.jpg", "other.jpg"])):
            for filename in filenames:
                mediafile = MediaFileFactory(parent=archive, sequence=sequence, original_filename=filename)
                AnimalObservationFactory(mediafile=mediafile)
                AnimalObservationFactory(mediafile=mediafile)

    def test_filter_uses_full_size_despite_filename_and_observation_joins(self):
        """A filename match and multiple observations must not change sequence sizes."""
        response = self.client.get(
            reverse("caidapp:sequences"),
            {"sequence_files_min": 2, "sequence_files_max": 2, "search": "match"},
        )
        self.assertEqual([seq.pk for seq in response.context["sequence_objects"]], [self.large.pk])
        self.assertEqual(response.context["number_of_mediafiles"], 1)
        self.assertContains(response, "2 media files")
        self.assertContains(response, "at least 2 individuals")

    def test_file_count_sort_differs_from_local_sequence_id(self):
        """Both sort choices must remain explicit and behave independently."""
        for sort, expected in (
            ("count_asc", [self.small.pk, self.large.pk]),
            ("count_desc", [self.large.pk, self.small.pk]),
            ("sequence_asc", [self.large.pk, self.small.pk]),
            ("sequence_desc", [self.small.pk, self.large.pk]),
        ):
            with self.subTest(sort=sort):
                response = self.client.get(reverse("caidapp:sequences"), {"sort": sort})
                self.assertEqual([seq.pk for seq in response.context["sequence_objects"]], expected)

    def test_list_displays_complete_sequence_summary(self):
        """Show the same summary in tabular and card views."""
        response = self.client.get(reverse("caidapp:sequences"), {"view": "list"})
        self.assertContains(response, "Sequence summary", count=2)
        self.assertNotContains(response, "ID within upload: 10")
        self.assertContains(response, "Media files in sequence: minimum")
        self.assertContains(response, "<h1>Media files and sequences</h1>")


class ObservationGroupSummaryTest(TestCase):
    """Ensure summaries follow the displayed grouping, filter, and page."""

    def setUp(self):
        """Create sibling observations and a second media file with distinct metadata."""
        self.owner = CaidUserFactory()
        self.client.force_login(self.owner.user)
        archive = UploadedArchiveFactory(owner=self.owner)
        sequence = SequenceFactory(uploaded_archive=archive)
        self.taxon = TaxonFactory(name="Lynx lynx")
        self.locality = LocalityFactory(owner=self.owner, name="Forest camera")
        self.identity = IndividualIdentityFactory(owner_workgroup=self.owner.workgroup)
        self.first = MediaFileFactory(parent=archive, sequence=sequence, locality=self.locality)
        second = MediaFileFactory(parent=archive, sequence=sequence, locality=self.locality)
        AnimalObservationFactory(mediafile=self.first, identity=self.identity, taxon=self.taxon)
        AnimalObservationFactory(mediafile=self.first, identity=None, taxon=self.taxon)
        AnimalObservationFactory(mediafile=second, identity=self.identity, taxon=self.taxon)

    def test_summary_follows_grouping_in_all_layouts(self):
        """Count each displayed group while leaving observation-level selection intact."""
        for view in ("cards", "large_cards", "list"):
            for grouping, title, sizes in (
                ("sequence", "Sequence summary", [2]),
                ("mediafile", "Media file summary", [1, 1]),
                ("identity", "Identity summary", [1, 2]),
            ):
                with self.subTest(view=view, grouping=grouping):
                    response = self.client.get(reverse("caidapp:observations"), {"view": view, "group": grouping})
                    groups = response.context["observation_groups"]
                    self.assertEqual(sorted(group["mediafile_count"] for group in groups), sizes)
                    self.assertContains(response, title)
                    self.assertContains(response, "Displayed observations · current filter and page")
                    self.assertContains(response, f'href="{reverse("caidapp:observations")}?taxon={self.taxon.pk}"')
                    self.assertContains(
                        response, f'href="{reverse("caidapp:observations")}?locality={self.locality.pk}"'
                    )
                    self.assertEqual(sum(len(group["observations"]) for group in groups), 3)

    def test_summary_excludes_sibling_observations_filtered_by_identity(self):
        """A known-identity filter must not count unidentified siblings in the same file."""
        response = self.client.get(reverse("caidapp:observations"), {"identity": self.identity.pk})
        summary = response.context["observation_groups"][0]["individuals_summary"]
        self.assertEqual(summary["minimum"], 1)
        self.assertTrue(summary["is_exact"])
        self.assertEqual(summary["unidentified_observations"], 0)

    def test_summary_is_limited_to_displayed_page(self):
        """A sequence continuing on another page must not silently summarize hidden observations."""
        for _ in range(23):
            AnimalObservationFactory(mediafile=self.first, identity=None, taxon=self.taxon)
        response = self.client.get(reverse("caidapp:observations"), {"per_page": 24})
        group = response.context["observation_groups"][0]
        self.assertEqual(len(group["observations"]), 24)
        self.assertEqual(group["individuals_summary"]["unidentified_observations"], 23)
        self.assertFalse(group["individuals_summary"]["is_exact"])
