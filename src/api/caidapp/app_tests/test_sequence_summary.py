"""Regression coverage for sequence sizes, ordering, and individual estimates."""

from types import SimpleNamespace

from caidapp.models import Sequence
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from .factories import (
    AnimalObservationFactory,
    CaidUserFactory,
    MediaFileFactory,
    SequenceFactory,
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
        self.assertContains(response, "Media files: 2")
        self.assertContains(response, "Individuals: at least 2")

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
        self.assertContains(response, "ID within upload: 10")
        self.assertContains(response, "Media files in sequence: minimum")
