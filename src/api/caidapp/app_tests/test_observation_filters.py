"""Separate observation geometry from counts across the containing media file."""

from types import SimpleNamespace

from caidapp.filters import AnimalObservationFilter
from caidapp.models import AnimalObservation
from django.test import TestCase
from django.urls import reverse

from .factories import AnimalObservationFactory, MediaFileFactory


class ObservationFilterTest(TestCase):
    def setUp(self):
        """Create a mixed file, a bbox-less object and an empty placeholder file."""
        self.mediafile = MediaFileFactory()
        self.user = self.mediafile.parent.owner
        self.boxed = AnimalObservationFactory(
            mediafile=self.mediafile, bbox_x_center=0.5, bbox_y_center=0.5, bbox_width=0.2, bbox_height=0.2
        )
        self.missing = AnimalObservationFactory(mediafile=self.mediafile)
        self.only_missing = AnimalObservationFactory(mediafile=MediaFileFactory(parent=self.mediafile.parent))
        self.placeholder = MediaFileFactory(parent=self.mediafile.parent).observations.get()
        self.request = SimpleNamespace(user=self.user.user)
        self.queryset = AnimalObservation.objects.filter(mediafile__parent=self.mediafile.parent)

    def ids(self, **data):
        """Evaluate only the requested filters against the fixture observations."""
        return set(
            AnimalObservationFilter(data, queryset=self.queryset, request=self.request).qs.values_list("pk", flat=True)
        )

    def test_bbox_presence_is_observation_local_but_count_is_mediafile_wide(self):
        """A bbox-less sibling still belongs to a media file with one bbox."""
        self.assertEqual(self.ids(has_bbox="false", objects_per_image="1"), {self.missing.pk})
        self.assertEqual(self.ids(has_bbox="true"), {self.boxed.pk})
        self.assertEqual(self.ids(objects_per_image="0"), {self.only_missing.pk, self.placeholder.pk})

    def test_object_count_includes_bboxless_objects_and_excludes_placeholders(self):
        """Object counts differ from bbox counts and do not depend on other filters."""
        self.assertEqual(self.ids(observations_per_mediafile="2"), {self.boxed.pk, self.missing.pk})
        self.assertEqual(self.ids(observations_per_mediafile="2", has_bbox="false"), {self.missing.pk})
        self.assertEqual(self.ids(observations_per_mediafile="0"), {self.placeholder.pk})
        self.assertEqual(self.ids(observations_per_mediafile_min="2"), {self.boxed.pk, self.missing.pk})
        self.assertEqual(self.ids(observations_per_mediafile_max="1"), {self.only_missing.pk, self.placeholder.pk})
        self.assertEqual(self.ids(observation_type="placeholder"), {self.placeholder.pk})
        self.assertEqual(self.ids(observation_type="object"), {self.boxed.pk, self.missing.pk, self.only_missing.pk})

    def test_incomplete_or_zero_size_bbox_is_missing_geometry(self):
        """A center value alone is not a usable bbox."""
        AnimalObservation.objects.filter(pk=self.only_missing.pk).update(bbox_x_center=0.5, bbox_width=0)
        self.assertIn(self.only_missing.pk, self.ids(has_bbox="false", objects_per_image="0"))
        self.assertNotIn(self.only_missing.pk, self.ids(has_bbox="true"))

    def test_legacy_multiple_filter_and_new_labels(self):
        """Keep bookmarked count filters while removing the redundant UI field."""
        self.assertEqual(self.ids(multiple_in_mediafile="true"), set())
        self.assertEqual(self.ids(multiple_in_mediafile="false"), set(self.queryset.values_list("pk", flat=True)))
        self.client.force_login(self.user.user)
        response = self.client.get(reverse("caidapp:observations"))
        self.assertContains(response, "BBox of this observation")
        self.assertContains(response, "Counts in the containing media file")
        self.assertContains(response, "Object observations per media file")
        self.assertNotContains(response, "Image has multiple observations")
        self.assertNotContains(response, 'name="multiple_in_mediafile"')
