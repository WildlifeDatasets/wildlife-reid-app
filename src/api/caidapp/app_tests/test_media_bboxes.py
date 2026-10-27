from caidapp.models import AnimalObservation
from caidapp.services.media_bboxes import mediafile_bbox_unions
from django.test import TestCase

from .factories import (
    AnimalObservationFactory,
    CaidUserFactory,
    IndividualIdentityFactory,
    MediaFileFactory,
    UploadedArchiveFactory,
)


class MediaBBoxUnionTest(TestCase):
    def setUp(self):
        """Create an image with multiple observations belonging to distinct identities."""
        self.user = CaidUserFactory(admin=True)
        self.archive = UploadedArchiveFactory(owner=self.user)
        self.media = MediaFileFactory(parent=self.archive, media_type="image")
        self.first, self.second = [IndividualIdentityFactory(owner_workgroup=self.user.workgroup) for _ in range(2)]

    def observation(self, identity, box, **kwargs):
        """Create an observation with explicit normalized bbox coordinates."""
        return AnimalObservationFactory(
            mediafile=self.media,
            identity=identity,
            bbox_x_center=box[0],
            bbox_y_center=box[1],
            bbox_width=box[2],
            bbox_height=box[3],
            **kwargs
        )

    def assertBox(self, actual, expected):
        """Compare normalized boxes without imposing floating-point arithmetic equality."""
        self.assertEqual(len(actual), 4)
        for value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(value, expected_value)

    def test_union_covers_all_valid_boxes_and_can_be_scoped_to_one_identity(self):
        """Use extreme edges across boxes while keeping identity-specific previews independent."""
        self.observation(self.first, [0.2, 0.3, 0.2, 0.2])
        self.observation(self.first, [0.7, 0.6, 0.2, 0.4])
        self.observation(self.second, [0.5, 0.5, 1, 1])

        with self.assertNumQueries(1):
            all_boxes = mediafile_bbox_unions([self.media.pk])
        identity_boxes = mediafile_bbox_unions([self.media.pk], by_identity=True)

        self.assertBox(all_boxes[self.media.pk], [0.5, 0.5, 1, 1])
        self.assertBox(identity_boxes[(self.media.pk, self.first.pk)], [0.45, 0.5, 0.7, 0.6])
        self.assertBox(identity_boxes[(self.media.pk, self.second.pk)], [0.5, 0.5, 1, 1])

    def test_invalid_boxes_and_placeholders_do_not_enlarge_the_union(self):
        """Ignore missing, nonfinite, nonpositive, off-image and placeholder boxes."""
        self.observation(self.first, [0.4, 0.5, 0.2, 0.3])
        for box in (
            [None, 0.5, 1, 1],
            [float("inf"), 0.5, 1, 1],
            [0.5, 0.5, 0, 1],
            [0.5, 0.5, -1, 1],
            [3, 3, 0.1, 0.1],
        ):
            self.observation(self.first, box)
        AnimalObservation.objects.bulk_create(
            [
                AnimalObservation(
                    mediafile=self.media,
                    identity=self.first,
                    is_no_detection_placeholder=True,
                    bbox_x_center=0.5,
                    bbox_y_center=0.5,
                    bbox_width=1,
                    bbox_height=1,
                )
            ]
        )

        self.assertBox(mediafile_bbox_unions([self.media.pk])[self.media.pk], [0.4, 0.5, 0.2, 0.3])

    def test_boxes_are_clipped_to_the_image_and_video_boxes_are_not_combined(self):
        """Clamp visible image edges and avoid combining video boxes without frame positions."""
        self.observation(self.first, [-0.05, 0.5, 0.3, 0.4])
        video = MediaFileFactory(parent=self.archive, media_type="video")
        AnimalObservationFactory(mediafile=video, bbox_x_center=0.5, bbox_y_center=0.5, bbox_width=0.4, bbox_height=0.4)

        boxes = mediafile_bbox_unions([self.media.pk, video.pk])

        self.assertBox(boxes[self.media.pk], [0.05, 0.5, 0.1, 0.4])
        self.assertNotIn(video.pk, boxes)
