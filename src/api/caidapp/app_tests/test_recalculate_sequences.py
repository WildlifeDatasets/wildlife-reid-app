from datetime import timedelta
from io import StringIO

from caidapp.app_tests.factories import LocalityFactory, UploadedArchiveFactory
from caidapp.models import MediaFile, Sequence
from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone


class RecalculateSequencesTest(TestCase):
    def test_retimed_media_bridge_sequences_without_crossing_localities(self):
        """Merge groups after a time correction and keep separate locality groups."""
        upload = UploadedArchiveFactory()
        locality = LocalityFactory(owner=upload.owner)
        other_locality = LocalityFactory(owner=upload.owner)
        start = timezone.now()

        def add_media(offset, locality):
            return MediaFile.objects.create(
                parent=upload,
                locality=locality,
                captured_at=None if offset is None else start + timedelta(seconds=offset),
                metadata_json={"sequence_number": 999},
            )

        first = add_media(0, locality)
        last = add_media(180, locality)
        moved = add_media(360, locality)
        other = add_media(90, other_locality)
        undated = add_media(None, locality)
        upload.make_sequences()
        first.refresh_from_db()
        first_sequence_id = first.sequence_id
        self.assertEqual(Sequence.objects.filter(uploaded_archive=upload).count(), 5)

        MediaFile.objects.filter(pk=moved.pk).update(captured_at=start + timedelta(seconds=90))
        call_command("recalculate_sequences", upload.pk, stdout=StringIO())

        for media in (first, last, moved, other, undated):
            media.refresh_from_db()
        self.assertEqual(first.sequence_id, first_sequence_id)
        self.assertEqual(last.sequence_id, first_sequence_id)
        self.assertEqual(moved.sequence_id, first_sequence_id)
        self.assertEqual(len({first.sequence_id, other.sequence_id, undated.sequence_id}), 3)
        self.assertEqual(Sequence.objects.filter(uploaded_archive=upload).count(), 3)
