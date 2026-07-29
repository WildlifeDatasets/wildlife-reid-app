from io import StringIO

from django.core.management import call_command
from django.test import TestCase

from .factories import AnimalObservationFactory, CaidUserFactory, IndividualIdentityFactory, MediaFileFactory, UploadedArchiveFactory


class AuditMediaFileAnimalLegacyCommandTest(TestCase):
    def test_reports_legacy_rows_without_changing_them(self):
        caiduser = CaidUserFactory()
        archive = UploadedArchiveFactory(owner=caiduser)
        identity = IndividualIdentityFactory(owner_workgroup=caiduser.workgroup)
        mediafile = MediaFileFactory(parent=archive, identity=identity)
        AnimalObservationFactory(mediafile=mediafile, identity=None)
        stdout = StringIO()

        call_command("audit_mediafile_animal_legacy", stdout=stdout)

        output = stdout.getvalue()
        self.assertIn("with_meaningful_legacy_data: 1", output)
        self.assertIn("legacy_with_one_observation_mismatch: 1", output)
        mediafile.refresh_from_db()
        self.assertEqual(mediafile.identity, identity)
