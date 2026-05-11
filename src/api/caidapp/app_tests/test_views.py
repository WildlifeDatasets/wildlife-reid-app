import logging
from io import StringIO

from caidapp import models
from caidapp import views
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
import pandas as pd

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

logger = logging.getLogger(__name__)


User = get_user_model()


class BasicFlowTest(TestCase):
    def setUp(self):
        """Initial setup for tests."""
        self.user = User.objects.create_user(
            username="testuser",
            password="secret123",
        )
        self.client = self.client  # vestavěný testovací klient
        self.wg = models.WorkGroup.objects.create(name="WG1")

    def test_login_and_view(self):
        """Do login and access to the home page."""
        # login
        login = self.client.login(username="testuser", password="secret123")
        self.assertTrue(login)  # ověř, že přihlášení fungovalo

        # přístup na redirect page
        response = self.client.get(reverse("caidapp:index"))
        self.assertEqual(response.status_code, 302)
        # self.assertContains(response, "WS1")

        response = self.client.get(reverse("caidapp:home"))
        self.assertEqual(response.status_code, 200)


class MediafileExportTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory()
        self.user = self.caiduser.user
        self.client.login(username=self.user.username, password="test123")

    def test_csv_export_uses_get_filter_scope(self):
        taxon_wolf = TaxonFactory(name="Wolf")
        taxon_lynx = TaxonFactory(name="Lynx")
        archive = UploadedArchiveFactory(owner=self.caiduser)
        locality = LocalityFactory(owner=self.caiduser)

        mediafile_a = MediaFileFactory(parent=archive, locality=locality, original_filename="wolf.jpg")
        mediafile_b = MediaFileFactory(parent=archive, locality=locality, original_filename="lynx.jpg")
        AnimalObservationFactory(mediafile=mediafile_a, taxon=taxon_wolf, identity=None)
        AnimalObservationFactory(mediafile=mediafile_b, taxon=taxon_lynx, identity=None)

        response = self.client.get(
            reverse("caidapp:download_csv_for_mediafiles"),
            {"taxon": taxon_wolf.id},
        )

        self.assertEqual(response.status_code, 200)
        df = pd.read_csv(StringIO(response.content.decode()))
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["original_path"], "wolf.jpg")

    def test_export_path_uses_observations(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        locality = LocalityFactory(owner=self.caiduser, name="North Meadow")
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha Female")
        taxon = TaxonFactory(name="Canis lupus")
        mediafile = MediaFileFactory(
            parent=archive,
            locality=locality,
            original_filename="wolf.jpg",
        )
        AnimalObservationFactory(mediafile=mediafile, taxon=taxon, identity=identity)

        path = views._render_mediafile_export_path(
            "{species}/{identity}/{hash}_{species}_{identity}{dotext}",
            mediafile,
        )

        self.assertIn("Canis_lupus", path)
        self.assertIn("Alpha_Female", path)
        self.assertTrue(path.endswith(".jpg"))


class SequenceViewTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory()
        self.user = self.caiduser.user
        self.client.login(username=self.user.username, password="test123")

    def test_sequence_view_groups_mediafiles_by_sequence(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        locality = LocalityFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        first_mediafile = MediaFileFactory(
            parent=archive,
            locality=locality,
            sequence=sequence,
            original_filename="first.jpg",
            captured_at=pd.Timestamp("2024-01-01T10:00:00Z").to_pydatetime(),
        )
        second_mediafile = MediaFileFactory(
            parent=archive,
            locality=locality,
            sequence=sequence,
            original_filename="second.jpg",
            captured_at=pd.Timestamp("2024-01-01T10:01:00Z").to_pydatetime(),
        )
        AnimalObservationFactory(mediafile=first_mediafile)
        AnimalObservationFactory(mediafile=second_mediafile)

        response = self.client.get(reverse("caidapp:sequences"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Sequence")
        self.assertContains(response, "first.jpg")
        self.assertContains(response, "second.jpg")

    def test_sequence_view_shows_primary_locality_and_expandable_extra_count(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        locality_primary = LocalityFactory(owner=self.caiduser, name="Xandovice")
        locality_secondary = LocalityFactory(owner=self.caiduser, name="Ypovice")
        sequence = SequenceFactory(uploaded_archive=archive)
        MediaFileFactory(parent=archive, locality=locality_primary, sequence=sequence, original_filename="first.jpg")
        MediaFileFactory(parent=archive, locality=locality_primary, sequence=sequence, original_filename="second.jpg")
        MediaFileFactory(parent=archive, locality=locality_secondary, sequence=sequence, original_filename="third.jpg")

        response = self.client.get(reverse("caidapp:sequences"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Xandovice")
        self.assertContains(response, "+1")
        self.assertContains(response, "Ypovice")


class IdentificationUploadsViewTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory()
        self.user = self.caiduser.user
        self.client.login(username=self.user.username, password="test123")

    def test_uploads_identities_uses_is_for_identification_without_taxon_requirement(self):
        visible_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            name="Visible identification upload",
            is_for_identification=True,
            contains_identities=False,
            taxon_for_identification=None,
        )
        UploadedArchiveFactory(
            owner=self.caiduser,
            name="Hidden non-identification upload",
            is_for_identification=False,
            contains_identities=False,
            taxon_for_identification=None,
        )

        response = self.client.get(reverse("caidapp:uploads_identities"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, visible_archive.name)
        self.assertNotContains(response, "Hidden non-identification upload")

    def test_uploads_identities_include_base_dataset_when_show_base_dataset_is_disabled(self):
        self.caiduser.show_base_dataset = False
        self.caiduser.save()
        UploadedArchiveFactory(
            owner=self.caiduser,
            name="Visible base dataset upload",
            is_for_identification=True,
            contains_identities=True,
            taxon_for_identification=None,
        )
        UploadedArchiveFactory(
            owner=self.caiduser,
            name="Visible ordinary identification upload",
            is_for_identification=True,
            contains_identities=False,
            taxon_for_identification=None,
        )

        response = self.client.get(reverse("caidapp:uploads_identities"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Visible base dataset upload")
        self.assertContains(response, "Visible ordinary identification upload")
        self.assertContains(response, "bi-star-fill")

    def test_uploads_identities_hide_base_dataset_when_show_base_dataset_is_enabled(self):
        self.caiduser.show_base_dataset = True
        self.caiduser.save()
        UploadedArchiveFactory(
            owner=self.caiduser,
            name="Hidden base dataset upload",
            is_for_identification=True,
            contains_identities=True,
            taxon_for_identification=None,
        )
        UploadedArchiveFactory(
            owner=self.caiduser,
            name="Visible ordinary identification upload",
            is_for_identification=True,
            contains_identities=False,
            taxon_for_identification=None,
        )

        response = self.client.get(reverse("caidapp:uploads_identities"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Hidden base dataset upload")
        self.assertContains(response, "Visible ordinary identification upload")

    def test_uploads_known_identities_uses_is_for_identification_without_taxon_requirement(self):
        visible_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            name="Visible base dataset upload",
            is_for_identification=True,
            contains_identities=True,
            taxon_for_identification=None,
        )
        UploadedArchiveFactory(
            owner=self.caiduser,
            name="Hidden taxonomy-only upload",
            is_for_identification=False,
            contains_identities=True,
            taxon_for_identification=None,
        )

        response = self.client.get(reverse("caidapp:uploads_known_identities"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, visible_archive.name)
        self.assertNotContains(response, "Hidden taxonomy-only upload")

    # def test_create_workstation(self):
    #     url = reverse("workstation-create")
    #     data = {
    #         "name": "TestWS",
    #         "description": "Workstation description",
    #         "group": self.group.pk,
    #         # "takt_time_seconds": 12.5,
    #     }
    #
    #     response = self.client.post(url, data)
    #
    #     # CreateView typicky přesměruje po success
    #     self.assertEqual(response.status_code, 302)
    #
    #     # ověř, že objekt je v DB
    #     ws = Workstation.objects.get(name="TestWS")
    #     self.assertEqual(ws.group, self.group)
    #     # self.assertEqual(ws.takt_time_seconds, 12.5)
    #
    #
    # def test_create_workstation_for_this_ip(self):
    #     url = reverse("workstation-create") + "?this_ip=True"
    #     data = {
    #         "name": "TestWS3",
    #         "description": "Workstation description",
    #         "group": self.group.pk,
    #         # "takt_time_seconds": 12.5,
    #     }
    #
    #     response = self.client.post(url, data)
    #
    #     # CreateView typicky přesměruje po success
    #     self.assertEqual(response.status_code, 302)
    #
    #     # ověř, že objekt je v DB
    #     ws = Workstation.objects.get(name="TestWS3")
    #     logger.debug(f"Workstation created with IP: {ws.ip_address}")
    #     self.assertEqual(ws.group, self.group)
    #     # self.assertEqual(ws.takt_time_seconds, 12.5)
