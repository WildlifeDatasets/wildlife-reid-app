import logging
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from caidapp import models
from caidapp import tasks
from caidapp import views
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse
import pandas as pd

from .factories import (
    AlbumFactory,
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
        self.assertContains(response, "Dashboard")
        self.assertContains(response, "Monthly Images vs Videos")


class HomeDashboardSnapshotTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory()
        self.user = self.caiduser.user
        self.client.login(username=self.user.username, password="test123")

    def test_refresh_home_dashboard_command_creates_snapshot(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        wolf = TaxonFactory(name="Wolf")
        image_mediafile = MediaFileFactory(parent=archive, media_type="image")
        video_mediafile = MediaFileFactory(parent=archive, media_type="video", original_filename="clip.mp4")
        AnimalObservationFactory(mediafile=image_mediafile, taxon=wolf, taxon_verified=True)
        AnimalObservationFactory(mediafile=video_mediafile, taxon=wolf, taxon_verified=True)

        output = StringIO()
        call_command("refresh_home_dashboard_stats", workgroup_id=self.caiduser.workgroup.id, stdout=output)

        snapshot = models.HomeDashboardSnapshot.objects.get(workgroup=self.caiduser.workgroup)
        summary = snapshot.payload["summary"]
        self.assertEqual(summary["total_mediafiles"], 2)
        self.assertEqual(summary["images_count"], 1)
        self.assertEqual(summary["videos_count"], 1)
        self.assertIn("Refreshed 1 home dashboard snapshot", output.getvalue())

    def test_home_view_uses_persisted_snapshot_when_available(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        wolf = TaxonFactory(name="Wolf")
        mediafile = MediaFileFactory(parent=archive, media_type="image")
        AnimalObservationFactory(mediafile=mediafile, taxon=wolf, taxon_verified=True)
        call_command("refresh_home_dashboard_stats", workgroup_id=self.caiduser.workgroup.id)

        response = self.client.get(reverse("caidapp:home"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "daily snapshot")
        self.assertContains(response, "Top Taxa by Media Files")
        self.assertContains(response, "Monthly Images vs Videos")


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


class MediaFileUpdateEmptyObservationTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory()
        self.user = self.caiduser.user
        self.client.login(username=self.user.username, password="test123")

    def _base_mediafile_update_post_data(self, mediafile, *, total_forms, initial_forms, extra_form_data=None):
        data = {
            "next": "",
            "observations-TOTAL_FORMS": str(total_forms),
            "observations-INITIAL_FORMS": str(initial_forms),
            "observations-MIN_NUM_FORMS": "0",
            "observations-MAX_NUM_FORMS": "1000",
            "locality": "",
            "location_0": "",
            "location_1": "",
            "captured_at": "",
            "note": "",
        }
        if extra_form_data:
            data.update(extra_form_data)
        return data

    def test_mark_empty_image_creates_nothing_observation_when_none_exist(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive)

        response = self.client.post(
            reverse("caidapp:media_file_update", args=[mediafile.id]),
            self._base_mediafile_update_post_data(
                mediafile,
                total_forms=0,
                initial_forms=0,
                extra_form_data={"mark_empty_image": "1"},
            ),
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("caidapp:media_file_update", args=[mediafile.id]))
        mediafile.refresh_from_db()
        observations = list(mediafile.observations.all())
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].taxon.name, "Nothing")
        self.assertTrue(observations[0].taxon_verified)
        self.assertIsNone(observations[0].bbox_x_center)
        self.assertIsNone(observations[0].bbox_y_center)
        self.assertIsNone(observations[0].bbox_width)
        self.assertIsNone(observations[0].bbox_height)

    def test_mark_empty_image_replaces_existing_observations_with_nothing(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive)
        taxon_wolf = TaxonFactory(name="Wolf")
        taxon_lynx = TaxonFactory(name="Lynx")
        obs_one = AnimalObservationFactory(mediafile=mediafile, taxon=taxon_wolf)
        obs_two = AnimalObservationFactory(mediafile=mediafile, taxon=taxon_lynx)

        response = self.client.post(
            reverse("caidapp:media_file_update", args=[mediafile.id]),
            self._base_mediafile_update_post_data(
                mediafile,
                total_forms=2,
                initial_forms=2,
                extra_form_data={
                    "observations-0-id": str(obs_one.id),
                    "observations-0-mediafile": str(mediafile.id),
                    "observations-0-taxon": str(taxon_wolf.id),
                    "observations-0-taxon_verified": "on",
                    "observations-0-identity": "",
                    "observations-0-identity_is_representative": "",
                    "observations-0-orientation": "N",
                    "observations-0-bbox_x_center": "",
                    "observations-0-bbox_y_center": "",
                    "observations-0-bbox_width": "",
                    "observations-0-bbox_height": "",
                    "observations-0-DELETE": "",
                    "observations-1-id": str(obs_two.id),
                    "observations-1-mediafile": str(mediafile.id),
                    "observations-1-taxon": str(taxon_lynx.id),
                    "observations-1-taxon_verified": "on",
                    "observations-1-identity": "",
                    "observations-1-identity_is_representative": "",
                    "observations-1-orientation": "N",
                    "observations-1-bbox_x_center": "",
                    "observations-1-bbox_y_center": "",
                    "observations-1-bbox_width": "",
                    "observations-1-bbox_height": "",
                    "observations-1-DELETE": "",
                    "mark_empty_image": "1",
                },
            ),
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("caidapp:media_file_update", args=[mediafile.id]))
        mediafile.refresh_from_db()
        observations = list(mediafile.observations.order_by("id"))
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].taxon.name, "Nothing")
        self.assertTrue(observations[0].taxon_verified)
        self.assertIsNone(observations[0].bbox_x_center)
        self.assertIsNone(observations[0].bbox_y_center)
        self.assertIsNone(observations[0].bbox_width)
        self.assertIsNone(observations[0].bbox_height)


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

    def test_sequence_view_has_collapsed_bulk_controls_and_expand_all_actions(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        one_file_sequence = SequenceFactory(uploaded_archive=archive)
        two_file_sequence = SequenceFactory(uploaded_archive=archive)
        three_file_sequence = SequenceFactory(uploaded_archive=archive)
        MediaFileFactory(parent=archive, sequence=one_file_sequence, original_filename="one.jpg")
        MediaFileFactory(parent=archive, sequence=two_file_sequence, original_filename="two-a.jpg")
        MediaFileFactory(parent=archive, sequence=two_file_sequence, original_filename="two-b.jpg")
        MediaFileFactory(parent=archive, sequence=three_file_sequence, original_filename="three-a.jpg")
        MediaFileFactory(parent=archive, sequence=three_file_sequence, original_filename="three-b.jpg")
        MediaFileFactory(parent=archive, sequence=three_file_sequence, original_filename="three-c.jpg")

        response = self.client.get(reverse("caidapp:sequences"))
        content = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'id="sequence-bulk-processing"')
        self.assertContains(response, "js-expand-all-sequences")
        self.assertContains(response, "js-collapse-all-sequences")
        self.assertContains(response, "Select whole sequence")
        self.assertNotContains(response, "Select all media files in sequence")
        self.assertContains(response, ".sequence-card-expanded{width:100%;max-width:100%}")
        self.assertContains(response, ".sequence-card-expanded.sequence-card-expanded-medium{grid-column:span 2;width:100%}")
        self.assertContains(response, ".sequence-card-expanded:not(.sequence-card-expanded-medium):not(.sequence-card-expanded-wide) .sequence-mediafile-card{flex:1 1 100%;width:100%;max-width:none}")
        self.assertContains(response, ".sequence-card-expanded .card-body{padding:.5rem}")
        self.assertContains(response, "border-top-left-radius:var(--bs-card-inner-border-radius)")
        self.assertContains(response, "background:var(--wrid-sequence-bg)")
        self.assertContains(response, ".sequence-cover{object-fit:contain;aspect-ratio:4/3;width:100%;background:var(--wrid-sequence-thumb-bg)")
        self.assertContains(response, "background:var(--wrid-media-card-bg)")
        self.assertContains(response, "border-color:var(--wrid-media-card-border-color)")
        self.assertContains(response, ".sequence-card-expanded>.sequence-card-footer-icon{display:none}")
        self.assertContains(response, "justify-content-start align-items-center gap-2")
        self.assertNotIn(f'id="sequence-card-{one_file_sequence.id}" class="sequence-card-expanded', content)
        self.assertContains(response, f"id=\"sequence-card-{two_file_sequence.id}\"")
        self.assertContains(response, "sequence-card-expanded-medium")
        self.assertContains(response, f"id=\"sequence-card-{three_file_sequence.id}\"")
        self.assertContains(response, "sequence-card-expanded-wide")

    def test_sequence_download_starts_from_selected_sequences(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        selected_sequence = SequenceFactory(uploaded_archive=archive)
        other_sequence = SequenceFactory(uploaded_archive=archive)
        selected_mediafile = MediaFileFactory(parent=archive, sequence=selected_sequence, original_filename="selected.jpg")
        other_mediafile = MediaFileFactory(parent=archive, sequence=other_sequence, original_filename="other.jpg")

        response = self.client.post(
            reverse("caidapp:sequences"),
            {
                "btnDownloadSequences": "1",
                "selected_sequence_ids": [str(selected_sequence.id)],
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("caidapp:download_sequences"))
        self.assertEqual(self.client.session["sequence_download_mediafile_ids"], [selected_mediafile.id])
        self.assertNotIn(other_mediafile.id, self.client.session["sequence_download_mediafile_ids"])

    def test_sequence_csv_export_uses_one_row_per_observation(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        taxon_wolf = TaxonFactory(name="Wolf")
        taxon_lynx = TaxonFactory(name="Lynx")
        observed_mediafile = MediaFileFactory(parent=archive, sequence=sequence, original_filename="observed.jpg")
        empty_mediafile = MediaFileFactory(parent=archive, sequence=sequence, original_filename="empty.jpg")
        AnimalObservationFactory(mediafile=observed_mediafile, taxon=taxon_wolf)
        AnimalObservationFactory(mediafile=observed_mediafile, taxon=taxon_lynx)
        session = self.client.session
        session["sequence_download_mediafile_ids"] = [observed_mediafile.id, empty_mediafile.id]
        session.save()

        response = self.client.get(reverse("caidapp:download_csv_for_sequences"))

        self.assertEqual(response.status_code, 200)
        df = pd.read_csv(StringIO(response.content.decode()))
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df["original_path"]), ["observed.jpg", "observed.jpg", "empty.jpg"])
        self.assertEqual(set(df[df["original_path"] == "observed.jpg"]["predicted_category"]), {"Wolf", "Lynx"})
        self.assertTrue(pd.isna(df[df["original_path"] == "empty.jpg"].iloc[0]["observation_id"]))

    def test_sequence_view_accepts_uploadedarchive_filter_alias(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        mediafile = MediaFileFactory(parent=archive, sequence=sequence, original_filename="first.jpg")
        other_archive = UploadedArchiveFactory(owner=self.caiduser)
        other_sequence = SequenceFactory(uploaded_archive=other_archive)
        MediaFileFactory(parent=other_archive, sequence=other_sequence, original_filename="second.jpg")

        response = self.client.get(
            reverse("caidapp:sequences"),
            {
                "search": "",
                "media_type": "",
                "orientation": "",
                "identity_is_representative": "unknown",
                "taxon": "",
                "uploadedarchive": str(archive.id),
                "captured_at_min": "",
                "captured_at_max": "",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, mediafile.original_filename)
        self.assertNotContains(response, "second.jpg")

    def test_sequence_view_shows_active_uploadedarchive_link(self):
        archive = UploadedArchiveFactory(owner=self.caiduser, name="Visible upload")
        sequence = SequenceFactory(uploaded_archive=archive)
        MediaFileFactory(parent=archive, sequence=sequence, original_filename="first.jpg")

        response = self.client.get(reverse("caidapp:sequences"), {"uploadedarchive_id": archive.id})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Filtered by:")
        self.assertContains(response, "Upload: Visible upload")
        self.assertContains(response, reverse("caidapp:uploadedarchive_detail", args=[archive.id]))

    def test_sequence_view_rejects_uploadedarchive_outside_user_workgroup(self):
        other_caiduser = CaidUserFactory()
        other_archive = UploadedArchiveFactory(owner=other_caiduser, name="Private outside upload")
        sequence = SequenceFactory(uploaded_archive=other_archive)
        MediaFileFactory(parent=other_archive, sequence=sequence, original_filename="private.jpg")

        response = self.client.get(reverse("caidapp:sequences"), {"uploadedarchive_id": other_archive.id})

        self.assertEqual(response.status_code, 404)
        self.assertNotContains(response, "Private outside upload", status_code=404)

    def test_sequence_view_shows_active_taxon_link_and_filters_accessible_mediafiles(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        wolf = TaxonFactory(name="Wolf")
        lynx = TaxonFactory(name="Lynx")
        wolf_sequence = SequenceFactory(uploaded_archive=archive)
        lynx_sequence = SequenceFactory(uploaded_archive=archive)
        wolf_mediafile = MediaFileFactory(parent=archive, sequence=wolf_sequence, original_filename="wolf.jpg")
        lynx_mediafile = MediaFileFactory(parent=archive, sequence=lynx_sequence, original_filename="lynx.jpg")
        AnimalObservationFactory(mediafile=wolf_mediafile, taxon=wolf)
        AnimalObservationFactory(mediafile=lynx_mediafile, taxon=lynx)

        response = self.client.get(reverse("caidapp:sequences"), {"taxon": wolf.id})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Taxon: Wolf")
        self.assertContains(response, f'{reverse("caidapp:media_files")}?taxon={wolf.id}')
        self.assertContains(response, "wolf.jpg")
        self.assertNotContains(response, "lynx.jpg")

    def test_sequence_view_shows_active_identity_link(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")
        other_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Beta")
        sequence = SequenceFactory(uploaded_archive=archive)
        other_sequence = SequenceFactory(uploaded_archive=archive)
        mediafile = MediaFileFactory(parent=archive, sequence=sequence, identity=identity, original_filename="alpha.jpg")
        other_mediafile = MediaFileFactory(parent=archive, sequence=other_sequence, identity=other_identity, original_filename="beta.jpg")

        response = self.client.get(reverse("caidapp:sequences"), {"individual_identity_id": identity.id})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Identity: Alpha")
        self.assertContains(response, reverse("caidapp:individual_identity_mediafiles", args=[identity.id]))
        self.assertContains(response, mediafile.original_filename)
        self.assertNotContains(response, other_mediafile.original_filename)

    def test_sequence_view_rejects_identity_outside_user_workgroup(self):
        other_caiduser = CaidUserFactory()
        identity = IndividualIdentityFactory(owner_workgroup=other_caiduser.workgroup, name="Private identity")

        response = self.client.get(reverse("caidapp:sequences"), {"individual_identity_id": identity.id})

        self.assertEqual(response.status_code, 404)
        self.assertNotContains(response, "Private identity", status_code=404)

    def test_sequence_view_shows_active_locality_link(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        locality = LocalityFactory(owner=self.caiduser, name="Visible locality")
        other_locality = LocalityFactory(owner=self.caiduser, name="Other locality")
        sequence = SequenceFactory(uploaded_archive=archive)
        other_sequence = SequenceFactory(uploaded_archive=archive)
        mediafile = MediaFileFactory(parent=archive, sequence=sequence, locality=locality, original_filename="locality.jpg")
        other_mediafile = MediaFileFactory(parent=archive, sequence=other_sequence, locality=other_locality, original_filename="other-locality.jpg")

        response = self.client.get(reverse("caidapp:sequences"), {"locality_hash": locality.hash})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Locality: Visible locality")
        self.assertContains(response, reverse("caidapp:media_files_locality", args=[locality.hash]))
        self.assertContains(response, mediafile.original_filename)
        self.assertNotContains(response, other_mediafile.original_filename)

    def test_sequence_view_rejects_locality_outside_user_workgroup(self):
        other_caiduser = CaidUserFactory()
        locality = LocalityFactory(owner=other_caiduser, name="Private locality")

        response = self.client.get(reverse("caidapp:sequences"), {"locality_hash": locality.hash})

        self.assertEqual(response.status_code, 404)
        self.assertNotContains(response, "Private locality", status_code=404)

    def test_sequence_view_shows_active_album_link(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        album = AlbumFactory(owner=self.caiduser, name="Visible album")
        other_album = AlbumFactory(owner=self.caiduser, name="Other album")
        sequence = SequenceFactory(uploaded_archive=archive)
        other_sequence = SequenceFactory(uploaded_archive=archive)
        mediafile = MediaFileFactory(parent=archive, sequence=sequence, original_filename="album.jpg")
        other_mediafile = MediaFileFactory(parent=archive, sequence=other_sequence, original_filename="other-album.jpg")
        album.mediafiles.add(mediafile)
        other_album.mediafiles.add(other_mediafile)

        response = self.client.get(reverse("caidapp:sequences"), {"album_hash": album.hash})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Album: Visible album")
        self.assertContains(response, reverse("caidapp:album", args=[album.hash]))
        self.assertContains(response, mediafile.original_filename)
        self.assertNotContains(response, other_mediafile.original_filename)

    def test_sequence_view_rejects_album_outside_user_access(self):
        other_caiduser = CaidUserFactory()
        album = AlbumFactory(owner=other_caiduser, name="Private album")

        response = self.client.get(reverse("caidapp:sequences"), {"album_hash": album.hash})

        self.assertEqual(response.status_code, 404)
        self.assertNotContains(response, "Private album", status_code=404)

    def test_taxon_list_links_to_sequences_for_taxon(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        wolf = TaxonFactory(name="Wolf")
        sequence = SequenceFactory(uploaded_archive=archive)
        mediafile = MediaFileFactory(parent=archive, sequence=sequence)
        AnimalObservationFactory(mediafile=mediafile, taxon=wolf)

        response = self.client.get(reverse("caidapp:show_taxons"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'{reverse("caidapp:sequences")}?taxon={wolf.id}')

    def test_identity_locality_and_album_lists_link_to_sequences(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")
        locality = LocalityFactory(owner=self.caiduser, name="Visible locality")
        album = AlbumFactory(owner=self.caiduser, name="Visible album")
        sequence = SequenceFactory(uploaded_archive=archive)
        mediafile = MediaFileFactory(parent=archive, sequence=sequence, identity=identity, locality=locality)
        album.mediafiles.add(mediafile)

        response = self.client.get(reverse("caidapp:individual_identities"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'{reverse("caidapp:sequences")}?individual_identity_id={identity.id}')

        response = self.client.get(reverse("caidapp:localities"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'{reverse("caidapp:sequences")}?locality_hash={locality.hash}')

        response = self.client.get(reverse("caidapp:albums"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'{reverse("caidapp:sequences")}?album_hash={album.hash}')

    def test_sequence_filename_metadata_uses_selected_mediafiles(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        selected_mediafile = MediaFileFactory(
            parent=archive,
            sequence=sequence,
            original_filename="Brdy/Charles/first.jpg",
        )
        other_mediafile = MediaFileFactory(
            parent=archive,
            sequence=sequence,
            original_filename="Brdy/Other/second.jpg",
        )

        response = self.client.post(
            reverse("caidapp:sequences"),
            {
                "selected_mediafile_ids": [str(selected_mediafile.id)],
                "btnExtractFilenameMetadata": "1",
            },
        )

        self.assertRedirects(response, reverse("caidapp:apply_filename_metadata_to_mediafiles"))

        response = self.client.post(
            reverse("caidapp:apply_filename_metadata_to_mediafiles"),
            {
                "path_regex": r"^(?:.*/)?(?P<locality>[^/]+)/(?P<unique_name>[^/]+)/[^/]+$",
            },
        )

        self.assertEqual(response.status_code, 302)
        selected_mediafile.refresh_from_db()
        other_mediafile.refresh_from_db()
        self.assertEqual(selected_mediafile.locality.name, "Brdy")
        self.assertEqual(selected_mediafile.identity.name, "Charles")
        self.assertIsNone(other_mediafile.identity)

    def test_sequence_filename_metadata_uses_directory_mapping_without_regex(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        mediafile = MediaFileFactory(
            parent=archive,
            sequence=sequence,
            original_filename="Brdy/Lynx/Charles/first.jpg",
        )

        self.client.post(
            reverse("caidapp:sequences"),
            {
                "selected_mediafile_ids": [str(mediafile.id)],
                "btnExtractFilenameMetadata": "1",
            },
        )
        response = self.client.post(
            reverse("caidapp:apply_filename_metadata_to_mediafiles"),
            {
                "directory_mapping": '{"locality": 0, "taxon": 1, "unique_name": 2}',
                "path_regex": "",
            },
        )

        self.assertEqual(response.status_code, 302)
        mediafile.refresh_from_db()
        self.assertEqual(mediafile.locality.name, "Brdy")
        self.assertEqual(mediafile.taxon.name, "Lynx")
        self.assertEqual(mediafile.identity.name, "Charles")

    def test_sequence_filename_metadata_chatgpt_prompt_includes_sample_paths(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        mediafiles = [
            MediaFileFactory(parent=archive, sequence=sequence, original_filename=f"Locality/Identity/file_{index}.jpg")
            for index in range(6)
        ]

        self.client.post(
            reverse("caidapp:sequences"),
            {
                "selected_mediafile_ids": [str(mediafile.id) for mediafile in mediafiles],
                "btnExtractFilenameMetadata": "1",
            },
        )
        response = self.client.get(reverse("caidapp:apply_filename_metadata_to_mediafiles"))

        self.assertEqual(response.status_code, 200)
        prompt = response.context["regex_chatgpt_prompt"]
        self.assertIn("Locality/Identity/file_0.jpg", prompt)
        self.assertIn("Locality/Identity/file_4.jpg", prompt)
        self.assertNotIn("Locality/Identity/file_5.jpg", prompt)
        self.assertIn("Use named groups only from: taxon, locality, unique_name", prompt)
        self.assertIn("The legacy group name identity is accepted", prompt)
        self.assertIn("The following is a description of the individual path parts", prompt)
        self.assertContains(response, "Ask ChatGPT")

    def test_sequence_filename_metadata_skips_manually_updated_mediafiles_by_default(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        mediafile = MediaFileFactory(
            parent=archive,
            sequence=sequence,
            original_filename="Brdy/Charles/first.jpg",
            updated_by=self.caiduser,
        )

        self.client.post(
            reverse("caidapp:sequences"),
            {
                "selected_mediafile_ids": [str(mediafile.id)],
                "btnExtractFilenameMetadata": "1",
            },
        )
        self.client.post(
            reverse("caidapp:apply_filename_metadata_to_mediafiles"),
            {
                "path_regex": r"^(?:.*/)?(?P<locality>[^/]+)/(?P<unique_name>[^/]+)/[^/]+$",
            },
        )

        mediafile.refresh_from_db()
        self.assertIsNone(mediafile.locality)
        self.assertIsNone(mediafile.identity)

        self.client.post(
            reverse("caidapp:sequences"),
            {
                "selected_mediafile_ids": [str(mediafile.id)],
                "btnExtractFilenameMetadata": "1",
            },
        )
        self.client.post(
            reverse("caidapp:apply_filename_metadata_to_mediafiles"),
            {
                "path_regex": r"^(?:.*/)?(?P<locality>[^/]+)/(?P<unique_name>[^/]+)/[^/]+$",
                "apply_to_manually_updated": "on",
            },
        )

        mediafile.refresh_from_db()
        self.assertEqual(mediafile.locality.name, "Brdy")
        self.assertEqual(mediafile.identity.name, "Charles")

    def test_sequence_filename_metadata_fills_only_empty_fields_by_default(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        old_locality = LocalityFactory(owner=self.caiduser, name="OldLocality")
        old_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="OldIdentity")
        mediafile = MediaFileFactory(
            parent=archive,
            sequence=sequence,
            locality=old_locality,
            identity=old_identity,
            original_filename="NewLocality/NewIdentity/first.jpg",
        )

        self.client.post(
            reverse("caidapp:sequences"),
            {
                "selected_mediafile_ids": [str(mediafile.id)],
                "btnExtractFilenameMetadata": "1",
            },
        )
        self.client.post(
            reverse("caidapp:apply_filename_metadata_to_mediafiles"),
            {
                "path_regex": r"^(?:.*/)?(?P<locality>[^/]+)/(?P<unique_name>[^/]+)/[^/]+$",
            },
        )

        mediafile.refresh_from_db()
        self.assertEqual(mediafile.locality, old_locality)
        self.assertEqual(mediafile.identity, old_identity)

        self.client.post(
            reverse("caidapp:sequences"),
            {
                "selected_mediafile_ids": [str(mediafile.id)],
                "btnExtractFilenameMetadata": "1",
            },
        )
        self.client.post(
            reverse("caidapp:apply_filename_metadata_to_mediafiles"),
            {
                "path_regex": r"^(?:.*/)?(?P<locality>[^/]+)/(?P<unique_name>[^/]+)/[^/]+$",
                "force_rewrite_filled_data": "on",
            },
        )

        mediafile.refresh_from_db()
        self.assertEqual(mediafile.locality.name, "NewLocality")
        self.assertEqual(mediafile.identity.name, "NewIdentity")

    def test_sequence_filename_metadata_uses_current_filter_when_nothing_is_selected(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        mediafile = MediaFileFactory(
            parent=archive,
            sequence=sequence,
            original_filename="Lynx/Charles/first.jpg",
        )

        self.client.post(
            reverse("caidapp:sequences"),
            {
                "taxon": "",
                "btnExtractFilenameMetadata": "1",
            },
        )
        self.client.post(
            reverse("caidapp:apply_filename_metadata_to_mediafiles"),
            {
                "path_regex": r'r"^(?P<taxon>[^/]+)/(?P<identity>[^/]+)/[^/]+$"',
            },
        )

        mediafile.refresh_from_db()
        self.assertEqual(mediafile.taxon.name, "Lynx")
        self.assertEqual(mediafile.identity.name, "Charles")


class IdentificationUploadsViewTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory()
        self.user = self.caiduser.user
        self.client.login(username=self.user.username, password="test123")

    def test_upload_lists_offer_filename_metadata_extraction(self):
        species_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            contains_single_taxon=False,
            taxon_for_identification=None,
        )
        identity_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
            contains_identities=False,
        )
        known_identity_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
            contains_identities=True,
        )

        response = self.client.get(reverse("caidapp:uploads"))
        self.assertContains(
            response,
            reverse("caidapp:apply_filename_metadata_to_uploadedarchive", args=[species_archive.id]),
        )
        self.assertContains(response, "Extract metadata from filenames")

        response = self.client.get(reverse("caidapp:uploads_identities"))
        self.assertContains(
            response,
            reverse("caidapp:apply_filename_metadata_to_uploadedarchive", args=[identity_archive.id]),
        )

        response = self.client.get(reverse("caidapp:uploads_known_identities"))
        self.assertContains(
            response,
            reverse("caidapp:apply_filename_metadata_to_uploadedarchive", args=[known_identity_archive.id]),
        )

    def test_upload_lists_link_to_sequences_for_archive(self):
        species_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            contains_single_taxon=False,
            taxon_for_identification=None,
        )
        identity_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
            contains_identities=False,
        )
        known_identity_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
            contains_identities=True,
        )

        response = self.client.get(reverse("caidapp:uploads"))
        self.assertContains(response, f'{reverse("caidapp:sequences")}?uploadedarchive_id={species_archive.id}')

        response = self.client.get(reverse("caidapp:uploads_identities"))
        self.assertContains(response, f'{reverse("caidapp:sequences")}?uploadedarchive_id={identity_archive.id}')

        response = self.client.get(reverse("caidapp:uploads_known_identities"))
        self.assertContains(response, f'{reverse("caidapp:sequences")}?uploadedarchive_id={known_identity_archive.id}')

    def test_uploadedarchive_detail_links_to_sequences_and_mediafiles(self):
        TaxonFactory(name=models.TAXON_NOT_CLASSIFIED)
        archive = UploadedArchiveFactory(
            owner=self.caiduser,
            contains_single_taxon=False,
            taxon_for_identification=None,
        )

        response = self.client.get(reverse("caidapp:uploadedarchive_detail", args=[archive.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, archive.name)
        self.assertContains(response, f'{reverse("caidapp:sequences")}?uploadedarchive_id={archive.id}')
        self.assertContains(response, reverse("caidapp:uploadedarchive_mediafiles", args=[archive.id]))
        self.assertContains(response, "bi-three-dots")

    def test_upload_filename_metadata_uses_all_mediafiles_in_archive(self):
        archive = UploadedArchiveFactory(owner=self.caiduser, is_for_identification=True, contains_identities=False)
        other_archive = UploadedArchiveFactory(owner=self.caiduser, is_for_identification=True, contains_identities=False)
        mediafile = MediaFileFactory(parent=archive, original_filename="Brdy/Charles/first.jpg")
        other_mediafile = MediaFileFactory(parent=other_archive, original_filename="Brdy/Other/second.jpg")

        response = self.client.get(
            reverse("caidapp:apply_filename_metadata_to_uploadedarchive", args=[archive.id]),
            {"next": reverse("caidapp:uploads_identities")},
        )

        self.assertRedirects(response, reverse("caidapp:apply_filename_metadata_to_mediafiles"))
        response = self.client.post(
            reverse("caidapp:apply_filename_metadata_to_mediafiles"),
            {
                "path_regex": r"^(?:.*/)?(?P<locality>[^/]+)/(?P<unique_name>[^/]+)/[^/]+$",
            },
        )

        self.assertRedirects(response, reverse("caidapp:uploads_identities"))
        mediafile.refresh_from_db()
        other_mediafile.refresh_from_db()
        self.assertEqual(mediafile.locality.name, "Brdy")
        self.assertEqual(mediafile.identity.name, "Charles")
        self.assertIsNone(other_mediafile.locality)
        self.assertIsNone(other_mediafile.identity)

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

    def test_uploads_identities_include_base_dataset_when_show_base_between_regular_uploads_is_enabled(self):
        self.caiduser.show_base_between_regular_uploads = True
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

    def test_uploads_identities_hide_base_dataset_when_show_base_between_regular_uploads_is_disabled(self):
        self.caiduser.show_base_between_regular_uploads = False
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


class IdentificationRerunTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory(admin=True)
        self.user = self.caiduser.user
        self.client.login(username=self.user.username, password="test123")
        self.workgroup = self.caiduser.workgroup
        self.identification_model = models.IdentificationModel.objects.create(
            name="Test model",
            public=True,
            model_path="/tmp/test-model.pth",
        )
        self.workgroup.identification_model = self.identification_model
        self.workgroup.default_taxon_for_identification = None
        self.workgroup.check_taxon_before_identification = False
        self.workgroup.save()

    @patch("caidapp.views.run_identification_bulk")
    def test_bulk_rerun_processes_identification_uploads_with_missing_identity_regardless_of_status(self, run_identification_bulk_mock):
        run_identification_bulk_mock.return_value = True

        archive_ready = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
            import_finished=True,
            identification_status="C",
        )
        MediaFileFactory(parent=archive_ready, identity=None)
        archive_ready_two = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
            import_finished=True,
            identification_status="IR",
        )
        MediaFileFactory(parent=archive_ready_two, identity=None)

        archive_with_identity = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
            import_finished=True,
            identification_status="IR",
        )
        MediaFileFactory(parent=archive_with_identity, with_identity=True)

        archive_importing = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
            import_finished=False,
            identification_status="IR",
        )
        MediaFileFactory(parent=archive_importing, identity=None)

        archive_outside_identification = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=False,
            import_finished=True,
            identification_status="IR",
        )
        MediaFileFactory(parent=archive_outside_identification, identity=None)

        tasks.run_identification_on_unidentified_for_workgroup(self.workgroup.id)

        run_identification_bulk_mock.assert_called_once()
        called_workgroup = run_identification_bulk_mock.call_args.args[0]
        called_archives = run_identification_bulk_mock.call_args.kwargs["uploaded_archives"]
        self.assertEqual(called_workgroup.id, self.workgroup.id)
        self.assertEqual({archive.id for archive in called_archives}, {archive_ready.id, archive_ready_two.id})

    @patch("caidapp.views.signature")
    @patch("caidapp.views._prepare_dataframe_for_identification")
    def test_run_identification_bulk_dispatches_one_job_for_multiple_uploads(
        self,
        prepare_dataframe_mock,
        signature_mock,
    ):
        archive_one = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
            import_finished=True,
            identification_status="IR",
        )
        mediafile_one = MediaFileFactory(parent=archive_one, identity=None)
        archive_two = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
            import_finished=True,
            identification_status="C",
        )
        mediafile_two = MediaFileFactory(parent=archive_two, identity=None)
        prepare_dataframe_mock.return_value = {
            "image_path": ["one.jpg", "two.jpg"],
            "class_id": [1, 2],
            "label": ["one", "two"],
        }

        signature_result = Mock()
        signature_mock.return_value = signature_result
        identify_task = Mock()
        identify_task.id = "bulk-identify-task-1"
        signature_result.apply_async.return_value = identify_task

        status_ok = views.run_identification_bulk(self.workgroup)

        self.assertTrue(status_ok)
        signature_mock.assert_called_once()
        bulk_kwargs = signature_mock.call_args.kwargs["kwargs"]
        self.assertEqual(bulk_kwargs["organization_id"], self.workgroup.id)
        self.assertNotIn("uploaded_archive_id", bulk_kwargs)
        signature_result.apply_async.assert_called_once()
        callback_kwargs = signature_result.apply_async.call_args.kwargs["link"].kwargs
        self.assertEqual(set(callback_kwargs["uploaded_archive_ids"]), {archive_one.id, archive_two.id})
        archive_one.refresh_from_db()
        archive_two.refresh_from_db()
        self.workgroup.refresh_from_db()
        self.assertEqual(archive_one.identification_status, "IAIP")
        self.assertEqual(archive_two.identification_status, "IAIP")
        self.assertEqual(self.workgroup.identification_reid_status, "Processing")
        self.assertEqual(self.workgroup.identification_scheduled_run_task_id, identify_task.id)
        self.assertIn("Running identification for 2 uploads", self.workgroup.identification_reid_message)

    @patch("caidapp.views.signature")
    @patch("caidapp.views._prepare_dataframe_for_identification")
    def test_run_identification_keeps_existing_queue_until_worker_success(
        self,
        prepare_dataframe_mock,
        signature_mock,
    ):
        archive = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
            import_finished=True,
            identification_status="IAID",
        )
        mediafile = MediaFileFactory(parent=archive, identity=None)
        prepare_dataframe_mock.return_value = {
            "image_path": ["image.jpg"],
            "class_id": [1],
            "label": ["unknown"],
        }

        queue_item = models.MediafilesForIdentification.objects.create(mediafile=mediafile)
        models.MediafileIdentificationSuggestion.objects.create(
            for_identification=queue_item,
            mediafile=mediafile,
            name="old suggestion",
            score=0.4,
        )
        (Path(settings.MEDIA_ROOT) / archive.outputdir).mkdir(parents=True, exist_ok=True)

        signature_result = Mock()
        signature_mock.return_value = signature_result
        signature_result.apply_async.return_value = Mock()

        status_ok = views.run_identification(archive, workgroup=self.workgroup)

        self.assertTrue(status_ok)
        self.assertTrue(models.MediafilesForIdentification.objects.filter(id=queue_item.id).exists())
        self.assertEqual(
            models.MediafileIdentificationSuggestion.objects.filter(for_identification=queue_item).count(),
            1,
        )
        signature_result.apply_async.assert_called_once()

    @patch("caidapp.tasks._prepare_mediafile_for_identification")
    def test_identify_on_success_clears_existing_queue_for_upload_on_success(self, prepare_mediafile_mock):
        archive = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
            import_finished=True,
            identification_status="IAIP",
        )
        mediafile = MediaFileFactory(parent=archive, identity=None)
        queue_item = models.MediafilesForIdentification.objects.create(mediafile=mediafile)
        models.MediafileIdentificationSuggestion.objects.create(
            for_identification=queue_item,
            mediafile=mediafile,
            name="old suggestion",
            score=0.4,
        )
        output_dir = Path(settings.MEDIA_ROOT) / archive.outputdir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json_file = output_dir / "identification_result.json"
        output_json_file.write_text(
            '{"mediafile_ids": [%d], "pred_image_paths": [[]], "pred_class_ids": [[]], "pred_labels": [[]], "scores": [[]], "keypoints": [[]]}'
            % mediafile.id,
            encoding="utf-8",
        )

        tasks.identify_on_success.run(
            {"status": "DONE", "output_json_file": str(output_json_file)},
            uploaded_archive_id=archive.id,
        )

        self.assertFalse(models.MediafilesForIdentification.objects.filter(id=queue_item.id).exists())
        self.assertEqual(
            models.MediafileIdentificationSuggestion.objects.filter(for_identification=queue_item).count(),
            0,
        )
        prepare_mediafile_mock.assert_called_once()

    @patch("caidapp.views.current_app.control.revoke")
    def test_stop_init_identification_revokes_running_reid_task(self, revoke_mock):
        self.workgroup.identification_reid_status = "Processing"
        self.workgroup.identification_scheduled_run_task_id = "reid-task-123"
        self.workgroup.save()

        response = self.client.get(reverse("caidapp:stop_init_identification"))

        self.assertEqual(response.status_code, 302)
        revoke_mock.assert_called_once_with("reid-task-123", terminate=True)
        self.workgroup.refresh_from_db()
        self.assertEqual(self.workgroup.identification_reid_status, "Not initiated")
        self.assertIsNone(self.workgroup.identification_scheduled_run_task_id)
        self.assertIn("stopped manually", self.workgroup.identification_reid_message)

    @patch("caidapp.views.current_app.control.revoke")
    def test_stop_init_identification_revokes_running_init_task(self, revoke_mock):
        self.workgroup.identification_init_status = "Processing"
        self.workgroup.identification_scheduled_init_task_id = "init-task-456"
        self.workgroup.save()

        response = self.client.get(reverse("caidapp:stop_init_identification"))

        self.assertEqual(response.status_code, 302)
        revoke_mock.assert_called_once_with("init-task-456", terminate=True)
        self.workgroup.refresh_from_db()
        self.assertEqual(self.workgroup.identification_init_status, "Not initiated")
        self.assertIsNone(self.workgroup.identification_scheduled_init_task_id)
        self.assertIn("stopped manually", self.workgroup.identification_init_message)

    @patch("caidapp.tasks.signature")
    @patch("caidapp.tasks._prepare_dataframe_for_identification")
    def test_init_identification_stores_running_worker_task_id(
        self,
        prepare_dataframe_mock,
        signature_mock,
    ):
        representative_archive = UploadedArchiveFactory(owner=self.caiduser, is_for_identification=True, import_finished=True)
        MediaFileFactory(
            parent=representative_archive,
            with_identity=True,
            identity_is_representative=True,
        )
        prepare_dataframe_mock.return_value = {
            "image_path": ["representative.jpg"],
            "class_id": [1],
            "label": ["known"],
        }

        signature_result = Mock()
        signature_mock.return_value = signature_result
        worker_task = Mock()
        worker_task.id = "init-worker-task-1"
        signature_result.apply_async.return_value = worker_task

        tasks.init_identification(self.workgroup.id)

        self.workgroup.refresh_from_db()
        self.assertEqual(self.workgroup.identification_init_status, "Processing")
        self.assertEqual(self.workgroup.identification_scheduled_init_task_id, worker_task.id)
        self.assertIsNone(self.workgroup.identification_scheduled_init_eta)

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
