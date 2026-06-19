import logging
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from caidapp import models
from caidapp import tasks
from caidapp import views
from caidapp import views_mediafile
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.files.uploadedfile import SimpleUploadedFile
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

    def test_home_suggests_send_to_identification_before_general_upload(self):
        archive = UploadedArchiveFactory(
            owner=self.caiduser,
            taxon_status="TV",
            is_for_identification=False,
            contains_single_taxon=False,
        )
        wolf = TaxonFactory(name="Wolf")
        mediafile = MediaFileFactory(parent=archive, media_type="image")
        AnimalObservationFactory(mediafile=mediafile, taxon=wolf, taxon_verified=True)

        response = self.client.get(reverse("caidapp:home"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Suggested next step:")
        self.assertContains(response, "Send to identification")
        self.assertContains(response, reverse("caidapp:uploads_ready_for_identification"))


    def test_home_suggests_manual_identification_for_unidentified_identification_upload(self):
        self.caiduser.workgroup.check_taxon_before_identification = False
        self.caiduser.workgroup.save(update_fields=["check_taxon_before_identification"])
        archive = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
            contains_single_taxon=True,
        )
        MediaFileFactory(parent=archive, identity=None)

        response = self.client.get(reverse("caidapp:home"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["home_next_step"]["label"], "Manual identification")
        self.assertContains(response, reverse("caidapp:manual_identification"))


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


class MediafileListViewTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory()
        self.user = self.caiduser.user
        self.client.login(username=self.user.username, password="test123")

    def test_media_files_view_shows_each_file_separately(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        first_mediafile = MediaFileFactory(
            parent=archive,
            sequence=sequence,
            original_filename="first.jpg",
        )
        second_mediafile = MediaFileFactory(
            parent=archive,
            sequence=sequence,
            original_filename="second.jpg",
        )

        response = self.client.get(reverse("caidapp:media_files"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, first_mediafile.original_filename)
        self.assertContains(response, second_mediafile.original_filename)
        self.assertNotContains(response, "toggle-sequence")
        self.assertNotContains(response, "data-sequence=")


class IdentityObservationAggregationTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory()
        self.client.login(username=self.caiduser.user.username, password="test123")

    def test_identity_list_uses_observation_identities_for_counts_and_cover(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")
        other_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Beta")
        first_locality = LocalityFactory(owner=self.caiduser, name="Forest")
        second_locality = LocalityFactory(owner=self.caiduser, name="Meadow")
        older_mediafile = MediaFileFactory(
            parent=archive,
            locality=first_locality,
            original_filename="older.jpg",
            captured_at=pd.Timestamp("2026-01-01T10:00:00Z").to_pydatetime(),
        )
        newer_mediafile = MediaFileFactory(
            parent=archive,
            locality=second_locality,
            original_filename="newer.jpg",
            captured_at=pd.Timestamp("2026-01-02T10:00:00Z").to_pydatetime(),
        )
        other_mediafile = MediaFileFactory(parent=archive, original_filename="other.jpg")

        AnimalObservationFactory(mediafile=older_mediafile, identity=identity, identity_is_representative=False)
        AnimalObservationFactory(mediafile=newer_mediafile, identity=identity, identity_is_representative=True)
        AnimalObservationFactory(mediafile=other_mediafile, identity=other_identity, identity_is_representative=True)

        response = self.client.get(reverse("caidapp:individual_identities"), {"view": "list"})

        self.assertEqual(response.status_code, 200)
        alpha = next(item for item in response.context["page_obj"] if item.id == identity.id)
        self.assertEqual(alpha.mediafile_count, 2)
        self.assertEqual(alpha.representative_mediafile_count, 1)
        self.assertEqual(alpha.locality_count, 2)
        self.assertEqual(alpha.cover_mediafile().id, newer_mediafile.id)
        self.assertEqual(alpha.last_seen.date().isoformat(), "2026-01-02")

    def test_identity_mediafiles_view_filters_by_observation_identity(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")
        other_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Beta")
        matching_mediafile = MediaFileFactory(parent=archive, original_filename="alpha.jpg")
        other_mediafile = MediaFileFactory(parent=archive, original_filename="beta.jpg")

        AnimalObservationFactory(mediafile=matching_mediafile, identity=identity)
        AnimalObservationFactory(mediafile=other_mediafile, identity=other_identity)

        response = self.client.get(reverse("caidapp:individual_identity_mediafiles", args=[identity.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "alpha.jpg")
        self.assertNotContains(response, "beta.jpg")

    def test_sequence_view_filters_by_observation_identity(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")
        other_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Beta")
        matching_sequence = SequenceFactory(uploaded_archive=archive)
        other_sequence = SequenceFactory(uploaded_archive=archive)
        matching_mediafile = MediaFileFactory(parent=archive, sequence=matching_sequence, original_filename="alpha-seq.jpg")
        other_mediafile = MediaFileFactory(parent=archive, sequence=other_sequence, original_filename="beta-seq.jpg")

        AnimalObservationFactory(mediafile=matching_mediafile, identity=identity)
        AnimalObservationFactory(mediafile=other_mediafile, identity=other_identity)

        response = self.client.get(reverse("caidapp:sequences"), {"individual_identity_id": identity.id})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Identity: Alpha")
        self.assertContains(response, "alpha-seq.jpg")
        self.assertNotContains(response, "beta-seq.jpg")



class ImageUploadGraphViewTest(TestCase):
    def test_upload_stats_graph_uses_date_axis_and_sorted_days(self):
        df = pd.DataFrame(
            [
                {"date": pd.Timestamp("2026-06-10").date(), "parent__owner__user__username": "alice"},
                {"date": pd.Timestamp("2026-06-01").date(), "parent__owner__user__username": "alice"},
                {"date": pd.Timestamp("2026-06-05").date(), "parent__owner__user__username": "alice"},
            ]
        )

        fig = views._build_image_upload_graph_figure(df)

        self.assertEqual(fig.layout.xaxis.type, "date")
        self.assertTrue(fig.layout.xaxis.rangeslider.visible)
        self.assertEqual(list(fig.data[0].x), [pd.Timestamp("2026-06-01").date(), pd.Timestamp("2026-06-05").date(), pd.Timestamp("2026-06-10").date()])


class AnimalObservationFormTest(TestCase):
    def test_identity_queryset_is_sorted_and_searchable(self):
        caiduser = CaidUserFactory()
        archive = UploadedArchiveFactory(owner=caiduser)
        mediafile = MediaFileFactory(parent=archive)
        IndividualIdentityFactory(owner_workgroup=caiduser.workgroup, name="Zebra")
        IndividualIdentityFactory(owner_workgroup=caiduser.workgroup, name="Antelope")
        observation = AnimalObservationFactory(mediafile=mediafile)

        form = views_mediafile.forms.AnimalObservationForm(instance=observation)

        self.assertEqual(
            list(form.fields["identity"].queryset.values_list("name", flat=True)),
            ["Antelope", "Identity0", "Zebra"],
        )
        self.assertIn("js-searchable-select", form.fields["identity"].widget.attrs["class"])
        self.assertIn("js-searchable-select", form.fields["taxon"].widget.attrs["class"])


class SpreadsheetIdentityLocalityImportExportTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory()
        self.user = self.caiduser.user
        self.client.login(username=self.user.username, password="test123")

    def _csv_upload(self, filename, rows):
        dataframe = pd.DataFrame(rows)
        return SimpleUploadedFile(
            filename,
            dataframe.to_csv(index=False).encode("utf-8"),
            content_type="text/csv",
        )

    def test_identity_export_includes_id_column(self):
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha", code="A-01")

        response = self.client.get(reverse("caidapp:export_identities_csv"))

        self.assertEqual(response.status_code, 200)
        df = pd.read_csv(StringIO(response.content.decode()))
        self.assertEqual(
            list(df.columns),
            ["Unnamed: 0", "id", "name", "code", "juv_code", "sex", "coat_type", "birth_date", "death_date", "note"],
        )
        self.assertEqual(int(df.iloc[0]["id"]), identity.id)

    def test_identity_import_prefers_id_for_rename(self):
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha", code="A-01")

        response = self.client.post(
            reverse("caidapp:import_identities"),
            {
                "spreadsheet_file": self._csv_upload(
                    "identities.csv",
                    [{"id": identity.id, "name": "Beta", "code": "A-01"}],
                )
            },
        )

        self.assertRedirects(response, reverse("caidapp:individual_identities"))
        identity.refresh_from_db()
        self.assertEqual(identity.name, "Beta")
        self.assertEqual(
            models.IndividualIdentity.objects.filter(owner_workgroup=self.caiduser.workgroup).count(),
            1,
        )

    def test_locality_export_includes_id_column(self):
        locality = LocalityFactory(owner=self.caiduser, name="North Meadow")

        response = self.client.get(reverse("caidapp:export_localities"))

        self.assertEqual(response.status_code, 200)
        df = pd.read_csv(StringIO(response.content.decode()))
        self.assertEqual(list(df.columns), ["Unnamed: 0", "id", "name", "location"])
        self.assertEqual(int(df.iloc[0]["id"]), locality.id)

    def test_locality_import_prefers_id_for_rename(self):
        locality = LocalityFactory(owner=self.caiduser, name="North Meadow")

        response = self.client.post(
            reverse("caidapp:import_localities"),
            {
                "spreadsheet_file": self._csv_upload(
                    "localities.csv",
                    [{"id": locality.id, "name": "South Meadow", "location": ""}],
                )
            },
        )

        self.assertRedirects(response, reverse("caidapp:localities"))
        locality.refresh_from_db()
        self.assertEqual(locality.name, "South Meadow")
        self.assertEqual(models.Locality.objects.filter(owner=self.caiduser).count(), 1)


class MediaFileListSearchTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory()
        self.user = self.caiduser.user
        self.client.login(username=self.user.username, password="test123")

    def test_mediafile_search_supports_explicit_regex(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        locality_matching = LocalityFactory(owner=self.caiduser, name="Site 1234")
        locality_non_matching = LocalityFactory(owner=self.caiduser, name="Site 12345")
        matching_mediafile = MediaFileFactory(
            parent=archive,
            locality=locality_matching,
            original_filename="matching.jpg",
        )
        non_matching_mediafile = MediaFileFactory(
            parent=archive,
            locality=locality_non_matching,
            original_filename="non-matching.jpg",
        )

        response = self.client.get(
            reverse("caidapp:media_files"),
            {
                "search": r"^\D*\d\D*\d\D*\d\D*\d\D*$",
                "search_regex": "true",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, matching_mediafile.original_filename)
        self.assertNotContains(response, non_matching_mediafile.original_filename)


class MediaFileUpdateEmptyObservationTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory()
        self.user = self.caiduser.user
        self.client.login(username=self.user.username, password="test123")
        self.caiduser.workgroup.check_taxon_before_identification = False
        self.caiduser.workgroup.save(update_fields=["check_taxon_before_identification"])

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

    def test_mediafile_update_links_to_uploadedarchive_detail(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive)

        response = self.client.get(reverse("caidapp:media_file_update", args=[mediafile.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, reverse("caidapp:uploadedarchive_detail", args=[archive.id]))
        self.assertNotContains(response, reverse("caidapp:uploadedarchive_mediafiles", args=[archive.id]))

    def test_predicted_taxon_select_uses_taxon_id_and_refreshes_searchable_dropdown(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        predicted_taxon = TaxonFactory(name="Panthera pardus")
        mediafile = MediaFileFactory(
            parent=archive,
            predicted_taxon=predicted_taxon,
            predicted_taxon_confidence=0.91,
        )
        AnimalObservationFactory(
            mediafile=mediafile,
            taxon=None,
            predicted_taxon=predicted_taxon,
            predicted_taxon_confidence=0.91,
        )

        response = self.client.get(reverse("caidapp:media_file_update", args=[mediafile.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            f"setCategoryAndSubmitInObservation('{predicted_taxon.id}', 0); return false;",
        )
        self.assertContains(response, "categoryDropdown.value = String(predictedTaxonId);")
        self.assertContains(response, 'categoryDropdown.dispatchEvent(new Event("change", {bubbles: true}));')

    def test_missing_taxon_sequence_carousel_keeps_annotation_mode(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        first_mediafile = MediaFileFactory(
            parent=archive,
            sequence=sequence,
            original_filename="first.jpg",
            static_thumbnail=SimpleUploadedFile("first.webp", b"first", content_type="image/webp"),
        )
        second_mediafile = MediaFileFactory(
            parent=archive,
            sequence=sequence,
            original_filename="second.jpg",
            static_thumbnail=SimpleUploadedFile("second.webp", b"second", content_type="image/webp"),
        )

        response = self.client.get(
            reverse("caidapp:missing_taxon_annotation_for_mediafile", args=[first_mediafile.id]),
            {"uploadedarchive_id": archive.id},
        )

        expected_url = (
            reverse("caidapp:missing_taxon_annotation_for_mediafile", args=[second_mediafile.id])
            + f"?uploadedarchive_id={archive.id}"
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, expected_url)

    def test_missing_taxon_mark_empty_keeps_annotation_mode(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive)

        response = self.client.post(
            reverse("caidapp:missing_taxon_annotation_for_mediafile", args=[mediafile.id])
            + f"?uploadedarchive_id={archive.id}",
            self._base_mediafile_update_post_data(
                mediafile,
                total_forms=0,
                initial_forms=0,
                extra_form_data={"mark_empty_image": "1"},
            ),
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.url,
            reverse("caidapp:missing_taxon_annotation_for_mediafile", args=[mediafile.id])
            + f"?uploadedarchive_id={archive.id}",
        )

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

    def test_mediafile_update_shows_confirm_identity_action_for_preidentified_mediafile(self):
        archive = UploadedArchiveFactory(owner=self.caiduser, is_for_identification=True)
        mediafile = MediaFileFactory(parent=archive)
        models.MediafilesForIdentification.objects.create(mediafile=mediafile)

        response = self.client.get(reverse("caidapp:media_file_update", args=[mediafile.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, reverse("caidapp:get_individual_identity_by_media_file", args=[mediafile.id]))
        self.assertContains(response, "Confirm identity")

    def test_save_sequence_uses_full_taxon_id_for_multi_digit_taxon_ids(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        first_mediafile = MediaFileFactory(parent=archive, sequence=sequence)
        second_mediafile = MediaFileFactory(parent=archive, sequence=sequence)
        observations = [
            AnimalObservationFactory(mediafile=first_mediafile, taxon=None, taxon_verified=False),
            AnimalObservationFactory(mediafile=second_mediafile, taxon=None, taxon_verified=False),
        ]
        taxon = None
        for index in range(12):
            created_taxon = TaxonFactory(name=f"SequenceTaxon{index}")
            if index == 11:
                taxon = created_taxon

        response = self.client.post(
            reverse("caidapp:media_file_update", args=[first_mediafile.id]),
            self._base_mediafile_update_post_data(
                first_mediafile,
                total_forms=1,
                initial_forms=1,
                extra_form_data={
                    "observations-0-id": str(observations[0].id),
                    "observations-0-mediafile": str(first_mediafile.id),
                    "observations-0-taxon": str(taxon.id),
                    "observations-0-taxon_verified": "on",
                    "observations-0-identity": "",
                    "observations-0-identity_is_representative": "",
                    "observations-0-orientation": "N",
                    "observations-0-bbox_x_center": "",
                    "observations-0-bbox_y_center": "",
                    "observations-0-bbox_width": "",
                    "observations-0-bbox_height": "",
                    "observations-0-DELETE": "",
                    "save_set_taxon_sequence": "1",
                },
            ),
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        for observation in observations:
            observation.refresh_from_db()
            self.assertEqual(observation.taxon_id, taxon.id)

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

    def test_manual_identification_filters_by_configured_taxon_when_enabled(self):
        wolf = TaxonFactory(name="Wolf")
        lynx = TaxonFactory(name="Lynx")
        bear = TaxonFactory(name="Bear")
        self.caiduser.workgroup.check_taxon_before_identification = True
        self.caiduser.workgroup.default_taxon_for_identification = wolf
        self.caiduser.workgroup.save()

        default_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
            taxon_for_identification=None,
        )
        matching_default = MediaFileFactory(parent=default_archive, identity=None)
        AnimalObservationFactory(mediafile=matching_default, taxon=wolf)
        wrong_default = MediaFileFactory(parent=default_archive, identity=None)
        AnimalObservationFactory(mediafile=wrong_default, taxon=lynx)

        override_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
            taxon_for_identification=bear,
        )
        matching_override = MediaFileFactory(parent=override_archive, identity=None)
        AnimalObservationFactory(mediafile=matching_override, taxon=bear)
        wrong_override = MediaFileFactory(parent=override_archive, identity=None)
        AnimalObservationFactory(mediafile=wrong_override, taxon=wolf)

        mediafile_ids = set(
            models.get_mediafiles_with_missing_identity(self.caiduser).values_list("id", flat=True)
        )

        self.assertEqual(mediafile_ids, {matching_default.id, matching_override.id})

    def test_manual_identification_does_not_filter_taxon_when_disabled(self):
        wolf = TaxonFactory(name="Wolf")
        lynx = TaxonFactory(name="Lynx")
        self.caiduser.workgroup.check_taxon_before_identification = False
        self.caiduser.workgroup.default_taxon_for_identification = wolf
        self.caiduser.workgroup.save()
        archive = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
            taxon_for_identification=wolf,
        )
        matching = MediaFileFactory(parent=archive, identity=None)
        AnimalObservationFactory(mediafile=matching, taxon=wolf)
        nonmatching = MediaFileFactory(parent=archive, identity=None)
        AnimalObservationFactory(mediafile=nonmatching, taxon=lynx)
        without_observation = MediaFileFactory(parent=archive, identity=None)

        mediafile_ids = set(
            models.get_mediafiles_with_missing_identity(self.caiduser).values_list("id", flat=True)
        )

        self.assertEqual(mediafile_ids, {matching.id, nonmatching.id, without_observation.id})

    def test_manual_identification_starts_with_accessible_unidentified_mediafile(self):
        archive = UploadedArchiveFactory(owner=self.caiduser, is_for_identification=True)
        first_mediafile = MediaFileFactory(parent=archive, identity=None)
        second_mediafile = MediaFileFactory(parent=archive, identity=None)
        other_user = CaidUserFactory()
        other_archive = UploadedArchiveFactory(owner=other_user, is_for_identification=True)
        MediaFileFactory(parent=other_archive, identity=None)

        response = self.client.get(reverse("caidapp:manual_identification"))

        self.assertRedirects(
            response,
            reverse("caidapp:manual_identification_mediafile", args=[first_mediafile.id]),
            fetch_redirect_response=False,
        )
        detail_response = self.client.get(response.url)
        self.assertEqual(detail_response.status_code, 200)
        self.assertContains(detail_response, "Manual identification")
        self.assertContains(
            detail_response,
            reverse("caidapp:manual_identification_mediafile", args=[second_mediafile.id]),
        )

    def test_manual_identification_skips_mediafile_with_observation_identity(self):
        archive = UploadedArchiveFactory(owner=self.caiduser, is_for_identification=True)
        identified_mediafile = MediaFileFactory(parent=archive, identity=None)
        unidentified_mediafile = MediaFileFactory(parent=archive, identity=None)
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup)
        AnimalObservationFactory(mediafile=identified_mediafile, identity=identity)

        response = self.client.get(reverse("caidapp:manual_identification"))

        self.assertRedirects(
            response,
            reverse("caidapp:manual_identification_mediafile", args=[unidentified_mediafile.id]),
            fetch_redirect_response=False,
        )

    def test_manual_identification_saves_identity_and_advances(self):
        archive = UploadedArchiveFactory(owner=self.caiduser, is_for_identification=True)
        current_mediafile = MediaFileFactory(parent=archive, identity=None)
        next_mediafile = MediaFileFactory(parent=archive, identity=None)
        observation = AnimalObservationFactory(mediafile=current_mediafile, identity=None)
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup)

        response = self.client.post(
            reverse("caidapp:manual_identification_mediafile", args=[current_mediafile.id]),
            self._base_mediafile_update_post_data(
                current_mediafile,
                total_forms=1,
                initial_forms=1,
                extra_form_data={
                    "observations-0-id": str(observation.id),
                    "observations-0-mediafile": str(current_mediafile.id),
                    "observations-0-taxon": "",
                    "observations-0-identity": str(identity.id),
                    "observations-0-orientation": "N",
                    "observations-0-bbox_x_center": "",
                    "observations-0-bbox_y_center": "",
                    "observations-0-bbox_width": "",
                    "observations-0-bbox_height": "",
                    "observations-0-DELETE": "",
                },
            ),
        )

        self.assertRedirects(
            response,
            reverse("caidapp:manual_identification_mediafile", args=[next_mediafile.id]),
            fetch_redirect_response=False,
        )
        observation.refresh_from_db()
        self.assertEqual(observation.identity, identity)


class IdentityListBulkActionsTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory()
        self.user = self.caiduser.user
        self.client.login(username=self.user.username, password="test123")

    def test_identity_search_supports_explicit_regex(self):
        matching = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha 1234")
        non_matching = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Beta 12345")

        response = self.client.get(
            reverse("caidapp:individual_identities"),
            {
                "search": r"^\D*\d\D*\d\D*\d\D*\d\D*$",
                "search_regex": "true",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, matching.name)
        self.assertNotContains(response, non_matching.name)

    def test_bulk_delete_confirm_and_delete_are_limited_to_user_workgroup(self):
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Delete me")
        other_caiduser = CaidUserFactory()
        other_identity = IndividualIdentityFactory(owner_workgroup=other_caiduser.workgroup, name="Private identity")
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive, identity=identity)

        confirm_response = self.client.post(
            reverse("caidapp:individual_identities"),
            {
                "bulk_action": "confirm_delete",
                "selected_identity_ids": [str(identity.id), str(other_identity.id)],
            },
        )

        self.assertEqual(confirm_response.status_code, 200)
        self.assertContains(confirm_response, "Delete me")
        self.assertNotContains(confirm_response, "Private identity")

        delete_response = self.client.post(
            reverse("caidapp:individual_identities"),
            {
                "bulk_action": "delete_selected",
                "confirm_delete": "yes",
                "selected_identity_ids": [str(identity.id), str(other_identity.id)],
            },
        )

        self.assertRedirects(delete_response, reverse("caidapp:individual_identities"))
        self.assertFalse(models.IndividualIdentity.objects.filter(id=identity.id).exists())
        self.assertTrue(models.IndividualIdentity.objects.filter(id=other_identity.id).exists())
        mediafile.refresh_from_db()
        self.assertIsNone(mediafile.identity)

    def test_bulk_open_sequences_redirects_to_multiple_identity_filter(self):
        first = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="First")
        second = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Second")

        response = self.client.post(
            reverse("caidapp:individual_identities"),
            {
                "bulk_action": "open_sequences",
                "selected_identity_ids": [str(first.id), str(second.id)],
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("caidapp:sequences"), response["Location"])
        self.assertIn(f"individual_identity_ids={first.id}", response["Location"])
        self.assertIn(f"individual_identity_ids={second.id}", response["Location"])

    def test_identity_update_links_to_sequences(self):
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")

        response = self.client.get(reverse("caidapp:individual_identity_update", args=[identity.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Sequences")
        self.assertContains(response, f'{reverse("caidapp:sequences")}?individual_identity_id={identity.id}')


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

    def test_sequence_view_search_matches_mediafile_filename_and_keeps_full_sequence(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        matching_sequence = SequenceFactory(uploaded_archive=archive)
        other_sequence = SequenceFactory(uploaded_archive=archive)
        matching_mediafile = MediaFileFactory(
            parent=archive,
            sequence=matching_sequence,
            original_filename="Brdy/alpha_1234.jpg",
        )
        sibling_mediafile = MediaFileFactory(
            parent=archive,
            sequence=matching_sequence,
            original_filename="Brdy/not-matching.jpg",
        )
        MediaFileFactory(
            parent=archive,
            sequence=other_sequence,
            original_filename="Brdy/unrelated.jpg",
        )

        response = self.client.get(reverse("caidapp:sequences"), {"search": "1234"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, matching_mediafile.original_filename)
        self.assertContains(response, sibling_mediafile.original_filename)
        self.assertContains(response, "sequence-nonmatching-mediafile")
        self.assertContains(response, "Select matching mediafiles on page")
        self.assertContains(response, "Select sequences with matching mediafile on page")
        self.assertNotContains(response, "Brdy/unrelated.jpg")

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
        one_mediafile = MediaFileFactory(parent=archive, sequence=one_file_sequence, original_filename="one.jpg")
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
        self.assertContains(response, 'name="btnDownloadSequences"')
        self.assertContains(response, "Change date and time")
        self.assertContains(response, "Dissolve to single-media sequences")
        self.assertContains(response, "js-requires-selection")
        self.assertContains(response, "No media files are selected. Select all media files on this page and continue?")
        self.assertContains(response, 'button.closest("form").requestSubmit(button)')
        self.assertContains(response, "Select whole sequence")
        self.assertNotContains(response, "Select all media files in sequence")
        self.assertContains(response, 'aria-label="Select"')
        self.assertLess(content.index('js-select-matching-mediafiles'), content.index('name="btnDownloadSequences"'))
        self.assertContains(response, "bg-body-tertiary p-3")
        self.assertContains(response, "Apply filters")
        self.assertContains(response, "border-top-left-radius:var(--bs-card-inner-border-radius)")
        self.assertContains(response, "background:var(--wrid-media-card-bg)")
        self.assertContains(response, "border-color:var(--wrid-media-card-border-color)")
        self.assertContains(response, ".sequence-grid{display:flex;flex-wrap:wrap")
        self.assertContains(response, ".sequence-grid .sequence-lead-card")
        self.assertContains(response, ".sequence-grid .sequence-extra-card{display:none}")
        self.assertContains(response, ".sequence-grid .sequence-extra-card.sequence-expanded-visible{display:flex}")
        self.assertContains(response, ".sequence-grid{column-gap:.25rem}")
        self.assertContains(response, ".sequence-grid-sequence-break{flex-basis:.45rem;width:.45rem}")
        self.assertContains(response, ".sequence-checkbox-prominent{width:1.3rem;height:1.3rem")
        self.assertContains(response, "js-sequence-checkbox sequence-checkbox-prominent")
        self.assertContains(response, ".sequence-grid .sequence-collapsed-card .sequence-mediafile-checkbox-overlay{display:none}")
        self.assertContains(response, "sequence-mediafile-checkbox-overlay")
        self.assertContains(response, 'js-sequence-checkbox[data-has-matching-mediafile="true"]')
        self.assertContains(response, "if(!hasIndividualMediafileCheckbox){ setCheckboxState(checkbox, true); }")
        self.assertContains(response, "justify-content-start align-items-center gap-2")
        self.assertNotIn(f'data-sequence-id="{one_file_sequence.id}" class="sequence-extra-card', content)
        self.assertContains(response, f'id="sequence-checkbox-{one_file_sequence.id}"')
        self.assertNotContains(response, f'id="mediafile-checkbox-{one_mediafile.id}"')
        self.assertContains(response, f'data-sequence-id="{two_file_sequence.id}"')
        self.assertContains(response, "sequence-lead-card")
        self.assertContains(response, "sequence-extra-card")
        self.assertContains(response, "sequence-end-card")
        self.assertContains(response, f'data-sequence-id="{three_file_sequence.id}"')

        list_response = self.client.get(reverse("caidapp:sequences"), {"view": "list"})
        self.assertContains(list_response, f'id="sequence-checkbox-list-{one_file_sequence.id}"')
        self.assertNotContains(list_response, f'id="mediafile-checkbox-list-{one_mediafile.id}"')

    def test_sequence_bulk_verify_taxon_handles_mediafile_without_observations(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        mediafile = MediaFileFactory(parent=archive, sequence=sequence, original_filename="empty-observation.jpg")

        response = self.client.post(
            reverse("caidapp:sequences") + "?show_overview_button=true&taxon_verified=false",
            {
                "selected_mediafile_ids": [str(mediafile.id)],
                "taxon_verified": "on",
                "btnBulkProcessing_set_taxon_verified": "1",
            },
        )

        self.assertEqual(response.status_code, 302)
        mediafile.refresh_from_db()
        self.assertTrue(mediafile.taxon_verified)
        observation = mediafile.observations.get()
        self.assertTrue(observation.taxon_verified)

    def test_sequence_verification_mode_groups_by_taxon_and_expands_sequences(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        taxon_wolf = TaxonFactory(name="Wolf")
        taxon_lynx = TaxonFactory(name="Lynx")
        wolf_sequence = SequenceFactory(uploaded_archive=archive)
        lynx_sequence = SequenceFactory(uploaded_archive=archive)
        wolf_first = MediaFileFactory(parent=archive, sequence=wolf_sequence, original_filename="wolf-a.jpg")
        wolf_second = MediaFileFactory(parent=archive, sequence=wolf_sequence, original_filename="wolf-b.jpg")
        lynx_mediafile = MediaFileFactory(parent=archive, sequence=lynx_sequence, original_filename="lynx.jpg")
        AnimalObservationFactory(mediafile=wolf_first, taxon=taxon_wolf, taxon_verified=False)
        AnimalObservationFactory(mediafile=wolf_second, taxon=taxon_wolf, taxon_verified=True)
        AnimalObservationFactory(mediafile=lynx_mediafile, taxon=taxon_lynx, taxon_verified=False)

        response = self.client.get(
            reverse("caidapp:sequences"),
            {"show_overview_button": "true", "taxon_verified": "false"},
        )

        self.assertEqual(response.status_code, 200)
        sequences = list(response.context["sequence_objects"])
        self.assertEqual([sequence.id for sequence in sequences], [lynx_sequence.id, wolf_sequence.id])
        self.assertEqual([sequence.verification_taxon_group_label for sequence in sequences], ["Lynx", "Wolf"])
        self.assertTrue(all(sequence.starts_verification_taxon_group for sequence in sequences))
        self.assertContains(response, "verification-taxon-heading")
        self.assertContains(response, "Verify taxon on selected media files")
        self.assertNotContains(response, "Verify taxon on all media files on this page")
        self.assertContains(response, "Media files view")
        self.assertContains(response, "sequence-expanded-visible")
        self.assertContains(response, "wolf-b.jpg")
        self.assertEqual(set(self.client.session["mediafile_ids_page"]), {wolf_first.id, wolf_second.id, lynx_mediafile.id})

    def test_sequence_verification_empty_state_links_home(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        taxon = TaxonFactory(name="Wolf")
        mediafile = MediaFileFactory(parent=archive, sequence=sequence, original_filename="verified.jpg", taxon_verified=True)
        AnimalObservationFactory(mediafile=mediafile, taxon=taxon, taxon_verified=True)

        response = self.client.get(
            reverse("caidapp:sequences"),
            {"show_overview_button": "true", "taxon_verified": "false"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No sequences matched the current filter.")
        self.assertContains(response, "Go home")
        self.assertContains(response, reverse("caidapp:home"))

    def test_legacy_verify_taxa_url_uses_sequence_verification_view(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        taxon = TaxonFactory(name="Bear")
        mediafile = MediaFileFactory(parent=archive, sequence=sequence, original_filename="bear.jpg")
        AnimalObservationFactory(mediafile=mediafile, taxon=taxon, taxon_verified=False)

        response = self.client.get(reverse("caidapp:verify_taxa"))

        self.assertEqual(response.status_code, 200)
        self.assertIn("sequence_objects", response.context)
        self.assertContains(response, "Media files view")
        self.assertContains(response, "bear.jpg")

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

    def test_dissolve_selected_sequences_splits_mediafiles_into_singletons(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        selected_sequence = SequenceFactory(uploaded_archive=archive, local_id=10)
        other_sequence = SequenceFactory(uploaded_archive=archive, local_id=20)
        first_mediafile = MediaFileFactory(parent=archive, sequence=selected_sequence, original_filename="first.jpg")
        second_mediafile = MediaFileFactory(parent=archive, sequence=selected_sequence, original_filename="second.jpg")
        untouched_mediafile = MediaFileFactory(parent=archive, sequence=other_sequence, original_filename="other.jpg")

        response = self.client.post(
            reverse("caidapp:sequences"),
            {
                "btnDissolveSequences": "1",
                "selected_sequence_ids": [str(selected_sequence.id)],
            },
        )

        self.assertEqual(response.status_code, 302)
        first_mediafile.refresh_from_db()
        second_mediafile.refresh_from_db()
        untouched_mediafile.refresh_from_db()
        self.assertNotEqual(first_mediafile.sequence_id, second_mediafile.sequence_id)
        self.assertNotEqual(first_mediafile.sequence_id, selected_sequence.id)
        self.assertNotEqual(second_mediafile.sequence_id, selected_sequence.id)
        self.assertEqual(untouched_mediafile.sequence_id, other_sequence.id)
        self.assertFalse(models.Sequence.objects.filter(id=selected_sequence.id).exists())
        self.assertEqual(models.MediaFile.objects.filter(sequence_id=first_mediafile.sequence_id).count(), 1)
        self.assertEqual(models.MediaFile.objects.filter(sequence_id=second_mediafile.sequence_id).count(), 1)

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

    def test_sequence_view_filters_by_multiple_identities(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        first_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")
        second_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Beta")
        other_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Gamma")
        first_sequence = SequenceFactory(uploaded_archive=archive)
        second_sequence = SequenceFactory(uploaded_archive=archive)
        other_sequence = SequenceFactory(uploaded_archive=archive)
        first_mediafile = MediaFileFactory(
            parent=archive,
            sequence=first_sequence,
            identity=first_identity,
            original_filename="alpha.jpg",
        )
        second_mediafile = MediaFileFactory(
            parent=archive,
            sequence=second_sequence,
            identity=second_identity,
            original_filename="beta.jpg",
        )
        other_mediafile = MediaFileFactory(
            parent=archive,
            sequence=other_sequence,
            identity=other_identity,
            original_filename="gamma.jpg",
        )

        response = self.client.get(
            reverse("caidapp:sequences"),
            {
                "individual_identity_ids": [str(first_identity.id), str(second_identity.id)],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Identities: 2")
        self.assertContains(response, first_mediafile.original_filename)
        self.assertContains(response, second_mediafile.original_filename)
        self.assertNotContains(response, other_mediafile.original_filename)

    def test_sequence_view_rejects_multiple_identity_filter_outside_user_workgroup(self):
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Visible identity")
        other_caiduser = CaidUserFactory()
        private_identity = IndividualIdentityFactory(owner_workgroup=other_caiduser.workgroup, name="Private identity")

        response = self.client.get(
            reverse("caidapp:sequences"),
            {
                "individual_identity_ids": [str(identity.id), str(private_identity.id)],
            },
        )

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

    def test_filename_metadata_skips_observation_metadata_for_multiple_observations(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        mediafile = MediaFileFactory(
            parent=archive,
            sequence=sequence,
            original_filename="Brdy/Lynx/Charles/first.jpg",
        )
        first_observation = AnimalObservationFactory(mediafile=mediafile, taxon=None, identity=None)
        second_observation = AnimalObservationFactory(mediafile=mediafile, taxon=None, identity=None)
        session = self.client.session
        session["filename_metadata_mediafile_ids"] = [mediafile.id]
        session["filename_metadata_return_url"] = reverse("caidapp:sequences")
        session["filename_metadata_source_label"] = "Sequences"
        session.save()

        response = self.client.post(
            reverse("caidapp:apply_filename_metadata_to_mediafiles"),
            {
                "path_regex": r"^(?P<locality>[^/]+)/(?P<taxon>[^/]+)/(?P<unique_name>[^/]+)/[^/]+$",
            },
        )

        self.assertRedirects(response, reverse("caidapp:sequences"))
        mediafile.refresh_from_db()
        self.assertEqual(mediafile.locality.name, "Brdy")
        self.assertIsNone(mediafile.taxon)
        self.assertIsNone(mediafile.identity)
        first_observation.refresh_from_db()
        second_observation.refresh_from_db()
        self.assertIsNone(first_observation.taxon)
        self.assertIsNone(second_observation.taxon)
        self.assertIsNone(first_observation.identity)
        self.assertIsNone(second_observation.identity)


class IdentificationUploadsViewTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory()
        self.user = self.caiduser.user
        self.client.login(username=self.user.username, password="test123")

    def test_taxon_dashboard_links_to_uploads_ready_for_identification(self):
        response = self.client.get(reverse("caidapp:taxon_processing"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, reverse("caidapp:uploads_ready_for_identification"))
        self.assertContains(response, "Send to identification")

    def test_taxon_dashboard_highlights_send_to_identification_when_upload_is_ready(self):
        archive = UploadedArchiveFactory(
            owner=self.caiduser,
            taxon_status="TV",
            is_for_identification=False,
            contains_single_taxon=False,
        )
        wolf = TaxonFactory(name="Wolf")
        mediafile = MediaFileFactory(parent=archive)
        AnimalObservationFactory(mediafile=mediafile, taxon=wolf, taxon_verified=True)

        response = self.client.get(reverse("caidapp:taxon_processing"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            f'class="btn btn-primary mt-2" href="{reverse("caidapp:uploads_ready_for_identification")}"',
            html=False,
        )

    def test_uploads_ready_for_identification_lists_verified_before_known(self):
        default_taxon = TaxonFactory(name="Lynx")
        self.caiduser.workgroup.default_taxon_for_identification = default_taxon
        self.caiduser.workgroup.save()
        wolf = TaxonFactory(name="Wolf")
        bear = TaxonFactory(name="Bear")
        known_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            name="Known archive",
            taxon_status="TKN",
            is_for_identification=False,
            contains_single_taxon=False,
        )
        verified_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            name="Verified archive",
            taxon_status="TV",
            is_for_identification=False,
            contains_single_taxon=False,
        )
        sent_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            name="Already sent archive",
            taxon_status="TV",
            is_for_identification=True,
            contains_single_taxon=False,
        )
        known_mediafile = MediaFileFactory(parent=known_archive)
        verified_mediafile = MediaFileFactory(parent=verified_archive)
        sent_mediafile = MediaFileFactory(parent=sent_archive)
        AnimalObservationFactory(mediafile=known_mediafile, taxon=bear)
        AnimalObservationFactory(mediafile=verified_mediafile, taxon=wolf)
        AnimalObservationFactory(mediafile=sent_mediafile, taxon=wolf)

        response = self.client.get(reverse("caidapp:uploads_ready_for_identification"))
        content = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Default identification taxon:")
        self.assertContains(response, "Lynx")
        self.assertContains(response, "Verified archive")
        self.assertContains(response, "Known archive")
        self.assertContains(response, "Wolf")
        self.assertContains(response, "Bear")
        self.assertContains(response, reverse("caidapp:select_taxon_for_identification", args=[verified_archive.id]))
        self.assertContains(response, reverse("caidapp:select_taxon_for_identification", args=[known_archive.id]))
        self.assertNotContains(response, "Already sent archive")
        self.assertLess(content.index("Verified archive"), content.index("Known archive"))

    def test_uploads_ready_for_identification_empty_state_links_to_taxon_dashboard(self):
        response = self.client.get(reverse("caidapp:uploads_ready_for_identification"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "There are no taxon uploads ready to send to identification.")
        self.assertContains(response, reverse("caidapp:taxon_processing"))

    def test_select_taxon_for_identification_uses_workgroup_default_and_explains_action(self):
        default_taxon = TaxonFactory(name="Lynx")
        self.caiduser.workgroup.default_taxon_for_identification = default_taxon
        self.caiduser.workgroup.save()
        archive = UploadedArchiveFactory(
            owner=self.caiduser,
            taxon_status="TV",
            is_for_identification=False,
            contains_single_taxon=False,
        )

        response = self.client.get(reverse("caidapp:select_taxon_for_identification", args=[archive.id]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["form"].initial["taxon_for_identification"], default_taxon)
        self.assertContains(response, "This screen sends media files with the selected taxon")
        self.assertContains(response, "Send to identification")

    def test_select_taxon_for_identification_prefers_archive_taxon_and_returns_to_home(self):
        default_taxon = TaxonFactory(name="Lynx")
        archive_taxon = TaxonFactory(name="Wolf")
        self.caiduser.workgroup.default_taxon_for_identification = default_taxon
        self.caiduser.workgroup.save()
        archive = UploadedArchiveFactory(
            owner=self.caiduser,
            taxon_status="TV",
            taxon_for_identification=archive_taxon,
            is_for_identification=False,
            contains_single_taxon=False,
        )
        next_url = reverse("caidapp:uploads_ready_for_identification")

        response = self.client.get(reverse("caidapp:select_taxon_for_identification", args=[archive.id]))
        self.assertEqual(response.context["form"].initial["taxon_for_identification"], archive_taxon)

        response = self.client.post(
            reverse("caidapp:select_taxon_for_identification", args=[archive.id]) + f"?next={next_url}",
            {
                "taxon_for_identification": str(archive_taxon.id),
                "next": next_url,
            },
        )

        self.assertRedirects(response, reverse("caidapp:home"))
        archive.refresh_from_db()
        self.assertEqual(archive.taxon_for_identification, archive_taxon)
        self.assertTrue(archive.is_for_identification)
        self.assertEqual(archive.identification_status, "IR")

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

    def test_uploads_identities_show_count_of_identified_individuals(self):
        archive = UploadedArchiveFactory(
            owner=self.caiduser,
            name="Upload with identities",
            is_for_identification=True,
            contains_identities=False,
            taxon_for_identification=None,
        )
        first_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")
        second_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Beta")
        MediaFileFactory(parent=archive, identity=first_identity)
        MediaFileFactory(parent=archive, identity=first_identity)
        MediaFileFactory(parent=archive, identity=second_identity)

        response = self.client.get(reverse("caidapp:uploads_identities"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Identities:")
        self.assertContains(response, "2 unique identified individuals in this upload")

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

    def test_dash_identities_upload_links_prefill_identification_modes(self):
        response = self.client.get(reverse("caidapp:dash_identities"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            reverse("caidapp:new_upload") + "?upload_target=identification&amp;contains_identities=1",
        )
        self.assertContains(
            response,
            reverse("caidapp:new_upload") + "?upload_target=identification",
        )


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

    @patch("caidapp.tasks.schedule_init_identification_for_workgroup")
    def test_completed_representative_upload_schedules_identification_init(self, schedule_mock):
        archive = UploadedArchiveFactory(
            owner=self.caiduser,
            contains_identities=True,
            import_finished=True,
        )
        MediaFileFactory(
            parent=archive,
            with_identity=True,
            identity_is_representative=True,
        )
        schedule_mock.reset_mock()

        scheduled = tasks.schedule_init_identification_after_representative_upload(archive)

        self.assertTrue(scheduled)
        schedule_mock.assert_called_once_with(self.workgroup)

    @patch("caidapp.tasks.schedule_init_identification_for_workgroup")
    def test_unfinished_representative_upload_does_not_schedule_identification_init(self, schedule_mock):
        archive = UploadedArchiveFactory(
            owner=self.caiduser,
            contains_identities=True,
            import_finished=False,
        )
        MediaFileFactory(
            parent=archive,
            with_identity=True,
            identity_is_representative=True,
        )
        schedule_mock.reset_mock()

        scheduled = tasks.schedule_init_identification_after_representative_upload(archive)

        self.assertFalse(scheduled)
        schedule_mock.assert_not_called()

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
