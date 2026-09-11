import datetime
import logging
import tempfile
from io import BytesIO, StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from caidapp import models
from caidapp import forms
from caidapp import tasks
from caidapp import views
from caidapp import views_mediafile
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone
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

    def test_user_menu_uses_personal_settings_label(self):
        response = self.client.get(reverse("caidapp:home"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Personal Settings")
        self.assertNotContains(response, "User Settings")

    def test_workgroup_settings_shows_add_model_from_huggingface_only_for_workgroup_admin_with_admin_access(self):
        self.caiduser.workgroup_admin = True
        self.caiduser.save(update_fields=["workgroup_admin"])

        response = self.client.get(reverse("caidapp:workgroup-update", args=[self.caiduser.workgroup.id]))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Add model from HuggingFace")

        self.user.is_staff = True
        self.user.save(update_fields=["is_staff"])
        response = self.client.get(reverse("caidapp:workgroup-update", args=[self.caiduser.workgroup.id]))

        self.assertContains(response, "Add model from HuggingFace")
        self.assertContains(
            response,
            reverse("admin:caidapp_identificationmodel_add") + f"?workgroup={self.caiduser.workgroup.id}",
        )

    def test_workgroup_settings_shows_link_to_personal_settings(self):
        self.caiduser.workgroup_admin = True
        self.caiduser.save(update_fields=["workgroup_admin"])

        response = self.client.get(reverse("caidapp:workgroup-update", args=[self.caiduser.workgroup.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Personal Settings")
        self.assertContains(response, reverse("caidapp:update_caiduser"))

    def test_personal_settings_shows_link_to_workgroup_settings_only_for_workgroup_admin(self):
        self.caiduser.workgroup_admin = False
        self.caiduser.save(update_fields=["workgroup_admin"])
        response = self.client.get(reverse("caidapp:update_caiduser"))

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Workgroup Settings", response.context["nav_dict"])

        self.caiduser.workgroup_admin = True
        self.caiduser.save(update_fields=["workgroup_admin"])

        response = self.client.get(reverse("caidapp:update_caiduser"))

        self.assertIn("Workgroup Settings", response.context["nav_dict"])
        self.assertContains(response, reverse("caidapp:workgroup-update", args=[self.caiduser.workgroup.id]))


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

    def test_card_image_prefers_static_variant_and_variant_urls_use_stored_fields(self):
        mediafile = MediaFileFactory(
            media_type="video",
            static_thumbnail=SimpleUploadedFile("static.webp", b"static", content_type="image/webp"),
            thumbnail=SimpleUploadedFile("motion.webp", b"motion", content_type="image/webp"),
            preview=SimpleUploadedFile("playback.mp4", b"video", content_type="video/mp4"),
        )

        self.assertEqual(mediafile.card_image.name, mediafile.static_thumbnail.name)
        self.assertEqual(mediafile.motion_thumbnail.name, mediafile.thumbnail.name)
        self.assertEqual(mediafile.static_thumbnail_url, mediafile.static_thumbnail.url)
        self.assertEqual(mediafile.thumbnail_url, mediafile.thumbnail.url)
        self.assertEqual(mediafile.preview_url, mediafile.preview.url)

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

    def test_media_files_video_card_uses_static_image_and_lazy_motion_preview(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(
            parent=archive,
            media_type="video",
            original_filename="clip.mp4",
            static_thumbnail=SimpleUploadedFile("clip-static.webp", b"static", content_type="image/webp"),
            thumbnail=SimpleUploadedFile("clip-motion.webp", b"motion", content_type="image/webp"),
        )

        response = self.client.get(reverse("caidapp:media_files"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'src="{mediafile.static_thumbnail.url}"')
        self.assertContains(response, f'data-motion-src="{mediafile.thumbnail.url}"')
        self.assertContains(response, "js-motion-preview")
        self.assertContains(response, "media-type-video-badge")

    def test_bulk_identity_select_is_limited_to_workgroup_and_searchable(self):
        UploadedArchiveFactory(owner=self.caiduser)
        zeta = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Zeta", code="Z-002")
        alpha = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha", code="A-001")
        IndividualIdentityFactory(name="Other Workgroup")

        response = self.client.get(reverse("caidapp:media_files"))

        form = response.context["form_bulk_processing"]
        identities = list(form.fields["identity"].queryset)
        self.assertEqual(identities, [alpha, zeta])
        self.assertEqual(form.fields["identity"].label_from_instance(alpha), "Alpha (A-001)")
        self.assertEqual(form.fields["identity"].label_from_instance(zeta), "Zeta (Z-002)")
        self.assertIn("js-searchable-select", form.fields["identity"].widget.attrs["class"])

    def test_init_identification_candidates_use_same_filter_as_init_worker(self):
        lynx = TaxonFactory(name="Lynx lynx")
        wolf = TaxonFactory(name="Canis lupus")
        self.caiduser.workgroup.check_taxon_before_identification = True
        self.caiduser.workgroup.default_taxon_for_identification = lynx
        self.caiduser.workgroup.save(update_fields=["check_taxon_before_identification", "default_taxon_for_identification"])
        archive = UploadedArchiveFactory(owner=self.caiduser)
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")
        init_mediafile = MediaFileFactory(parent=archive, original_filename="init-candidate.jpg")
        skipped_mediafile = MediaFileFactory(parent=archive, original_filename="other-taxon.jpg")
        missing_identity_mediafile = MediaFileFactory(parent=archive, original_filename="representative-without-identity.jpg")
        AnimalObservationFactory(
            mediafile=init_mediafile,
            taxon=lynx,
            identity=identity,
            identity_is_representative=True,
        )
        AnimalObservationFactory(
            mediafile=skipped_mediafile,
            taxon=wolf,
            identity=identity,
            identity_is_representative=True,
        )
        AnimalObservationFactory(
            mediafile=missing_identity_mediafile,
            taxon=lynx,
            identity=None,
            identity_is_representative=True,
        )
        AnimalObservationFactory(
            mediafile=missing_identity_mediafile,
            taxon=lynx,
            identity=identity,
            identity_is_representative=False,
        )

        response = self.client.get(reverse("caidapp:media_files"), {"init_identification_candidates": "true"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["number_of_mediafiles"], 1)
        self.assertContains(response, "init-candidate.jpg")
        self.assertNotContains(response, "other-taxon.jpg")
        self.assertNotContains(response, "representative-without-identity.jpg")


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

    def test_identity_card_displays_animal_code(self):
        IndividualIdentityFactory(
            owner_workgroup=self.caiduser.workgroup,
            name="Alpha",
            code="A-01",
        )

        response = self.client.get(reverse("caidapp:individual_identities"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "<strong>Code:</strong> A-01", html=True)

    def test_identity_list_sorts_by_mediafile_count(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        alpha = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")
        beta = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Beta")
        gamma = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Gamma")

        alpha_mediafile = MediaFileFactory(parent=archive, original_filename="alpha.jpg")
        beta_first_mediafile = MediaFileFactory(parent=archive, original_filename="beta-1.jpg")
        beta_second_mediafile = MediaFileFactory(parent=archive, original_filename="beta-2.jpg")

        AnimalObservationFactory(mediafile=alpha_mediafile, identity=alpha)
        AnimalObservationFactory(mediafile=beta_first_mediafile, identity=beta)
        AnimalObservationFactory(mediafile=beta_second_mediafile, identity=beta)

        response = self.client.get(
            reverse("caidapp:individual_identities"),
            {"view": "list", "sort": "mediafile_count", "dir": "desc"},
        )

        self.assertEqual(response.status_code, 200)
        ordered_ids = [identity.id for identity in response.context["page_obj"]]
        self.assertEqual(ordered_ids[:3], [beta.id, alpha.id, gamma.id])

    def test_identity_list_uses_per_page_query_parameter(self):
        for index in range(7):
            IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name=f"Identity {index:02d}")

        response = self.client.get(
            reverse("caidapp:individual_identities"),
            {"view": "list", "per_page": "6"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context["page_obj"].object_list), 6)
        self.assertEqual(response.context["records_per_page"], 6)

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
            ["Antelope", "Zebra"],
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
        another_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Beta")
        mediafile = MediaFileFactory(parent=UploadedArchiveFactory(owner=self.caiduser))
        AnimalObservationFactory(mediafile=mediafile, identity=identity)
        AnimalObservationFactory(mediafile=mediafile, identity=identity)
        AnimalObservationFactory(mediafile=mediafile, identity=another_identity)

        response = self.client.get(reverse("caidapp:export_identities_csv"))

        self.assertEqual(response.status_code, 200)
        df = pd.read_csv(StringIO(response.content.decode()))
        self.assertEqual(
            list(df.columns),
            [
                "Unnamed: 0",
                "id",
                "name",
                "code",
                "juv_code",
                "sex",
                "coat_type",
                "birth_date",
                "death_date",
                "note",
                "mediafile_count",
            ],
        )
        self.assertEqual(int(df.iloc[0]["id"]), identity.id)
        self.assertEqual(int(df.loc[df["id"] == identity.id, "mediafile_count"].iloc[0]), 1)
        self.assertEqual(int(df.loc[df["id"] == another_identity.id, "mediafile_count"].iloc[0]), 1)

    def test_identity_export_xlsx_includes_mediafile_count(self):
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")
        mediafile = MediaFileFactory(parent=UploadedArchiveFactory(owner=self.caiduser))
        AnimalObservationFactory(mediafile=mediafile, identity=identity)

        response = self.client.get(reverse("caidapp:export_identities_xlsx"))

        self.assertEqual(response.status_code, 200)
        df = pd.read_excel(BytesIO(response.content))
        self.assertEqual(
            list(df.columns),
            ["id", "name", "code", "juv_code", "sex", "coat_type", "birth_date", "death_date", "note", "mediafile_count"],
        )
        self.assertEqual(int(df.iloc[0]["mediafile_count"]), 1)

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

    def test_identity_import_blank_code_and_juv_code_keep_existing_values(self):
        identity = IndividualIdentityFactory(
            owner_workgroup=self.caiduser.workgroup,
            name="Alpha",
            code="A-01",
            juv_code="J-01",
        )

        response = self.client.post(
            reverse("caidapp:import_identities"),
            {
                "spreadsheet_file": self._csv_upload(
                    "identities.csv",
                    [{"id": identity.id, "name": "Alpha", "code": "", "juv_code": ""}],
                )
            },
        )

        self.assertRedirects(response, reverse("caidapp:individual_identities"))
        identity.refresh_from_db()
        self.assertEqual(identity.code, "A-01")
        self.assertEqual(identity.juv_code, "J-01")

    def test_identity_import_clear_token_removes_code_and_juv_code(self):
        identity = IndividualIdentityFactory(
            owner_workgroup=self.caiduser.workgroup,
            name="Alpha",
            code="A-01",
            juv_code="J-01",
        )

        response = self.client.post(
            reverse("caidapp:import_identities"),
            {
                "spreadsheet_file": self._csv_upload(
                    "identities.csv",
                    [
                        {
                            "id": identity.id,
                            "name": "Alpha",
                            "code": forms.SPREADSHEET_CLEAR_TOKEN,
                            "juv_code": forms.SPREADSHEET_CLEAR_TOKEN,
                        }
                    ],
                )
            },
        )

        self.assertRedirects(response, reverse("caidapp:individual_identities"))
        identity.refresh_from_db()
        self.assertIsNone(identity.code)
        self.assertIsNone(identity.juv_code)

    def test_locality_export_includes_id_column(self):
        locality = LocalityFactory(owner=self.caiduser, name="North Meadow")

        response = self.client.get(reverse("caidapp:export_localities"))

        self.assertEqual(response.status_code, 200)
        df = pd.read_csv(StringIO(response.content.decode()))
        self.assertEqual(list(df.columns), ["Unnamed: 0", "id", "name", "location"])
        self.assertEqual(int(df.iloc[0]["id"]), locality.id)

    def test_locality_import_prefers_id_for_rename(self):
        locality = LocalityFactory(owner=self.caiduser, name="North Meadow", location="49.123,16.456")

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
        self.assertEqual(locality.location, "49.123,16.456")
        self.assertEqual(models.Locality.objects.filter(owner=self.caiduser).count(), 1)

    def test_locality_import_clear_token_removes_location(self):
        locality = LocalityFactory(owner=self.caiduser, name="North Meadow", location="49.123,16.456")

        response = self.client.post(
            reverse("caidapp:import_localities"),
            {
                "spreadsheet_file": self._csv_upload(
                    "localities.csv",
                    [{"id": locality.id, "name": "North Meadow", "location": forms.SPREADSHEET_CLEAR_TOKEN}],
                )
            },
        )

        self.assertRedirects(response, reverse("caidapp:localities"))
        locality.refresh_from_db()
        self.assertIsNone(locality.location)


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

    def test_mediafile_update_lists_owned_albums_and_updates_membership(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive)
        selected_album = AlbumFactory(owner=self.caiduser)
        removed_album = AlbumFactory(owner=self.caiduser)
        removed_album.mediafiles.add(mediafile)

        response = self.client.get(reverse("caidapp:media_file_update", args=[mediafile.id]))

        self.assertContains(response, 'data-allow-new-albums="true"')
        self.assertContains(response, selected_album.name)
        self.assertContains(response, removed_album.name)

        response = self.client.post(
            reverse("caidapp:media_file_update", args=[mediafile.id]),
            self._base_mediafile_update_post_data(
                mediafile,
                total_forms=0,
                initial_forms=0,
                extra_form_data={
                    "album_hashes": [str(selected_album.hash), "New album from media file"],
                },
            ),
        )

        self.assertEqual(response.status_code, 302)
        self.assertTrue(selected_album.mediafiles.filter(pk=mediafile.pk).exists())
        self.assertFalse(removed_album.mediafiles.filter(pk=mediafile.pk).exists())
        self.assertTrue(
            models.Album.objects.filter(
                owner=self.caiduser,
                name="New album from media file",
                mediafiles=mediafile,
            ).exists()
        )

    def test_mediafile_update_uses_referer_as_next_when_next_is_missing(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive)
        source_url = reverse("caidapp:sequences") + "?media_type=image&page=2"

        response = self.client.get(
            reverse("caidapp:media_file_update", args=[mediafile.id]),
            HTTP_REFERER=source_url,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["next"], source_url)

    def test_mediafile_update_save_redirects_to_next(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive)
        source_url = reverse("caidapp:sequences") + "?media_type=image&page=2"

        response = self.client.post(
            reverse("caidapp:media_file_update", args=[mediafile.id]),
            self._base_mediafile_update_post_data(
                mediafile,
                total_forms=0,
                initial_forms=0,
                extra_form_data={"next": source_url},
            ),
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, source_url)

    def test_mediafile_update_add_related_links_keep_observation_prefix(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive)
        AnimalObservationFactory(mediafile=mediafile)

        response = self.client.get(reverse("caidapp:media_file_update", args=[mediafile.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "select_observation_prefix=observations-0")
        self.assertContains(response, "select_observation_prefix=observations-__prefix__")

    def test_create_taxon_returns_to_mediafile_with_created_selection_params(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive)
        next_url = reverse("caidapp:media_file_update", args=[mediafile.id])

        response = self.client.post(
            reverse("caidapp:add_taxon"),
            {"name": "Caracal", "parent": "", "next": next_url},
            QUERY_STRING="select_observation_prefix=observations-2",
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn(next_url, response.url)
        self.assertIn("created_taxon_id=", response.url)
        self.assertIn("select_observation_prefix=observations-2", response.url)

    def test_create_identity_returns_to_mediafile_with_created_selection_params(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive)
        next_url = reverse("caidapp:media_file_update", args=[mediafile.id])

        response = self.client.post(
            reverse("caidapp:individual_identity_create"),
            {"name": "Alpha", "code": "", "juv_code": "", "sex": "U", "coat_type": "U", "note": "", "next": next_url},
            QUERY_STRING="select_observation_prefix=observations-1",
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn(next_url, response.url)
        self.assertIn("created_identity_id=", response.url)
        self.assertIn("select_observation_prefix=observations-1", response.url)
        mediafile.refresh_from_db()
        self.assertFalse(hasattr(mediafile, "identity"))

    def test_predicted_taxon_select_uses_taxon_id_and_refreshes_searchable_dropdown(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        predicted_taxon = TaxonFactory(name="Panthera pardus")
        mediafile = MediaFileFactory(parent=archive)
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

    def test_video_detail_has_static_animated_and_video_preview_tabs(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(
            parent=archive,
            media_type="video",
            original_filename="clip.mp4",
            static_thumbnail=SimpleUploadedFile("clip-static.webp", b"static", content_type="image/webp"),
            thumbnail=SimpleUploadedFile("clip-animated.webp", b"animated", content_type="image/webp"),
        )
        AnimalObservationFactory(mediafile=mediafile)

        response = self.client.get(reverse("caidapp:media_file_update", args=[mediafile.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Static frame + bbox")
        self.assertContains(response, "Animated preview")
        self.assertContains(response, f'id="video-pane{mediafile.id}"')
        self.assertContains(response, 'id="annotCanvas"')
        self.assertContains(response, reverse("caidapp:stream_video", args=[mediafile.id]))

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
        self.assertFalse(mediafile.observations.exclude(identity=None).exists())

    def test_bulk_delete_redirects_to_first_page_when_last_page_disappears(self):
        identities = [
            IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name=f"Identity {index:02d}")
            for index in range(25)
        ]
        list_url = reverse("caidapp:individual_identities")
        return_url = f"{list_url}?view=list&sort=mediafile_count&dir=desc&page=2"

        response = self.client.post(
            return_url,
            {
                "bulk_action": "delete_selected",
                "confirm_delete": "yes",
                "selected_identity_ids": [str(identities[-1].id)],
                "return_url": return_url,
            },
        )

        expected_url = f"{list_url}?view=list&sort=mediafile_count&dir=desc&page=1"
        self.assertRedirects(response, expected_url)
        self.assertFalse(models.IndividualIdentity.objects.filter(id=identities[-1].id).exists())

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

    def test_list_view_edit_links_preserve_tabular_layout_in_next(self):
        for index in range(25):
            IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name=f"Identity {index:02d}")
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Zulu")

        response = self.client.get(reverse("caidapp:individual_identities"), {"view": "list", "page": 2})

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            f'{reverse("caidapp:individual_identity_update", args=[identity.id])}'
            "?next=/caidapp/individual_identities/%3Fview%3Dlist%26page%3D2",
        )

    def test_identity_update_links_to_sequences(self):
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")

        response = self.client.get(reverse("caidapp:individual_identity_update", args=[identity.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'{reverse("caidapp:observations")}?identity={identity.id}')
        self.assertContains(response, "Sequences")
        self.assertContains(response, f'{reverse("caidapp:sequences")}?individual_identity_id={identity.id}')

    def test_identity_update_redirects_back_to_tabular_list_when_next_is_set(self):
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha", code="A-1")
        next_url = f"{reverse('caidapp:individual_identities')}?view=list&page=2"

        response = self.client.post(
            reverse("caidapp:individual_identity_update", args=[identity.id]),
            {
                "next": next_url,
                "name": "Alpha edited",
                "code": "A-1",
                "juv_code": "",
                "sex": "U",
                "coat_type": "U",
                "note": "",
                "birth_date": "",
                "death_date": "",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], next_url)
        identity.refresh_from_db()
        self.assertEqual(identity.name, "Alpha edited")

    def test_identity_update_shows_mediafile_count_link(self):
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile_one = MediaFileFactory(parent=archive)
        mediafile_two = MediaFileFactory(parent=archive)
        AnimalObservationFactory(mediafile=mediafile_one, identity=identity)
        AnimalObservationFactory(mediafile=mediafile_two, identity=identity)

        response = self.client.get(reverse("caidapp:individual_identity_update", args=[identity.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Media Files (2)")
        self.assertContains(response, reverse("caidapp:individual_identity_mediafiles", args=[identity.id]))

    def test_identity_update_nav_shows_mediafile_and_sequence_counts(self):
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        mediafile_one = MediaFileFactory(parent=archive, sequence=sequence)
        mediafile_two = MediaFileFactory(parent=archive, sequence=sequence)
        AnimalObservationFactory(mediafile=mediafile_one, identity=identity)
        AnimalObservationFactory(mediafile=mediafile_two, identity=identity)

        response = self.client.get(reverse("caidapp:individual_identity_update", args=[identity.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Media Files (2)")
        self.assertContains(response, "Sequences (1)")


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

    def test_sequence_bbox_preview_uses_image_area_wrapper(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        mediafile = MediaFileFactory(parent=archive, sequence=sequence, original_filename="bbox.jpg")
        AnimalObservationFactory(
            mediafile=mediafile,
            bbox_x_center=0.5,
            bbox_y_center=0.5,
            bbox_width=0.2,
            bbox_height=0.3,
        )

        response = self.client.get(reverse("caidapp:sequences"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "bbox-preview-frame")
        self.assertContains(response, "bbox-image-area")
        self.assertContains(response, "syncBboxPreviewFrame")

    def test_video_cards_use_static_image_and_lazy_motion_preview(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        mediafile = MediaFileFactory(
            parent=archive,
            sequence=sequence,
            media_type="video",
            original_filename="clip.mp4",
            static_thumbnail=SimpleUploadedFile("clip-static.webp", b"static", content_type="image/webp"),
            thumbnail=SimpleUploadedFile("clip-motion.webp", b"motion", content_type="image/webp"),
        )
        AnimalObservationFactory(mediafile=mediafile, bbox_x_center=0.5, bbox_y_center=0.5)

        response = self.client.get(reverse("caidapp:sequences"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'src="{mediafile.static_thumbnail.url}"')
        self.assertContains(response, f'data-motion-src="{mediafile.thumbnail.url}"')
        self.assertContains(response, "media-type-video-badge")


    def test_bulk_identity_post_rejects_identity_from_other_workgroup(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive, original_filename="selected.jpg")
        other_identity = IndividualIdentityFactory(name="Foreign Identity")

        response = self.client.post(
            reverse("caidapp:sequences"),
            {
                "selected_mediafile_ids": [str(mediafile.id)],
                "identity": str(other_identity.id),
                "btnBulkProcessing_id_identity": "Apply to selection",
            },
        )

        self.assertEqual(response.status_code, 200)
        mediafile.refresh_from_db()
        self.assertTrue(mediafile.observations.get().is_no_detection_placeholder)

    def test_mediafiles_bbox_preview_uses_image_area_wrapper(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive, original_filename="bbox.jpg")
        AnimalObservationFactory(
            mediafile=mediafile,
            bbox_x_center=0.5,
            bbox_y_center=0.5,
            bbox_width=0.2,
            bbox_height=0.3,
        )

        response = self.client.get(reverse("caidapp:media_files"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "bbox-preview-frame")
        self.assertContains(response, "bbox-image-area")
        self.assertContains(response, "syncBboxPreviewFrame")

    def test_mediafiles_star_uses_first_observation_representative_flag(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")
        mediafile = MediaFileFactory(parent=archive)
        AnimalObservationFactory(mediafile=mediafile, identity=identity, identity_is_representative=True)

        response = self.client.get(reverse("caidapp:media_files"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'data-mediafile-id="{mediafile.id}"')
        self.assertContains(response, "bi-star-fill")
        self.assertContains(response, "Alpha")

    def test_mediafiles_card_shows_observation_identities_with_expand_badge(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        first_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")
        second_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Beta")
        third_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Gamma")
        mediafile = MediaFileFactory(parent=archive, identity=None, original_filename="multi-identity.jpg")
        AnimalObservationFactory(mediafile=mediafile, identity=first_identity)
        AnimalObservationFactory(mediafile=mediafile, identity=second_identity)
        AnimalObservationFactory(mediafile=mediafile, identity=third_identity)

        response = self.client.get(reverse("caidapp:media_files"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "multi-identity.jpg")
        self.assertContains(response, "Alpha")
        self.assertContains(response, "Beta")
        self.assertContains(response, "+1")
        self.assertContains(response, f'id="mediafile-identities-{mediafile.id}"')
        self.assertContains(response, "Gamma")

    def test_mediafiles_star_is_informational_for_multiple_observations(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        first_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")
        second_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Beta")
        mediafile = MediaFileFactory(parent=archive, identity=None, identity_is_representative=False)
        AnimalObservationFactory(mediafile=mediafile, identity=first_identity, identity_is_representative=True)
        AnimalObservationFactory(mediafile=mediafile, identity=second_identity, identity_is_representative=False)

        response = self.client.get(reverse("caidapp:media_files"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, f'data-mediafile-id="{mediafile.id}"')
        self.assertContains(response, "multiple observations")
        self.assertContains(response, "Edit representative flags in the media file detail.")

    def test_representative_mediafiles_star_is_informational_for_multiple_observations(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        taxon = TaxonFactory(name="Lynx")
        archive.taxon_for_identification = taxon
        archive.save(update_fields=["taxon_for_identification"])
        first_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")
        second_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Beta")
        mediafile = MediaFileFactory(parent=archive, identity=None, identity_is_representative=False)
        AnimalObservationFactory(mediafile=mediafile, taxon=taxon, identity=first_identity, identity_is_representative=True)
        AnimalObservationFactory(mediafile=mediafile, taxon=taxon, identity=second_identity, identity_is_representative=False)

        response = self.client.get(reverse("caidapp:media_files"), {"identity_is_representative": "true"})

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, f'data-mediafile-id="{mediafile.id}"')
        self.assertContains(response, "multiple observations")
        self.assertContains(response, "Edit representative flags in the media file detail.")

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

    def test_sequence_bulk_verify_taxon_skips_no_detection_placeholder_without_taxon(self):
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
        observation = mediafile.observations.get()
        self.assertTrue(observation.is_no_detection_placeholder)
        self.assertFalse(observation.taxon_verified)

    def test_sequence_bulk_set_full_image_bbox_updates_selected_mediafiles(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        selected_sequence = SequenceFactory(uploaded_archive=archive)
        other_sequence = SequenceFactory(uploaded_archive=archive)
        selected_mediafile = MediaFileFactory(parent=archive, sequence=selected_sequence, original_filename="selected.jpg")
        selected_without_observation = MediaFileFactory(
            parent=archive,
            sequence=selected_sequence,
            original_filename="selected-no-obs.jpg",
        )
        untouched_mediafile = MediaFileFactory(parent=archive, sequence=other_sequence, original_filename="other.jpg")
        selected_observation = AnimalObservationFactory(mediafile=selected_mediafile)
        AnimalObservationFactory(mediafile=untouched_mediafile)

        response = self.client.post(
            reverse("caidapp:sequences"),
            {
                "selected_sequence_ids": [str(selected_sequence.id)],
                "btnBulkProcessing_set_full_image_bbox": "1",
            },
        )

        self.assertEqual(response.status_code, 302)
        selected_observation.refresh_from_db()
        self.assertEqual(selected_observation.bbox_x_center, 0.5)
        self.assertEqual(selected_observation.bbox_y_center, 0.5)
        self.assertEqual(selected_observation.bbox_width, 1.0)
        self.assertEqual(selected_observation.bbox_height, 1.0)

        created_observation = selected_without_observation.observations.get()
        self.assertEqual(created_observation.bbox_x_center, 0.5)
        self.assertEqual(created_observation.bbox_y_center, 0.5)
        self.assertEqual(created_observation.bbox_width, 1.0)
        self.assertEqual(created_observation.bbox_height, 1.0)

        untouched_observation = untouched_mediafile.observations.get()
        self.assertIsNone(untouched_observation.bbox_x_center)
        self.assertIsNone(untouched_observation.bbox_y_center)
        self.assertIsNone(untouched_observation.bbox_width)
        self.assertIsNone(untouched_observation.bbox_height)

    def test_sequence_bulk_remove_bbox_keeps_identities(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        selected_sequence = SequenceFactory(uploaded_archive=archive)
        other_sequence = SequenceFactory(uploaded_archive=archive)
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup)
        selected_mediafile = MediaFileFactory(parent=archive, sequence=selected_sequence)
        selected_without_observation = MediaFileFactory(parent=archive, sequence=selected_sequence)
        untouched_mediafile = MediaFileFactory(parent=archive, sequence=other_sequence)
        selected_observation = AnimalObservationFactory(
            mediafile=selected_mediafile,
            identity=identity,
            bbox_x_center=0.5,
            bbox_y_center=0.5,
            bbox_width=0.4,
            bbox_height=0.3,
        )
        untouched_observation = AnimalObservationFactory(
            mediafile=untouched_mediafile,
            bbox_x_center=0.5,
            bbox_y_center=0.5,
            bbox_width=0.4,
            bbox_height=0.3,
        )

        response = self.client.post(
            reverse("caidapp:sequences"),
            {
                "selected_sequence_ids": [str(selected_sequence.id)],
                "btnBulkProcessing_remove_bbox": "1",
            },
        )

        self.assertEqual(response.status_code, 302)
        selected_observation.refresh_from_db()
        self.assertEqual(selected_observation.identity, identity)
        self.assertIsNone(selected_observation.bbox_x_center)
        self.assertIsNone(selected_observation.bbox_y_center)
        self.assertIsNone(selected_observation.bbox_width)
        self.assertIsNone(selected_observation.bbox_height)
        self.assertTrue(selected_without_observation.observations.get().is_no_detection_placeholder)

        untouched_observation.refresh_from_db()
        self.assertEqual(untouched_observation.bbox_x_center, 0.5)
        self.assertEqual(untouched_observation.bbox_y_center, 0.5)
        self.assertEqual(untouched_observation.bbox_width, 0.4)
        self.assertEqual(untouched_observation.bbox_height, 0.3)

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
        mediafile = MediaFileFactory(parent=archive, sequence=sequence, original_filename="verified.jpg")
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

    def test_sequence_download_configuration_uses_sequence_urls(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive, original_filename="selected.jpg")
        session = self.client.session
        session["sequence_download_mediafile_ids"] = [mediafile.id]
        session.save()

        response = self.client.get(reverse("caidapp:download_sequences"))

        self.assertContains(response, "Download sequences")
        self.assertContains(response, reverse("caidapp:download_zip_for_sequences"))

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

    def test_create_sequence_combines_selected_mediafiles_and_removes_empty_sequences(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        first_sequence = SequenceFactory(uploaded_archive=archive, local_id=10)
        second_sequence = SequenceFactory(uploaded_archive=archive, local_id=20)
        remaining_sequence = SequenceFactory(uploaded_archive=archive, local_id=30)
        first_mediafile = MediaFileFactory(parent=archive, sequence=first_sequence, original_filename="first.jpg")
        second_mediafile = MediaFileFactory(parent=archive, sequence=second_sequence, original_filename="second.jpg")
        remaining_mediafile = MediaFileFactory(parent=archive, sequence=remaining_sequence, original_filename="third.jpg")
        sibling_mediafile = MediaFileFactory(parent=archive, sequence=remaining_sequence, original_filename="fourth.jpg")

        response = self.client.post(
            reverse("caidapp:sequences"),
            {
                "btnCreateSequence": "1",
                "selected_mediafile_ids": [str(first_mediafile.id), str(remaining_mediafile.id)],
            },
        )

        self.assertEqual(response.status_code, 302)
        first_mediafile.refresh_from_db()
        second_mediafile.refresh_from_db()
        remaining_mediafile.refresh_from_db()
        sibling_mediafile.refresh_from_db()

        self.assertEqual(first_mediafile.sequence_id, remaining_mediafile.sequence_id)
        self.assertNotEqual(first_mediafile.sequence_id, first_sequence.id)
        self.assertFalse(models.Sequence.objects.filter(id=first_sequence.id).exists())
        self.assertTrue(models.Sequence.objects.filter(id=remaining_sequence.id).exists())
        self.assertEqual(sibling_mediafile.sequence_id, remaining_sequence.id)
        self.assertEqual(second_mediafile.sequence_id, second_sequence.id)
        self.assertEqual(models.MediaFile.objects.filter(sequence_id=first_mediafile.sequence_id).count(), 2)

    def test_mediafiles_view_create_sequence_uses_form_selection(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        first_sequence = SequenceFactory(uploaded_archive=archive, local_id=10)
        second_sequence = SequenceFactory(uploaded_archive=archive, local_id=20)
        first_mediafile = MediaFileFactory(parent=archive, sequence=first_sequence, original_filename="first.jpg")
        second_mediafile = MediaFileFactory(parent=archive, sequence=second_sequence, original_filename="second.jpg")

        response = self.client.post(
            reverse("caidapp:media_files"),
            {
                "form-TOTAL_FORMS": "2",
                "form-INITIAL_FORMS": "2",
                "form-MIN_NUM_FORMS": "0",
                "form-MAX_NUM_FORMS": "1000",
                "form-0-id": str(first_mediafile.id),
                "form-0-selected": "on",
                "form-1-id": str(second_mediafile.id),
                "form-1-selected": "on",
                "btnCreateSequence": "1",
            },
        )

        self.assertEqual(response.status_code, 302)
        first_mediafile.refresh_from_db()
        second_mediafile.refresh_from_db()

        self.assertEqual(first_mediafile.sequence_id, second_mediafile.sequence_id)
        self.assertNotEqual(first_mediafile.sequence_id, first_sequence.id)
        self.assertFalse(models.Sequence.objects.filter(id=first_sequence.id).exists())
        self.assertFalse(models.Sequence.objects.filter(id=second_sequence.id).exists())
        self.assertEqual(models.MediaFile.objects.filter(sequence_id=first_mediafile.sequence_id).count(), 2)

    def test_mediafiles_bulk_identity_update_updates_all_selected_observations(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        target_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")
        first_mediafile = MediaFileFactory(parent=archive, original_filename="first.jpg")
        second_mediafile = MediaFileFactory(parent=archive, original_filename="second.jpg")
        first_observation = AnimalObservationFactory(mediafile=first_mediafile)
        second_observation = AnimalObservationFactory(mediafile=second_mediafile)

        response = self.client.post(
            reverse("caidapp:media_files"),
            {
                "form-TOTAL_FORMS": "2",
                "form-INITIAL_FORMS": "2",
                "form-MIN_NUM_FORMS": "0",
                "form-MAX_NUM_FORMS": "1000",
                "form-0-id": str(first_mediafile.id),
                "form-0-selected": "on",
                "form-1-id": str(second_mediafile.id),
                "form-1-selected": "on",
                "identity": str(target_identity.id),
                "btnBulkProcessing_id_identity": "Apply to selection",
            },
        )

        self.assertEqual(response.status_code, 200)
        first_mediafile.refresh_from_db()
        second_mediafile.refresh_from_db()
        first_observation.refresh_from_db()
        second_observation.refresh_from_db()

        self.assertEqual(first_observation.identity, target_identity)
        self.assertEqual(second_observation.identity, target_identity)

    def test_mediafiles_bulk_identity_update_skips_multiple_observations(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        target_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")
        mediafile = MediaFileFactory(parent=archive, original_filename="multiple.jpg")
        first_observation = AnimalObservationFactory(mediafile=mediafile)
        second_observation = AnimalObservationFactory(mediafile=mediafile)

        response = self.client.post(
            reverse("caidapp:media_files"),
            {
                "form-TOTAL_FORMS": "1",
                "form-INITIAL_FORMS": "1",
                "form-MIN_NUM_FORMS": "0",
                "form-MAX_NUM_FORMS": "1000",
                "form-0-id": str(mediafile.id),
                "form-0-selected": "on",
                "identity": str(target_identity.id),
                "btnBulkProcessing_id_identity": "Apply to selection",
            },
        )

        self.assertEqual(response.status_code, 200)
        first_observation.refresh_from_db()
        second_observation.refresh_from_db()
        self.assertIsNone(first_observation.identity)
        self.assertIsNone(second_observation.identity)
        self.assertContains(response, "Identity was not changed")
        self.assertContains(response, "multiple observations")

    def test_mediafiles_bulk_representative_false_updates_single_observation(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive, original_filename="selected.jpg")
        observation = AnimalObservationFactory(mediafile=mediafile, identity_is_representative=True)

        response = self.client.post(
            reverse("caidapp:media_files"),
            {
                "form-TOTAL_FORMS": "1",
                "form-INITIAL_FORMS": "1",
                "form-MIN_NUM_FORMS": "0",
                "form-MAX_NUM_FORMS": "1000",
                "form-0-id": str(mediafile.id),
                "form-0-selected": "on",
                "btnBulkProcessing_id_identity_is_representative": "Apply to selection",
            },
        )

        self.assertEqual(response.status_code, 200)
        mediafile.refresh_from_db()
        observation.refresh_from_db()
        self.assertFalse(observation.identity_is_representative)

    def test_mediafiles_bulk_representative_skips_multiple_observations(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive, original_filename="multi.jpg")
        first_observation = AnimalObservationFactory(mediafile=mediafile, identity_is_representative=True)
        second_observation = AnimalObservationFactory(mediafile=mediafile, identity_is_representative=True)

        response = self.client.post(
            reverse("caidapp:media_files"),
            {
                "form-TOTAL_FORMS": "1",
                "form-INITIAL_FORMS": "1",
                "form-MIN_NUM_FORMS": "0",
                "form-MAX_NUM_FORMS": "1000",
                "form-0-id": str(mediafile.id),
                "form-0-selected": "on",
                "btnBulkProcessing_id_identity_is_representative": "Apply to selection",
            },
        )

        self.assertEqual(response.status_code, 200)
        mediafile.refresh_from_db()
        first_observation.refresh_from_db()
        second_observation.refresh_from_db()
        self.assertTrue(first_observation.identity_is_representative)
        self.assertTrue(second_observation.identity_is_representative)
        self.assertContains(response, "Representative identity was not changed")
        self.assertContains(response, "multiple observations")

    def test_mediafiles_bulk_set_full_image_bbox_updates_selected_mediafiles(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        selected_mediafile = MediaFileFactory(parent=archive, original_filename="selected.jpg")
        second_selected_mediafile = MediaFileFactory(parent=archive, original_filename="selected-no-obs.jpg")
        untouched_mediafile = MediaFileFactory(parent=archive, original_filename="untouched.jpg")
        selected_observation = AnimalObservationFactory(mediafile=selected_mediafile)
        AnimalObservationFactory(mediafile=untouched_mediafile)

        response = self.client.post(
            reverse("caidapp:media_files"),
            {
                "form-TOTAL_FORMS": "3",
                "form-INITIAL_FORMS": "3",
                "form-MIN_NUM_FORMS": "0",
                "form-MAX_NUM_FORMS": "1000",
                "form-0-id": str(selected_mediafile.id),
                "form-0-selected": "on",
                "form-1-id": str(second_selected_mediafile.id),
                "form-1-selected": "on",
                "form-2-id": str(untouched_mediafile.id),
                "btnBulkProcessing_set_full_image_bbox": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        selected_observation.refresh_from_db()
        self.assertEqual(selected_observation.bbox_x_center, 0.5)
        self.assertEqual(selected_observation.bbox_y_center, 0.5)
        self.assertEqual(selected_observation.bbox_width, 1.0)
        self.assertEqual(selected_observation.bbox_height, 1.0)

        created_observation = second_selected_mediafile.observations.get()
        self.assertEqual(created_observation.bbox_x_center, 0.5)
        self.assertEqual(created_observation.bbox_y_center, 0.5)
        self.assertEqual(created_observation.bbox_width, 1.0)
        self.assertEqual(created_observation.bbox_height, 1.0)

        untouched_observation = untouched_mediafile.observations.get()
        self.assertIsNone(untouched_observation.bbox_x_center)
        self.assertIsNone(untouched_observation.bbox_y_center)
        self.assertIsNone(untouched_observation.bbox_width)
        self.assertIsNone(untouched_observation.bbox_height)

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

    def test_mediafile_export_uses_observation_values_and_relative_bbox(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        observed_taxon = TaxonFactory(name="Observed taxon")
        mediafile = MediaFileFactory(
            parent=archive,
            original_filename="two.jpg",
            note="Mother with three juveniles",
        )
        first = AnimalObservationFactory(
            mediafile=mediafile,
            taxon=observed_taxon,
            bbox_x_center=0.5,
            bbox_y_center=0.4,
            bbox_width=0.2,
            bbox_height=0.4,
            identity_is_representative=True,
        )
        second = AnimalObservationFactory(mediafile=mediafile, taxon=None)

        response = self.client.get(reverse("caidapp:download_csv_for_mediafiles"))

        self.assertEqual(response.status_code, 200)
        df = pd.read_csv(StringIO(response.content.decode()))
        rows = df[df["mediafile_id"] == mediafile.id].sort_values("observation_id")
        self.assertEqual(list(rows["observation_id"]), [first.id, second.id])
        self.assertEqual(rows.iloc[0]["predicted_category"], "Observed taxon")
        self.assertTrue(pd.isna(rows.iloc[1]["predicted_category"]))
        self.assertEqual(rows.iloc[0]["bbox_cx"], 0.5)
        self.assertEqual(rows.iloc[0]["bbox_cy"], 0.4)
        self.assertEqual(rows.iloc[0]["bbox_w"], 0.2)
        self.assertEqual(rows.iloc[0]["bbox_h"], 0.4)
        self.assertTrue(rows.iloc[0]["identity_is_representative"])
        self.assertEqual(set(rows["mediafile_note"]), {"Mother with three juveniles"})
        self.assertNotIn("note", df.columns)

    def test_mediafile_csv_and_xlsx_exports_have_same_observation_rows(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive, original_filename="same.jpg")
        AnimalObservationFactory(mediafile=mediafile)
        AnimalObservationFactory(mediafile=mediafile)

        csv_response = self.client.get(reverse("caidapp:download_csv_for_mediafiles"))
        xlsx_response = self.client.get(reverse("caidapp:download_xlsx_for_mediafiles"))
        csv_df = pd.read_csv(StringIO(csv_response.content.decode()))
        xlsx_df = pd.read_excel(BytesIO(xlsx_response.content))

        self.assertEqual(list(csv_df.columns), list(xlsx_df.columns))
        self.assertEqual(list(csv_df["observation_id"]), list(xlsx_df["observation_id"]))

    def test_observation_import_updates_and_creates(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        taxon = TaxonFactory(name="Imported taxon")
        mediafile = MediaFileFactory(parent=archive)
        observation = AnimalObservationFactory(mediafile=mediafile, taxon=None)
        frame = pd.DataFrame(
            [
                {
                    "mediafile_id": mediafile.id,
                    "observation_id": observation.id,
                    "taxon_id": taxon.id,
                    "bbox_cx": 0.5,
                    "bbox_cy": 0.5,
                    "bbox_w": 0.4,
                    "bbox_h": 0.2,
                    "mediafile_note": "Original path description",
                },
                {
                    "mediafile_id": mediafile.id,
                    "observation_id": "",
                    "taxon_id": taxon.id,
                    "mediafile_note": "Original path description",
                },
            ]
        )

        created, updated = views._import_observation_dataframe(frame, self.caiduser)

        self.assertEqual((created, updated), (1, 1))
        observation.refresh_from_db()
        self.assertEqual(observation.taxon, taxon)
        self.assertEqual(observation.bbox_width, 0.4)
        self.assertEqual(mediafile.observations.count(), 2)
        mediafile.refresh_from_db()
        self.assertEqual(mediafile.note, "Original path description")

    def test_observation_import_creates_rows_for_multiple_identities(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        first_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup)
        second_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup)
        mediafile = MediaFileFactory(parent=archive)
        observation = AnimalObservationFactory(mediafile=mediafile, identity=None)
        frame = pd.DataFrame(
            [
                {
                    "mediafile_id": mediafile.id,
                    "observation_id": observation.id,
                    "identity_id": first_identity.id,
                },
                {
                    "mediafile_id": mediafile.id,
                    "observation_id": "",
                    "identity_id": second_identity.id,
                },
            ]
        )

        created, updated = views._import_observation_dataframe(frame, self.caiduser)

        self.assertEqual((created, updated), (1, 1))
        self.assertEqual(
            set(mediafile.observations.values_list("identity_id", flat=True)),
            {first_identity.id, second_identity.id},
        )

    def test_observation_import_accepts_legacy_note_and_clear_token(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive, note="Old note")
        observation = AnimalObservationFactory(mediafile=mediafile)

        views._import_observation_dataframe(
            pd.DataFrame(
                [
                    {
                        "mediafile_id": mediafile.id,
                        "observation_id": observation.id,
                        "note": forms.SPREADSHEET_CLEAR_TOKEN,
                    }
                ]
            ),
            self.caiduser,
        )

        mediafile.refresh_from_db()
        self.assertEqual(mediafile.note, "")

    def test_observation_import_rejects_conflicting_notes_for_same_mediafile(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive, note="Unchanged")
        observation = AnimalObservationFactory(mediafile=mediafile)
        frame = pd.DataFrame(
            [
                {
                    "mediafile_id": mediafile.id,
                    "observation_id": observation.id,
                    "mediafile_note": "First note",
                },
                {
                    "mediafile_id": mediafile.id,
                    "observation_id": "",
                    "mediafile_note": "Conflicting note",
                },
            ]
        )

        with self.assertRaisesRegex(ValueError, "Row 3: conflicting mediafile_note"):
            views._import_observation_dataframe(frame, self.caiduser)

        mediafile.refresh_from_db()
        self.assertEqual(mediafile.note, "Unchanged")
        self.assertEqual(mediafile.observations.count(), 1)

    def test_observation_import_invalid_bbox_rolls_back_all_rows(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        first = AnimalObservationFactory(mediafile=MediaFileFactory(parent=archive), taxon=None)
        second = AnimalObservationFactory(mediafile=MediaFileFactory(parent=archive), taxon=None)
        taxon = TaxonFactory()
        frame = pd.DataFrame(
            [
                {"mediafile_id": first.mediafile_id, "observation_id": first.id, "taxon_id": taxon.id},
                {
                    "mediafile_id": second.mediafile_id,
                    "observation_id": second.id,
                    "bbox_cx": 0.95,
                    "bbox_cy": 0.5,
                    "bbox_w": 0.2,
                    "bbox_h": 0.2,
                },
            ]
        )

        with self.assertRaisesRegex(ValueError, "Row 3: bbox must fit"):
            views._import_observation_dataframe(frame, self.caiduser)

        first.refresh_from_db()
        self.assertIsNone(first.taxon)

    @patch("caidapp.tasks.import_observations_task.delay", return_value=Mock(id="observation-import-task"))
    def test_observation_import_explains_identity_path_and_cancels_entire_file(self, delay_mock):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        observation = AnimalObservationFactory(mediafile=MediaFileFactory(parent=archive), identity=None)
        upload = SimpleUploadedFile(
            "observations.csv",
            (
                "mediafile_id,observation_id,unique_name\n"
                f"{observation.mediafile_id},{observation.id},LY20/Zadni_paste/2020-09-20_B514_Julien_P.JPG\n"
            ).encode(),
            content_type="text/csv",
        )

        response = self.client.post(reverse("caidapp:import_observations"), {"spreadsheet_file": upload})
        self.assertEqual(response.status_code, 302)
        observation_import = models.ObservationImport.objects.get()
        delay_mock.assert_called_once_with(observation_import.id)
        with patch.object(tasks.import_observations_task, "update_state"):
            tasks.import_observations_task.run(observation_import.id)
        response = self.client.get(response.url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Import cancelled — no rows were changed.")
        self.assertContains(response, "matched 0 records; it must identify exactly one record")
        self.assertContains(response, "looks like an original file path")
        observation.refresh_from_db()
        self.assertIsNone(observation.identity)

    @patch("caidapp.tasks.import_observations_task.delay", return_value=Mock(id="observation-import-task"))
    def test_observation_import_runs_on_worker_and_persists_result(self, delay_mock):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive)
        upload = SimpleUploadedFile(
            "observations.csv",
            f"mediafile_id\n{mediafile.id}\n".encode(),
            content_type="text/csv",
        )

        response = self.client.post(reverse("caidapp:import_observations"), {"spreadsheet_file": upload})

        self.assertEqual(response.status_code, 302)
        observation_import = models.ObservationImport.objects.get()
        delay_mock.assert_called_once_with(observation_import.id)
        with patch.object(tasks.import_observations_task, "update_state"):
            tasks.import_observations_task.run(observation_import.id)
        observation_import.refresh_from_db()
        self.assertEqual(observation_import.status, models.ObservationImport.STATUS_SUCCEEDED)
        self.assertEqual((observation_import.created_count, observation_import.updated_count), (1, 0))
        self.assertEqual(mediafile.observations.count(), 1)

    def test_observation_import_diagnostics_collects_errors_from_all_rows(self):
        errors = views._collect_observation_import_errors(
            pd.DataFrame([{"mediafile_id": "bad-first"}, {"mediafile_id": "bad-second"}]),
            self.caiduser,
        )

        self.assertEqual(len(errors), 2)
        self.assertIn("Row 2: mediafile_id must be a positive integer", errors[0])
        self.assertIn("Row 3: mediafile_id must be a positive integer", errors[1])

    def test_observation_import_option_creates_missing_locality(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive, locality=None)
        locality_name = "New import locality"

        created, updated = views._import_observation_dataframe(
            pd.DataFrame([{"mediafile_id": mediafile.id, "locality name": locality_name}]),
            self.caiduser,
            create_missing_localities=True,
        )

        self.assertEqual((created, updated), (1, 0))
        mediafile.refresh_from_db()
        self.assertEqual(mediafile.locality.name, locality_name)
        self.assertEqual(mediafile.locality.owner, self.caiduser)

    def test_observation_import_missing_locality_requires_checkbox(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive, locality=None)
        frame = pd.DataFrame([{"mediafile_id": mediafile.id, "locality name": "New import locality"}])

        with self.assertRaisesRegex(ValueError, "locality name 'New import locality' matched 0 records"):
            views._import_observation_dataframe(frame, self.caiduser)

        self.assertFalse(models.Locality.objects.filter(name="New import locality", owner=self.caiduser).exists())

    def test_observation_import_option_creates_missing_identity(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive)

        created, updated = views._import_observation_dataframe(
            pd.DataFrame([{"mediafile_id": mediafile.id, "unique_name": "New import identity", "code": "NI-01"}]),
            self.caiduser,
            create_missing_identities=True,
        )

        self.assertEqual((created, updated), (1, 0))
        observation = mediafile.observations.get()
        self.assertEqual(observation.identity.name, "New import identity")
        self.assertEqual(observation.identity.code, "NI-01")
        self.assertEqual(observation.identity.owner_workgroup, self.caiduser.workgroup)

    def test_observation_import_identity_path_cannot_create_identity(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive)
        identity_name = "LY20/Zadni_paste/2020-09-20_B514_Julien_P.JPG"

        with self.assertRaisesRegex(ValueError, "looks like an original file path"):
            views._import_observation_dataframe(
                pd.DataFrame([{"mediafile_id": mediafile.id, "unique_name": identity_name}]),
                self.caiduser,
                create_missing_identities=True,
            )

        self.assertFalse(
            models.IndividualIdentity.objects.filter(name=identity_name, owner_workgroup=self.caiduser.workgroup).exists()
        )

    def test_observation_import_invalid_id_does_not_create(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive)
        before = mediafile.observations.count()
        frame = pd.DataFrame([{"mediafile_id": mediafile.id, "observation_id": "1.5"}])

        with self.assertRaisesRegex(ValueError, "observation_id must be a positive integer"):
            views._import_observation_dataframe(frame, self.caiduser)

        self.assertEqual(mediafile.observations.count(), before)

    def test_observation_import_rejects_partial_bbox(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        observation = AnimalObservationFactory(mediafile=MediaFileFactory(parent=archive))
        frame = pd.DataFrame(
            [
                {
                    "mediafile_id": observation.mediafile_id,
                    "observation_id": observation.id,
                    "bbox_cx": 0.4,
                    "bbox_cy": 0.5,
                    "bbox_w": 0.2,
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "bbox requires bbox_cx"):
            views._import_observation_dataframe(frame, self.caiduser)

    def test_sequence_select_all_filtered_uses_all_pages(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        selected_ids = []
        for index in range(25):
            sequence = SequenceFactory(uploaded_archive=archive)
            mediafile = MediaFileFactory(parent=archive, sequence=sequence, original_filename=f"page-{index}.jpg")
            selected_ids.append(mediafile.id)
        excluded_sequence = SequenceFactory(uploaded_archive=archive)
        excluded = MediaFileFactory(parent=archive, sequence=excluded_sequence, media_type="video")

        response = self.client.post(
            f'{reverse("caidapp:sequences")}?media_type=image',
            {"btnDownloadSequences": "1", "select_all_filtered": "on"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(set(self.client.session["sequence_download_mediafile_ids"]), set(selected_ids))
        self.assertNotIn(excluded.id, self.client.session["sequence_download_mediafile_ids"])

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
        mediafile = MediaFileFactory(parent=archive, sequence=sequence, original_filename="alpha.jpg")
        other_mediafile = MediaFileFactory(parent=archive, sequence=other_sequence, original_filename="beta.jpg")
        AnimalObservationFactory(mediafile=mediafile, identity=identity)
        AnimalObservationFactory(mediafile=other_mediafile, identity=other_identity)

        response = self.client.get(reverse("caidapp:sequences"), {"individual_identity_id": identity.id})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Identity: Alpha")
        self.assertContains(response, reverse("caidapp:individual_identity_update", args=[identity.id]))
        self.assertContains(response, mediafile.original_filename)
        self.assertNotContains(response, other_mediafile.original_filename)

    def test_sequence_view_shows_active_locality_detail_link(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        locality = LocalityFactory(owner=self.caiduser, name="Forest Camp")
        other_locality = LocalityFactory(owner=self.caiduser, name="River Bank")
        sequence = SequenceFactory(uploaded_archive=archive)
        other_sequence = SequenceFactory(uploaded_archive=archive)
        mediafile = MediaFileFactory(parent=archive, sequence=sequence, locality=locality, original_filename="forest.jpg")
        other_mediafile = MediaFileFactory(
            parent=archive,
            sequence=other_sequence,
            locality=other_locality,
            original_filename="river.jpg",
        )

        response = self.client.get(reverse("caidapp:sequences"), {"locality_hash": locality.hash})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Locality: Forest Camp")
        self.assertContains(response, reverse("caidapp:update_locality", args=[locality.id]))
        self.assertContains(response, mediafile.original_filename)
        self.assertNotContains(response, other_mediafile.original_filename)

    def test_media_files_view_shows_active_filter_detail_links(self):
        archive = UploadedArchiveFactory(owner=self.caiduser, name="Visible upload")
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")
        locality = LocalityFactory(owner=self.caiduser, name="Forest Camp")
        MediaFileFactory(parent=archive, identity=identity, locality=locality, original_filename="alpha.jpg")

        response = self.client.get(
            reverse("caidapp:media_files"),
            {
                "uploadedarchive_id": archive.id,
                "individual_identity_id": identity.id,
                "locality_hash": locality.hash,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Filtered by:")
        self.assertContains(response, "Upload: Visible upload")
        self.assertContains(response, "Identity: Alpha")
        self.assertContains(response, "Locality: Forest Camp")
        self.assertContains(response, reverse("caidapp:uploadedarchive_detail", args=[archive.id]))
        self.assertContains(response, f'{reverse("caidapp:observations")}?uploadedarchive={archive.id}')
        self.assertContains(response, f'{reverse("caidapp:sequences")}?uploadedarchive_id={archive.id}')
        self.assertContains(response, reverse("caidapp:individual_identity_update", args=[identity.id]))
        self.assertContains(response, reverse("caidapp:update_locality", args=[locality.id]))

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
            original_filename="alpha.jpg",
        )
        second_mediafile = MediaFileFactory(
            parent=archive,
            sequence=second_sequence,
            original_filename="beta.jpg",
        )
        other_mediafile = MediaFileFactory(
            parent=archive,
            sequence=other_sequence,
            original_filename="gamma.jpg",
        )
        AnimalObservationFactory(mediafile=first_mediafile, identity=first_identity)
        AnimalObservationFactory(mediafile=second_mediafile, identity=second_identity)
        AnimalObservationFactory(mediafile=other_mediafile, identity=other_identity)

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
        self.assertContains(response, f'{reverse("caidapp:observations")}?taxon={wolf.id}')
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
        self.assertContains(response, f'{reverse("caidapp:observations")}?locality={locality.id}')
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
        self.assertEqual(selected_mediafile.observations.get().identity.name, "Charles")
        self.assertTrue(other_mediafile.observations.get().is_no_detection_placeholder)

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
        observation = mediafile.observations.get()
        self.assertEqual(mediafile.locality.name, "Brdy")
        self.assertEqual(observation.taxon.name, "Lynx")
        self.assertEqual(observation.identity.name, "Charles")

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

    def test_filename_metadata_warns_about_mediafiles_with_multiple_observations(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        single_observation_mediafile = MediaFileFactory(parent=archive, sequence=sequence)
        multiple_observation_mediafile = MediaFileFactory(parent=archive, sequence=sequence)
        AnimalObservationFactory(mediafile=multiple_observation_mediafile)
        AnimalObservationFactory(mediafile=multiple_observation_mediafile)
        session = self.client.session
        session["filename_metadata_mediafile_ids"] = [
            single_observation_mediafile.id,
            multiple_observation_mediafile.id,
        ]
        session["filename_metadata_return_url"] = reverse("caidapp:sequences")
        session["filename_metadata_source_label"] = "Sequences"
        session.save()

        response = self.client.get(reverse("caidapp:apply_filename_metadata_to_mediafiles"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["multiple_observation_mediafile_count"], 1)
        self.assertContains(response, "1 media file contains multiple observations")
        self.assertContains(response, "those values will be skipped")

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
        self.assertTrue(mediafile.observations.get().is_no_detection_placeholder)

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
        self.assertEqual(mediafile.observations.get().identity.name, "Charles")

    def test_sequence_filename_metadata_fills_only_empty_fields_by_default(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        old_locality = LocalityFactory(owner=self.caiduser, name="OldLocality")
        old_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="OldIdentity")
        mediafile = MediaFileFactory(
            parent=archive,
            sequence=sequence,
            locality=old_locality,
            original_filename="NewLocality/NewIdentity/first.jpg",
        )
        observation = AnimalObservationFactory(mediafile=mediafile, identity=old_identity)

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
        observation.refresh_from_db()
        self.assertEqual(mediafile.locality, old_locality)
        self.assertEqual(observation.identity, old_identity)

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
        observation.refresh_from_db()
        self.assertEqual(mediafile.locality.name, "NewLocality")
        self.assertEqual(observation.identity.name, "NewIdentity")

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
        observation = mediafile.observations.get()
        self.assertEqual(observation.taxon.name, "Lynx")
        self.assertEqual(observation.identity.name, "Charles")

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
        first_observation.refresh_from_db()
        second_observation.refresh_from_db()
        self.assertIsNone(first_observation.taxon)
        self.assertIsNone(second_observation.taxon)
        self.assertIsNone(first_observation.identity)
        self.assertIsNone(second_observation.identity)


class ObservationViewTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory()
        self.user = self.caiduser.user
        self.client.login(username=self.user.username, password="test123")

    def test_observations_view_lists_each_observation_separately(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive, local_id=22)
        mediafile = MediaFileFactory(parent=archive, sequence=sequence, original_filename="three-animals.jpg")
        first = AnimalObservationFactory(mediafile=mediafile, bbox_x_center=0.2, bbox_y_center=0.4, bbox_width=0.2, bbox_height=0.3)
        second = AnimalObservationFactory(mediafile=mediafile, bbox_x_center=0.7, bbox_y_center=0.5, bbox_width=0.2, bbox_height=0.3)

        response = self.client.get(reverse("caidapp:observations"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'id="observation-{first.id}"')
        self.assertContains(response, f'id="observation-{second.id}"')
        self.assertContains(response, "Sequence 22")
        self.assertContains(response, "observation-bbox")
        self.assertContains(response, "syncObservationBboxFrame")
        self.assertContains(response, f"#observation-{first.id}")
        self.assertContains(response, "observation-group-collapsed")
        self.assertContains(response, "observation-group-collapsed-card")
        self.assertContains(response, "observation-group-break")
        self.assertContains(response, "observation-group-hover-highlight")
        self.assertContains(response, "animateObservationGridChange")
        self.assertContains(response, "animateObservationCollapse")
        self.assertContains(response, "js-toggle-observation-group")
        self.assertContains(response, "js-expand-all-observation-groups")
        self.assertContains(response, "groupCheckbox.indeterminate")
        self.assertContains(response, 'id="observation-action-toolbar"')
        self.assertContains(response, "Sort by")
        self.assertContains(response, ">Media file</a>", html=False)
        self.assertContains(response, "2 observations · 1 media file")
        self.assertContains(response, "Containing media files")
        self.assertContains(response, "Download containing media files")
        self.assertContains(response, "Captured: oldest")
        self.assertContains(response, 'id="observation-select-page"')
        self.assertContains(response, "All matching · 2 observations · 1 media file · 1 sequence")
        self.assertNotContains(response, "Select all 2 matching filter")
        self.assertContains(response, "Edit metadata…")
        self.assertContains(response, "Edit observations")
        self.assertContains(response, "Sequences")
        self.assertContains(response, 'class="dropdown-item" type="button" data-bs-toggle="collapse" data-bs-target="#observation-bulk-fields"')
        self.assertNotContains(response, 'class="btn btn-sm btn-outline-primary" type="button" data-bs-toggle="collapse" data-bs-target="#observation-bulk-fields"')
        self.assertContains(response, '<i class="bi bi-lightning"></i> Actions')
        self.assertNotContains(response, '<i class="bi bi-three-dots"></i> Actions')
        self.assertContains(response, "> Data</button>")
        self.assertContains(response, "Metadata uses one row per observation")
        self.assertContains(response, "A media file without observations")
        self.assertNotContains(response, "Export includes every observation belonging to a matching media file")
        self.assertContains(response, 'id="observation-scope-modal"')
        self.assertContains(response, "scopeModalMessage.textContent")
        self.assertContains(response, 'data-action-scope="sequences" data-confirm-always="true"')
        self.assertContains(response, "Split complete containing sequences")
        self.assertContains(response, "Media files outside the current filter may also be affected")
        self.assertContains(response, "directSequenceMediafileCount")
        self.assertContains(response, "additional media files")
        self.assertContains(response, f'data-mediafile-id="{mediafile.id}"')
        self.assertContains(response, f'data-sequence-id="{sequence.id}"')
        self.assertNotContains(response, "Observation selection")
        self.assertNotContains(response, "Observation fields")
        self.assertContains(response, 'title="2 observations grouped by Sequence 22 · 1 media file on this page"')
        self.assertContains(response, 'class="observation-group-controls"', count=1)
        self.assertNotContains(response, "observation-group-toggle-label")
        self.assertContains(response, ".observation-group-collapsed .observation-group-lead .observation-image-select{display:none}")
        self.assertContains(response, ".observation-group-expanded .observation-group-select{display:none}")

    def test_single_observation_group_has_no_group_card_controls(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive, local_id=23)
        observation = AnimalObservationFactory(mediafile=MediaFileFactory(parent=archive, sequence=sequence))

        response = self.client.get(reverse("caidapp:observations"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'id="observation-{observation.id}"')
        self.assertNotContains(response, 'class="observation-group-controls"')

    def test_observations_large_cards_mode_uses_two_column_card_layout(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        observation = AnimalObservationFactory(mediafile=MediaFileFactory(parent=archive))

        response = self.client.get(reverse("caidapp:observations"), {"view": "large_cards"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["view_mode"], "large_cards")
        self.assertContains(response, f'id="observation-{observation.id}"')
        self.assertContains(response, 'class="observation-groups observation-groups-large"')
        self.assertContains(response, ".observation-groups-large .observation-group-member{flex-basis:calc(50% - 1.5rem)")
        self.assertContains(response, 'title="Large cards view" aria-label="Large cards view" aria-current="page"')
        self.assertContains(response, '<i class="bi bi-grid-3x3-gap" aria-hidden="true"></i>')
        self.assertContains(response, '<i class="bi bi-grid" aria-hidden="true"></i>')
        self.assertContains(response, '<i class="bi bi-list-ul" aria-hidden="true"></i>')

    def test_observation_card_shows_icon_fields_and_copyable_details_modal(self):
        locality = LocalityFactory(owner=self.caiduser, name="Forest locality")
        archive = UploadedArchiveFactory(owner=self.caiduser, name="Camera upload")
        mediafile = MediaFileFactory(
            parent=archive,
            locality=locality,
            captured_at=timezone.make_aware(datetime.datetime(2025, 4, 3, 14, 25)),
            original_filename="forest/lynx.jpg",
        )
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Lynx A")
        taxon = TaxonFactory(name="Lynx lynx")
        observation = AnimalObservationFactory(mediafile=mediafile, taxon=taxon, identity=identity)

        response = self.client.get(reverse("caidapp:observations"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, f"Observation #{observation.id}")
        self.assertContains(response, '<i class="bi bi-tag"')
        self.assertContains(response, '<i class="bi bi-fingerprint"')
        self.assertContains(response, '<i class="bi bi-image"')
        self.assertContains(response, '<i class="bi bi-geo-alt"')
        self.assertContains(response, '<i class="bi bi-calendar3"')
        self.assertContains(response, "Lynx lynx")
        self.assertContains(response, "Lynx A")
        self.assertContains(response, "Forest locality")
        self.assertContains(response, f'href="{reverse("caidapp:observations")}?taxon={taxon.id}"')
        self.assertContains(response, f'href="{reverse("caidapp:observations")}?identity={identity.id}"')
        self.assertContains(response, f'href="{reverse("caidapp:observations")}?locality={locality.id}"')
        self.assertContains(response, "2025-04-03 14:25")
        self.assertContains(response, 'id="observation-info-modal"')
        self.assertContains(response, f'id="observation-info-template-{observation.id}"')
        self.assertContains(response, 'class="btn btn-sm btn-link text-secondary observation-info-button')
        self.assertContains(response, "Values can be selected and copied.")
        self.assertContains(response, "observationInfoModalBody.replaceChildren")

    def test_sequence_link_opens_only_that_sequences_observations(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        selected_sequence = SequenceFactory(uploaded_archive=archive, local_id=24)
        other_sequence = SequenceFactory(uploaded_archive=archive, local_id=25)
        selected = AnimalObservationFactory(
            mediafile=MediaFileFactory(parent=archive, sequence=selected_sequence)
        )
        other = AnimalObservationFactory(
            mediafile=MediaFileFactory(parent=archive, sequence=other_sequence)
        )

        unfiltered_response = self.client.get(reverse("caidapp:observations"))
        sequence_url = f'{reverse("caidapp:observations")}?sequence={selected_sequence.id}'
        self.assertContains(unfiltered_response, f'href="{sequence_url}"')

        filtered_response = self.client.get(sequence_url)

        self.assertEqual(filtered_response.status_code, 200)
        self.assertContains(filtered_response, f'id="observation-{selected.id}"')
        self.assertNotContains(filtered_response, f'id="observation-{other.id}"')
        self.assertEqual(filtered_response.context["number_of_observations"], 1)

    def test_observations_sort_sequences_by_newest_captured_date(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        older_sequence = SequenceFactory(uploaded_archive=archive, local_id=40)
        newer_sequence = SequenceFactory(uploaded_archive=archive, local_id=41)
        older = AnimalObservationFactory(
            mediafile=MediaFileFactory(
                parent=archive,
                sequence=older_sequence,
                captured_at=timezone.now() - datetime.timedelta(days=2),
            )
        )
        newer = AnimalObservationFactory(
            mediafile=MediaFileFactory(
                parent=archive,
                sequence=newer_sequence,
                captured_at=timezone.now() - datetime.timedelta(days=1),
            )
        )

        response = self.client.get(reverse("caidapp:observations"), {"sort": "captured_desc"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["sort_by"], "captured_desc")
        self.assertEqual(response.context["observation_groups"][0]["observations"][0].id, newer.id)
        self.assertEqual(response.context["observation_groups"][1]["observations"][0].id, older.id)

    def test_taxon_sort_disables_ambiguous_grouping(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        taxon_z = TaxonFactory(name="Zebra taxon")
        taxon_a = TaxonFactory(name="Aardvark taxon")
        observation_z = AnimalObservationFactory(mediafile=MediaFileFactory(parent=archive, sequence=sequence), taxon=taxon_z)
        observation_a = AnimalObservationFactory(mediafile=MediaFileFactory(parent=archive, sequence=sequence), taxon=taxon_a)

        response = self.client.get(
            reverse("caidapp:observations"),
            {"group": "sequence", "sort": "taxon"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["group_by"], "none")
        self.assertTrue(response.context["grouping_disabled_for_sort"])
        self.assertEqual([group["observations"][0].id for group in response.context["observation_groups"]], [observation_a.id, observation_z.id])
        self.assertContains(response, "Grouping was disabled because one media file or sequence can contain multiple taxa")

    def test_grouped_context_sort_modes_render(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        AnimalObservationFactory(
            mediafile=MediaFileFactory(
                parent=archive,
                sequence=sequence,
                locality=LocalityFactory(owner=self.caiduser),
                original_filename="sorted.jpg",
                captured_at=timezone.now(),
            )
        )

        for sort_by in ("captured_asc", "captured_desc", "locality", "filename", "observation_id"):
            with self.subTest(sort_by=sort_by):
                response = self.client.get(reverse("caidapp:observations"), {"sort": sort_by})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.context["sort_by"], sort_by)

    def test_observations_list_mode_renders_one_collapsible_row_per_group(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive, local_id=31)
        mediafile = MediaFileFactory(parent=archive, sequence=sequence, original_filename="two.jpg")
        first = AnimalObservationFactory(mediafile=mediafile, bbox_x_center=0.3)
        second = AnimalObservationFactory(mediafile=mediafile, bbox_x_center=0.7)

        response = self.client.get(reverse("caidapp:observations"), {"view": "list"})

        self.assertEqual(response.status_code, 200)
        group = response.context["observation_groups"][0]
        self.assertContains(response, f"#{first.id}")
        self.assertContains(response, f"#{second.id}")
        self.assertContains(response, "2 observations")
        self.assertContains(response, f'data-observation-id="{first.id}"')
        self.assertContains(response, f'data-observation-id="{second.id}"')
        self.assertContains(response, "js-observation-group-checkbox")
        self.assertContains(response, "js-toggle-observation-list-group")
        self.assertContains(response, 'class="observation-list-group-shell"', count=1)
        self.assertContains(response, 'class="observation-list-group-block"', count=1)
        self.assertContains(response, f'aria-controls="observation-list-group-details-{group["id"]}"')
        self.assertContains(response, f'id="observation-list-group-details-{group["id"]}" class="observation-list-group-details" data-group-id="{group["id"]}" hidden')
        self.assertContains(response, f'id="observation-list-{first.id}" class="observation-list-member" data-group-id=', count=1)
        self.assertContains(response, f'id="observation-list-{second.id}" class="observation-list-member" data-group-id=', count=1)
        self.assertContains(response, "observation-list-detail-table")
        self.assertContains(response, "observation-list-detail-table td{border:0!important")
        self.assertContains(response, "setObservationListGroupExpanded")
        self.assertContains(response, 'details.hidden = !expanded')
        self.assertContains(response, "js-expand-all-observation-groups")
        self.assertContains(response, "js-collapse-all-observation-groups")
        self.assertContains(response, "Merge containing media files into new sequence")

    def test_observations_ungrouped_list_keeps_one_plain_row_per_observation(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive, original_filename="two.jpg")
        first = AnimalObservationFactory(mediafile=mediafile, bbox_x_center=0.3)
        second = AnimalObservationFactory(mediafile=mediafile, bbox_x_center=0.7)

        response = self.client.get(reverse("caidapp:observations"), {"view": "list", "group": "none"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "<th>Observation</th>", html=True)
        self.assertContains(response, "<th>Media file</th>", html=True)
        self.assertContains(response, f'id="observation-list-{first.id}" class="observation-list-member"', count=1)
        self.assertContains(response, f'id="observation-list-{second.id}" class="observation-list-member"', count=1)
        self.assertNotContains(response, 'class="observation-list-group-shell"')
        self.assertNotContains(response, 'aria-controls="observation-list-group-details-')

    def test_observation_bulk_bbox_updates_selected_observation_only(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        selected_mediafile = MediaFileFactory(parent=archive)
        selected_first = AnimalObservationFactory(mediafile=selected_mediafile, bbox_x_center=0.2)
        selected_second = AnimalObservationFactory(mediafile=selected_mediafile, bbox_x_center=0.8)
        untouched = AnimalObservationFactory(mediafile=MediaFileFactory(parent=archive), bbox_x_center=0.4)

        response = self.client.post(
            reverse("caidapp:observations"),
            {
                "selected_observation_ids": [str(selected_first.id)],
                "btnBulkProcessing_set_full_image_bbox": "1",
            },
        )

        self.assertEqual(response.status_code, 302)
        selected_first.refresh_from_db()
        selected_second.refresh_from_db()
        untouched.refresh_from_db()
        self.assertEqual(selected_first.bbox_width, 1.0)
        self.assertNotEqual(selected_second.bbox_width, 1.0)
        self.assertNotEqual(untouched.bbox_width, 1.0)

    def test_observation_bulk_taxon_updates_selected_observation_only(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive)
        selected = AnimalObservationFactory(mediafile=mediafile, taxon_verified=True, taxon_verified_at=timezone.now())
        sibling = AnimalObservationFactory(mediafile=mediafile)
        sibling_taxon_id = sibling.taxon_id
        taxon = TaxonFactory()

        response = self.client.post(
            reverse("caidapp:observations"),
            {
                "selected_observation_ids": [str(selected.id)],
                "taxon": str(taxon.id),
                "btnBulkProcessing_id_taxon": "1",
            },
        )

        self.assertEqual(response.status_code, 302)
        selected.refresh_from_db()
        sibling.refresh_from_db()
        self.assertEqual(selected.taxon_id, taxon.id)
        self.assertFalse(selected.taxon_verified)
        self.assertIsNone(selected.taxon_verified_at)
        self.assertEqual(sibling.taxon_id, sibling_taxon_id)

    def test_observation_bulk_identity_updates_selected_observation_only(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive)
        old_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup)
        selected = AnimalObservationFactory(
            mediafile=mediafile,
            identity=old_identity,
            identity_is_representative=True,
        )
        sibling = AnimalObservationFactory(mediafile=mediafile)
        sibling_identity_id = sibling.identity_id
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup)

        response = self.client.post(
            reverse("caidapp:observations"),
            {
                "selected_observation_ids": [str(selected.id)],
                "identity": str(identity.id),
                "btnBulkProcessing_id_identity": "1",
            },
        )

        self.assertEqual(response.status_code, 302)
        selected.refresh_from_db()
        sibling.refresh_from_db()
        self.assertEqual(selected.identity_id, identity.id)
        self.assertFalse(selected.identity_is_representative)
        self.assertEqual(sibling.identity_id, sibling_identity_id)

    def test_observation_representative_requires_identity(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        observation = AnimalObservationFactory(
            mediafile=MediaFileFactory(parent=archive),
            identity=None,
            identity_is_representative=False,
        )

        response = self.client.post(
            reverse("caidapp:observations"),
            {
                "selected_observation_ids": [str(observation.id)],
                "identity_is_representative": "on",
                "btnBulkProcessing_id_identity_is_representative": "1",
            },
        )

        self.assertEqual(response.status_code, 302)
        observation.refresh_from_db()
        self.assertFalse(observation.identity_is_representative)

    def test_observation_bulk_action_without_explicit_selection_is_noop(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        observation = AnimalObservationFactory(
            mediafile=MediaFileFactory(parent=archive),
            bbox_x_center=0.4,
            bbox_y_center=0.4,
            bbox_width=0.2,
            bbox_height=0.2,
        )

        response = self.client.post(
            reverse("caidapp:observations"),
            {"btnBulkProcessing_remove_bbox": "1"},
        )

        self.assertEqual(response.status_code, 302)
        observation.refresh_from_db()
        self.assertEqual(observation.bbox_x_center, 0.4)

    def test_delete_selected_observations_deletes_only_exact_selection(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive)
        selected = AnimalObservationFactory(mediafile=mediafile, bbox_x_center=0.3)
        sibling = AnimalObservationFactory(mediafile=mediafile, bbox_x_center=0.7)

        response = self.client.post(
            reverse("caidapp:observations"),
            {
                "selected_observation_ids": [str(selected.id)],
                "btnDeleteObservations": "1",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertFalse(AnimalObservation.objects.filter(id=selected.id).exists())
        self.assertTrue(AnimalObservation.objects.filter(id=sibling.id).exists())

    def test_delete_last_observation_keeps_no_detection_placeholder(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive)
        selected = AnimalObservationFactory(mediafile=mediafile, bbox_x_center=0.3)

        response = self.client.post(
            reverse("caidapp:observations"),
            {
                "selected_observation_ids": [str(selected.id)],
                "btnDeleteObservations": "1",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertFalse(AnimalObservation.objects.filter(id=selected.id).exists())
        remaining = AnimalObservation.objects.get(mediafile=mediafile)
        self.assertTrue(remaining.is_no_detection_placeholder)

    def test_observation_select_all_mutates_only_workgroup_images(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        editable_observation = AnimalObservationFactory(
            mediafile=MediaFileFactory(parent=archive),
            bbox_x_center=0.5,
            bbox_y_center=0.5,
            bbox_width=0.2,
            bbox_height=0.2,
        )
        owner = CaidUserFactory()
        shared_mediafile = MediaFileFactory(parent=UploadedArchiveFactory(owner=owner))
        shared_observation = AnimalObservationFactory(
            mediafile=shared_mediafile,
            bbox_x_center=0.5,
            bbox_y_center=0.5,
            bbox_width=0.2,
            bbox_height=0.2,
        )
        album = AlbumFactory(owner=owner)
        album.mediafiles.add(shared_mediafile)
        models.AlbumShareRole.objects.create(album=album, user=self.caiduser)

        response = self.client.post(
            reverse("caidapp:observations"),
            {"select_all_filtered": "on", "btnBulkProcessing_remove_bbox": "1"},
        )

        self.assertEqual(response.status_code, 302)
        editable_observation.refresh_from_db()
        shared_observation.refresh_from_db()
        self.assertIsNone(editable_observation.bbox_x_center)
        self.assertEqual(shared_observation.bbox_x_center, 0.5)

    def test_observation_select_all_respects_observation_filter(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        selected_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup)
        matching = AnimalObservationFactory(
            mediafile=MediaFileFactory(parent=archive),
            identity=selected_identity,
            bbox_x_center=0.3,
        )
        outside = AnimalObservationFactory(
            mediafile=MediaFileFactory(parent=archive),
            identity=None,
            bbox_x_center=0.7,
        )

        response = self.client.post(
            reverse("caidapp:observations") + f"?identity={selected_identity.id}",
            {"select_all_filtered": "on", "btnBulkProcessing_remove_bbox": "1"},
        )

        self.assertEqual(response.status_code, 302)
        matching.refresh_from_db()
        outside.refresh_from_db()
        self.assertIsNone(matching.bbox_x_center)
        self.assertEqual(outside.bbox_x_center, 0.7)

    def test_observation_bulk_action_rejects_editable_id_outside_current_filter(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        selected_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Selected")
        matching_observation = AnimalObservationFactory(
            mediafile=MediaFileFactory(parent=archive),
            identity=selected_identity,
            bbox_x_center=0.3,
            bbox_y_center=0.5,
            bbox_width=0.2,
            bbox_height=0.2,
        )
        outside_observation = AnimalObservationFactory(
            mediafile=MediaFileFactory(parent=archive),
            bbox_x_center=0.7,
            bbox_y_center=0.5,
            bbox_width=0.2,
            bbox_height=0.2,
        )

        response = self.client.post(
            reverse("caidapp:observations") + f"?identity={selected_identity.id}",
            {
                "selected_observation_ids": [str(outside_observation.id)],
                "btnBulkProcessing_remove_bbox": "1",
            },
        )

        self.assertEqual(response.status_code, 302)
        matching_observation.refresh_from_db()
        outside_observation.refresh_from_db()
        self.assertEqual(matching_observation.bbox_x_center, 0.3)
        self.assertEqual(outside_observation.bbox_x_center, 0.7)

    def test_observation_sequence_action_merges_selected_images(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        first_mediafile = MediaFileFactory(parent=archive, sequence=SequenceFactory(uploaded_archive=archive))
        second_mediafile = MediaFileFactory(parent=archive, sequence=SequenceFactory(uploaded_archive=archive))
        first_observation = AnimalObservationFactory(mediafile=first_mediafile)
        first_sibling = AnimalObservationFactory(mediafile=first_mediafile)
        second_observation = AnimalObservationFactory(mediafile=second_mediafile)

        response = self.client.post(
            reverse("caidapp:observations"),
            {
                "selected_observation_ids": [str(first_observation.id), str(first_sibling.id), str(second_observation.id)],
                "btnCreateSequence": "1",
            },
        )

        self.assertEqual(response.status_code, 302)
        first_mediafile.refresh_from_db()
        second_mediafile.refresh_from_db()
        self.assertEqual(first_mediafile.sequence_id, second_mediafile.sequence_id)

    def test_observation_sequence_action_splits_complete_containing_sequences(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        sequence = SequenceFactory(uploaded_archive=archive)
        unrelated_sequence = SequenceFactory(uploaded_archive=archive)
        first_mediafile = MediaFileFactory(parent=archive, sequence=sequence)
        second_mediafile = MediaFileFactory(parent=archive, sequence=sequence)
        unrelated_mediafile = MediaFileFactory(parent=archive, sequence=unrelated_sequence)
        selected_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup)
        other_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup)
        first_observation = AnimalObservationFactory(mediafile=first_mediafile, identity=selected_identity)
        AnimalObservationFactory(mediafile=second_mediafile, identity=other_identity)
        AnimalObservationFactory(mediafile=unrelated_mediafile)

        response = self.client.post(
            reverse("caidapp:observations") + f"?identity={selected_identity.id}",
            {
                "selected_observation_ids": [str(first_observation.id)],
                "btnDissolveSequences": "1",
            },
        )

        self.assertEqual(response.status_code, 302)
        first_mediafile.refresh_from_db()
        second_mediafile.refresh_from_db()
        unrelated_mediafile.refresh_from_db()
        self.assertNotEqual(first_mediafile.sequence_id, second_mediafile.sequence_id)
        self.assertEqual(unrelated_mediafile.sequence_id, unrelated_sequence.id)

    def test_observation_selection_download_stores_only_selected_editable_images(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        selected_mediafile = MediaFileFactory(parent=archive)
        other_mediafile = MediaFileFactory(parent=archive)
        selected_observation = AnimalObservationFactory(mediafile=selected_mediafile)
        selected_sibling = AnimalObservationFactory(mediafile=selected_mediafile)
        AnimalObservationFactory(mediafile=other_mediafile)

        response = self.client.post(
            reverse("caidapp:observations"),
            {
                "selected_observation_ids": [str(selected_observation.id), str(selected_sibling.id)],
                "btnDownloadObservations": "1",
            },
        )

        self.assertRedirects(response, reverse("caidapp:download_observations"))
        self.assertEqual(self.client.session["observation_download_mediafile_ids"], [selected_mediafile.id])

    def test_identity_filter_does_not_include_sibling_observation_from_same_image(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive, original_filename="two-animals.jpg")
        selected_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Selected")
        other_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Other")
        selected = AnimalObservationFactory(mediafile=mediafile, identity=selected_identity)
        sibling = AnimalObservationFactory(mediafile=mediafile, identity=other_identity)

        response = self.client.get(reverse("caidapp:observations"), {"identity": selected_identity.id})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'id="observation-{selected.id}"')
        self.assertNotContains(response, f'id="observation-{sibling.id}"')

    def test_observation_export_uses_existing_round_trip_rows_for_matching_mediafile(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive, original_filename="two-animals.jpg")
        selected_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Selected")
        other_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Other")
        selected = AnimalObservationFactory(mediafile=mediafile, identity=selected_identity)
        sibling = AnimalObservationFactory(mediafile=mediafile, identity=other_identity)

        response = self.client.get(reverse("caidapp:download_csv_for_observations"), {"identity": selected_identity.id})

        self.assertEqual(response.status_code, 200)
        df = pd.read_csv(StringIO(response.content.decode()))
        self.assertEqual(set(df["observation_id"]), {selected.id, sibling.id})
        self.assertEqual(set(df["original_path"]), {"two-animals.jpg"})
        self.assertContains(
            self.client.get(reverse("caidapp:observations")),
            reverse("caidapp:import_observations"),
        )

    def test_observation_image_export_uses_current_filter_and_download_configuration(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        selected_mediafile = MediaFileFactory(parent=archive, original_filename="selected.jpg")
        other_mediafile = MediaFileFactory(parent=archive, original_filename="other.jpg")
        selected_identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Selected")
        AnimalObservationFactory(mediafile=selected_mediafile, identity=selected_identity)
        AnimalObservationFactory(mediafile=other_mediafile)

        response = self.client.get(reverse("caidapp:prepare_observation_download"), {"identity": selected_identity.id})

        self.assertRedirects(response, reverse("caidapp:download_observations"))
        download_response = self.client.get(reverse("caidapp:download_observations"))
        self.assertContains(download_response, "Export 1 media files")
        self.assertContains(download_response, reverse("caidapp:download_zip_for_observations"))
        self.assertNotContains(download_response, "Import observation metadata")
        self.assertNotContains(download_response, "Download sequences")

    def test_observation_export_includes_blank_row_for_no_detection_placeholder(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        observed_mediafile = MediaFileFactory(parent=archive, original_filename="observed.jpg")
        empty_mediafile = MediaFileFactory(parent=archive, original_filename="empty.jpg")
        AnimalObservationFactory(mediafile=observed_mediafile)

        response = self.client.get(reverse("caidapp:download_csv_for_observations"))

        self.assertEqual(response.status_code, 200)
        df = pd.read_csv(StringIO(response.content.decode()))
        self.assertEqual(list(df["original_path"]), ["observed.jpg", "empty.jpg"])
        self.assertTrue(pd.isna(df[df["original_path"] == "empty.jpg"].iloc[0]["observation_id"]))
        created, updated = views._import_observation_dataframe(df, self.caiduser)
        self.assertEqual((created, updated), (0, 1))
        self.assertTrue(empty_mediafile.observations.get().is_no_detection_placeholder)

    def test_observation_export_search_includes_matching_empty_mediafile(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        empty_mediafile = MediaFileFactory(parent=archive, original_filename="empty-search-target.jpg")

        response = self.client.get(reverse("caidapp:download_csv_for_observations"), {"search": "search-target"})

        self.assertEqual(response.status_code, 200)
        df = pd.read_csv(StringIO(response.content.decode()))
        self.assertEqual(list(df["mediafile_id"]), [empty_mediafile.id])
        self.assertTrue(pd.isna(df.iloc[0]["observation_id"]))

    def test_observation_import_accepts_mediafile_exported_from_shared_album(self):
        owner = CaidUserFactory()
        archive = UploadedArchiveFactory(owner=owner)
        mediafile = MediaFileFactory(parent=archive, original_filename="shared.jpg")
        observation = AnimalObservationFactory(mediafile=mediafile)
        album = AlbumFactory(owner=owner)
        album.mediafiles.add(mediafile)
        models.AlbumShareRole.objects.create(album=album, user=self.caiduser)

        response = self.client.get(reverse("caidapp:download_csv_for_observations"))

        self.assertEqual(response.status_code, 200)
        df = pd.read_csv(StringIO(response.content.decode()))
        shared_rows = df[df["mediafile_id"] == mediafile.id]
        created, updated = views._import_observation_dataframe(shared_rows, self.caiduser)
        self.assertEqual((created, updated), (0, 1))
        self.assertTrue(models.AnimalObservation.objects.filter(id=observation.id).exists())

    def test_has_bbox_filter_excludes_legacy_observation_without_bbox(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive)
        with_bbox = AnimalObservationFactory(mediafile=mediafile, bbox_x_center=0.5, bbox_y_center=0.5, bbox_width=0.3, bbox_height=0.3)
        without_bbox = AnimalObservationFactory(mediafile=MediaFileFactory(parent=archive))

        response = self.client.get(reverse("caidapp:observations"), {"has_bbox": "true"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'id="observation-{with_bbox.id}"')
        self.assertNotContains(response, f'id="observation-{without_bbox.id}"')

    def test_objects_per_image_filters_count_only_real_bounding_boxes(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        no_bbox = AnimalObservationFactory(mediafile=MediaFileFactory(parent=archive))
        one_bbox = AnimalObservationFactory(
            mediafile=MediaFileFactory(parent=archive),
            bbox_x_center=0.5,
            bbox_y_center=0.5,
            bbox_width=0.3,
            bbox_height=0.3,
        )
        multi_mediafile = MediaFileFactory(parent=archive)
        two_bbox_first = AnimalObservationFactory(
            mediafile=multi_mediafile,
            bbox_x_center=0.3,
            bbox_y_center=0.5,
            bbox_width=0.2,
            bbox_height=0.3,
        )
        two_bbox_second = AnimalObservationFactory(
            mediafile=multi_mediafile,
            bbox_x_center=0.7,
            bbox_y_center=0.5,
            bbox_width=0.2,
            bbox_height=0.3,
        )

        zero_response = self.client.get(reverse("caidapp:observations"), {"objects_per_image": "0"})
        self.assertContains(zero_response, f'id="observation-{no_bbox.id}"')
        self.assertNotContains(zero_response, f'id="observation-{one_bbox.id}"')

        one_response = self.client.get(reverse("caidapp:observations"), {"objects_per_image": "1"})
        self.assertContains(one_response, f'id="observation-{one_bbox.id}"')
        self.assertNotContains(one_response, f'id="observation-{no_bbox.id}"')

        multi_response = self.client.get(reverse("caidapp:observations"), {"objects_per_image": "2"})
        self.assertContains(multi_response, f'id="observation-{two_bbox_first.id}"')
        self.assertContains(multi_response, f'id="observation-{two_bbox_second.id}"')
        self.assertNotContains(multi_response, f'id="observation-{one_bbox.id}"')

        minimum_response = self.client.get(reverse("caidapp:observations"), {"objects_per_image_min": "2"})
        self.assertContains(minimum_response, f'id="observation-{two_bbox_first.id}"')
        self.assertNotContains(minimum_response, f'id="observation-{one_bbox.id}"')

        maximum_response = self.client.get(reverse("caidapp:observations"), {"objects_per_image_max": "1"})
        self.assertContains(maximum_response, f'id="observation-{no_bbox.id}"')
        self.assertContains(maximum_response, f'id="observation-{one_bbox.id}"')
        self.assertNotContains(maximum_response, f'id="observation-{two_bbox_first.id}"')

    def test_observations_view_restricts_other_workgroup(self):
        foreign_user = CaidUserFactory()
        foreign_archive = UploadedArchiveFactory(owner=foreign_user)
        foreign_observation = AnimalObservationFactory(mediafile=MediaFileFactory(parent=foreign_archive))

        response = self.client.get(reverse("caidapp:observations"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, f'id="observation-{foreign_observation.id}"')


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

    def test_manual_identification_links_unknown_media_to_its_detail(self):
        archive = UploadedArchiveFactory(owner=self.caiduser, is_for_identification=True)
        mediafile = MediaFileFactory(
            parent=archive,
            image_file="output/missing/images/unknown.jpg",
            original_filename="unknown.jpg",
        )
        queue_item = models.MediafilesForIdentification.objects.create(mediafile=mediafile)

        response = self.client.get(reverse("caidapp:get_individual_identity", args=[queue_item.id]))

        self.assertEqual(response.status_code, 200)
        detail_url = reverse("caidapp:media_file_update", args=[mediafile.id])
        self.assertContains(response, f'href="{detail_url}?next=', html=False)
        self.assertContains(response, "Open media detail")

    def test_reid_selection_updates_target_observation_without_legacy_identity(self):
        archive = UploadedArchiveFactory(owner=self.caiduser, is_for_identification=True)
        mediafile = MediaFileFactory(parent=archive, metadata_json={})
        observation = AnimalObservationFactory(mediafile=mediafile)
        mediafile.metadata_json["reid_observation_id"] = observation.id
        mediafile.save(update_fields=["metadata_json"])
        queue_item = models.MediafilesForIdentification.objects.create(mediafile=mediafile)
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Candidate")

        response = self.client.get(reverse("caidapp:set_individual_identity", args=[queue_item.id, identity.id]))

        self.assertRedirects(response, reverse("caidapp:get_individual_identity"))
        observation.refresh_from_db()
        mediafile.refresh_from_db()
        self.assertEqual(observation.identity, identity)
        self.assertFalse(hasattr(mediafile, "identity"))
        self.assertFalse(models.MediafilesForIdentification.objects.filter(id=queue_item.id).exists())

    def test_reid_selection_with_multiple_observations_requires_manual_choice(self):
        archive = UploadedArchiveFactory(owner=self.caiduser, is_for_identification=True)
        mediafile = MediaFileFactory(parent=archive, metadata_json={})
        AnimalObservationFactory(mediafile=mediafile)
        AnimalObservationFactory(mediafile=mediafile)
        queue_item = models.MediafilesForIdentification.objects.create(mediafile=mediafile)
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Candidate")

        response = self.client.get(reverse("caidapp:set_individual_identity", args=[queue_item.id, identity.id]))

        self.assertRedirects(response, reverse("caidapp:media_file_update", args=[mediafile.id]))
        self.assertTrue(models.MediafilesForIdentification.objects.filter(id=queue_item.id).exists())
        self.assertFalse(mediafile.observations.filter(identity=identity).exists())

    def test_manual_identification_suggestion_menu_links_candidate_media_to_its_detail(self):
        archive = UploadedArchiveFactory(owner=self.caiduser, is_for_identification=True)
        unknown_mediafile = MediaFileFactory(parent=archive, identity=None, metadata_json={})
        queue_item = models.MediafilesForIdentification.objects.create(mediafile=unknown_mediafile)
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Candidate")
        candidate_mediafile = MediaFileFactory(parent=archive, identity=identity, metadata_json={})
        models.MediafileIdentificationSuggestion.objects.create(
            for_identification=queue_item,
            mediafile=candidate_mediafile,
            identity=identity,
            score=0.9,
            name=identity.name,
        )

        response = self.client.get(reverse("caidapp:get_individual_identity", args=[queue_item.id]))

        self.assertEqual(response.status_code, 200)
        detail_url = reverse("caidapp:media_file_update", args=[candidate_mediafile.id])
        self.assertContains(response, 'aria-label="Suggestion actions"', html=False)
        self.assertContains(response, f'href="{detail_url}?next=', html=False)

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

    def test_taxonomy_uploads_keep_archives_sent_to_identification(self):
        identification_taxon = TaxonFactory(name="Lynx")
        taxonomy_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            name="Taxonomy only archive",
            contains_single_taxon=False,
            contains_identities=False,
            is_for_identification=False,
            taxon_for_identification=None,
        )
        sent_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            name="Taxonomy archive sent to identification",
            contains_single_taxon=False,
            contains_identities=False,
            is_for_identification=True,
            taxon_for_identification=identification_taxon,
        )
        direct_identification_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            name="Direct identification archive",
            contains_single_taxon=False,
            contains_identities=False,
            is_for_identification=True,
            taxon_for_identification=None,
        )

        response = self.client.get(reverse("caidapp:uploads"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, taxonomy_archive.name)
        self.assertContains(response, sent_archive.name)
        self.assertNotContains(response, direct_identification_archive.name)

    def test_uploadedarchive_detail_links_to_observations_sequences_and_mediafiles(self):
        TaxonFactory(name=models.TAXON_NOT_CLASSIFIED)
        archive = UploadedArchiveFactory(
            owner=self.caiduser,
            contains_single_taxon=False,
            taxon_for_identification=None,
        )

        response = self.client.get(reverse("caidapp:uploadedarchive_detail", args=[archive.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, archive.name)
        self.assertContains(response, f'{reverse("caidapp:observations")}?uploadedarchive={archive.id}')
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
        self.assertEqual(mediafile.identity_from_observations.name, "Charles")
        self.assertIsNone(other_mediafile.locality)
        self.assertIsNone(other_mediafile.identity_from_observations)

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

    def test_dash_identities_shows_init_identification_csv_download_only_for_admin(self):
        self.caiduser.workgroup_admin = False
        self.caiduser.save(update_fields=["workgroup_admin"])

        response = self.client.get(reverse("caidapp:dash_identities"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, reverse("caidapp:download_init_identification_csv"))
        self.assertNotContains(response, reverse("caidapp:download_run_identification_csv"))

        self.caiduser.workgroup_admin = True
        self.caiduser.save(update_fields=["workgroup_admin"])

        response = self.client.get(reverse("caidapp:dash_identities"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, reverse("caidapp:download_init_identification_csv"))
        self.assertContains(response, reverse("caidapp:download_run_identification_csv"))

    def test_dash_identities_hides_run_and_model_settings_controls_for_non_admin(self):
        self.caiduser.workgroup_admin = False
        self.caiduser.save(update_fields=["workgroup_admin"])

        response = self.client.get(reverse("caidapp:dash_identities"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, reverse("caidapp:train_identification"))
        self.assertNotContains(response, reverse("caidapp:pre_identify"))
        self.assertNotContains(response, reverse("caidapp:workgroup-update", args=[self.caiduser.workgroup.id]))

    def test_dash_identities_shows_run_and_model_settings_controls_for_workgroup_admin(self):
        self.caiduser.workgroup_admin = True
        self.caiduser.save(update_fields=["workgroup_admin"])

        response = self.client.get(reverse("caidapp:dash_identities"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, reverse("caidapp:train_identification"))
        self.assertContains(response, reverse("caidapp:pre_identify"))
        self.assertContains(response, reverse("caidapp:workgroup-update", args=[self.caiduser.workgroup.id]))


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
        self.workgroup.identification_initialized_model = self.identification_model
        self.workgroup.default_taxon_for_identification = None
        self.workgroup.check_taxon_before_identification = False
        self.workgroup.save()

    def test_identification_information_is_workgroup_admin_only(self):
        self.caiduser.workgroup_admin = False
        self.caiduser.save(update_fields=["workgroup_admin"])

        response = self.client.get(reverse("caidapp:identification_information"))

        self.assertEqual(response.status_code, 405)

    def test_identification_information_shows_models_runs_messages_and_counts(self):
        identity = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="Lynx A")
        archive = UploadedArchiveFactory(owner=self.caiduser, is_for_identification=True)
        mediafile = MediaFileFactory(parent=archive, used_for_init_identification=True)
        AnimalObservationFactory(
            mediafile=mediafile,
            identity=identity,
            identity_is_representative=True,
        )
        models.IdentificationRunStatistic.objects.create(
            workgroup=self.workgroup,
            operation="init",
            status="finished",
            image_number=12,
            video_number=2,
            duration_seconds=9.5,
            finished_at=timezone.now(),
            task_id="init-info-task",
        )
        models.IdentificationRunStatistic.objects.create(
            workgroup=self.workgroup,
            operation="identify",
            status="failed",
            image_number=7,
            video_number=1,
            duration_seconds=3.0,
            finished_at=timezone.now(),
            task_id="identify-info-task",
        )
        self.workgroup.identification_init_status = "Finished"
        self.workgroup.identification_init_message = "Reference embeddings created."
        self.workgroup.identification_reid_status = "Failed"
        self.workgroup.identification_reid_message = "Worker returned an error."
        self.workgroup.save(
            update_fields=[
                "identification_init_status",
                "identification_init_message",
                "identification_reid_status",
                "identification_reid_message",
            ]
        )

        response = self.client.get(reverse("caidapp:identification_information"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, self.identification_model.name)
        self.assertContains(response, "Reference embeddings created.")
        self.assertContains(response, "Worker returned an error.")
        self.assertContains(response, "init-info-task")
        self.assertContains(response, "identify-info-task")
        self.assertEqual(response.context["counts"]["identities"], 1)
        self.assertEqual(response.context["counts"]["representative_mediafiles"], 1)
        self.assertEqual(response.context["counts"]["initialized_reference_mediafiles"], 1)

    def test_identification_dashboard_links_to_information_for_admin(self):
        response = self.client.get(reverse("caidapp:dash_identities"))

        self.assertContains(response, reverse("caidapp:identification_information"))

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

    @patch("caidapp.tasks.schedule_reid_identification_for_workgroup")
    @patch("caidapp.tasks.init_identification.apply_async")
    def test_schedule_init_identification_does_not_pre_schedule_reid(self, apply_async_mock, schedule_reid_mock):
        task = Mock()
        task.id = "init-task-1"
        apply_async_mock.return_value = task

        tasks.schedule_init_identification_for_workgroup(self.workgroup, delay_minutes=0)

        schedule_reid_mock.assert_not_called()
        self.workgroup.refresh_from_db()
        self.assertEqual(self.workgroup.identification_scheduled_init_task_id, "init-task-1")
        self.assertEqual(self.workgroup.identification_init_status, "Scheduled")

    @patch("caidapp.tasks.schedule_reid_identification_for_workgroup")
    def test_failed_init_identification_does_not_schedule_reid(self, schedule_reid_mock):
        tasks.init_identification_on_success(
            {"status": "ERROR", "error": "Could not initialize."},
            workgroup_id=self.workgroup.id,
        )

        schedule_reid_mock.assert_not_called()
        self.workgroup.refresh_from_db()
        self.assertEqual(self.workgroup.identification_init_status, "ERROR")

    @patch("caidapp.tasks.schedule_init_identification_for_workgroup")
    @patch("caidapp.tasks.schedule_reid_identification_for_workgroup")
    def test_successful_init_records_initialized_model_after_completion(self, schedule_reid_mock, schedule_init_mock):
        tasks.init_identification_on_success(
            {"status": "DONE", "message": "Initialized."},
            workgroup_id=self.workgroup.id,
            identification_model_id=self.identification_model.id,
        )

        self.workgroup.refresh_from_db()
        self.assertEqual(self.workgroup.identification_initialized_model, self.identification_model)
        schedule_reid_mock.assert_called_once()
        schedule_init_mock.assert_not_called()

    @patch("caidapp.tasks.schedule_init_identification_for_workgroup")
    @patch("caidapp.tasks.schedule_reid_identification_for_workgroup")
    def test_completed_stale_init_schedules_init_for_current_model(self, schedule_reid_mock, schedule_init_mock):
        previous_model = self.identification_model
        current_model = models.IdentificationModel.objects.create(
            name="Current model",
            public=True,
            model_path="hf-hub:example/current-model",
        )
        self.workgroup.identification_model = current_model
        self.workgroup.save(update_fields=["identification_model"])

        tasks.init_identification_on_success(
            {"status": "DONE", "message": "Initialized."},
            workgroup_id=self.workgroup.id,
            identification_model_id=previous_model.id,
        )

        self.workgroup.refresh_from_db()
        self.assertEqual(self.workgroup.identification_initialized_model, previous_model)
        schedule_reid_mock.assert_not_called()
        schedule_init_mock.assert_called_once()

    def test_dash_identities_keeps_init_enabled_without_representatives_when_idle(self):
        response = self.client.get(reverse("caidapp:dash_identities"))

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("disabled", response.context["btn_styles"]["init_identification"]["class"])

    def test_dash_identities_disables_init_when_identification_is_running(self):
        self.workgroup.identification_reid_status = "Processing"
        self.workgroup.save(update_fields=["identification_reid_status"])

        response = self.client.get(reverse("caidapp:dash_identities"))

        self.assertEqual(response.status_code, 200)
        self.assertIn("disabled", response.context["btn_styles"]["init_identification"]["class"])

    def test_dash_identities_shows_stop_when_init_is_scheduled(self):
        self.workgroup.identification_init_status = "Scheduled"
        self.workgroup.identification_scheduled_init_task_id = "queued-init-task"
        self.workgroup.save(update_fields=["identification_init_status", "identification_scheduled_init_task_id"])

        response = self.client.get(reverse("caidapp:dash_identities"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, reverse("caidapp:stop_init_identification"))
        self.assertIn("disabled", response.context["btn_styles"]["init_identification"]["class"])

    def test_download_init_identification_csv_returns_latest_workgroup_csv(self):
        csv_path = Path(settings.MEDIA_ROOT) / self.workgroup.name / "init_identification.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("image_path,mediafile_id\nfirst.webp,1\n", encoding="utf-8")

        response = self.client.get(reverse("caidapp:download_init_identification_csv"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv")
        self.assertIn("attachment", response["Content-Disposition"])
        self.assertEqual(b"".join(response.streaming_content), b"image_path,mediafile_id\nfirst.webp,1\n")

    def test_download_init_identification_csv_is_admin_only(self):
        self.caiduser.workgroup_admin = False
        self.caiduser.save(update_fields=["workgroup_admin"])

        response = self.client.get(reverse("caidapp:download_init_identification_csv"))

        self.assertEqual(response.status_code, 405)

    def test_download_run_identification_csv_returns_latest_workgroup_csv(self):
        reid_runs_dir = Path(settings.MEDIA_ROOT) / self.workgroup.name / "reid_runs"
        older_csv_path = reid_runs_dir / "20260625-120000" / "identification_metadata.csv"
        latest_csv_path = reid_runs_dir / "20260626-120000" / "identification_metadata.csv"
        older_csv_path.parent.mkdir(parents=True, exist_ok=True)
        latest_csv_path.parent.mkdir(parents=True, exist_ok=True)
        older_csv_path.write_text("image_path,mediafile_id\nold.webp,1\n", encoding="utf-8")
        latest_csv_path.write_text("image_path,mediafile_id\nlatest.webp,2\n", encoding="utf-8")

        response = self.client.get(reverse("caidapp:download_run_identification_csv"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv")
        self.assertIn("attachment", response["Content-Disposition"])
        self.assertEqual(b"".join(response.streaming_content), b"image_path,mediafile_id\nlatest.webp,2\n")

    def test_download_run_identification_csv_is_admin_only(self):
        self.caiduser.workgroup_admin = False
        self.caiduser.save(update_fields=["workgroup_admin"])

        response = self.client.get(reverse("caidapp:download_run_identification_csv"))

        self.assertEqual(response.status_code, 405)

    def test_pre_identify_is_admin_only(self):
        self.caiduser.workgroup_admin = False
        self.caiduser.save(update_fields=["workgroup_admin"])

        response = self.client.get(reverse("caidapp:pre_identify"))

        self.assertEqual(response.status_code, 405)

    def test_run_identification_on_unidentified_is_admin_only(self):
        self.caiduser.workgroup_admin = False
        self.caiduser.save(update_fields=["workgroup_admin"])

        response = self.client.get(reverse("caidapp:run_identification_on_unidentified"))

        self.assertEqual(response.status_code, 405)

    def test_stop_init_identification_is_admin_only(self):
        self.caiduser.workgroup_admin = False
        self.caiduser.save(update_fields=["workgroup_admin"])

        response = self.client.get(reverse("caidapp:stop_init_identification"))

        self.assertEqual(response.status_code, 405)

    @patch("caidapp.views.current_app.control.revoke")
    @patch("caidapp.views.tasks.run_identification_on_unidentified_for_workgroup")
    def test_do_suggestions_now_runs_batch_without_api_celery_worker(self, run_now_mock, revoke_mock):
        self.workgroup.identification_reid_status = "Scheduled"
        self.workgroup.identification_scheduled_run_task_id = "scheduled-run-task"
        self.workgroup.identification_scheduled_run_eta = timezone.now() + timezone.timedelta(minutes=30)
        self.workgroup.save(
            update_fields=[
                "identification_reid_status",
                "identification_scheduled_run_task_id",
                "identification_scheduled_run_eta",
            ]
        )

        response = self.client.get(reverse("caidapp:run_identification_on_unidentified"))

        self.assertEqual(response.status_code, 302)
        revoke_mock.assert_called_once_with("scheduled-run-task", terminate=True)
        run_now_mock.assert_called_once()
        self.assertEqual(run_now_mock.call_args.args[0], self.workgroup.id)
        self.assertEqual(run_now_mock.call_args.kwargs["request"], response.wsgi_request)

    @patch("caidapp.views.AsyncResult")
    @patch("caidapp.views.current_app.control.revoke")
    @patch("caidapp.views.tasks.run_identification_on_unidentified_for_workgroup")
    def test_do_suggestions_now_does_not_duplicate_running_worker_task(
        self,
        run_now_mock,
        revoke_mock,
        async_result_mock,
    ):
        async_result_mock.return_value.state = "PROGRESS"
        self.workgroup.identification_reid_status = "Processing"
        self.workgroup.identification_scheduled_run_task_id = "running-identify-task"
        self.workgroup.save(update_fields=["identification_reid_status", "identification_scheduled_run_task_id"])

        response = self.client.get(reverse("caidapp:run_identification_on_unidentified"))

        self.assertEqual(response.status_code, 302)
        async_result_mock.assert_called_once_with("running-identify-task")
        revoke_mock.assert_not_called()
        run_now_mock.assert_not_called()

    def test_dash_identities_counts_representatives_by_workgroup_default_taxon(self):
        self.workgroup.check_taxon_before_identification = True
        self.workgroup.default_taxon_for_identification = TaxonFactory(name="Lynx lynx")
        self.workgroup.save(update_fields=["check_taxon_before_identification", "default_taxon_for_identification"])

        representative_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
            import_finished=True,
            taxon_for_identification=None,
        )
        representative_mediafile = MediaFileFactory(
            parent=representative_archive,
            with_identity=True,
            identity_is_representative=False,
        )
        AnimalObservationFactory(
            mediafile=representative_mediafile,
            identity=representative_mediafile.identity_from_observations,
            taxon=self.workgroup.default_taxon_for_identification,
            identity_is_representative=True,
        )

        response = self.client.get(reverse("caidapp:dash_identities"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["btn_styles"]["n_representative"], 1)
        self.assertEqual(self.workgroup.number_of_representative_media_files(), 1)

    @patch("caidapp.views.signature")
    @patch("caidapp.views._prepare_dataframe_for_identification")
    def test_train_identification_uses_base_model_weights_and_output_weights(self, prepare_dataframe_mock, signature_mock):
        self.caiduser.workgroup_admin = True
        self.caiduser.save(update_fields=["workgroup_admin"])
        self.identification_model.model_path = "/models/base/source-checkpoint.pth"
        self.identification_model.base_model_path = "hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256"
        self.identification_model.save(update_fields=["model_path", "base_model_path"])
        prepare_dataframe_mock.return_value = {
            "image_path": ["representative-1.jpg", "representative-2.jpg"],
            "class_id": [1, 1],
            "label": ["lynx-alpha", "lynx-alpha"],
        }
        signature_result = Mock()
        signature_mock.return_value = signature_result

        with tempfile.TemporaryDirectory() as media_root, override_settings(MEDIA_ROOT=media_root):
            response = self.client.get(
                reverse("caidapp:train_identification"),
                HTTP_REFERER=reverse("caidapp:dash_identities"),
            )

        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse("caidapp:dash_identities"))
        signature_mock.assert_called_once()
        payload = signature_mock.call_args.kwargs["kwargs"]["identification_model"]
        self.assertEqual(payload["base_model_source"], "hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256")
        self.assertEqual(payload["initial_weights_path"], "/models/base/source-checkpoint.pth")
        self.assertTrue(payload["output_weights_path"].endswith(".pth"))
        self.assertNotIn("init_path", payload)
        self.assertNotIn("init_checkpoint_path", payload)
        self.assertNotIn("path", payload)
        self.assertNotIn("source_path", payload)
        signature_result.apply_async.assert_called_once()

    def test_mediafiles_representative_filter_uses_workgroup_default_taxon(self):
        self.workgroup.check_taxon_before_identification = True
        self.workgroup.default_taxon_for_identification = TaxonFactory(name="Lynx lynx")
        self.workgroup.save(update_fields=["check_taxon_before_identification", "default_taxon_for_identification"])

        representative_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
            import_finished=True,
            taxon_for_identification=None,
        )
        representative_mediafile = MediaFileFactory(
            parent=representative_archive,
            with_identity=True,
            identity_is_representative=False,
            original_filename="representative.jpg",
        )
        AnimalObservationFactory(
            mediafile=representative_mediafile,
            identity=representative_mediafile.identity_from_observations,
            taxon=self.workgroup.default_taxon_for_identification,
            identity_is_representative=True,
        )

        response = self.client.get(reverse("caidapp:media_files"), {"identity_is_representative": "true"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "representative.jpg")

    def test_representative_mediafiles_url_redirects_to_mediafiles_filter(self):
        response = self.client.get(reverse("caidapp:representative_mediafiles"))

        self.assertRedirects(
            response,
            f"{reverse('caidapp:media_files')}?identity_is_representative=true",
            fetch_redirect_response=False,
        )

    def test_toggle_identity_representative_updates_observation_flag(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        identity = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="Alpha")
        mediafile = MediaFileFactory(parent=archive, identity=None, identity_is_representative=False)
        observation = AnimalObservationFactory(
            mediafile=mediafile,
            identity=identity,
            identity_is_representative=False,
        )

        response = self.client.post(reverse("caidapp:toggle_identity_representative", args=[mediafile.id]))

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["representative"])
        observation.refresh_from_db()
        self.assertTrue(observation.identity_is_representative)

    def test_toggle_identity_representative_rejects_multiple_observations(self):
        archive = UploadedArchiveFactory(owner=self.caiduser)
        identity = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="Alpha")
        mediafile = MediaFileFactory(parent=archive)
        first = AnimalObservationFactory(mediafile=mediafile, identity=identity)
        second = AnimalObservationFactory(mediafile=mediafile, identity=identity)

        response = self.client.post(reverse("caidapp:toggle_identity_representative", args=[mediafile.id]))

        self.assertEqual(response.status_code, 400)
        first.refresh_from_db()
        second.refresh_from_db()
        self.assertFalse(first.identity_is_representative)
        self.assertFalse(second.identity_is_representative)

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
        self.identification_model.model_path = "/tmp/reid-weights.pth"
        self.identification_model.base_model_path = "hf-hub:BVRA/MegaDescriptor-T-224"
        self.identification_model.save(update_fields=["model_path", "base_model_path"])

        signature_result = Mock()
        signature_mock.return_value = signature_result
        identify_task = Mock()
        identify_task.id = "bulk-identify-task-1"
        signature_result.apply_async.return_value = identify_task

        status_ok = views.run_identification_bulk(self.workgroup)

        self.assertTrue(status_ok)
        signature_mock.assert_called_once()
        self.assertEqual(signature_mock.call_args.args[0], "identify")
        bulk_kwargs = signature_mock.call_args.kwargs["kwargs"]
        self.assertEqual(bulk_kwargs["organization_id"], self.workgroup.id)
        self.assertNotIn("uploaded_archive_id", bulk_kwargs)
        self.assertEqual(bulk_kwargs["identification_model"]["model_source"], "hf-hub:BVRA/MegaDescriptor-T-224")
        self.assertEqual(bulk_kwargs["identification_model"]["weights_path"], "/tmp/reid-weights.pth")
        self.assertNotIn("path", bulk_kwargs["identification_model"])
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
        identify_task = Mock()
        identify_task.id = "rerun-identify-task-1"
        signature_result.apply_async.return_value = identify_task

        status_ok = views.run_identification(archive, workgroup=self.workgroup)

        self.assertTrue(status_ok)
        self.assertEqual(signature_mock.call_args.args[0], "identify")
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

        response = self.client.get(
            reverse("caidapp:stop_init_identification"),
            HTTP_REFERER=reverse("caidapp:dash_identities"),
        )

        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse("caidapp:dash_identities"))
        revoke_mock.assert_called_once_with("reid-task-123", terminate=True)
        self.workgroup.refresh_from_db()
        self.assertEqual(self.workgroup.identification_reid_status, "Not initiated")
        self.assertIsNone(self.workgroup.identification_scheduled_run_task_id)
        self.assertIn("stopped manually", self.workgroup.identification_reid_message)

    @patch("caidapp.views.signature")
    def test_identification_is_blocked_until_selected_model_is_initialized(self, signature_mock):
        new_model = models.IdentificationModel.objects.create(
            name="Selected model",
            public=True,
            model_path="hf-hub:example/selected-model",
        )
        self.workgroup.identification_model = new_model
        self.workgroup.save(update_fields=["identification_model"])

        started = views.run_identification_bulk(self.workgroup, uploaded_archives=[])

        self.assertFalse(started)
        signature_mock.assert_not_called()

    @patch("caidapp.views.run_identification_bulk")
    def test_scheduled_identification_waits_for_selected_model_initialization(self, run_bulk_mock):
        new_model = models.IdentificationModel.objects.create(
            name="Selected model",
            public=True,
            model_path="hf-hub:example/selected-model",
        )
        self.workgroup.identification_model = new_model
        self.workgroup.identification_reid_status = "Scheduled"
        self.workgroup.identification_scheduled_run_task_id = "scheduled-identification"
        self.workgroup.save(
            update_fields=[
                "identification_model",
                "identification_reid_status",
                "identification_scheduled_run_task_id",
            ]
        )

        started = tasks.run_identification_on_unidentified_for_workgroup(self.workgroup.id)

        self.assertFalse(started)
        run_bulk_mock.assert_not_called()
        self.workgroup.refresh_from_db()
        self.assertEqual(self.workgroup.identification_reid_status, "Not initiated")
        self.assertIsNone(self.workgroup.identification_scheduled_run_task_id)
        self.assertIn("Waiting for initialization", self.workgroup.identification_reid_message)

    @patch("caidapp.tasks.schedule_init_identification_for_workgroup")
    def test_changing_workgroup_model_schedules_initialization_after_commit(self, schedule_init_mock):
        new_model = models.IdentificationModel.objects.create(
            name="New model",
            public=True,
            model_path="hf-hub:example/new-model",
        )
        data = {
            "name": self.workgroup.name,
            "default_taxon_for_identification": self.workgroup.default_taxon_for_identification_id,
            "sequence_time_limit": self.workgroup.sequence_time_limit,
            "identity_code_regex": self.workgroup.identity_code_regex,
            "identity_merge_distinguishing_regex": self.workgroup.identity_merge_distinguishing_regex,
            "identification_model": new_model.id,
            "detection_model_path": self.workgroup.detection_model_path,
            "detection_model_architecture": self.workgroup.detection_model_architecture,
        }

        with self.captureOnCommitCallbacks(execute=True):
            response = self.client.post(
                reverse("caidapp:workgroup-update", args=[self.workgroup.id]),
                data,
            )

        self.assertEqual(response.status_code, 302)
        self.workgroup.refresh_from_db()
        self.assertEqual(self.workgroup.identification_model, new_model)
        schedule_init_mock.assert_called_once()

    @patch("caidapp.views.current_app.control.revoke")
    def test_stop_init_identification_revokes_running_init_task(self, revoke_mock):
        self.workgroup.identification_init_status = "Processing"
        self.workgroup.identification_scheduled_init_task_id = "init-task-456"
        self.workgroup.save()

        response = self.client.get(reverse("caidapp:stop_init_identification"))

        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse("caidapp:dash_identities"))
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
        self.identification_model.model_path = "/tmp/test-model.pth"
        self.identification_model.base_model_path = "hf-hub:BVRA/MegaDescriptor-T-224"
        self.identification_model.save(update_fields=["model_path", "base_model_path"])

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
        self.assertEqual(signature_mock.call_args.args[0], "init_identification")
        payload = signature_mock.call_args.kwargs["kwargs"]["identification_model"]
        self.assertEqual(payload["model_source"], "hf-hub:BVRA/MegaDescriptor-T-224")
        self.assertEqual(payload["weights_path"], "/tmp/test-model.pth")
        self.assertNotIn("path", payload)

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
