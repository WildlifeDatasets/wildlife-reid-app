import logging
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from caidapp import models
from caidapp import tasks
from caidapp import views
from django.conf import settings
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
