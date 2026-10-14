from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from caidapp import tasks
from caidapp.app_tests.factories import CaidUserFactory, IndividualIdentityFactory, UploadedArchiveFactory
from caidapp.models import (
    IdentificationOutlierSuggestionResult,
    IdentificationSimilarPairResult,
    MediaFile,
    NotificationRecipient,
)
from django.test import TestCase
from django.urls import reverse


class SimilarIdentityPairsTest(TestCase):
    def setUp(self):
        """Prepare staff, a regular member, and temporary media for similar-identity pair searches."""
        self.staff = CaidUserFactory(admin=False)
        self.staff.user.is_staff = True
        self.staff.user.save(update_fields=["is_staff"])
        self.workgroup = self.staff.workgroup
        self.nonstaff = CaidUserFactory(admin=True, workgroup=self.workgroup)
        self.client.force_login(self.staff.user)
        self.temporary_media = TemporaryDirectory()
        self.addCleanup(self.temporary_media.cleanup)
        media_settings = self.settings(MEDIA_ROOT=self.temporary_media.name)
        media_settings.enable()
        self.addCleanup(media_settings.disable)

    def test_start_queues_csv_preparation_without_scanning_media_in_http_request(self):
        """Queue CSV preparation outside the request and dispatch the similarity worker from that task."""
        csv_data = {
            "image_path": [
                str(Path(self.temporary_media.name) / "images/a.jpg"),
                str(Path(self.temporary_media.name) / "images/b.jpg"),
            ],
            "class_id": [11, 22],
            "label": ["A", "B"],
        }
        with patch("caidapp.tasks.prepare_similar_identity_pairs.apply_async") as prepare_mock, patch(
            "caidapp.tasks._prepare_dataframe_for_identification",
            side_effect=AssertionError("CSV prepared in HTTP request"),
        ):
            response = self.client.post(reverse("caidapp:run_similar_identity_pairs"))

        result = IdentificationSimilarPairResult.objects.get()
        self.assertRedirects(
            response,
            reverse("caidapp:similar_identity_pairs_result", args=[result.id]),
            fetch_redirect_response=False,
        )
        self.assertEqual(result.status, "processing")
        self.assertTrue(result.task_id)
        self.assertIn("Preparing", result.message)
        self.assertFalse(self.workgroup.file_path(result.input_csv_filename).exists())
        prepare_mock.assert_called_once_with(args=[result.id], task_id=result.task_id, queue="celery", retry=False)

        job = MagicMock(id="similar-job-1")
        with patch("caidapp.tasks._prepare_dataframe_for_identification", return_value=csv_data), patch(
            "caidapp.tasks.signature"
        ) as signature_mock:
            signature_mock.return_value.apply_async.return_value = job
            tasks.prepare_similar_identity_pairs(result.id)

        result.refresh_from_db()
        self.assertEqual(result.task_id, "similar-job-1")
        self.assertTrue(self.workgroup.file_path(result.input_csv_filename).is_file())
        self.assertEqual(result.output_csv_filename, f"identification_similar_pairs_{result.id}_most_similar.csv")
        signature_mock.assert_called_once()
        self.assertEqual(signature_mock.call_args.args[0], "detect_most_similar")
        self.assertEqual(signature_mock.call_args.kwargs["kwargs"]["organization_id"], self.workgroup.id)
        self.assertEqual(signature_mock.return_value.apply_async.call_count, 1)
        recipients = NotificationRecipient.objects.values_list("user_id", flat=True)
        self.assertEqual(list(recipients), [self.staff.id])

    def test_csv_preparation_error_is_saved_by_background_task(self):
        """Persist CSV preparation errors without dispatching the similarity worker."""
        with patch("caidapp.tasks.prepare_similar_identity_pairs.apply_async"):
            result = tasks.run_similar_identity_pairs_for_workgroup(self.workgroup)
        with patch(
            "caidapp.tasks._prepare_dataframe_for_identification", side_effect=OSError("image store unavailable")
        ), patch("caidapp.tasks.signature") as signature_mock:
            tasks.prepare_similar_identity_pairs(result.id)
        result.refresh_from_db()
        self.assertEqual(result.status, "error")
        self.assertIn("image store unavailable", result.message)
        self.assertIsNotNone(result.finished_at)
        signature_mock.assert_not_called()

    def test_callback_stores_both_sources_and_links_to_result(self):
        """Save both similarity sources, notify staff, and link pair results to comparison and merge review."""
        first_identity = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="First")
        second_identity = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="Second")
        archive = UploadedArchiveFactory(owner=self.staff)
        first_media = MediaFile.objects.create(
            parent=archive,
            media_type="image",
            mediafile="images/a.jpg",
            image_file="images/a.jpg",
            preview="previews/a.jpg",
            original_filename="a.jpg",
        )
        second_media = MediaFile.objects.create(
            parent=archive,
            media_type="image",
            mediafile="images/b.jpg",
            image_file="images/b.jpg",
            preview="previews/b.jpg",
            original_filename="b.jpg",
        )
        result = IdentificationSimilarPairResult.objects.create(workgroup=self.workgroup)
        pair = {
            "path_a": str(Path(self.temporary_media.name) / first_media.image_file.name),
            "path_b": str(Path(self.temporary_media.name) / second_media.image_file.name),
            "class_id_a": first_identity.id,
            "class_id_b": second_identity.id,
            "label_a": first_identity.name,
            "label_b": second_identity.name,
            "similarity": 0.91,
        }

        tasks.similar_identity_pairs_on_success(
            {
                "status": "DONE",
                "message": "finished",
                "mega_pairs": [pair],
                "local_pairs": [{**pair, "similarity": 18}],
            },
            result_id=result.id,
        )
        result.refresh_from_db()
        self.assertEqual(result.status, "done")
        self.assertEqual(result.mega_pairs[0]["mediafile_a_id"], first_media.id)
        self.assertEqual(result.local_pairs[0]["mediafile_b_id"], second_media.id)
        self.assertEqual(result.local_pairs[0]["similarity"], 18)
        self.assertIsNotNone(result.finished_at)
        recipients = NotificationRecipient.objects.select_related("notification").all()
        self.assertEqual(recipients.count(), 1)
        self.assertEqual(recipients[0].user_id, self.staff.id)
        self.assertEqual(
            recipients[0].notification.get_link_url(),
            reverse("caidapp:similar_identity_pairs_result", args=[result.id]),
        )

        response = self.client.get(reverse("caidapp:similar_identity_pairs_result", args=[result.id]))
        self.assertContains(response, "Cosine similarity: 0.910")
        self.assertContains(response, "Matched points: 18")
        self.assertContains(
            response,
            f'{reverse("caidapp:comparison")}?a_kind=mediafile&amp;a_id={first_media.id}'
            f"&amp;b_kind=mediafile&amp;b_id={second_media.id}",
        )
        self.assertContains(response, "Review A into B")
        self.assertContains(response, f"from_mediafile_id={first_media.id}")
        self.assertContains(response, f"to_mediafile_id={second_media.id}")
        info = self.client.get(reverse("caidapp:identification_information"))
        self.assertContains(info, "Similar identity pairs")
        self.assertContains(info, "1 Mega pairs, 1 Local pairs")

    def test_nonstaff_cannot_start_view_or_download_results(self):
        """Deny pair search, result pages, and CSV downloads to non-staff members."""
        result = IdentificationSimilarPairResult.objects.create(workgroup=self.workgroup)
        self.client.force_login(self.nonstaff.user)
        dashboard = self.client.get(reverse("caidapp:dash_identities"))
        self.assertNotContains(dashboard, "Find similar identity pairs")
        self.assertEqual(self.client.post(reverse("caidapp:run_similar_identity_pairs")).status_code, 403)
        self.assertEqual(
            self.client.get(reverse("caidapp:similar_identity_pairs_result", args=[result.id])).status_code, 403
        )
        self.assertEqual(
            self.client.get(
                reverse("caidapp:download_similar_identity_pairs_csv", args=[result.id, "input"])
            ).status_code,
            403,
        )

    def test_staff_cannot_view_another_workgroups_result(self):
        """Hide pair results and downloads from staff in another workgroup."""
        other_workgroup_user = CaidUserFactory()
        result = IdentificationSimilarPairResult.objects.create(workgroup=other_workgroup_user.workgroup)
        self.assertEqual(
            self.client.get(reverse("caidapp:similar_identity_pairs_result", args=[result.id])).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(
                reverse("caidapp:download_similar_identity_pairs_csv", args=[result.id, "input"])
            ).status_code,
            404,
        )

    def test_worker_error_is_reported_to_staff(self):
        """Store worker error details and notify staff without notifying regular members."""
        result = IdentificationSimilarPairResult.objects.create(workgroup=self.workgroup)
        tasks.similar_identity_pairs_on_success({"status": "ERROR", "error": "No embeddings"}, result_id=result.id)
        result.refresh_from_db()
        self.assertEqual(result.status, "error")
        self.assertIn("No embeddings", result.message)
        self.assertEqual(NotificationRecipient.objects.filter(user=self.staff).count(), 1)
        self.assertFalse(NotificationRecipient.objects.filter(user=self.nonstaff).exists())

    def test_outlier_completion_notification_reaches_only_staff(self):
        """Send outlier completion notifications only to staff with a link to the saved result."""
        result = IdentificationOutlierSuggestionResult.objects.create(workgroup=self.workgroup, suggestions=[])
        tasks.identification_outlier_detection_on_success(
            {"status": "DONE", "message": "finished", "suggestions": []},
            result_id=result.id,
        )
        recipients = NotificationRecipient.objects.select_related("notification").all()
        self.assertEqual(recipients.count(), 1)
        self.assertEqual(recipients[0].user_id, self.staff.id)
        self.assertEqual(
            recipients[0].notification.get_link_url(),
            reverse("caidapp:identification_outlier_suggestions_result", args=[result.id]),
        )
