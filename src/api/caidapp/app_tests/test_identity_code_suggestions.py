from django.test import TestCase
from django.urls import reverse
from unittest.mock import MagicMock, patch

from caidapp.app_tests.factories import CaidUserFactory, IndividualIdentityFactory


class IdentityCodeSuggestionsViewTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory(admin=True)
        self.workgroup = self.caiduser.workgroup
        self.workgroup.identity_code_regex = r"B\d+"
        self.workgroup.save()
        self.client.force_login(self.caiduser.user)

    @patch("caidapp.views._celery_worker_available", return_value=False)
    def test_view_filters_suggestions_by_workgroup_regex(self, _worker_available_mock):
        matching = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="B75 Cumel")
        non_matching = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="NoCode Identity")
        other_workgroup = IndividualIdentityFactory(name="B610 Fernet")

        response = self.client.get(reverse("caidapp:show_identity_code_suggestions"))

        self.assertEqual(response.status_code, 200)
        identities = response.context["identities"]
        self.assertEqual([identity.id for identity in identities], [matching.id])
        self.assertContains(response, "B75 Cumel")
        self.assertNotContains(response, non_matching.name)
        self.assertNotContains(response, other_workgroup.name)

    @patch("caidapp.views._celery_worker_available", return_value=False)
    def test_view_uses_custom_workgroup_regex(self, _worker_available_mock):
        self.workgroup.identity_code_regex = r"[A-Z]{2}\d{2}"
        self.workgroup.save()
        matching = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="AB12 Kira")
        IndividualIdentityFactory(owner_workgroup=self.workgroup, name="B75 Cumel")

        response = self.client.get(reverse("caidapp:show_identity_code_suggestions"))

        self.assertEqual(response.status_code, 200)
        identities = response.context["identities"]
        self.assertEqual([identity.id for identity in identities], [matching.id])
        self.assertContains(response, "AB12 Kira")
        self.assertContains(response, self.workgroup.identity_code_regex)

    @patch("caidapp.views._celery_worker_available", return_value=False)
    def test_view_renders_bulk_selection_controls(self, _worker_available_mock):
        matching = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="B75 Cumel")

        response = self.client.get(reverse("caidapp:show_identity_code_suggestions"))

        self.assertContains(response, 'id="select-all-identities"')
        self.assertContains(response, 'name="identity_ids"')
        self.assertContains(response, f'value="{matching.id}"')
        self.assertContains(response, "Apply to selected")

    def test_bulk_apply_updates_only_selected_identities(self):
        selected = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="B75 Cumel", code="old-selected")
        not_selected = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="B90 Kira", code="old-other")

        response = self.client.post(
            reverse("caidapp:apply_selected_identity_code_suggestions"),
            {"identity_ids": [str(selected.id)]},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        selected.refresh_from_db()
        not_selected.refresh_from_db()
        self.assertEqual(selected.code, "B75")
        self.assertEqual(selected.name, "Cumel")
        self.assertIn("former code: old-selected", selected.note)
        self.assertEqual(not_selected.code, "old-other")
        self.assertEqual(not_selected.name, "B90 Kira")
        messages = list(response.context["messages"])
        self.assertTrue(any("Applied code suggestions to 1 identities." in str(message) for message in messages))

    @patch("caidapp.views.tasks.compute_identity_code_suggestions_task.delay")
    @patch("caidapp.views._celery_worker_available")
    def test_view_starts_background_job_when_worker_is_available(self, worker_available_mock, delay_mock):
        worker_available_mock.return_value = True
        delay_mock.return_value = MagicMock(id="job-123")

        response = self.client.get(reverse("caidapp:show_identity_code_suggestions"))

        self.assertEqual(response.status_code, 200)
        delay_mock.assert_called_once_with(self.workgroup.id)
        session = self.client.session
        self.assertEqual(session["identity_code_suggestions_job_id"], "job-123")
        self.assertContains(response, "Generating suggestions")
        self.assertContains(response, "Cancel generation")
        self.assertContains(response, reverse("caidapp:cancel_identity_code_suggestions"))

    @patch("caidapp.views.AsyncResult")
    def test_status_endpoint_returns_progress_meta(self, async_result_mock):
        session = self.client.session
        session["identity_code_suggestions_job_id"] = "job-123"
        session.save()

        result = MagicMock()
        result.state = "PROGRESS"
        result.info = {"current": 25, "total": 100, "matches": 7, "message": "Checked 25 of 100 identities."}
        async_result_mock.return_value = result

        response = self.client.get(reverse("caidapp:identity_code_suggestions_status"))

        self.assertEqual(response.status_code, 200)
        self.assertJSONEqual(
            response.content,
            {
                "status": "progress",
                "progress": {
                    "current": 25,
                    "total": 100,
                    "matches": 7,
                    "message": "Checked 25 of 100 identities.",
                },
                "redirect_url": "",
            },
        )

    @patch("caidapp.views.current_app.control.revoke")
    def test_cancel_view_revokes_running_job(self, revoke_mock):
        session = self.client.session
        session["identity_code_suggestions_job_id"] = "job-123"
        session.save()

        response = self.client.post(reverse("caidapp:cancel_identity_code_suggestions"))

        self.assertEqual(response.status_code, 302)
        revoke_mock.assert_called_once_with("job-123", terminate=True)
        self.assertNotIn("identity_code_suggestions_job_id", self.client.session)

    @patch("caidapp.views._compute_identity_code_suggestions_sync")
    @patch("caidapp.views._celery_worker_available", return_value=False)
    @patch("caidapp.views.AsyncResult")
    def test_view_recovers_from_stale_job_when_worker_is_missing(
        self,
        async_result_mock,
        _worker_available_mock,
        compute_sync_mock,
    ):
        matching = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="B75 Cumel")
        compute_sync_mock.side_effect = lambda workgroup: [matching]
        session = self.client.session
        session["identity_code_suggestions_job_id"] = "stale-job"
        session["identity_code_suggestions_started_at"] = "2026-06-22T12:00:00+00:00"
        session.save()

        result = MagicMock()
        result.state = "PENDING"
        result.info = {}
        async_result_mock.return_value = result

        response = self.client.get(reverse("caidapp:show_identity_code_suggestions"))

        self.assertEqual(response.status_code, 200)
        compute_sync_mock.assert_called_once()
        self.assertContains(response, "B75 Cumel")
        self.assertNotIn("identity_code_suggestions_job_id", self.client.session)
