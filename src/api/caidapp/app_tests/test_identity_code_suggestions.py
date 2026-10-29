import json
from unittest.mock import MagicMock, patch
from urllib.parse import urlencode

from caidapp import views
from caidapp.app_tests.factories import (
    AnimalObservationFactory,
    CaidUserFactory,
    IndividualIdentityFactory,
    MediaFileFactory,
    UploadedArchiveFactory,
)
from django.test import TestCase, override_settings
from django.urls import reverse


class IdentityCodeSuggestionsViewTest(TestCase):
    def setUp(self):
        """Authenticate an admin whose workgroup extracts identity codes with a custom regex."""
        self.caiduser = CaidUserFactory(admin=True)
        self.workgroup = self.caiduser.workgroup
        self.workgroup.identity_code_regex = r"B\d+"
        self.workgroup.save()
        self.client.force_login(self.caiduser.user)

    @patch("caidapp.views._celery_worker_available", return_value=False)
    def test_view_filters_suggestions_by_workgroup_regex(self, _worker_available_mock):
        """Show matching identity names only from the current workgroup."""
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
        """Use the workgroup's custom two-letter code pattern when generating suggestions."""
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
        """Render selection controls, an apply action, and the identity's media-file count."""
        matching = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="B75 Cumel")
        archive = UploadedArchiveFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=archive)
        AnimalObservationFactory(mediafile=mediafile, identity=matching)

        response = self.client.get(reverse("caidapp:show_identity_code_suggestions"))

        self.assertContains(response, 'id="select-all-identities"')
        self.assertContains(response, 'name="identity_ids"')
        self.assertContains(response, f'value="{matching.id}"')
        self.assertContains(response, "Apply to selected")
        self.assertContains(response, "1 media file")

    def test_bulk_apply_updates_only_selected_identities(self):
        """Apply extracted codes only to selected identities and retain the former code in notes."""
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

    @patch("caidapp.views._celery_worker_available", return_value=False)
    def test_suggestions_paginate_by_two_hundred_without_regenerating(self, _worker_available_mock):
        """Page synchronous and cached suggestions in stable groups of 200 without losing entries."""
        identities = IndividualIdentityFactory.build_batch(401, owner_workgroup=self.workgroup, name="B75 Cumel")
        type(identities[0]).objects.bulk_create(identities)
        url = reverse("caidapp:show_identity_code_suggestions")
        with patch(
            "caidapp.views._compute_identity_code_suggestions_sync", wraps=views._compute_identity_code_suggestions_sync
        ) as compute:
            first = self.client.get(url)
            second = self.client.get(url, {"page": 2})
            third = self.client.get(url, {"page": 3})
            invalid = self.client.get(url, {"page": "invalid"})
            beyond_last = self.client.get(url, {"page": 999})

        compute.assert_called_once_with(self.workgroup)
        self.assertEqual([len(page.context["identities"]) for page in (first, second, third)], [200, 200, 1])
        actual_ids = [identity.id for page in (first, second, third) for identity in page.context["identities"]]
        self.assertCountEqual(actual_ids, [identity.id for identity in identities])
        self.assertEqual(invalid.context["page_obj"].number, 1)
        self.assertEqual(beyond_last.context["page_obj"].number, 3)
        self.assertContains(first, "of 401 suggestions.")
        self.assertContains(first, "Select all on this page")
        self.assertContains(second, 'href="?page=3"')
        self.assertContains(second, 'href="?page=1"')
        for page in (first, second, third):
            self.assertContains(page, 'name="identity_ids"', count=len(page.context["identities"]))

    @override_settings(DATA_UPLOAD_MAX_NUMBER_FIELDS=1000)
    def test_bulk_apply_compact_selection_exceeds_a_thousand_identities(self):
        """Apply more than a thousand codes using one field while preserving workgroup isolation."""
        identities = IndividualIdentityFactory.build_batch(1001, owner_workgroup=self.workgroup, name="B75 Cumel")
        identity_model = type(identities[0])
        identity_model.objects.bulk_create(identities)
        other = IndividualIdentityFactory(name="B90 Other", code="unchanged")
        selected_ids = [str(identity.pk) for identity in identities] + [str(other.pk), str(identities[0].pk)]

        response = self.client.post(
            reverse("caidapp:apply_selected_identity_code_suggestions"),
            urlencode({"identity_ids_json": json.dumps(selected_ids)}),
            content_type="application/x-www-form-urlencoded",
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(identity_model.objects.filter(owner_workgroup=self.workgroup, code="B75").count(), 1001)
        other.refresh_from_db()
        self.assertEqual(other.code, "unchanged")
        identities[0].refresh_from_db()
        self.assertEqual(identities[0].note.count("former code:"), 1)

    def test_bulk_apply_rejects_invalid_compact_selection_before_changing_codes(self):
        """Reject malformed JSON and invalid IDs without partially applying valid entries."""
        identity = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="B75 Cumel", code="original")
        for payload in ("broken", "null", "{}", json.dumps([identity.pk, "invalid"]), "[true]", "[-1]"):
            with self.subTest(payload=payload):
                response = self.client.post(
                    reverse("caidapp:apply_selected_identity_code_suggestions"), {"identity_ids_json": payload}
                )
                self.assertEqual(response.status_code, 302)
                identity.refresh_from_db()
                self.assertEqual(identity.code, "original")

    @patch("caidapp.views.tasks.compute_identity_code_suggestions_task.delay")
    @patch("caidapp.views._celery_worker_available")
    def test_view_starts_background_job_when_worker_is_available(self, worker_available_mock, delay_mock):
        """Queue code suggestions when a worker is available and retain the task ID in the session."""
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

    @patch("caidapp.views._celery_worker_available", return_value=True)
    @patch("caidapp.views.AsyncResult")
    def test_status_endpoint_returns_progress_meta(self, async_result_mock, _worker_available_mock):
        """Return the running code-suggestion job's counters and message as progress metadata."""
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
        """Revoke the running code-suggestion task and remove its session reference."""
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
        """Recover a stale background job by computing suggestions synchronously when no worker is available."""
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
