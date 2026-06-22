from unittest.mock import MagicMock, patch

from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from caidapp import views
from caidapp.app_tests.factories import CaidUserFactory, IndividualIdentityFactory
from caidapp.model_extra import compute_identity_suggestions
from caidapp.models import MergeIdentitySuggestionResult


class MergeIdentitySuggestionsComputationTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory(admin=True)
        self.workgroup = self.caiduser.workgroup

    def test_compute_identity_suggestions_ignores_empty_codes(self):
        IndividualIdentityFactory(owner_workgroup=self.workgroup, name="VeryLongDifferentIdentityNameAlpha", code=None)
        IndividualIdentityFactory(owner_workgroup=self.workgroup, name="Ox", code=None)

        result_id = compute_identity_suggestions(self.workgroup.id)
        result = MergeIdentitySuggestionResult.objects.get(id=result_id)

        self.assertEqual(result.suggestions, [])

    def test_compute_identity_suggestions_matches_same_non_empty_code(self):
        first = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="Alpha Identity", code="B75")
        second = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="Omega Identity", code="B75")

        result_id = compute_identity_suggestions(self.workgroup.id)
        result = MergeIdentitySuggestionResult.objects.get(id=result_id)

        self.assertEqual(len(result.suggestions), 1)
        identity_a_id, identity_b_id, distance = result.suggestions[0]
        self.assertEqual({identity_a_id, identity_b_id}, {first.id, second.id})
        self.assertEqual(distance, 0)

    def test_compute_identity_suggestions_reports_pair_progress(self):
        for name in ["Alpha", "Bravo", "Charlie"]:
            IndividualIdentityFactory(owner_workgroup=self.workgroup, name=name)
        progress_updates = []

        compute_identity_suggestions(self.workgroup.id, progress_callback=lambda **data: progress_updates.append(data))

        self.assertGreaterEqual(len(progress_updates), 2)
        self.assertEqual(progress_updates[0]["total"], 3)
        self.assertEqual(progress_updates[-1]["current"], 3)
        self.assertEqual(progress_updates[-1]["total"], 3)


class MergeIdentitySuggestionsViewTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory(admin=True)
        self.client.force_login(self.caiduser.user)

    @patch("caidapp.views.tasks.refresh_identities_suggestions_task.delay")
    @patch("caidapp.views._celery_worker_available", return_value=True)
    def test_suggest_merge_identities_starts_generation_when_missing(self, _worker_mock, delay_mock):
        delay_mock.return_value = MagicMock(id="job-123")

        response = self.client.get(reverse("caidapp:suggest_merge_identities"))

        self.assertEqual(response.status_code, 200)
        delay_mock.assert_called_once_with(self.caiduser.workgroup.id, 100)
        self.assertEqual(self.client.session["refresh_job_id"], "job-123")
        self.assertContains(response, "Generating merge suggestions")
        self.assertContains(response, "Cancel generation")

    @patch("caidapp.views.compute_identity_suggestions")
    @patch("caidapp.views.current_app.control.inspect")
    def test_refresh_identities_suggestions_falls_back_to_sync_without_worker(self, inspect_mock, compute_mock):
        inspect_instance = MagicMock()
        inspect_instance.stats.return_value = None
        inspect_mock.return_value = inspect_instance
        compute_mock.return_value = 123

        request = self.client.request().wsgi_request
        request.user = self.caiduser.user
        request.session = self.client.session

        result_id = views.refresh_identities_suggestions(request)

        self.assertEqual(result_id, 123)
        compute_mock.assert_called_once_with(self.caiduser.workgroup.id, 100)
        self.assertEqual(request.session["refresh_result_id"], 123)
        self.assertNotIn("refresh_job_id", request.session)

    def test_suggest_merge_identities_renders_bulk_selection_controls(self):
        first = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha", code="B75")
        second = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Omega", code="B75")
        result = MergeIdentitySuggestionResult.objects.create(
            workgroup=self.caiduser.workgroup,
            suggestions=[[first.id, second.id, 0]],
        )

        session = self.client.session
        session["refresh_result_id"] = result.id
        session.save()

        response = self.client.get(reverse("caidapp:suggest_merge_identities"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'id="select-all-merge-suggestions"')
        self.assertContains(response, 'id="select-distance-zero-merge-suggestions"')
        self.assertContains(response, 'data-distance="0"')

    def test_suggestions_are_paginated_without_recomputation(self):
        first = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Alpha")
        second = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Beta")
        result = MergeIdentitySuggestionResult.objects.create(
            workgroup=self.caiduser.workgroup,
            suggestions=[[first.id, second.id, distance] for distance in range(205)],
        )
        session = self.client.session
        session["refresh_result_id"] = result.id
        session.save()

        with patch("caidapp.views.refresh_identities_suggestions") as refresh_mock:
            response = self.client.get(reverse("caidapp:suggest_merge_identities"), {"page": 2})

        self.assertEqual(response.status_code, 200)
        refresh_mock.assert_not_called()
        self.assertEqual(response.context["page_obj"].number, 2)
        self.assertEqual(response.context["page_obj"].paginator.count, 205)
        self.assertEqual(len(response.context["suggestions"]), 100)
        self.assertContains(response, "Showing 101-200 of 205 saved suggestions.")
        self.assertContains(response, "?page=3")

    @patch("caidapp.views._celery_worker_available", return_value=True)
    @patch("caidapp.views.AsyncResult")
    def test_status_endpoint_returns_progress(self, async_result_mock, _worker_mock):
        session = self.client.session
        session["refresh_job_id"] = "job-123"
        session.save()
        result = MagicMock()
        result.state = "PROGRESS"
        result.info = {
            "current": 20,
            "total": 50,
            "suggestions_count": 7,
            "message": "Compared 20 of 50 identities.",
        }
        async_result_mock.return_value = result

        response = self.client.get(reverse("caidapp:merge_identity_suggestions_status"))

        self.assertEqual(response.status_code, 200)
        self.assertJSONEqual(
            response.content,
            {
                "status": "progress",
                "progress": result.info,
                "redirect_url": "",
            },
        )

    @patch("caidapp.views.current_app.control.revoke")
    def test_cancel_view_revokes_running_job(self, revoke_mock):
        session = self.client.session
        session["refresh_job_id"] = "job-123"
        session.save()

        response = self.client.post(reverse("caidapp:cancel_merge_identity_suggestions"))

        self.assertEqual(response.status_code, 302)
        revoke_mock.assert_called_once_with("job-123", terminate=True)
        self.assertNotIn("refresh_job_id", self.client.session)

    @patch("caidapp.views._celery_worker_available", return_value=False)
    @patch("caidapp.views.AsyncResult")
    def test_view_recovers_from_stale_job_without_worker(self, async_result_mock, _worker_mock):
        session = self.client.session
        session["refresh_job_id"] = "stale-job"
        session["refresh_job_started_at"] = "2026-06-22T12:00:00+00:00"
        session.save()
        stale_result = MagicMock()
        stale_result.state = "PENDING"
        stale_result.info = {}
        async_result_mock.return_value = stale_result

        with patch("caidapp.views.compute_identity_suggestions", return_value=123) as compute_mock:
            response = self.client.get(reverse("caidapp:suggest_merge_identities"))

        self.assertEqual(response.status_code, 200)
        compute_mock.assert_called_once_with(self.caiduser.workgroup.id, 100)
        self.assertNotIn("refresh_job_id", self.client.session)

    @patch("caidapp.views._celery_worker_available", return_value=False)
    @patch("caidapp.views.AsyncResult")
    def test_fresh_job_survives_temporary_worker_inspect_failure(self, async_result_mock, _worker_mock):
        session = self.client.session
        session["refresh_job_id"] = "fresh-job"
        session["refresh_job_started_at"] = timezone.now().isoformat()
        session.save()
        pending_result = MagicMock()
        pending_result.state = "PENDING"
        pending_result.info = {}
        async_result_mock.return_value = pending_result

        response = self.client.get(reverse("caidapp:merge_identity_suggestions_status"))

        self.assertEqual(response.json()["status"], "pending")
        self.assertEqual(self.client.session["refresh_job_id"], "fresh-job")

    @patch("caidapp.views.tasks.refresh_identities_suggestions_task.delay")
    @patch("caidapp.views._celery_worker_available", return_value=True)
    @patch("caidapp.views.AsyncResult")
    def test_start_does_not_enqueue_duplicate_job(self, async_result_mock, _worker_mock, delay_mock):
        session = self.client.session
        session["refresh_job_id"] = "job-123"
        session.save()
        running_result = MagicMock()
        running_result.state = "PROGRESS"
        running_result.info = {}
        async_result_mock.return_value = running_result

        response = self.client.post(reverse("caidapp:start_merge_identity_suggestions"))

        self.assertEqual(response.status_code, 302)
        delay_mock.assert_not_called()
