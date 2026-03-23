from unittest.mock import MagicMock, patch

from django.test import TestCase
from django.urls import reverse

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


class MergeIdentitySuggestionsViewTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory(admin=True)
        self.client.force_login(self.caiduser.user)

    @patch("caidapp.views.refresh_identities_suggestions")
    def test_suggest_merge_identities_starts_generation_when_missing(self, refresh_mock):
        response = self.client.get(reverse("caidapp:suggest_merge_identities"))

        self.assertEqual(response.status_code, 200)
        refresh_mock.assert_called_once()
        self.assertContains(response, "Generation of merge suggestions has started.")

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
