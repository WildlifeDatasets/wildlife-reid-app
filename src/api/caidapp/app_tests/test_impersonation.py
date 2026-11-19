from caidapp.app_tests.factories import CaidUserFactory
from caidapp.forms import UserSelectForm
from caidapp.user_labels import format_user_label
from django.contrib.auth import SESSION_KEY, get_user_model
from django.test import TestCase
from django.urls import reverse


class ImpersonationTest(TestCase):
    def setUp(self):
        """Create an original administrator, another administrator, and a user."""
        self.admin = CaidUserFactory(
            user__is_superuser=True,
            user__is_staff=True,
            user__first_name="Original",
            user__last_name="Admin",
        ).user
        self.other_admin = CaidUserFactory(user__is_superuser=True, user__is_staff=True).user
        self.member = CaidUserFactory().user
        self.client.force_login(self.admin)

    def test_choices_include_name_email_and_username(self):
        """Distinguish accounts even when their display names match."""
        label = UserSelectForm().fields["user"].label_from_instance(self.admin)
        for value in ("Original Admin", self.admin.email, self.admin.username):
            self.assertIn(value, label)

    def test_label_handles_missing_name_and_email(self):
        """Keep accounts without profile details identifiable."""
        user = get_user_model()(username="unique-login")
        self.assertEqual(format_user_label(user), "unique-login")

    def test_repeated_impersonation_returns_to_authenticated_account(self):
        """Switch through a second administrator without losing the original login."""
        for target in (self.other_admin, self.member):
            response = self.client.post(reverse("caidapp:impersonate_user"), {"user": target.pk})
            self.assertRedirects(response, reverse("caidapp:home"), fetch_redirect_response=False)
            self.assertEqual(self.client.session["original_user_id"], self.admin.pk)
            self.assertEqual(self.client.session["impersonate_user_id"], target.pk)
            self.assertEqual(self.client.session[SESSION_KEY], str(self.admin.pk))

        response = self.client.get(reverse("caidapp:notifications"))
        self.assertEqual(response.wsgi_request.user.pk, self.member.pk)
        self.assertContains(response, f'title="Return to {format_user_label(self.admin)}"')

        response = self.client.get(reverse("caidapp:stop_impersonation"))
        self.assertRedirects(response, reverse("caidapp:home"), fetch_redirect_response=False)
        self.assertNotIn("impersonate_user_id", self.client.session)
        self.assertNotIn("original_user_id", self.client.session)
        response = self.client.get(reverse("caidapp:notifications"))
        self.assertEqual(response.wsgi_request.user.pk, self.admin.pk)

    def test_legacy_overwritten_original_does_not_change_return_account(self):
        """Use the authenticated account even for sessions created before the fix."""
        session = self.client.session
        session["original_user_id"] = self.other_admin.pk
        session["impersonate_user_id"] = self.member.pk
        session.save()
        response = self.client.get(reverse("caidapp:notifications"))
        self.assertContains(response, f'title="Return to {format_user_label(self.admin)}"')
        self.client.get(reverse("caidapp:stop_impersonation"))
        self.assertEqual(self.client.session[SESSION_KEY], str(self.admin.pk))

    def test_impersonated_non_superuser_cannot_start_another_impersonation(self):
        """Retain the existing superuser restriction when switching accounts."""
        self.client.post(reverse("caidapp:impersonate_user"), {"user": self.member.pk})
        response = self.client.post(reverse("caidapp:impersonate_user"), {"user": self.other_admin.pk})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.client.session["impersonate_user_id"], self.member.pk)
