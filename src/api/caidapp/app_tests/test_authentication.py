from urllib.parse import parse_qs, urlparse

from django.contrib.auth import SESSION_KEY, get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse


@override_settings(
    ACCOUNT_DEFAULT_HTTP_PROTOCOL="https",
    SOCIALACCOUNT_PROVIDERS={
        "google": {
            "APP": {"client_id": "migration-test-client", "secret": "migration-test-secret", "key": ""},
            "SCOPE": ["profile", "email"],
            "AUTH_PARAMS": {"access_type": "online"},
        }
    },
)
class AuthenticationTest(TestCase):
    def test_login_page_offers_both_login_methods(self):
        """Show Google and password login links on the account login page."""
        response = self.client.get(reverse("account_login"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, reverse("google_login"))
        self.assertContains(response, reverse("caidapp:djangologin"))

    def test_google_login_redirects_to_provider_with_callback(self):
        """Send the configured client, HTTPS callback, and state to Google."""
        response = self.client.get(reverse("google_login"))

        self.assertEqual(response.status_code, 302)
        destination = urlparse(response.url)
        self.assertEqual(destination.hostname, "accounts.google.com")
        parameters = parse_qs(destination.query)
        self.assertEqual(parameters["client_id"], ["migration-test-client"])
        self.assertEqual(parameters["redirect_uri"], ["https://testserver" + reverse("google_callback")])
        self.assertIn("state", parameters)

    def test_password_login_and_logout_preserve_session_behavior(self):
        """Create an authenticated session on login and clear it on logout."""
        user = get_user_model().objects.create_user(username="migration-test", password="test-password-52")

        response = self.client.post(
            reverse("caidapp:djangologin"),
            {"username": user.username, "password": "test-password-52"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.client.session[SESSION_KEY], str(user.pk))
        response = self.client.post(reverse("caidapp:logout_view"))
        self.assertEqual(response.status_code, 302)
        self.assertNotIn(SESSION_KEY, self.client.session)

    def test_invalid_password_does_not_create_authenticated_session(self):
        """Keep the session anonymous when password authentication fails."""
        get_user_model().objects.create_user(username="migration-test", password="test-password-52")

        response = self.client.post(
            reverse("caidapp:djangologin"),
            {"username": "migration-test", "password": "wrong-password"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertNotIn(SESSION_KEY, self.client.session)
