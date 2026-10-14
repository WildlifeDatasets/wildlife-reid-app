from caidapp.app_tests.factories import CaidUserFactory, NotificationFactory, NotificationRecipientFactory
from django.test import TestCase
from django.urls import reverse


class NotificationNavbarTest(TestCase):
    def setUp(self):
        """Authenticate a notification recipient for navigation and read-state tests."""
        self.caiduser = CaidUserFactory()
        self.client.force_login(self.caiduser.user)

    def test_unread_notification_shows_indicator_in_top_navigation(self):
        """Show one accessible bell with an unread indicator when unread notifications exist."""
        NotificationRecipientFactory(
            notification=NotificationFactory(),
            user=self.caiduser,
            read=False,
        )

        response = self.client.get(reverse("caidapp:notifications"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "notification-unread-indicator")
        self.assertContains(response, 'aria-label="Notifications, unread notifications available"')
        self.assertEqual(response.content.count(b'<i class="bi bi-bell"></i>'), 1)

    def test_read_notifications_do_not_show_indicator(self):
        """Omit the unread badge when all notifications have been read."""
        NotificationRecipientFactory(
            notification=NotificationFactory(),
            user=self.caiduser,
            read=True,
        )

        response = self.client.get(reverse("caidapp:notifications"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "notification-unread-indicator")
        self.assertContains(response, 'aria-label="Notifications"')

    def test_mark_all_notifications_as_read_only_updates_current_user(self):
        """Mark only the current user's notifications as read and record read timestamps."""
        own_unread = NotificationRecipientFactory(
            notification=NotificationFactory(),
            user=self.caiduser,
            read=False,
        )
        other_user = CaidUserFactory()
        other_unread = NotificationRecipientFactory(
            notification=NotificationFactory(),
            user=other_user,
            read=False,
        )

        response = self.client.post(reverse("caidapp:notifications-mark-all-as-read"))

        self.assertRedirects(response, reverse("caidapp:notifications"))
        own_unread.refresh_from_db()
        other_unread.refresh_from_db()
        self.assertTrue(own_unread.read)
        self.assertIsNotNone(own_unread.read_at)
        self.assertFalse(other_unread.read)

    def test_notification_list_includes_mark_all_as_read_button(self):
        """Offer a mark-all-read action on the notification list."""
        response = self.client.get(reverse("caidapp:notifications"))

        self.assertContains(response, "Mark all as read")
        self.assertContains(
            response,
            reverse("caidapp:notifications-mark-all-as-read"),
        )
