from django.test import TestCase
from django.urls import reverse

from caidapp.app_tests.factories import CaidUserFactory, NotificationFactory, NotificationRecipientFactory


class NotificationNavbarTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory()
        self.client.force_login(self.caiduser.user)

    def test_unread_notification_shows_indicator_in_top_navigation(self):
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
        NotificationRecipientFactory(
            notification=NotificationFactory(),
            user=self.caiduser,
            read=True,
        )

        response = self.client.get(reverse("caidapp:notifications"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "notification-unread-indicator")
        self.assertContains(response, 'aria-label="Notifications"')
