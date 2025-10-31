from django.test import TestCase
from django.urls import reverse
from django.contrib.auth import get_user_model
from caidapp import models
from caidapp.models import WorkGroup
import logging

logger = logging.getLogger(__name__)


User = get_user_model()


class BasicFlowTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="testuser",
            password="secret123",
        )
        self.client = self.client  # vestavěný testovací klient
        self.wg = models.WorkGroup.objects.create(name="WG1")

    def test_login_and_view(self):
        # login
        login = self.client.login(username="testuser", password="secret123")
        self.assertTrue(login)  # ověř, že přihlášení fungovalo

        # přístup na redirect page
        response = self.client.get(reverse("caidapp:index"))
        self.assertEqual(response.status_code, 302)
        # self.assertContains(response, "WS1")

        response = self.client.get(reverse("caidapp:home"))
        self.assertEqual(response.status_code, 200)

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
