from django.test import TestCase
from django.urls import reverse

from caidapp.app_tests.factories import CaidUserFactory, LocalityFactory, MediaFileFactory


class DeleteLocalityTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory(admin=True)
        self.client.force_login(self.caiduser.user)

    def test_delete_locality_keeps_linked_mediafiles_and_clears_locality(self):
        locality = LocalityFactory(owner=self.caiduser, name="Old Meadow")
        mediafile = MediaFileFactory(parent__owner=self.caiduser, locality=locality)
        url = reverse("caidapp:delete_locality", kwargs={"locality_id": locality.id})

        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "You are about to delete locality")
        self.assertContains(response, "1")
        self.assertTrue(locality.__class__.objects.filter(id=locality.id).exists())

        response = self.client.post(url)

        self.assertRedirects(response, reverse("caidapp:localities"))
        self.assertFalse(locality.__class__.objects.filter(id=locality.id).exists())
        mediafile.refresh_from_db()
        self.assertIsNone(mediafile.locality)
        self.assertEqual(mediafile.parent.owner, self.caiduser)
