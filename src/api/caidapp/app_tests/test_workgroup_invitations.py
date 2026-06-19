from django.test import TestCase
from django.urls import reverse
from django.core.exceptions import ValidationError

from caidapp import models
from caidapp.app_tests.factories import (
    AlbumFactory,
    CaidUserFactory,
    IndividualIdentityFactory,
    LocalityFactory,
    MediaFileFactory,
    UploadedArchiveFactory,
    WorkGroupFactory,
)
from caidapp.services.workgroup_migration import migrate_user_to_workgroup


class WorkgroupMigrationTest(TestCase):
    def setUp(self):
        self.source = WorkGroupFactory()
        self.target = WorkGroupFactory()
        self.user = CaidUserFactory()
        self.user.workgroup = self.source
        self.user.workgroup_admin = True
        self.user.save(update_fields=["workgroup", "workgroup_admin"])
        self.archive = UploadedArchiveFactory(owner=self.user, identification_status="IAIP")
        self.mediafile = MediaFileFactory(parent=self.archive)
        self.locality = LocalityFactory(owner=self.user)
        self.album = AlbumFactory(owner=self.user)
        self.identity = IndividualIdentityFactory(owner_workgroup=self.source)
        self.identification = models.MediafilesForIdentification.objects.create(mediafile=self.mediafile)

    def test_migrates_data_and_deletes_personal_workgroup(self):
        migrate_user_to_workgroup(user=self.user, target_workgroup=self.target)

        self.user.refresh_from_db()
        self.identity.refresh_from_db()
        self.archive.refresh_from_db()
        self.assertEqual(self.user.workgroup, self.target)
        self.assertFalse(self.user.workgroup_admin)
        self.assertFalse(self.user.can_edit_taxon_data)
        self.assertFalse(self.user.can_edit_identity_data)
        self.assertTrue(self.user.can_edit_other_records)
        self.assertFalse(self.user.is_observer)
        self.assertEqual(self.identity.owner_workgroup, self.target)
        self.assertFalse(models.WorkGroup.objects.filter(pk=self.source.pk).exists())
        self.assertFalse(models.MediafilesForIdentification.objects.filter(pk=self.identification.pk).exists())
        self.assertEqual(self.archive.identification_status, "C")
        self.assertTrue(models.Locality.objects.filter(pk=self.locality.pk, owner=self.user).exists())
        self.assertTrue(models.Album.objects.filter(pk=self.album.pk, owner=self.user).exists())

    def test_rejects_migration_from_multi_user_workgroup(self):
        other_user = CaidUserFactory()
        other_user.workgroup = self.source
        other_user.save(update_fields=["workgroup"])
        with self.assertRaisesMessage(ValidationError, "one-member personal workgroup"):
            migrate_user_to_workgroup(user=self.user, target_workgroup=self.target)
        self.user.refresh_from_db()
        self.assertEqual(self.user.workgroup, self.source)


class WorkgroupInvitationViewTest(TestCase):
    def setUp(self):
        self.target = WorkGroupFactory()
        self.admin = CaidUserFactory()
        self.admin.workgroup = self.target
        self.admin.workgroup_admin = True
        self.admin.save(update_fields=["workgroup", "workgroup_admin"])
        self.invited = CaidUserFactory()
        self.client.force_login(self.admin.user)

    def test_create_by_email_sends_linked_notification(self):
        response = self.client.post(
            reverse("caidapp:workgroup_invitation"),
            {"user_identifier": self.invited.user.email},
        )
        self.assertRedirects(response, reverse("caidapp:workgroup_invitations"))
        invitation = models.WorkGroupInvitation.objects.get(invited_user=self.invited)
        notification = models.Notification.objects.get(recipients__user=self.invited)
        self.assertEqual(
            notification.get_link_url(),
            reverse("caidapp:workgroup_invitation_detail", kwargs={"pk": invitation.pk}),
        )
