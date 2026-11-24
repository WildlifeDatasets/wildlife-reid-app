from unittest.mock import patch

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
from django.core.exceptions import ValidationError
from django.db import transaction
from django.test import TestCase
from django.urls import reverse


class WorkgroupMigrationTest(TestCase):
    def setUp(self):
        """Prepare a personal workgroup with owned data and a target workgroup for migration tests."""
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
        self.identification = models.MediafilesForIdentification.objects.create(
            observation=self.mediafile.observations.first()
        )

    def test_migrates_data_and_deletes_personal_workgroup(self):
        """Move personal workgroup data, reset access and identification state, and delete the emptied group."""
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

    @patch("caidapp.tasks.init_identification.apply_async")
    def test_rejects_migration_from_multi_user_workgroup(self, dispatch):
        """Reject migration out of a multi-user workgroup without moving the member."""
        other_user = CaidUserFactory()
        other_user.workgroup = self.source
        other_user.save(update_fields=["workgroup"])
        with self.captureOnCommitCallbacks(execute=True):
            with self.assertRaisesMessage(ValidationError, "one-member personal workgroup"):
                migrate_user_to_workgroup(user=self.user, target_workgroup=self.target)
        dispatch.assert_not_called()
        self.user.refresh_from_db()
        self.assertEqual(self.user.workgroup, self.source)

    @patch("caidapp.tasks.init_identification.apply_async")
    def test_schedules_target_reid_refresh_after_commit(self, dispatch):
        """Queue a target-group rebuild only after the transferred data is committed."""
        dispatch.return_value.id = "migration-init-task"
        with self.captureOnCommitCallbacks(execute=True):
            migrate_user_to_workgroup(user=self.user, target_workgroup=self.target)
            dispatch.assert_not_called()

        dispatch.assert_called_once()
        self.assertEqual(dispatch.call_args.kwargs["args"], [self.target.pk])
        self.target.refresh_from_db()
        self.assertEqual(self.target.identification_scheduled_init_task_id, "migration-init-task")
        self.assertEqual(self.target.identification_init_status, "Scheduled")
        self.identity.refresh_from_db()
        self.assertEqual(self.identity.owner_workgroup, self.target)

    @patch("caidapp.tasks.init_identification.apply_async")
    def test_rollback_does_not_schedule_reid_refresh(self, dispatch):
        """Discard the rebuild when an enclosing invitation transaction rolls back."""
        with self.captureOnCommitCallbacks(execute=True):
            with self.assertRaisesMessage(ValueError, "Rollback invitation"):
                with transaction.atomic():
                    migrate_user_to_workgroup(user=self.user, target_workgroup=self.target)
                    raise ValueError("Rollback invitation")

        dispatch.assert_not_called()
        self.user.refresh_from_db()
        self.assertEqual(self.user.workgroup, self.source)
        self.assertTrue(models.WorkGroup.objects.filter(pk=self.source.pk).exists())

    @patch("caidapp.tasks.init_identification.apply_async", side_effect=RuntimeError("Broker unavailable"))
    def test_dispatch_failure_keeps_completed_migration(self, dispatch):
        """Log a scheduling failure without reporting a completed migration as failed."""
        # captureOnCommitCallbacks logs robust callback errors via django.test.
        with self.assertLogs("django.test", level="ERROR"):
            with self.captureOnCommitCallbacks(execute=True):
                migrate_user_to_workgroup(user=self.user, target_workgroup=self.target)

        dispatch.assert_called_once()
        self.user.refresh_from_db()
        self.assertEqual(self.user.workgroup, self.target)


class WorkgroupInvitationViewTest(TestCase):
    def setUp(self):
        """Authenticate an admin of the target group and create an external user to invite."""
        self.target = WorkGroupFactory()
        self.admin = CaidUserFactory()
        self.admin.workgroup = self.target
        self.admin.workgroup_admin = True
        self.admin.save(update_fields=["workgroup", "workgroup_admin"])
        self.invited = CaidUserFactory()
        self.client.force_login(self.admin.user)

    def test_create_by_email_sends_linked_notification(self):
        """Resolve an invitee by email and create a notification linked to the invitation detail."""
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

    @patch("caidapp.tasks.schedule_init_identification_for_workgroup")
    def test_acceptance_schedules_refresh_after_invitation_is_accepted(self, schedule):
        """Schedule the rebuild after both membership and invitation status are saved."""
        invitation = models.WorkGroupInvitation.objects.create(
            invited_user=self.invited, invited_by=self.admin, target_workgroup=self.target
        )
        self.client.force_login(self.invited.user)

        def check_accepted_state(workgroup):
            invitation.refresh_from_db()
            self.invited.refresh_from_db()
            self.assertEqual(invitation.status, "accepted")
            self.assertEqual(self.invited.workgroup, workgroup)

        schedule.side_effect = check_accepted_state
        with self.captureOnCommitCallbacks(execute=True):
            response = self.client.post(reverse("caidapp:workgroup_invitation_accept", args=[invitation.pk]))
            schedule.assert_not_called()

        self.assertRedirects(response, reverse("caidapp:workgroup_invitations_for_user"))
        schedule.assert_called_once_with(self.target)
