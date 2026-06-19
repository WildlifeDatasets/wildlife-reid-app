from django.test import TestCase
from django.urls import reverse

from caidapp import forms
from caidapp.app_tests.factories import (
    AnimalObservationFactory,
    CaidUserFactory,
    IndividualIdentityFactory,
    MediaFileFactory,
    TaxonFactory,
    UploadedArchiveFactory,
    WorkGroupFactory,
)


class WorkgroupMemberManagementTest(TestCase):
    def setUp(self):
        self.workgroup = WorkGroupFactory(name="Research team")
        self.admin = CaidUserFactory(admin=True)
        self.admin.workgroup = self.workgroup
        self.admin.workgroup_admin = True
        self.admin.save(update_fields=["workgroup", "workgroup_admin"])
        self.member = CaidUserFactory()
        self.member.workgroup = self.workgroup
        self.member.workgroup_admin = False
        self.member.save(update_fields=["workgroup", "workgroup_admin"])
        self.outsider = CaidUserFactory()

    def test_workgroup_settings_no_longer_contains_user_selector(self):
        form = forms.WorkgroupForm(instance=self.workgroup)

        self.assertNotIn("caidusers", form.fields)

    def test_edit_permissions_are_enabled_by_default(self):
        user = CaidUserFactory()

        self.assertTrue(user.can_edit_taxon_data)
        self.assertTrue(user.can_edit_identity_data)
        self.assertTrue(user.can_edit_other_records)
        self.assertFalse(user.is_observer)

    def test_admin_sees_only_users_from_own_workgroup(self):
        self.client.force_login(self.admin.user)

        response = self.client.get(reverse("caidapp:workgroup_members"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, self.admin.user.username)
        self.assertContains(response, self.member.user.username)
        self.assertNotContains(response, self.outsider.user.username)
        self.assertContains(response, reverse("caidapp:workgroup_member_update", args=[self.member.pk]))

    def test_regular_member_cannot_open_workgroup_user_list(self):
        self.client.force_login(self.member.user)

        response = self.client.get(reverse("caidapp:workgroup_members"))

        self.assertEqual(response.status_code, 403)

    def test_admin_can_update_member_workflow_access(self):
        self.client.force_login(self.admin.user)

        response = self.client.post(
            reverse("caidapp:workgroup_member_update", args=[self.member.pk]),
            {"show_taxon_classification": "on"},
        )

        self.assertRedirects(response, reverse("caidapp:workgroup_members"))
        self.member.refresh_from_db()
        self.assertTrue(self.member.show_taxon_classification)
        self.assertFalse(self.member.show_reid)

    def test_admin_can_make_member_read_only_observer(self):
        self.client.force_login(self.admin.user)

        response = self.client.post(
            reverse("caidapp:workgroup_member_update", args=[self.member.pk]),
            {"is_observer": "on"},
        )

        self.assertRedirects(response, reverse("caidapp:workgroup_members"))
        self.member.refresh_from_db()
        self.assertTrue(self.member.is_observer)
        self.assertFalse(self.member.can_edit_taxon_data)
        self.assertFalse(self.member.can_edit_identity_data)
        self.assertFalse(self.member.can_edit_other_records)

    def test_workgroup_admin_cannot_be_made_observer(self):
        self.client.force_login(self.admin.user)

        response = self.client.post(
            reverse("caidapp:workgroup_member_update", args=[self.admin.pk]),
            {"is_observer": "on"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "A workgroup admin cannot be a read-only observer.")
        self.admin.refresh_from_db()
        self.assertFalse(self.admin.is_observer)

    def test_observer_cannot_submit_changes(self):
        self.member.is_observer = True
        self.member.save(update_fields=["is_observer"])
        self.client.force_login(self.member.user)

        response = self.client.post(reverse("caidapp:update_caiduser"), {"timezone": "UTC"})

        self.assertEqual(response.status_code, 403)
        self.assertContains(response, "Editing not allowed", status_code=403)
        self.assertContains(response, "alert alert-warning", status_code=403)
        self.assertEqual(response.content.count(b"Read-only observers cannot modify records."), 1)

    def test_member_without_other_permission_cannot_change_other_records(self):
        self.member.can_edit_other_records = False
        self.member.save(update_fields=["can_edit_other_records"])
        self.client.force_login(self.member.user)

        response = self.client.post(reverse("caidapp:update_caiduser"), {"timezone": "UTC"})

        self.assertEqual(response.status_code, 403)

    def test_member_without_taxon_permission_cannot_update_taxon(self):
        taxon = TaxonFactory()
        self.member.can_edit_taxon_data = False
        self.member.save(update_fields=["can_edit_taxon_data"])
        self.client.force_login(self.member.user)

        response = self.client.post(reverse("caidapp:update_taxon", args=[taxon.pk]), {"name": "Changed"})

        self.assertEqual(response.status_code, 403)
        taxon.refresh_from_db()
        self.assertNotEqual(taxon.name, "Changed")

    def test_member_without_identity_permission_cannot_update_identity(self):
        identity = IndividualIdentityFactory(owner_workgroup=self.workgroup)
        self.member.can_edit_identity_data = False
        self.member.save(update_fields=["can_edit_identity_data"])
        self.client.force_login(self.member.user)

        response = self.client.post(
            reverse("caidapp:individual_identity_update", args=[identity.pk]),
            {"name": "Changed"},
        )

        self.assertEqual(response.status_code, 403)
        identity.refresh_from_db()
        self.assertNotEqual(identity.name, "Changed")

    def test_member_without_identity_permission_cannot_tamper_via_taxon_form(self):
        archive = UploadedArchiveFactory(owner=self.member)
        mediafile = MediaFileFactory(parent=archive)
        taxon = TaxonFactory()
        current_identity = IndividualIdentityFactory(owner_workgroup=self.workgroup)
        other_identity = IndividualIdentityFactory(owner_workgroup=self.workgroup)
        observation = AnimalObservationFactory(
            mediafile=mediafile,
            taxon=taxon,
            identity=current_identity,
        )
        self.member.can_edit_taxon_data = True
        self.member.can_edit_identity_data = False
        self.member.save(update_fields=["can_edit_taxon_data", "can_edit_identity_data"])
        self.client.force_login(self.member.user)

        response = self.client.post(
            reverse("caidapp:missing_taxon_annotation_for_mediafile", args=[mediafile.pk]),
            {
                "observations-TOTAL_FORMS": "1",
                "observations-INITIAL_FORMS": "1",
                "observations-MIN_NUM_FORMS": "0",
                "observations-MAX_NUM_FORMS": "1000",
                "observations-0-id": str(observation.pk),
                "observations-0-taxon": str(taxon.pk),
                "observations-0-identity": str(other_identity.pk),
            },
        )

        self.assertEqual(response.status_code, 403)
        observation.refresh_from_db()
        self.assertEqual(observation.identity, current_identity)

    def test_admin_cannot_update_user_from_another_workgroup(self):
        self.client.force_login(self.admin.user)

        response = self.client.post(
            reverse("caidapp:workgroup_member_update", args=[self.outsider.pk]),
            {"show_taxon_classification": "on", "show_reid": "on"},
        )

        self.assertEqual(response.status_code, 404)

    def test_regular_member_cannot_change_own_workflow_access(self):
        self.client.force_login(self.member.user)
        self.member.refresh_from_db()
        self.member.show_taxon_classification = True
        self.member.show_reid = True
        self.member.save(update_fields=["show_taxon_classification", "show_reid"])

        response = self.client.post(
            reverse("caidapp:update_caiduser"),
            {
                "show_taxon_classification": "",
                "show_reid": "",
                "timezone": self.member.timezone,
            },
        )

        self.assertEqual(response.status_code, 302)
        self.member.refresh_from_db()
        self.assertTrue(self.member.show_taxon_classification)
        self.assertTrue(self.member.show_reid)

    def test_admin_can_change_own_workflow_access(self):
        self.client.force_login(self.admin.user)
        self.admin.refresh_from_db()
        self.admin.show_taxon_classification = True
        self.admin.show_reid = True
        self.admin.save(update_fields=["show_taxon_classification", "show_reid"])

        response = self.client.post(
            reverse("caidapp:update_caiduser"),
            {"timezone": self.admin.timezone},
        )

        self.assertEqual(response.status_code, 302)
        self.admin.refresh_from_db()
        self.assertFalse(self.admin.show_taxon_classification)
        self.assertFalse(self.admin.show_reid)
