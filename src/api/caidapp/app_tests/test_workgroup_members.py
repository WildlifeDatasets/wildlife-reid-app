from django.test import TestCase
from django.urls import reverse

from caidapp import forms
from caidapp.app_tests.factories import CaidUserFactory, WorkGroupFactory


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
