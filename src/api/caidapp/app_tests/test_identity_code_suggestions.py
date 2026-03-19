from django.test import TestCase
from django.urls import reverse

from caidapp.app_tests.factories import CaidUserFactory, IndividualIdentityFactory


class IdentityCodeSuggestionsViewTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory(admin=True)
        self.workgroup = self.caiduser.workgroup
        self.workgroup.identity_code_regex = r"B\d+"
        self.workgroup.save()
        self.client.force_login(self.caiduser.user)

    def test_view_filters_suggestions_by_workgroup_regex(self):
        matching = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="B75 Cumel")
        non_matching = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="NoCode Identity")
        other_workgroup = IndividualIdentityFactory(name="B610 Fernet")

        response = self.client.get(reverse("caidapp:show_identity_code_suggestions"))

        self.assertEqual(response.status_code, 200)
        identities = response.context["identities"]
        self.assertEqual([identity.id for identity in identities], [matching.id])
        self.assertContains(response, "B75 Cumel")
        self.assertNotContains(response, non_matching.name)
        self.assertNotContains(response, other_workgroup.name)

    def test_view_uses_custom_workgroup_regex(self):
        self.workgroup.identity_code_regex = r"[A-Z]{2}\d{2}"
        self.workgroup.save()
        matching = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="AB12 Kira")
        IndividualIdentityFactory(owner_workgroup=self.workgroup, name="B75 Cumel")

        response = self.client.get(reverse("caidapp:show_identity_code_suggestions"))

        self.assertEqual(response.status_code, 200)
        identities = response.context["identities"]
        self.assertEqual([identity.id for identity in identities], [matching.id])
        self.assertContains(response, "AB12 Kira")
        self.assertContains(response, self.workgroup.identity_code_regex)
