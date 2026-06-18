from django.test import TestCase
from django.urls import reverse

from caidapp import models
from caidapp.app_tests.factories import (
    CaidUserFactory,
    IndividualIdentityFactory,
    MediaFileFactory,
    UploadedArchiveFactory,
)
from caidapp.services.workgroup_next_steps import build_next_steps


class WorkgroupNextStepsServiceTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory(admin=True)
        self.workgroup = self.caiduser.workgroup
        self.workgroup.check_taxon_before_identification = False
        self.workgroup.save(update_fields=["check_taxon_before_identification"])

    def test_build_next_steps_for_empty_workgroup(self):
        steps = build_next_steps(self.workgroup)

        self.assertEqual([step.code for step in steps[:2]], ["no_uploads", "no_identities"])
        self.assertEqual(steps[0].url, reverse("caidapp:new_upload"))

    def test_manual_identification_is_next_for_unidentified_identification_upload(self):
        archive = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
        )
        MediaFileFactory(parent=archive, identity=None)

        steps = build_next_steps(self.workgroup)

        self.assertEqual(steps[0].code, "manual_identification")
        self.assertEqual(steps[0].url, reverse("caidapp:manual_identification"))

    def test_build_next_steps_for_existing_identities(self):
        UploadedArchiveFactory(owner=self.caiduser)
        identity_without_media = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="Solo Identity")
        identity_with_code = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="B75 Cumel")
        identity_with_media = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="Fernet")
        models.MediaFile.objects.create(
            parent=UploadedArchiveFactory(owner=self.caiduser),
            identity=identity_with_media,
            media_type="image",
            mediafile="images/test.jpg",
            image_file="images/test.jpg",
            preview="previews/test.jpg",
            original_filename="test.jpg",
        )
        models.MergeIdentitySuggestionResult.objects.create(
            workgroup=self.workgroup,
            suggestions=[(identity_with_code.id, identity_with_media.id, 1)],
        )

        steps = build_next_steps(self.workgroup)
        step_codes = [step.code for step in steps]

        self.assertIn("identities_without_mediafiles", step_codes)
        self.assertIn("identities_without_representatives", step_codes)
        self.assertIn("identity_code_suggestions_available", step_codes)
        self.assertIn("merge_suggestions_available", step_codes)
        self.assertNotIn("no_uploads", step_codes)
        self.assertNotIn("no_identities", step_codes)
        self.assertEqual(steps[0].code, "identities_without_mediafiles")
        self.assertEqual(identity_without_media.count_of_mediafiles(), 0)


class DashIdentitiesNextStepViewTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory(admin=True)
        self.caiduser.workgroup.check_taxon_before_identification = False
        self.caiduser.workgroup.save(update_fields=["check_taxon_before_identification"])
        self.client.force_login(self.caiduser.user)

    def test_dashboard_offers_manual_identification_for_unidentified_upload(self):
        archive = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
        )
        MediaFileFactory(parent=archive, identity=None)

        response = self.client.get(reverse("caidapp:dash_identities"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["primary_next_step"].code, "manual_identification")
        self.assertEqual(response.context["manual_identification_count"], 1)
        self.assertContains(response, reverse("caidapp:manual_identification"))
        self.assertContains(response, "Manual identification (1)")

    def test_dashboard_contains_primary_next_step(self):
        response = self.client.get(reverse("caidapp:dash_identities"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["primary_next_step"].code, "no_uploads")
        self.assertContains(response, "Suggested next step")
