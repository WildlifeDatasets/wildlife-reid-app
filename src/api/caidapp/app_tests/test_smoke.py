import logging

from caidapp.models import WorkGroup, WorkGroupInvitation
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import NoReverseMatch, URLPattern, URLResolver, reverse

from caidapp.app_tests.factories import (
    UserFactory,
    WorkGroupFactory,
    WorkGroupInvitationFactory,
    UploadedArchiveFactory,
    MediaFileFactory, AnimalObservationFactory, IndividualIdentityFactory, LocalityFactory,
    NotificationFactory, NotificationRecipientFactory
)

logger = logging.getLogger(__name__)

User = get_user_model()


class UrlSmokeTest(TestCase):
    # def setUp(self):
    #     """Initial data for tests."""
    #     # User
    #     self.user = User.objects.create_user(
    #         username="smoke",
    #         password="secret123",
    #     )
    #     self.user2 = User.objects.create_user(
    #         username="smoke2",
    #         password="secret123",
    #     )
    #
    #     # self.client.login(
    #     #     username="smoke", password="secret123"
    #     # )
    #     self.client.force_login(self.user) # faster
    #
    #     # WorkGroup
    #     self.workgroup = WorkGroup.objects.create(name="WG1")
    #
    #     logger.debug(f"{self.user=}, {self.workgroup=}")
    #     assert self.user.caiduser is not None
    #     self.user.caiduser.workgroup = self.workgroup
    #     self.user.caiduser.workgroup_admin = True
    #
    #     self.user.caiduser.save()
    #
    #     self.wg_invitation = WorkGroupInvitation.objects.create(
    #         invited_user=self.user.caiduser,
    #         invited_by=self.user2.caiduser,
    #         target_workgroup=self.workgroup,
    #         status="pending",
    #     )

    def setUp(self):

        self.user = UserFactory()
        self.user2 = UserFactory()
        self.user.is_staff = True
        self.user.save()

        self.caiduser = self.user.caiduser
        self.caiduser2 = self.user2.caiduser


        self.workgroup = WorkGroupFactory()

        self.caiduser.workgroup = self.workgroup
        self.caiduser.workgroup_admin = True
        self.caiduser.import_dir = "/tmp/caid_import"  # nastavit import_dir pro testy
        self.caiduser.save()

        self.client.force_login(self.user)


        # invitation
        self.wg_invitation = WorkGroupInvitationFactory(
            invited_user=self.caiduser,
            invited_by=self.caiduser2,
            target_workgroup=self.workgroup,
        )

        # minimální dataset pro smoke test
        self.archive = UploadedArchiveFactory(owner=self.caiduser)
        self.mediafile = MediaFileFactory(parent=self.archive)
        self.observation = AnimalObservationFactory(mediafile=self.mediafile)
        self.identity = IndividualIdentityFactory(owner_workgroup=self.workgroup)
        MediaFileFactory.create_batch(3, parent=self.archive, identity=self.identity)
        self.identities = IndividualIdentityFactory.create_batch(3, owner_workgroup=self.workgroup)
        self.locality = LocalityFactory(owner=self.caiduser)
        self.notification = NotificationFactory()

        self.recipient = NotificationRecipientFactory(
            notification=self.notification,
            user=self.caiduser
        )
        # self.locality = LocalityFactory(owner_workgroup=self.workgroup)


        # self.wg_invitation.save()

    # def test_all_named_urls(self):
    #     """Projde všechny pojmenované URL a zkusí GET.
    #     Akceptuje status 200 nebo redirect (302)."""
    #
    #     resolver = get_resolver()
    #
    #     for pattern in resolver.url_patterns:
    #         print(f"Pattern: {pattern}")
    #         name = getattr(pattern, "name", None)
    #         logger.debug(f"resolver {name=}, {pattern=}")
    #         if not name:
    #             continue
    #
    #         try:
    #             url = reverse(name, kwargs=self._build_kwargs(pattern))
    #         except NoReverseMatch:
    #             continue
    #
    #         response = self.client.get(url)
    #         self.assertIn(
    #             response.status_code,
    #             [200, 302],
    #             f"{url} ({name}) failed with status {response.status_code}"
    #         )

    def test_all_named_caidapp_urls(self):
        """Go over all named URLs in caidapp and try GET."""
        from caidapp import urls as caidapp_urls

        patterns = list(self._walk_patterns(caidapp_urls.urlpatterns))
        tested = 0

        for pattern in patterns:
            name = getattr(pattern, "name", None)
            if not name:
                continue
            if "logout" in name:  # logout URL může být přístupná, ale není potřeba ji testovat
                continue
            if "sample_data" in name:  # sample data URL může být přístupná, ale není potřeba ji testovat
                continue

            try:
                url = reverse(f"caidapp:{name}", kwargs=self._build_kwargs(pattern))
            except NoReverseMatch:
                continue

            response = self.client.get(url)
            self.assertIn(
                response.status_code,
                [200, 302, 403],  # 403 pro neautorizované přístupy, které jsou v pořádku
                f"{url} ({name}) failed with {response.status_code}",
            )
            tested += 1
            print(f"✓ {name}: {response.status_code}")

        print(f"Tested {tested} caidapp URLs")

    def _walk_patterns(self, patterns):
        """Rekurzivně projde všechny URLPattern a URLResolver."""
        for pattern in patterns:
            if isinstance(pattern, URLPattern):
                yield pattern
            elif isinstance(pattern, URLResolver):
                yield from self._walk_patterns(pattern.url_patterns)

    def _build_kwargs(self, pattern):
        """Vrátí základní kwargs pro URL s parametry."""
        kwargs = {}
        regex = str(pattern.pattern)
        if "<int:pk>" in regex:
            kwargs["pk"] = 1
        if "<int:id>" in regex:
            kwargs["id"] = 1
        if "<int:workgroup_pk>" in regex:
            kwargs["workgroup_pk"] = self.workgroup.pk
        # if "<int:workstation_pk>" in regex:
        #     kwargs["workstation_pk"] = self.ws.pk
        # if "<int:process_pk>" in regex:
        #     kwargs["process_pk"] = self.process.pk
        # if "<int:issue_id>" in regex:
        #     kwargs["issue_id"] = self.issue.pk
        # if "<int:test_id>" in regex:
        #     kwargs["test_id"] = self.tech_test.pk
        return kwargs
