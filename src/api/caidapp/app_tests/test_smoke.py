from django.test import TestCase
from django.urls import get_resolver, reverse, NoReverseMatch, URLPattern, URLResolver
from django.contrib.auth import get_user_model

from caidapp.models import (
    WorkGroup, UploadedArchive, MediaFile, IndividualIdentity,
)
import logging

logger = logging.getLogger(__name__)

User = get_user_model()




class UrlSmokeTest(TestCase):
    def setUp(self):
        # User
        self.user = User.objects.create_user(
            username="smoke",
            password="secret123",
        )
        self.client.login(username="smoke", password="secret123")

        # WorkGroup
        self.workgroup = WorkGroup.objects.create(name="WG1")

        logger.debug(f"{self.user=}, {self.workgroup=}")
        assert self.user.caiduser is not None
        self.user.caiduser.workgroup = self.workgroup

        self.user.caiduser.save()

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
        from caidapp import urls as caidapp_urls

        patterns = list(self._walk_patterns(caidapp_urls.urlpatterns))
        tested = 0

        for pattern in patterns:
            name = getattr(pattern, "name", None)
            if not name:
                continue

            try:
                url = reverse(f"caidapp:{name}", kwargs=self._build_kwargs(pattern))
            except NoReverseMatch:
                continue

            response = self.client.get(url)
            self.assertIn(
                response.status_code,
                [200, 302],
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