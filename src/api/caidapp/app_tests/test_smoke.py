import logging

from caidapp.models import ObservationImport, WorkGroup, WorkGroupInvitation
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import NoReverseMatch, URLPattern, URLResolver, reverse

from caidapp.app_tests.factories import (
    UserFactory,
    WorkGroupFactory,
    WorkGroupInvitationFactory,
    UploadedArchiveFactory,
    MediaFileFactory, AnimalObservationFactory, IndividualIdentityFactory, LocalityFactory,
    NotificationFactory, NotificationRecipientFactory, TaxonFactory
)

logger = logging.getLogger(__name__)

User = get_user_model()


class UrlSmokeTest(TestCase):
    MUTATING_NAMES = {
        "clear_identity_suggestions",
        "exclude_merge_identity_suggestion",
        "notifications-mark-all-as-read",
        "toggle_identity_representative",
    }
    SKIPPED_VIEWS = {
        "stream_video",
        # These downloads correctly return 404 until an identification run has
        # produced its workgroup-specific CSV artifact.
        "download_init_identification_csv",
        "download_run_identification_csv",
        # A ZIP export requires an explicit sequence selection in the request.
        "download_zip_for_sequences",
    }

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
        self.manual_identification_taxon = TaxonFactory()
        self.manual_identification_archive = UploadedArchiveFactory(
            owner=self.caiduser,
            is_for_identification=True,
            import_finished=True,
            taxon_for_identification=self.manual_identification_taxon,
        )
        self.manual_identification_mediafile = MediaFileFactory(parent=self.manual_identification_archive)
        AnimalObservationFactory(
            mediafile=self.manual_identification_mediafile,
            taxon=self.manual_identification_taxon,
            identity=None,
        )
        self.observation_import = ObservationImport.objects.create(
            caiduser=self.caiduser,
            source_filename="smoke-test.csv",
            stored_file="observation_imports/smoke-test.csv",
        )
        self.identity = IndividualIdentityFactory(owner_workgroup=self.workgroup)
        MediaFileFactory.create_batch(3, parent=self.archive, identity=self.identity)
        self.identities = IndividualIdentityFactory.create_batch(3, owner_workgroup=self.workgroup)
        self.locality = LocalityFactory(owner=self.caiduser)
        self.notification = NotificationFactory()

        self.recipient = NotificationRecipientFactory(
            notification=self.notification,
            user=self.caiduser
        )

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

            if name in self.MUTATING_NAMES:
                continue
            if name in self.SKIPPED_VIEWS:
                continue
            if "delete" in name:
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
            kwargs["pk"] = self._get_existing_pk_for_pattern(pattern)
        if "<int:id>" in regex:
            kwargs["id"] = 1
        if "<int:mediafile_id>" in regex:
            kwargs["mediafile_id"] = self.mediafile.pk
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

    def _get_existing_pk_for_pattern(self, pattern):
        """Return an existing primary key for URLs that need a real object."""
        name = getattr(pattern, "name", "")
        if name == "media_file_update":
            return self.mediafile.pk
        if name == "manual_identification_mediafile":
            return self.manual_identification_mediafile.pk
        if name == "missing_taxon_annotation_for_mediafile":
            return self.mediafile.pk
        if name == "toggle_identity_representative":
            return self.mediafile.pk
        if name == "individual_identity_update":
            return self.identity.pk
        if name == "notification-update":
            return self.notification.pk
        if name == "notification-detail":
            return self.notification.pk
        if name == "notification-delete":
            return self.notification.pk
        if name == "observation_delete":
            return self.observation.pk
        if name == "workgroup_member_update":
            return self.caiduser.pk
        if name in {
            "workgroup_invitation_detail",
            "workgroup_invitation_accept",
            "workgroup_invitation_decline",
        }:
            return self.wg_invitation.pk
        if name == "observation_import_status":
            return self.observation_import.pk
        if name == "workgroup-update":
            return self.workgroup.pk
        return 1
